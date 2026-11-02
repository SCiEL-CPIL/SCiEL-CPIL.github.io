import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 115.0

    n, m, p = 3, 4, 6

    def time_ok():
        return time.time() - start_time < DEADLINE

    def add_strassen_2x2x2(i1, i2, j1, j2, k1, k2, terms):
        """Standard Strassen 2x2x2"""
        a, b = (i1, j1), (i1, j2)
        c, d = (i2, j1), (i2, j2)
        e, f = (j1, k1), (j1, k2)
        g, h = (j2, k1), (j2, k2)
        C11, C12 = (i1, k1), (i1, k2)
        C21, C22 = (i2, k1), (i2, k2)
        
        terms.append(({a: 1, d: 1}, {e: 1, h: 1}, {C11: 1, C22: 1}))
        terms.append(({c: 1, d: 1}, {e: 1}, {C21: 1, C22: -1}))
        terms.append(({a: 1}, {f: 1, h: -1}, {C12: 1, C22: 1}))
        terms.append(({d: 1}, {g: 1, e: -1}, {C11: 1, C21: 1}))
        terms.append(({a: 1, b: 1}, {h: 1}, {C11: -1, C12: 1}))
        terms.append(({c: 1, a: -1}, {e: 1, f: 1}, {C22: 1}))
        terms.append(({b: 1, d: -1}, {g: 1, h: 1}, {C11: 1}))

    def add_schoolbook(i, j, k, terms):
        terms.append((
            {(i, j): 1}, {(j, k): 1}, {(i, k): 1}
        ))

    def build_and_verify(strategy):
        """Build decomposition, verify, return (U, V, W, rank) or None"""
        terms = []
        
        if strategy == 0:
            # Current best: row 0 schoolbook, rows 1-2 Strassen
            for j in range(m):
                for k in range(p):
                    add_schoolbook(0, j, k, terms)
            for j1 in [0, 2]:
                for k1 in [0, 2, 4]:
                    add_strassen_2x2x2(1, 2, j1, j1 + 1, k1, k1 + 1, terms)
        
        elif strategy == 1:
            # Alternative: rows 0-1 Strassen, row 2 schoolbook
            for j in range(m):
                for k in range(p):
                    add_schoolbook(2, j, k, terms)
            for j1 in [0, 2]:
                for k1 in [0, 2, 4]:
                    add_strassen_2x2x2(0, 1, j1, j1 + 1, k1, k1 + 1, terms)
        
        elif strategy == 2:
            # Try: use Strassen on rows 0-1 for cols 0-1, rows 1-2 for cols 2-3
            # This creates overlap on row 1 which we need to handle
            # Actually this won't work cleanly due to overlap
            # Instead: try a different partition
            # Rows 0-1: Strassen on 2x2x2 blocks
            # Row 2: schoolbook
            # But try different col/row splits
            for j in range(m):
                for k in range(p):
                    add_schoolbook(2, j, k, terms)
            # Rows 0-1, cols 0-1
            for k1 in [0, 2, 4]:
                add_strassen_2x2x2(0, 1, 0, 1, k1, k1 + 1, terms)
            # Rows 0-1, cols 2-3
            for k1 in [0, 2, 4]:
                add_strassen_2x2x2(0, 1, 2, 3, k1, k1 + 1, terms)
        
        elif strategy == 3:
            # Try: 3x2x2 blocks using a custom decomposition
            # For A[0:3, 0:2] * B[0:2, k1:k2] -> C[0:3, k1:k2]
            # Schoolbook: 3*2*2 = 12 terms
            # Can we do better? Try a hybrid
            # Use Strassen for rows 0-1, schoolbook for row 2
            for k1 in [0, 2, 4]:
                add_strassen_2x2x2(0, 1, 0, 1, k1, k1 + 1, terms)
                add_strassen_2x2x2(0, 1, 2, 3, k1, k1 + 1, terms)
                for j in range(2):
                    for k in range(6):
                        add_schoolbook(2, j, k, terms)
                # But we double-counted row 2... need to fix
                # This approach is flawed. Let me try differently.
            # Actually, let's just try: all 3 rows with Strassen where possible
            # Rows 0-1: Strassen, Row 2: schoolbook (same as strategy 1)
            # Let me try a genuinely different approach:
            # Use the fact that 3x4x6 = 3x(4x6) and try to optimize 4x6 first
            # 4x6 can be seen as 2x2 x 2x3 blocks
            # For each 2x2 A block and 2x3 B block, find min rank
            # 2x2x3 schoolbook = 12, Strassen-like = ?
            # Let's just use the same as strategy 0 for now
            terms.clear()
            for j in range(m):
                for k in range(p):
                    add_schoolbook(0, j, k, terms)
            for j1 in [0, 2]:
                for k1 in [0, 2, 4]:
                    add_strassen_2x2x2(1, 2, j1, j1 + 1, k1, k1 + 1, terms)
        
        rank = len(terms)
        
        # Build matrices
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j][col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k][col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i][col] = coeff
        
        # Verify
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                    if abs(total - 1.0) > 1e-9:
                        return None
        
        return U, V, W, rank

    # Try all strategies
    best_result = None
    best_rank = float('inf')
    
    for strategy in range(4):
        if not time_ok():
            break
        
        result = build_and_verify(strategy)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_result = (U, V, W, rank)
                best_rank = rank

    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U, "V": V, "W": W}

    # Fallback: pure schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
