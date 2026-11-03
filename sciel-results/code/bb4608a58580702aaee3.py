import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0

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
            # Row 0 schoolbook, rows 1-2 Strassen
            for j in range(m):
                for k in range(p):
                    add_schoolbook(0, j, k, terms)
            for j1 in [0, 2]:
                for k1 in [0, 2, 4]:
                    add_strassen_2x2x2(1, 2, j1, j1 + 1, k1, k1 + 1, terms)
        
        elif strategy == 1:
            # Rows 0-1 Strassen, row 2 schoolbook
            for j in range(m):
                for k in range(p):
                    add_schoolbook(2, j, k, terms)
            for j1 in [0, 2]:
                for k1 in [0, 2, 4]:
                    add_strassen_2x2x2(0, 1, j1, j1 + 1, k1, k1 + 1, terms)
        
        elif strategy == 2:
            # Try: Strassen on rows 0-1 for all col/row blocks, row 2 schoolbook
            # But with different col splits
            for j in range(m):
                for k in range(p):
                    add_schoolbook(2, j, k, terms)
            # Rows 0-1, cols 0-1,3
            for k1 in [0, 2, 4]:
                add_strassen_2x2x2(0, 1, 0, 1, k1, k1 + 1, terms)
                add_strassen_2x2x2(0, 1, 2, 3, k1, k1 + 1, terms)
        
        elif strategy == 3:
            # Try: use 3x2x2 blocks with schoolbook for the 3rd row
            # For each 2-col block of A and 2-col block of C:
            # A[0:3, j1:j2] * B[j1:j2, k1:k2] -> C[0:3, k1:k2]
            # Use Strassen for rows 0-1, schoolbook for row 2
            for j1 in [0, 2]:
                for k1 in [0, 2, 4]:
                    add_strassen_2x2x2(0, 1, j1, j1 + 1, k1, k1 + 1, terms)
                    for j in [j1, j1+1]:
                        for k in [k1, k1+1]:
                            add_schoolbook(2, j, k, terms)
        
        elif strategy == 4:
            # Try: rows 0-1 Strassen for cols 0-3, then handle cols 4-5 separately
            # Actually same as strategy 1/2
            # Let me try: interleave Strassen blocks
            for k1 in [0, 2, 4]:
                add_strassen_2x2x2(0, 1, 0, 1, k1, k1 + 1, terms)
                add_strassen_2x2x2(0, 1, 2, 3, k1, k1 + 1, terms)
                add_strassen_2x2x2(1, 2, 0, 1, k1, k1 + 1, terms)
                add_strassen_2x2x2(1, 2, 2, 3, k1, k1 + 1, terms)
            # This double-counts row 1, so it's wrong. Skip.
            terms.clear()
            # Fallback to strategy 0
            for j in range(m):
                for k in range(p):
                    add_schoolbook(0, j, k, terms)
            for j1 in [0, 2]:
                for k1 in [0, 2, 4]:
                    add_strassen_2x2x2(1, 2, j1, j1 + 1, k1, k1 + 1, terms)
        
        elif strategy == 5:
            # Try: 3x4x6 = 3x(2x3) x (2x3) split differently
            # A: 3x4 = [3x2 | 3x2], B: 4x6 = [2x3; 2x3]
            # C = A1*B1 + A2*B2 where A1 is 3x2, B1 is 2x3, etc.
            # Each 3x2x3: use best known decomposition
            # For 3x2x3, try: Strassen for rows 0-1 (2x2x3 part) + schoolbook for row 2
            # 2x2x3: split into three 2x2x1, each rank 4, total 12
            # Or: use a 2x2x3 specific decomposition
            # Actually 2x2x3 schoolbook = 12, and I don't think we can do better
            # So 3x2x3 = 12 + 6 = 18 (schoolbook for row 2: 1x2x3 = 6)
            # Two such blocks: 36, plus we need to handle the full 6 columns
            # Wait, this is getting complicated. Let me just do:
            # Block 1: A[0:3, 0:2] * B[0:2, 0:6] -> C[0:3, 0:6]
            # Block 2: A[0:3, 2:4] * B[2:4, 0:6] -> C[0:3, 0:6]
            # For each 3x2x6: use Strassen for rows 0-1, schoolbook for row 2
            # 3x2x6: rows 0-1: 2x2x6 = 3 blocks of 2x2x2 = 21 terms
            # Row 2: 1x2x6 = 12 terms
            # Total per 3x2x6: 33, two blocks: 66. Same as before.
            terms.clear()
            for j1 in [0, 2]:
                for k1 in [0, 2, 4]:
                    add_strassen_2x2x2(0, 1, j1, j1 + 1, k1, k1 + 1, terms)
                    add_strassen_2x2x2(1, 2, j1, j1 + 1, k1, k1 + 1, terms)
                for j in [j1, j1+1]:
                    for k in range(p):
                        add_schoolbook(0, j, k, terms)
            # This double-counts. Let me just keep it simple.
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
    
    for strategy in range(6):
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
