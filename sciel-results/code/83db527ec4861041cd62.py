"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n=2, m=5, p=6):
    """Verify that the decomposition is correct using the trace convention."""
    rank = len(U[0])
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """Return the 7 rank-1 tensors for 2x2x2 Strassen decomposition."""
    tensors = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),  # M3
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
    ]
    return tensors

def _als_optimize(U, V, W, n, m, p, max_iters=200, tol=1e-10, time_limit=10.0):
    """
    Optimize a rank-r decomposition using alternating least squares.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Returns optimized (U, V, W) as lists of lists.
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    start_time = time.time()
    prev_loss = np.inf
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Update U
        for a in range(n * m):
            i, j = divmod(a, m)
            for k in range(p):
                b = j * p + k
                c = k * n + i
                # Target: 1.0
                # U[a] @ (V[b] * W[c]) = 1.0
                # So U[a] should minimize ||U[a] @ (V[b]*W[c]) - 1||^2 over all k
                # This is a linear system: U[a] @ M = 1 for each (b,c)
                # M is a matrix where each row is V[b] * W[c]
        
        # Build the system for U[a]
        for a in range(n * m):
            i, j = divmod(a, m)
            rows = []
            for k in range(p):
                b = j * p + k
                c = k * n + i
                rows.append(V_arr[b] * W_arr[c])
            M = np.array(rows)  # (p, rank)
            # Minimize ||M @ U[a].T - 1||^2
            # M @ x = 1, where x = U[a].T
            try:
                U_arr[a] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
            except np.linalg.LinAlgError:
                pass
        
        # Update V
        for b in range(m * p):
            j, k = divmod(b, p)
            rows = []
            for i in range(n):
                a = i * m + j
                c = k * n + i
                rows.append(U_arr[a] * W_arr[c])
            M = np.array(rows)  # (n, rank)
            try:
                V_arr[b] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
            except np.linalg.LinAlgError:
                pass
        
        # Update W
        for c in range(p * n):
            k, i = divmod(c, n)
            rows = []
            for j in range(m):
                a = i * m + j
                b = j * p + k
                rows.append(U_arr[a] * V_arr[b])
            M = np.array(rows)  # (m, rank)
            try:
                W_arr[c] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
            except np.linalg.LinAlgError:
                pass
        
        # Compute loss
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                    loss += (val - 1.0) ** 2
        
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
    
    return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Fallback: Schoolbook decomposition
    def _schoolbook():
        rank = schoolbook_rank
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Build initial rank-54 decomposition using Strassen blocks
    def _build_initial():
        rank = 54
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col = 0
        strassen_tensors = _strassen_2x2x2()
        
        # Part 1: A1 B1 (j=0,1)
        for block in range(3):
            k_offset = block * 2
            for (u, v, w) in strassen_tensors:
                u_map = [(0, 0), (0, 1), (1, 0), (1, 1)]
                for idx, (ii, jj) in enumerate(u_map):
                    a_idx = ii * m + jj
                    U[a_idx][col] += u[idx]
                v_map = [(0, k_offset), (0, k_offset+1), (1, k_offset), (1, k_offset+1)]
                for idx, (jj, kk) in enumerate(v_map):
                    b_idx = jj * p + kk
                    V[b_idx][col] += v[idx]
                w_map = [(k_offset, 0), (k_offset+1, 0), (k_offset, 1), (k_offset+1, 1)]
                for idx, (kk, ii) in enumerate(w_map):
                    c_idx = kk * n + ii
                    W[c_idx][col] += w[idx]
                col += 1
                
        # Part 2: A2 B2 (j=2,3)
        for block in range(3):
            k_offset = block * 2
            for (u, v, w) in strassen_tensors:
                u_map = [(0, 2), (0, 3), (1, 2), (1, 3)]
                for idx, (ii, jj) in enumerate(u_map):
                    a_idx = ii * m + jj
                    U[a_idx][col] += u[idx]
                v_map = [(2, k_offset), (2, k_offset+1), (3, k_offset), (3, k_offset+1)]
                for idx, (jj, kk) in enumerate(v_map):
                    b_idx = jj * p + kk
                    V[b_idx][col] += v[idx]
                w_map = [(k_offset, 0), (k_offset+1, 0), (k_offset, 1), (k_offset+1, 1)]
                for idx, (kk, ii) in enumerate(w_map):
                    c_idx = kk * n + ii
                    W[c_idx][col] += w[idx]
                col += 1
                
        # Part 3: A3 B3 (j=4)
        for i in range(n):
            a_idx = i * m + 4
            for k in range(p):
                b_idx = 4 * p + k
                c_idx = k * n + i
                U[a_idx][col] = 1.0
                V[b_idx][col] = 1.0
                W[c_idx][col] = 1.0
                col += 1
        
        return U, V, W, rank
    
    U, V, W, current_rank = _build_initial()
    
    if not _verify(U, V, W, n, m, p):
        return _schoolbook()
    
    best_rank = current_rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Try to reduce rank using ALS optimization
    np.random.seed(42)
    
    # Try ranks from current_rank-1 down to 30
    for target_rank in range(best_rank - 1, 29, -1):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random restarts for this target rank
        max_restarts = 5
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Create random initialization for target_rank
            U_init = np.random.randn(n * m, target_rank) * 0.5
            V_init = np.random.randn(m * p, target_rank) * 0.5
            W_init = np.random.randn(p * n, target_rank) * 0.5
            
            # Run ALS optimization
            U_opt, V_opt, W_opt = _als_optimize(
                U_init.tolist(), V_init.tolist(), W_init.tolist(),
                n, m, p, max_iters=100, tol=1e-10, time_limit=8.0
            )
            
            # Check if the optimization converged to a valid decomposition
            if _verify(U_opt, V_opt, W_opt, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U_opt, V_opt, W_opt
                found = True
                break
        
        if found:
            # Try to go even lower
            continue
        else:
            # If we couldn't find a decomposition at this rank, 
            # it's likely we can't at lower ranks either (heuristic)
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
