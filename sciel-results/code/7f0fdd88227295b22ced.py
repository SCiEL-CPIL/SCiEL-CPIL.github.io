import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T is 1 at (i*m+j, j*p+k, k*n+i) for all i,j,k
    # We represent T implicitly or as a sparse structure.
    # For ALS, we need to access T[i, j, k] or flattened indices.
    
    # Precompute indices
    # U index: u = i*m + j
    # V index: v = j*p + k
    # W index: w = k*n + i
    
    # T is 1 at (u, v, w) if they form a valid tuple.
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    
    # We can store T as a dictionary or just compute on the fly.
    # Since n,m,p are small, we can just loop.
    
    def get_T_u_v_w(u, v, w):
        # Check if u, v, w correspond to a valid i, j, k
        i = u // m
        j = u % m
        j2 = v // p
        k = v % p
        k2 = w // n
        i2 = w % n
        if i == i2 and j == j2 and k == k2:
            return 1.0
        return 0.0

    def check_solution(U, V, W):
        # Check if sum_r U[u,r] V[v,r] W[w,r] == T[u,v,w] for all u,v,w
        # We only need to check the non-zero entries of T
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[u, r] * V[v, r] * W[w, r]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
                    # Also check some zero entries?
                    # The tensor is zero elsewhere.
                    # We should check a few random zero entries to ensure they are 0.
        # Check random zero entries
        for _ in range(100):
            u = np.random.randint(0, n*m)
            v = np.random.randint(0, m*p)
            w = np.random.randint(0, p*n)
            if get_T_u_v_w(u, v, w) == 0:
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[u, r] * V[v, r] * W[w, r]
                err = abs(val)
                if err > max_err:
                    max_err = err
        return max_err

    def als_step(U, V, W, rank):
        # Update U
        for i in range(n):
            for j in range(m):
                u = i * m + j
                # We want to minimize sum_{k} (T[i,j,k] - sum_r U[u,r] V[j*p+k,r] W[k*n+i,r])^2
                # Note: T[i,j,k] is 1 for all k.
                # Let's form the matrix B for this u.
                # B[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # b[k] = T[i,j,k] = 1
                B = np.zeros((p, rank))
                b = np.ones(p)
                for k in range(p):
                    v = j * p + k
                    w = k * n + i
                    for r in range(rank):
                        B[k, r] = V[v, r] * W[w, r]
                # Solve B @ U[u, :] = b
                U[u, :], residuals, rank_b, s = np.linalg.lstsq(B, b, rcond=None)
        
        # Update V
        for j in range(m):
            for k in range(p):
                v = j * p + k
                # Minimize sum_{i} (T[i,j,k] - sum_r U[i*m+j,r] V[v,r] W[k*n+i,r])^2
                # T[i,j,k] = 1 for all i.
                B = np.zeros((n, rank))
                b = np.ones(n)
                for i in range(n):
                    u = i * m + j
                    w = k * n + i
                    for r in range(rank):
                        B[i, r] = U[u, r] * W[w, r]
                V[v, :], residuals, rank_b, s = np.linalg.lstsq(B, b, rcond=None)
        
        # Update W
        for k in range(p):
            for i in range(n):
                w = k * n + i
                # Minimize sum_{j} (T[i,j,k] - sum_r U[i*m+j,r] V[j*p+k,r] W[w,r])^2
                # T[i,j,k] = 1 for all j.
                B = np.zeros((m, rank))
                b = np.ones(m)
                for j in range(m):
                    u = i * m + j
                    v = j * p + k
                    for r in range(rank):
                        B[j, r] = U[u, r] * V[v, r]
                W[w, :], residuals, rank_b, s = np.linalg.lstsq(B, b, rcond=None)

    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try ranks from 19 upwards
    for rank in range(19, 37):
        if time.time() > deadline:
            break
            
        # Try a few random restarts
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize
            # Use small random values
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Run ALS
            for iter in range(100):
                if time.time() > deadline:
                    break
                als_step(U, V, W, rank)
                
                # Check convergence
                err = check_solution(U, V, W)
                if err < 1e-9:
                    # Check if entries are close to integers/half-integers
                    # If not, we might still return it if the evaluator accepts floats?
                    # The task says "integer/half-integer entries".
                    # We should try to snap to nearest integer/half-integer and re-check.
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    err_snap = check_solution(U_snap, V_snap, W_snap)
                    if err_snap < 1e-9:
                        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    else:
                        # If snapping fails, maybe the float solution is valid?
                        # The task says "exactly decompose". Floats might be OK if they are exact?
                        # But "integer/half-integer" is specified.
                        # Let's return the float version if snapping fails, as it's better than nothing?
                        # No, invalid result scores 0.
                        # We should only return if it satisfies the constraint.
                        # If the ALS solution is not integer/half-integer, it might be invalid.
                        # However, for this problem, the minimal rank decomposition IS known to have integer entries.
                        # So if we found a solution, it should be close to integers.
                        pass

    # If no solution found, return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
