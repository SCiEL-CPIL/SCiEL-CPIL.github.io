import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 125
    
    def verify_exact(U, V, W):
        """Verify decomposition exactly. Returns True if valid."""
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(len(U[i*m+j])):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Fast vectorized verification using einsum."""
        U_mat = np.array(U).reshape(n, m, -1)
        V_mat = np.array(V).reshape(m, p, -1)
        W_mat = np.array(W).reshape(p, n, -1)
        T = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
        return float(np.max(np.abs(T - 1.0)))
    
    def schoolbook_decomposition():
        """Return the standard schoolbook decomposition of rank n*m*p."""
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
    
    def try_alternating_minimization(rank, max_iters=200, seed=42):
        """Try to find a rank decomposition using alternating minimization."""
        if time.time() > deadline - 15:
            return None
        
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n*m, rank) * 0.3
        V = np.random.randn(m*p, rank) * 0.3
        W = np.random.randn(p*n, rank) * 0.3
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            # Update U: for each (i,j), solve least squares
            for i in range(n):
                for j in range(m):
                    # B[c, k] = V[j*p+k, c] * W[k*n+i, c]
                    B = np.zeros((rank, p))
                    for k in range(p):
                        B[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), solve least squares
            for j in range(m):
                for k in range(p):
                    A = np.zeros((rank, n))
                    for i in range(n):
                        A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), solve least squares
            for k in range(p):
                for i in range(n):
                    C = np.zeros((rank, m))
                    for j in range(m):
                        C[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Compute error
            max_err = verify_fast(U, V, W)
            
            if max_err < 1e-10:
                # Found a solution, now try to snap to rational values
                U_list = [[float(x) for x in U[row]] for row in range(n*m)]
                V_list = [[float(x) for x in V[row]] for row in range(m*p)]
                W_list = [[float(x) for x in W[row]] for row in range(p*n)]
                
                # Try to snap to simple fractions
                def snap_to_rational(x, tol=1e-6):
                    # Try integers and simple fractions
                    for denom in [1, 2, 3, 4, 5]:
                        num = round(x * denom)
                        if abs(x - num/denom) < tol:
                            return num/denom
                    return x
                
                U_snap = [[snap_to_rational(x) for x in row] for row in U_list]
                V_snap = [[snap_to_rational(x) for x in row] for row in V_list]
                W_snap = [[snap_to_rational(x) for x in row] for row in W_list]
                
                if verify_exact(U_snap, V_snap, W_snap):
                    return {"U": U_snap, "V": V_snap, "W": W_snap}
                else:
                    # Return the floating point solution
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            if max_err > prev_err * 1.05 and it > 20:
                break
            prev_err = max_err
        
        return None
    
    # Try to find a lower-rank decomposition
    # For 5x5 matrix multiplication, the best known rank is 125
    # But let's try a few lower ranks just in case
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Try ranks from 100 to 124 (lower than schoolbook)
    for target_rank in range(100, min(125, rank_schoolbook)):
        if time.time() > deadline - 20:
            break
        
        result = try_alternating_minimization(target_rank, max_iters=100, seed=target_rank)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break  # Found a lower rank, use it
    
    # If no lower rank found, use schoolbook
    if best_result is None:
        best_result = schoolbook_decomposition()
    
    return best_result
# EVOLVE-BLOCK-END
