import time
import random
import numpy as np
from scipy.optimize import least_squares

# EVOLVE-BLOCK-START

def _check_decomposition(U, V, W, n, m, p, num_tests=10):
    """Check that the decomposition is exact for random inputs."""
    rank = len(U[0])
    for seed in range(num_tests):
        random.seed(seed * 7 + 13)
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        AB = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        X = [[0] * n for _ in range(rank)]
        for r in range(rank):
            for i in range(n):
                s = 0
                for j in range(m):
                    s += U[i * m + j][r] * A[i][j]
                X[r][i] = s
        
        Y = [[0] * p for _ in range(rank)]
        for r in range(rank):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += V[j * p + k][r] * B[j][k]
                Y[r][k] = s
        
        C = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += W[k * n + i][r] * X[r][i] * Y[r][k]
                C[i][k] = s
        
        if C != AB:
            return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _strassen_2x2x6():
    """
    Strassen-based decomposition for 2x2x6 matrix multiplication.
    Rank = 3 * 7 = 21.
    """
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_s = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    
    V_s = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    
    W_s = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    
    for block in range(3):
        for s in range(7):
            r = block * 7 + s
            for ij in range(4):
                U[ij][r] = U_s[s][ij]
            for j in range(2):
                for k in range(2):
                    v_idx = j * 6 + (block * 2 + k)
                    v_s_idx = j * 2 + k
                    V[v_idx][r] = V_s[s][v_s_idx]
            for k in range(2):
                for i in range(2):
                    w_idx = (block * 2 + k) * 2 + i
                    w_s_idx = k * 2 + i
                    W[w_idx][r] = W_s[s][w_s_idx]
                    
    return {"U": U, "V": V, "W": W}

def _find_rank_20_decomposition(n, m, p, deadline):
    """
    Attempt to find a rank-20 decomposition using local search.
    """
    target = np.zeros((n, p, m)) # T[i,k,j] = 1 if j is the intermediate index? 
    # Tensor T such that C[i,k] = sum_j A[i,j] B[j,k] T[i,k,j]
    # So T[i,k,j] = 1 for all i,k,j? No.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Tensor entries T_{i,j,k} where A is i,j, B is j,k, C is i,k.
    # T[i,j,k] = 1.
    
    # We want to decompose T into sum_r u_r (n) v_r (m) w_r (p) ?
    # No, standard form: sum_r u_r[i] v_r[j] w_r[k] A[i,j] B[j,k] ...
    # The decomposition is for the multiplication tensor.
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # The check function uses:
    # X[r,i] = sum_j U[i*m+j, r] A[i,j]
    # Y[r,k] = sum_j V[j*p+k, r] B[j,k]
    # C[i,k] = sum_r W[k*n+i, r] X[r,i] Y[r,k]
    
    # This is equivalent to C = sum_r W_r * (U_r^T A) * (V_r^T B) ? No.
    # It is a bilinear decomposition.
    
    # We will try to fit 20 terms.
    # Initialize with the first 20 terms of Strassen.
    strassen = _strassen_2x2x6()
    U0 = np.array(strassen["U"], dtype=float).T # (rank, n*m)
    V0 = np.array(strassen["V"], dtype=float).T # (rank, m*p)
    W0 = np.array(strassen["W"], dtype=float).T # (rank, p*n)
    
    # We drop the last term (index 20) and try to fit the remaining 20.
    # Actually, we should try dropping different terms.
    # For simplicity, drop the last one.
    U_init = U0[:20]
    V_init = V0[:20]
    W_init = W0[:20]
    
    # Flatten parameters
    # U: 20 * 4 = 80
    # V: 20 * 12 = 240
    # W: 20 * 12 = 240
    # Total 560 parameters.
    
    def compute_C(params):
        # Unpack
        U_flat = params[0:80]
        V_flat = params[80:320]
        W_flat = params[320:560]
        
        U = U_flat.reshape(20, 4)
        V = V_flat.reshape(20, 12)
        W = W_flat.reshape(20, 12)
        
        # Compute C for a random A, B? No, we need to match the tensor.
        # The tensor is defined by the mapping (A,B) -> C.
        # We can check the error on the tensor itself.
        # T[i,j,k] = 1.
        # The decomposition computes C[i,k] = sum_r W[k*n+i, r] (sum_j U[i*m+j, r] A[i,j]) (sum_l V[l*p+k, r] B[l,k]).
        # This is a 3rd order tensor contraction.
        # We can form the tensor of the decomposition and compare to the target tensor.
        
        # Target Tensor T_target[i, j, k] = 1.
        # Decomposition Tensor T_dec[i, j, k] = sum_r U[i*m+j, r] V[l*p+k, r] W[k*n+i, r] ?
        # Note the indices in U, V, W.
        # U index: i*m+j. V index: l*p+k. W index: k*n+i.
        # Sum over r.
        # This defines a tensor D[i, j, l, k].
        # We want D[i, j, l, k] to act such that sum_j,l D[i,j,l,k] A[i,j] B[l,k] = C[i,k].
        # This requires D[i,j,l,k] = delta_{j,l}.
        # So we want sum_r U[i*m+j, r] V[l*p+k, r] W[k*n+i, r] = delta_{j,l}.
        
        # We can flatten this into a vector and compare to the target vector.
        # Target vector: 1 if j==l, 0 otherwise.
        # Shape: n * m * m * p = 2 * 2 * 2 * 6 = 48.
        
        D = np.zeros((n, m, m, p))
        for r in range(20):
            u_r = U[r, :] # (4,)
            v_r = V[r, :] # (12,)
            w_r = W[r, :] # (12,)
            
            # u_r is indexed by i*m+j
            # v_r is indexed by l*p+k
            # w_r is indexed by k*n+i
            
            for i in range(n):
                for j in range(m):
                    u_val = u_r[i*m+j]
                    if u_val == 0: continue
                    for l in range(m):
                        for k in range(p):
                            v_val = v_r[l*p+k]
                            if v_val == 0: continue
                            w_val = w_r[k*n+i]
                            D[i, j, l, k] += u_val * v_val * w_val
                            
        return D.flatten()

    # Target vector
    target_vec = np.zeros(n*m*m*p)
    for i in range(n):
        for j in range(m):
            for l in range(m):
                for k in range(p):
                    if j == l:
                        target_vec[(i*m+j)*m*p + l*p + k] = 1.0
                    # Wait, the flattening order must match D.flatten()
                    # D is (n, m, m, p).
                    # Index: i, j, l, k.
                    # Flatten: i * (m*m*p) + j * (m*p) + l * p + k.
                    # My target_vec index: (i*m+j)*m*p ... is wrong.
                    # Correct: i * (m*m*p) + j * (m*p) + l * p + k.
                    # Let's re-verify D.flatten() order.
                    # D[i,j,l,k].
                    # C order: i, then j, then l, then k.
                    # Yes.
                    
    # Re-define target_vec correctly
    target_vec = np.zeros(n*m*m*p)
    for i in range(n):
        for j in range(m):
            for l in range(m):
                for k in range(p):
                    if j == l:
                        idx = i * (m*m*p) + j * (m*p) + l * p + k
                        target_vec[idx] = 1.0

    def residuals(params):
        return compute_C(params) - target_vec

    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    # Use least squares
    try:
        res = least_squares(residuals, x0, method='lm', max_nfev=1000)
        if res.cost < 1e-9:
            # Found a solution
            U_sol = res.x[0:80].reshape(20, 4).T
            V_sol = res.x[80:320].reshape(20, 12).T
            W_sol = res.x[320:560].reshape(20, 12).T
            
            # Round to integers
            U_out = [[int(round(x)) for x in row] for row in U_sol]
            V_out = [[int(round(x)) for x in row] for row in V_sol]
            W_out = [[int(round(x)) for x in row] for row in W_sol]
            
            if _check_decomposition(U_out, V_out, W_out, n, m, p):
                return {"U": U_out, "V": V_out, "W": W_out}
    except Exception:
        pass
        
    return None

def construct():
    n, m, p = 2, 2, 6
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to find rank 20
    if time.time() < deadline - 20:
        result = _find_rank_20_decomposition(n, m, p, deadline)
        if result:
            return result
            
    # Fall back to Strassen rank 21
    result = _strassen_2x2x6()
    return result

# EVOLVE-BLOCK-END
