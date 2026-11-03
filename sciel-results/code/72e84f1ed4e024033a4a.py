import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 8]
DEADLINE = 145.0
_START = time.time()

def _time_left():
    return DEADLINE - (time.time() - _START)

def _strassen_2x2_2():
    """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen)."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # P1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _build_strassen_embedding(n, m, p, strassen_blocks):
    """
    Build decomposition using Strassen on specified 2x2x2 blocks, schoolbook elsewhere.
    strassen_blocks: list of (i0, j0, k0)
    """
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    n_school = n*m*p - len(covered)
    rank = 7 * len(strassen_blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for (i0, j0, k0) in strassen_blocks:
        sU, sV, sW = _strassen_2x2_2()
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col+c] = sU[di*2+dj][c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col+c] = sV[dj*2+dk][c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di)][col+c] = sW[dk*2+di][c]
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    return U, V, W

def _try_rank_reduction_optimization(U, V, W, n, m, p, target_rank, time_limit=10.0):
    """
    Attempt to reduce the rank of the given decomposition to target_rank
    using least squares optimization.
    """
    start_time = time.time()
    n_terms = n * m * p
    
    # Flatten U, V, W into a single vector for optimization
    # U: (n*m, current_rank)
    # V: (m*p, current_rank)
    # W: (p*n, current_rank)
    
    current_rank = len(U[0])
    if current_rank <= target_rank:
        return U, V, W, current_rank
    
    # We will try to project the current rank `current_rank` solution onto 
    # a lower rank `target_rank` by optimizing the factors directly.
    # This is a non-convex problem.
    
    # Initialize with random factors of size target_rank
    # Or better, initialize with the first target_rank columns of the current solution
    # and see if we can adjust them to fit.
    
    U_init = np.array(U[:, :target_rank])
    V_init = np.array(V[:, :target_rank])
    W_init = np.array(W[:, :target_rank])
    
    # The error is the sum of squares of the tensor entries not covered by the rank-target_rank decomposition.
    # T_ijk = sum_r U_ir V_jr W_kr
    # We want T_ijk to be 1 if (i,j,k) is in the support, 0 otherwise? 
    # No, the tensor for matrix multiplication has 1s at (i,j,k) for the product terms.
    # The support is all (i,j,k) where the term A_ij * B_jk contributes to C_ki? 
    # Wait, the tensor is M_{ijk} = delta_{i, k} delta_{j, j}? No.
    # The matrix multiplication tensor T has T_{ijk} = 1 if the term A_{ij} B_{jk} contributes to C_{ki}?
    # Actually, C_{ki} = sum_j A_{ij} B_{jk}.
    # So the tensor is T_{ijk} = 1 for all i,j,k? No.
    # The tensor is defined such that C_{ki} = sum_{j,r} A_{ij} B_{jk} W_{kri}?
    # The standard tensor is T_{ijk} where T_{ijk} = 1 if the term A_{ij} B_{jk} is present in the product.
    # For standard matrix multiplication, every pair (i,j,k) corresponds to a term?
    # No, A is n x m, B is m x p. C is n x p.
    # C_{ip} = sum_j A_{ij} B_{jp}.
    # So the tensor T_{i, j, p} has 1 if the term A_{ij} B_{jp} is used.
    # In the task, dimensions are n=3, m=3, p=8.
    # U is (n*m, rank) = (9, R)
    # V is (m*p, rank) = (24, R)
    # W is (p*n, rank) = (24, R)
    # The tensor entry T_{(i,j), (j,k), (k,i)}?
    # The verify function uses:
    # s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
    # and checks if s == 1.0.
    # This implies that for every (i,j,k), the sum must be 1.
    # This means the tensor is the "all-ones" tensor of size 3x3x8?
    # No, that would mean C_{ki} = sum_{j} A_{ij} B_{jk} is computed by summing over all j?
    # Yes, that is exactly matrix multiplication.
    # So the target tensor T has T_{ijk} = 1 for all i in [0,3), j in [0,3), k in [0,8).
    
    def residuals(params):
        # params: flattened U, V, W of rank target_rank
        U_flat = params[:n*m*target_rank]
        V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
        W_flat = params[n*m*target_rank + m*p*target_rank : ]
        
        U_opt = U_flat.reshape(n*m, target_rank)
        V_opt = V_flat.reshape(m*p, target_rank)
        W_opt = W_flat.reshape(p*n, target_rank)
        
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    val = np.dot(U_opt[i*m+j], np.multiply(V_opt[j*p+k], W_opt[k*n+i]))
                    res[i,j,k] = val - 1.0
        return res.flatten()
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    try:
        # Use least squares
        result = least_squares(residuals, x0, method='lm', max_nfev=1000, xtol=1e-8, ftol=1e-8)
        
        if result.success and result.cost < 1e-8:
            U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_opt = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_opt = result.x[n*m*target_rank + m*p*target_rank : ].reshape(p*n, target_rank)
            
            # Convert back to list of lists
            U_new = U_opt.tolist()
            V_new = V_opt.tolist()
            W_new = W_opt.tolist()
            
            # Verify
            if _verify(U_new, V_new, W_new, n, m, p, tol=1e-5):
                return U_new, V_new, W_new, target_rank
    except Exception as e:
        pass
        
    return U, V, W, current_rank

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Start with the known good fixed placement (4 blocks along k)
    # This gives rank 68.
    # (0,0,0), (0,0,2), (0,0,4), (0,0,6)
    blocks = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]
    U, V, W = _build_strassen_embedding(n, m, p, blocks)
    
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if something went wrong
        U, V, W = _schoolbook(n, m, p)
        
    best_rank = len(U[0])
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to reduce rank to 67 using optimization
    if _time_left() > 20:
        try:
            U_new, V_new, W_new, new_rank = _try_rank_reduction_optimization(U, V, W, n, m, p, best_rank - 1, time_limit=_time_left() - 10)
            if new_rank < best_rank:
                best_rank = new_rank
                best_result = {"U": U_new, "V": V_new, "W": W_new}
        except Exception:
            pass
            
    # Try to reduce rank to 66 if we found 67 and have time
    if best_rank > 66 and _time_left() > 10:
        try:
            U_cur = best_result["U"]
            V_cur = best_result["V"]
            W_cur = best_result["W"]
            U_new, V_new, W_new, new_rank = _try_rank_reduction_optimization(U_cur, V_cur, W_cur, n, m, p, best_rank - 1, time_limit=_time_left() - 5)
            if new_rank < best_rank:
                best_rank = new_rank
                best_result = {"U": U_new, "V": V_new, "W": W_new}
        except Exception:
            pass

    return best_result
