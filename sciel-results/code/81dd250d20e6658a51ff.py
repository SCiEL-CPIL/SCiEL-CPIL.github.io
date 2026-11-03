"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication.
    """
    terms = []
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1}
    ))
    terms.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1}
    ))
    terms.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (1,0): 1}
    ))
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (0,1): 1}
    ))
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    return terms

def _verify_decomposition(result, n=2, m=4, p=6, trials=100):
    """Verify a decomposition numerically with integer arithmetic."""
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    
    for trial in range(trials):
        A = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
        
        C_exp = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        C_comp = [[0] * p for _ in range(n)]
        for col in range(rank):
            u_val = sum(U[i * m + j][col] * A[i][j] for i in range(n) for j in range(m))
            v_val = sum(V[j * p + k][col] * B[j][k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_comp[i][k] += W[k * n + i][col] * u_val * v_val
        
        if C_comp != C_exp:
            return False
    return True

def _strassen_block_decomposition():
    """
    Construct a rank-42 decomposition using Strassen's algorithm.
    """
    n, m, p = DIMENSIONS
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = _strassen_2x2x2_terms()
    
    col = 0
    for a_block in range(2):
        for b_col_block in range(3):
            for t in range(7):
                U_coeffs, V_coeffs, W_coeffs = strassen_terms[t]
                
                for (i_loc, j_loc), coeff in U_coeffs.items():
                    i = i_loc
                    j = a_block * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                for (j_loc, k_loc), coeff in V_coeffs.items():
                    j = a_block * 2 + j_loc
                    k = b_col_block * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                for (i_loc, k_loc), coeff in W_coeffs.items():
                    i = i_loc
                    k = b_col_block * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _try_find_lower_rank(n, m, p, target_rank, time_limit):
    """
    Attempt to find a decomposition of lower rank using ALS.
    Returns None if not found or if time limit exceeded.
    """
    start_time = time.time()
    # Tensors: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # Target: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Reshape indices for easier handling
    # U_flat[i, r] where i in [0, n*m)
    # V_flat[j, r] where j in [0, m*p)
    # W_flat[k, r] where k in [0, p*n)
    
    # We want to minimize || sum_r U[:,r] V[:,r]^T W[:,r]^T - T ||
    # T is the tensor of all ones, shape (n*m, m*p, p*n) -> actually T[i,j,k] = 1
    
    # ALS:
    # Fix U, V -> solve W
    # Fix V, W -> solve U
    # Fix U, W -> solve V
    
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    for trial in range(10): # 10 random restarts
        if time.time() - start_time > time_limit:
            break
            
        np.random.seed(trial * 12345 + target_rank)
        
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        for iteration in range(500):
            if time.time() - start_time > time_limit:
                break
            
            # 1. Solve for W
            # For each (i, j), we have: sum_r U[i, r] V[j, r] W[:, r] = 1 (vector of length p*n)
            # Wait, the equation is:
            # C[i, k] = sum_j A[i, j] B[j, k]
            # Decomposition: C[i, k] = sum_r U[i, r] V[j, r] W[k, r] ??
            # No, the standard form is:
            # C[i, k] = sum_r U[i, r] V[j, r] W[k, r] is not correct because j is summed over.
            # The tensor decomposition is:
            # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
            # Here T[i, j, k] is 1 if it's a valid multiplication term?
            # Actually, the multiplication tensor M_{n,m,p} has entries M_{i,j,k} = 1.
            # We need sum_r U[i, r] V[j, r] W[k, r] = 1 for all i, j, k.
            
            # So, for fixed U, V:
            # For each k (index of W), we have a system:
            # sum_r U[i, r] V[j, r] W[k, r] = 1 for all i, j.
            # Let A_k be a matrix of size (n*m) x rank.
            # A_k[i*m+j, r] = U[i, r] V[j, r].
            # We want A_k W[k, :]^T = 1 (vector of length n*m).
            # This is a linear system for W[k, :].
            
            # However, U and V are fixed, so A_k is fixed for all k?
            # No, A_k depends on i and j, which are independent of k.
            # So the matrix A is the same for all k?
            # A[i*m+j, r] = U[i, r] V[j, r].
            # Yes, A is (n*m) x rank.
            # We want to find W[k, r] such that sum_r A[i*m+j, r] W[k, r] = 1 for all i, j, k.
            # This means W[k, :] must be the same for all k?
            # No, the equation is sum_r U[i, r] V[j, r] W[k, r] = 1.
            # If W[k, r] is the same for all k, say w_r, then sum_r U[i, r] V[j, r] w_r = 1.
            # This is a single system for w.
            # But W has p*n rows. The equation must hold for each k.
            # So for each k, W[k, :] is a solution to A x = 1.
            # The system A x = 1 might have multiple solutions.
            # We can pick the minimum norm solution.
            
            # Build A
            # A[i*m+j, r] = U[i, r] * V[j, r]
            # We can construct A using broadcasting.
            # U is (n*m, rank). V is (m*p, rank).
            # We need to map indices.
            # Let's reshape U to (n, m, rank) and V to (m, p, rank).
            # Then A[i, j, r] = U[i, j, r] * V[j, r] ?? No, V index is j*p+k.
            # V[j, k, r] = V_flat[j*p+k, r].
            # The term in the sum is U[i, j, r] * V[j, k, r] * W[k, i, r]?
            # No, W index is k*n+i.
            
            # Let's stick to the flat indices.
            # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This is not a simple linear system in W for fixed U, V because of the k in V's index.
            # For a fixed (i, j), we have:
            # sum_r U[i*m+j, r] * (sum_k V[j*p+k, r] W[k*n+i, r]) = 1 ?? No.
            # The sum is over r. i, j, k are fixed.
            # So for each (i, j, k), we have one equation.
            # There are n*m*p equations.
            # Unknowns are W[k*n+i, r] for all k, i, r. Total p*n*rank unknowns.
            
            # Let's group by (i, k).
            # For fixed i, k, we have m equations (for j=0..m-1):
            # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
            
            # Let w_ik = W[k*n+i, :] (vector of length rank).
            # Let M_ik be a matrix of size m x rank.
            # M_ik[j, r] = U[i*m+j, r] * V[j*p+k, r].
            # Then sum_j M_ik[j, :] dot w_ik = 1 (scalar).
            # This is (M_ik^T 1_m) dot w_ik = 1.
            # Let v_ik = sum_j M_ik[j, :] (vector of length rank).
            # Then v_ik dot w_ik = 1.
            
            # This gives one equation for w_ik (length rank vector).
            # We have n*p such equations (for each i, k).
            # Total unknowns: n*p*rank.
            # Total equations: n*p.
            # This is underdetermined.
            # We can choose w_ik to be the minimum norm solution to v_ik dot w_ik = 1.
            # w_ik = v_ik / (v_ik dot v_ik).
            
            # So the update for W is:
            # For each i, k:
            #   v_ik[r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            #   W[k*n+i, r] = v_ik[r] / (sum_r' v_ik[r']^2)
            
            # Similarly for U and V.
            
            # Update W
            W_new = np.zeros((p * n, target_rank))
            for i in range(n):
                for k in range(p):
                    # v_ik[r] = sum_j U[i*m+j, r] * V[j*p+k, r]
                    # We can compute this using matrix multiplication if we reshape.
                    # U_i = U[i*m : (i+1)*m, :] (m x rank)
                    # V_k = V[:, k::p] ? No, V is (m*p, rank).
                    # V_jk = V[j*p+k, :] for j in 0..m-1.
                    # We can extract V_k as (m, rank) by taking rows j*p+k.
                    # indices = [j*p+k for j in range(m)]
                    # V_k = V[indices, :]
                    # v_ik = sum_j U_i[j, :] * V_k[j, :] (element-wise sum over j)
                    # v_ik = np.sum(U_i * V_k, axis=0)
                    
                    indices_V = np.array([j * p + k for j in range(m)])
                    V_k = V[indices_V, :] # (m, rank)
                    U_i = U[i * m : (i + 1) * m, :] # (m, rank)
                    
                    v_ik = np.sum(U_i * V_k, axis=0) # (rank,)
                    norm_sq = np.dot(v_ik, v_ik)
                    if norm_sq > 1e-10:
                        W_new[k * n + i, :] = v_ik / norm_sq
                    else:
                        W_new[k * n + i, :] = 0
                        
            W = W_new
            
            # Update U
            # Symmetric logic.
            # For fixed j, k? No, for fixed j, r?
            # Equation: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
            # Group by (j, i)?
            # For fixed j, i:
            # sum_r U[i*m+j, r] * (sum_k V[j*p+k, r] W[k*n+i, r]) = 1
            # Let u_ji = U[i*m+j, :]
            # Let v_ji[r] = sum_k V[j*p+k, r] W[k*n+i, r]
            # Then u_ji dot v_ji = 1.
            # u_ji = v_ji / (v_ji dot v_ji).
            
            U_new = np.zeros((n * m, target_rank))
            for i in range(n):
                for j in range(m):
                    # v_ji[r] = sum_k V[j*p+k, r] W[k*n+i, r]
                    # V_j = V[j*p : (j+1)*p, :] (p x rank)
                    # W_i = W[:, i::n] ? No.
                    # W_ki = W[k*n+i, :] for k in 0..p-1.
                    # indices_W = [k*n+i for k in range(p)]
                    # W_i = W[indices_W, :] (p x rank)
                    # v_ji = sum_k V_j[k, :] * W_i[k, :]
                    
                    V_j = V[j * p : (j + 1) * p, :] # (p, rank)
                    indices_W = np.array([k * n + i for k in range(p)])
                    W_i = W[indices_W, :] # (p, rank)
                    
                    v_ji = np.sum(V_j * W_i, axis=0) # (rank,)
                    norm_sq = np.dot(v_ji, v_ji)
                    if norm_sq > 1e-10:
                        U_new[i * m + j, :] = v_ji / norm_sq
                    else:
                        U_new[i * m + j, :] = 0
                        
            U = U_new
            
            # Update V
            # For fixed j, k:
            # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
            # Group by (j, k)?
            # For fixed j, k:
            # sum_r V[j*p+k, r] * (sum_i U[i*m+j, r] W[k*n+i, r]) = 1
            # Let v_jk = V[j*p+k, :]
            # Let w_jk[r] = sum_i U[i*m+j, r] W[k*n+i, r]
            # Then v_jk dot w_jk = 1.
            # v_jk = w_jk / (w_jk dot w_jk).
            
            V_new = np.zeros((m * p, target_rank))
            for j in range(m):
                for k in range(p):
                    # w_jk[r] = sum_i U[i*m+j, r] W[k*n+i, r]
                    # U_j = U[:, j::m] ? No.
                    # U_ij = U[i*m+j, :] for i in 0..n-1.
                    # indices_U = [i*m+j for i in range(n)]
                    # U_j = U[indices_U, :] (n x rank)
                    # W_k = W[k*n : (k+1)*n, :] (n x rank)
                    # w_jk = sum_i U_j[i, :] * W_k[i, :]
                    
                    indices_U = np.array([i * m + j for i in range(n)])
                    U_j = U[indices_U, :] # (n, rank)
                    W_k = W[k * n : (k + 1) * n, :] # (n, rank)
                    
                    w_jk = np.sum(U_j * W_k, axis=0) # (rank,)
                    norm_sq = np.dot(w_jk, w_jk)
                    if norm_sq > 1e-10:
                        V_new[j * p + k, :] = w_jk / norm_sq
                    else:
                        V_new[j * p + k, :] = 0
                        
            V = V_new
            
            # Check error
            # err = || sum_r U V W - 1 ||
            # We can check a few random points or compute the full tensor.
            # Full tensor is n*m*p = 48 elements.
            # Compute C_comp
            C_comp = np.zeros((n, p))
            # Let's compute the error for a few random (i, j, k)
            err = 0
            for _ in range(10):
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                val = 0
                for r in range(target_rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                err += (val - 1) ** 2
            err = np.sqrt(err / 10)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if err < 1e-6:
                break
                
    if best_err < 1e-4:
        # Round to nearest integers or half-integers
        # The problem allows integer/half-integer entries.
        # Let's try rounding to nearest 0.5
        U_int = np.round(best_U * 2) / 2
        V_int = np.round(best_V * 2) / 2
        W_int = np.round(best_W * 2) / 2
        
        # Convert to lists
        U_list = U_int.tolist()
        V_list = V_int.tolist()
        W_list = W_int.tolist()
        
        result = {"U": U_list, "V": V_list, "W": W_list}
        
        # Verify
        if _verify_decomposition(result, n, m, p, trials=100):
            return result
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    # Try to find rank 40
    if time.time() < deadline - 20:
        res_40 = _try_find_lower_rank(2, 4, 6, 40, time_limit=80)
        if res_40:
            return res_40
            
    # Fallback to rank 42
    return _strassen_block_decomposition()
# EVOLVE-BLOCK-END
