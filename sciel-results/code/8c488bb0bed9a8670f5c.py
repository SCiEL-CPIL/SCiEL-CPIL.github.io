"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
from itertools import product, combinations

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _build_decomposition(rank_terms):
    """
    Build U, V, W factor matrices from a list of term specifications.
    Each term is (U_dict, V_dict, W_dict) where each dict maps (i,j) or (j,k) or (k,i) to a scalar.
    """
    n, m, p = DIMENSIONS
    R = len(rank_terms)
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), val in u_dict.items():
            U[i * m + j, r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k, r] = val
        for (k, i), val in w_dict.items():
            W[k * n + i, r] = val
            
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    R = U.shape[1]
    U_reshaped = U.reshape((n, m, R))
    V_reshaped = V.reshape((m, p, R))
    W_reshaped = W.reshape((p, n, R))
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_recon, T_target, atol=atol)

def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for the 2x2 core (indices 0,1).
    """
    terms = []
    
    # m1 = (a00 + a11)(b00 + b11)
    terms.append(({(0,0): 1, (1,1): 1},
                  {(0,0): 1, (1,1): 1},
                  {(0,0): 1, (1,1): 1}))
    
    # m2 = (a10 + a11)b00
    terms.append(({(1,0): 1, (1,1): 1},
                  {(0,0): 1},
                  {(0,1): 1, (1,1): -1}))
    
    # m3 = a00(b01 - b11)
    terms.append(({(0,0): 1},
                  {(0,1): 1, (1,1): -1},
                  {(1,0): 1, (1,1): 1}))
    
    # m4 = a11(b10 - b00)
    terms.append(({(1,1): 1},
                  {(1,0): 1, (0,0): -1},
                  {(0,0): 1, (0,1): 1}))
    
    # m5 = (a00 + a01)b11
    terms.append(({(0,0): 1, (0,1): 1},
                  {(1,1): 1},
                  {(0,0): -1, (1,0): 1}))
    
    # m6 = (a10 - a00)(b00 + b01)
    terms.append(({(1,0): 1, (0,0): -1},
                  {(0,0): 1, (0,1): 1},
                  {(1,1): 1}))
    
    # m7 = (a01 - a11)(b10 + b11)
    terms.append(({(0,1): 1, (1,1): -1},
                  {(1,0): 1, (1,1): 1},
                  {(0,0): 1}))
    
    return terms

def _border_terms_standard():
    """
    Return standard multiplication terms for the border (at least one index is 2).
    """
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    return terms

def _get_all_possible_supports(n, m, p):
    """
    Get all possible (i,j), (j,k), (k,i) index pairs.
    Returns lists: U_supports, V_supports, W_supports
    """
    U_supports = [(i, j) for i in range(n) for j in range(m)]
    V_supports = [(j, k) for j in range(m) for k in range(p)]
    W_supports = [(k, i) for k in range(p) for i in range(n)]
    return U_supports, V_supports, W_supports

def _solve_milp_for_rank(R_max, n, m, p, timeout=60):
    """
    Try to find a rank-R decomposition using MILP.
    Binary variables x_u, x_v, x_w select which supports are used.
    Continuous variables solve for the actual values.
    
    The constraint is that for each (i,j,k):
    sum over r of U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    This is non-linear. Instead, we use a different approach:
    Fix the supports (which entries are non-zero) and solve for values.
    
    For a given rank R, we need to select R "terms" where each term has:
    - A subset of U indices (non-zero entries in column r)
    - A subset of V indices
    - A subset of W indices
    
    Then solve for the values such that the tensor equation holds.
    
    This is still complex. Let's use a simpler approach:
    Try all combinations of R "base terms" from a pool, and for each,
    solve a linear system to find if valid values exist.
    """
    start_time = time.time()
    
    U_supports, V_supports, W_supports = _get_all_possible_supports(n, m, p)
    
    # For each rank R, we need to find R vectors (u_r, v_r, w_r) such that
    # sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all (i,j,k)
    
    # This is equivalent to finding a rank-R decomposition of the tensor T[i,j,k]=1
    # where each rank-1 component is a tensor product of three vectors.
    
    # Approach: Use iterative refinement with random initialization
    # and alternating least squares (ALS)
    
    best_result = None
    best_rank = R_max + 1
    
    for R in range(23, R_max + 1):  # Try ranks from 23 upward
        if time.time() - start_time > timeout:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() - start_time > timeout:
                break
            
            # Random initialization
            U_rand = np.random.randn(n * m, R)
            V_rand = np.random.randn(m * p, R)
            W_rand = np.random.randn(p * n, R)
            
            # ALS iterations
            converged = False
            for it in range(200):
                if time.time() - start_time > timeout:
                    break
                
                # Update U
                # Actually, let's use a simpler projection method
                # Residual approach:
                U_reshaped = U_rand.reshape((n, m, R))
                V_reshaped = V_rand.reshape((m, p, R))
                W_reshaped = W_rand.reshape((p, n, R))
                
                T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
                residual = np.ones((n, m, p)) - T_recon
                
                # Update U: for each (i,j), U[i,j,:] = argmin sum_r U[i,j,r]*V[j,:,r]@W[:,i,r] vs 1
                # This is a linear system for U[i,j,:]
                # Let A[i,j] be a matrix where A[i,j][r, r'] = sum_k V[j,k,r]*W[k,i,r']
                # Then we want: A[i,j] @ U[i,j,:] ≈ 1 (scalar)
                # This is underdetermined. We minimize ||A[i,j]@U[i,j,:] - 1||^2
                # Solution: U[i,j,:] = A[i,j].T @ (A[i,j] @ A[i,j].T)^{-1} * 1
                
                for i in range(n):
                    for j in range(m):
                        # Build A matrix: A[r, r'] = sum_k V[j,k,r] * W[k,i,r']
                        A = np.zeros((R, R))
                        for k in range(p):
                            A[:, r] = A[:, r]  # placeholder
                            A = A + np.outer(V_reshaped[j, k, :], W_reshaped[k, i, :])
                        
                        # Solve A @ u = 1 (scalar), minimize ||A@u - 1||^2
                        # u = A.T @ (A @ A.T)^{-1} * 1
                        try:
                            AT = A.T
                            AT_inv = np.linalg.pinv(AT @ A)
                            u_val = AT @ AT_inv @ np.ones(R)
                            U_rand[i * m + j, :] = u_val
                        except:
                            pass
                
                # Update V similarly
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((R, R))
                        for i in range(n):
                            A = A + np.outer(U_reshaped[i, j, :], W_reshaped[k, i, :])
                        try:
                            AT = A.T
                            AT_inv = np.linalg.pinv(AT @ A)
                            v_val = AT @ AT_inv @ np.ones(R)
                            V_rand[j * p + k, :] = v_val
                        except:
                            pass
                
                # Update W similarly
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((R, R))
                        for j in range(m):
                            A = A + np.outer(U_reshaped[i, j, :], V_reshaped[j, k, :])
                        try:
                            AT = A.T
                            AT_inv = np.linalg.pinv(AT @ A)
                            w_val = AT @ AT_inv @ np.ones(R)
                            W_rand[k * n + i, :] = w_val
                        except:
                            pass
                
                # Check convergence
                T_recon = np.einsum('ijr,jkr,kir->ijk', 
                                    U_rand.reshape((n, m, R)), 
                                    V_rand.reshape((m, p, R)), 
                                    W_rand.reshape((p, n, R)))
                if np.allclose(T_recon, np.ones((n, m, p)), atol=1e-6):
                    converged = True
                    break
                
                # Regularization to prevent blowup
                U_rand *= 0.99
                V_rand *= 0.99
                W_rand *= 0.99
            
            if converged:
                # Verify with tighter tolerance
                if _verify_decomposition(U_rand, V_rand, W_rand, n, m, p, atol=1e-8):
                    # Try to round to half-integers
                    U_rounded = np.round(U_rand * 2) / 2
                    V_rounded = np.round(V_rand * 2) / 2
                    W_rounded = np.round(W_rand * 2) / 2
                    
                    if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, atol=1e-8):
                        best_result = (U_rounded, V_rounded, W_rounded)
                        best_rank = R
                        break
                    else:
                        best_result = (U_rand, V_rand, W_rand)
                        best_rank = R
                        break
        
        if best_rank <= R:
            break
    
    if best_result is not None:
        return best_result
    return None

def construct():
    """
    Returns a minimal-rank decomposition for 3x3 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 40s safety margin
    
    n, m, p = DIMENSIONS
    
    # First try MILP/ALS approach to find lower rank
    try:
        result = _solve_milp_for_rank(26, n, m, p, timeout=90)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p, atol=1e-8):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass
    
    # Fallback: Strassen core + Standard border (Rank 26)
    terms = _strassen_2x2_terms() + _border_terms_standard()
    U, V, W = _build_decomposition(terms)
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Final fallback: Standard multiplication (Rank 27)
    terms_std = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms_std.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    U, V, W = _build_decomposition(terms_std)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
