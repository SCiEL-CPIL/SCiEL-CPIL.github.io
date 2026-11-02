"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import itertools

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
    terms.append(({(0,0): 1, (1,1): 1},
                  {(0,0): 1, (1,1): 1},
                  {(0,0): 1, (1,1): 1}))
    terms.append(({(1,0): 1, (1,1): 1},
                  {(0,0): 1},
                  {(0,1): 1, (1,1): -1}))
    terms.append(({(0,0): 1},
                  {(0,1): 1, (1,1): -1},
                  {(1,0): 1, (1,1): 1}))
    terms.append(({(1,1): 1},
                  {(1,0): 1, (0,0): -1},
                  {(0,0): 1, (0,1): 1}))
    terms.append(({(0,0): 1, (0,1): 1},
                  {(1,1): 1},
                  {(0,0): -1, (1,0): 1}))
    terms.append(({(1,0): 1, (0,0): -1},
                  {(0,0): 1, (0,1): 1},
                  {(1,1): 1}))
    terms.append(({(0,1): 1, (1,1): -1},
                  {(1,0): 1, (1,1): 1},
                  {(0,0): 1}))
    return terms

def _border_terms_standard():
    """Standard border terms (at least one index is 2)."""
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    return terms

def _try_standard_decomposition():
    """Standard schoolbook: rank 27."""
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    U, V, W = _build_decomposition(terms)
    return U, V, W

def _find_rank_reduction(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank by finding if any column can be expressed as 
    a combination that allows elimination. Uses linear algebra approach.
    """
    R = U.shape[1]
    best_rank = R
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try each column for elimination
    for col in range(R):
        if time.time() > deadline:
            break
        
        # Check if removing this column and adjusting others can still work
        # This is a complex optimization, so we use a heuristic:
        # Try to see if the tensor contribution of this column can be 
        # matched by a linear combination of other columns
        
        if R <= 1:
            break
        
        # Create a copy without this column
        mask = [i for i in range(R) if i != col]
        U_reduced = U[:, mask]
        V_reduced = V[:, mask]
        W_reduced = W[:, mask]
        
        # Check if the tensor can be represented with reduced rank
        # by solving for adjustments. This is NP-hard in general, so 
        # we use a simple check: can we find coefficients such that 
        # the missing term is covered?
        
        # For a quick heuristic, check if the column is "redundant"
        # by seeing if its contribution is already covered
        U_r = U_reduced.reshape(n, m, len(mask))
        V_r = V_reduced.reshape(m, p, len(mask))
        W_r = W_reduced.reshape(p, n, len(mask))
        
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        T_target = np.ones((n, m, p))
        
        # The difference is what the removed column provided
        T_missing = T_target - T_recon
        
        # Check if T_missing can be represented as a rank-1 tensor
        # T_missing[i,j,k] = U_col[i,j] * V_col[j,k] * W_col[k,i]
        # This is true iff T_missing has rank 1 in the tensor sense
        
        # Simple check: if all entries of T_missing are zero, the column was redundant
        if np.allclose(T_missing, 0, atol=1e-9):
            best_rank = len(mask)
            best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
            U, V, W = U_reduced, V_reduced, W_reduced
            R = len(mask)
            col = -1  # restart from beginning since indices changed
            continue
        
        # Try a more sophisticated approach: see if we can adjust 
        # the remaining columns to cover the missing term
        # This requires solving a system, which is expensive
        # For now, skip and try next column
    
    return best_U, best_V, best_W, best_rank

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Strassen core + standard border (rank 26)
    if time.time() < deadline:
        terms = _strassen_2x2_terms() + _border_terms_standard()
        U, V, W = _build_decomposition(terms)
        if _verify_decomposition(U, V, W, n, m, p):
            R = U.shape[1]
            if R < best_rank:
                best_rank = R
                best_result = (U, V, W)
    
    # Strategy 2: Try rank reduction on the rank-26 solution
    if best_result is not None and time.time() < deadline:
        U, V, W = best_result
        U_red, V_red, W_red, rank_red = _find_rank_reduction(U, V, W, n, m, p, deadline)
        if rank_red < best_rank:
            if _verify_decomposition(U_red, V_red, W_red, n, m, p):
                best_rank = rank_red
                best_result = (U_red, V_red, W_red)
    
    # Strategy 3: Pure standard (rank 27) as fallback
    if time.time() < deadline:
        U, V, W = _try_standard_decomposition()
        if _verify_decomposition(U, V, W, n, m, p):
            R = U.shape[1]
            if R < best_rank:
                best_rank = R
                best_result = (U, V, W)
    
    if best_result is None:
        U, V, W = _try_standard_decomposition()
        best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
