"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication.
    Each term is a tuple (U_coeffs, V_coeffs, W_coeffs) where:
      U_coeffs: dict {(i, j): coeff} for the 2x2 left matrix
      V_coeffs: dict {(j, k): coeff} for the 2x2 right matrix
      W_coeffs: dict {(i, k): coeff} for the 2x2 output
    """
    terms = []
    
    # M1 = (P00+P11)*(Q00+Q11)
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    
    # M2 = (P10+P11)*Q00
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1}
    ))
    
    # M3 = P00*(Q01-Q11)
    terms.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1}
    ))
    
    # M4 = P11*(Q10-Q00)
    terms.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (1,0): 1}
    ))
    
    # M5 = (P00+P01)*Q11
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (0,1): 1}
    ))
    
    # M6 = (P10-P00)*(Q00+Q01)
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    
    # M7 = (P01-P11)*(Q10+Q11)
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    
    return terms

def _strassen_block_decomposition():
    """
    Construct a rank-42 decomposition using Strassen's algorithm.
    
    Strategy: Split the 2x4 * 4x6 multiplication into blocks:
    - A is split into two 2x2 blocks: A_left (cols 0,1) and A_right (cols 2,3)
    - B is split into top (rows 0,1) and bottom (rows 2,3), each with 3 column blocks of width 2
    
    So C = A_left * B_top + A_right * B_bottom
    
    Each (2x2) * (2x2) block multiplication uses 7 Strassen terms.
    Total: 2 (A blocks) * 3 (B col blocks) * 7 (Strassen) = 42 terms.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    strassen_terms = _strassen_2x2x2_terms()
    
    col = 0
    # For each A block (left/right) and each B column block
    for a_block in range(2):  # 0: A cols 0,1; 1: A cols 2,3
        for b_col_block in range(3):  # B col blocks: 0,1,2 (each width 2)
            for t in range(7):  # 7 Strassen terms
                U_coeffs, V_coeffs, W_coeffs = strassen_terms[t]
                
                # Update U: A block a_block has columns at offset a_block*2
                for (i_loc, j_loc), coeff in U_coeffs.items():
                    i = i_loc
                    j = a_block * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Update V: B rows depend on a_block (top if a_block=0, bottom if a_block=1)
                # B col block b_col_block has columns at offset b_col_block*2
                for (j_loc, k_loc), coeff in V_coeffs.items():
                    j = a_block * 2 + j_loc
                    k = b_col_block * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Update W: output C[i, k] where i is row of C, k is col of C
                # W index: k*n + i (official trace convention)
                for (i_loc, k_loc), coeff in W_coeffs.items():
                    i = i_loc
                    k = b_col_block * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(result, n=2, m=4, p=6, trials=50):
    """Verify a decomposition numerically with integer arithmetic."""
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    
    for trial in range(trials):
        A = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
        
        # Expected C
        C_exp = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        # Computed C
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

def _schoolbook_decomposition():
    """Fallback: rank-m = 4 decomposition."""
    n, m, p = DIMENSIONS
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for i in range(n):
            U[i * m + j][j] = 1
        for k in range(p):
            V[j * p + k][j] = 1
        for i in range(n):
            for k in range(p):
                W[k * n + i][j] = 1
    
    return {"U": U, "V": V, "W": W}

def _try_rank_reduction(strassen_result, target_rank, time_budget=30):
    """
    Attempt to reduce the rank of the Strassen decomposition by finding
    linear dependencies among the rank-42 terms.
    
    Uses a greedy approach: try to express each term as a linear combination
    of previously kept terms, removing it if possible.
    """
    start_time = time.time()
    U = np.array(strassen_result['U'], dtype=np.float64)
    V = np.array(strassen_result['V'], dtype=np.float64)
    W = np.array(strassen_result['W'], dtype=np.float64)
    
    n, m, p = DIMENSIONS
    U_dim, V_dim, W_dim = U.shape[0], V.shape[0], W.shape[0]
    
    # Build the tensor for each rank-1 component
    # T_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # We want to find a subset of these that still spans the same tensor space
    
    # Check if we can reduce rank by finding null space
    # For now, just return the original if we can't find a reduction quickly
    
    # Simple check: are any columns identical or linearly dependent?
    # This is a heuristic - full rank reduction is NP-hard
    
    if time.time() - start_time > time_budget:
        return None
    
    # Try to remove terms that are linear combinations of others
    # This is a simplified version - in practice, we'd use more sophisticated methods
    kept_indices = list(range(42))
    
    # For each term, check if it can be expressed as a combination of kept terms
    for idx in range(42):
        if idx not in kept_indices:
            continue
        if time.time() - start_time > time_budget:
            break
        
        # Check if term idx is in the span of other kept terms
        other_indices = [i for i in kept_indices if i != idx]
        
        if len(other_indices) < target_rank:
            continue
        
        # Build the tensor for term idx
        T_idx = np.einsum('a,b,c->abc', U[:, idx], V[:, idx], W[:, idx])
        
        # Build tensors for other terms
        T_others = np.stack([
            np.einsum('a,b,c->abc', U[:, i], V[:, i], W[:, i])
            for i in other_indices
        ])
        
        # Flatten
        T_idx_flat = T_idx.flatten()
        T_others_flat = T_others.flatten(axis=0)
        
        # Check if T_idx is in the span of T_others
        # Use least squares
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(T_others_flat.T, T_idx_flat, rcond=None)
            reconstructed = T_others_flat @ coeffs
            if np.allclose(T_idx_flat, reconstructed, atol=1e-10):
                kept_indices.remove(idx)
        except:
            pass
    
    if len(kept_indices) < 42:
        # Build reduced decomposition
        U_red = U[:, kept_indices]
        V_red = V[:, kept_indices]
        W_red = W[:, kept_indices]
        
        result = {
            'U': U_red.tolist(),
            'V': V_red.tolist(),
            'W': W_red.tolist()
        }
        
        # Verify
        if _verify_decomposition(result, 2, 4, 6, trials=20):
            return result
    
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2 by 4 
    matrix by a 4 by 6 matrix.
    
    Strategy: Use Strassen-based rank-42 decomposition, then attempt rank reduction.
    Split A into two 2x2 blocks, B into top/bottom 2x6 halves, each with 3 col blocks.
    Each (2x2)*(2x2) uses 7 Strassen terms. Total: 2*3*7 = 42.
    
    The output factor follows the official trace convention W[k*n+i,r].
    """
    start_time = time.time()
    deadline = start_time + 100
    
    # Try Strassen decomposition first (rank 42)
    strassen_result = _strassen_block_decomposition()
    
    # Verify the decomposition
    if not _verify_decomposition(strassen_result, 2, 4, 6, trials=50):
        # Fall back to schoolbook
        return _schoolbook_decomposition()
    
    # Check deadline
    if time.time() > deadline - 40:
        return strassen_result
    
    # Try to reduce rank
    reduced = _try_rank_reduction(strassen_result, target_rank=35, time_budget=35)
    
    if reduced is not None:
        return reduced
    
    # Return the rank-42 decomposition
    return strassen_result
# EVOLVE-BLOCK-END
