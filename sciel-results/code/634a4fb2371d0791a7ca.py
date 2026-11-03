"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication.
    """
    terms = [
        # M1 = (P00+P11)*(Q00+Q11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (P10+P11)*Q00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = P00*(Q01-Q11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = P11*(Q10-Q00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (P00+P01)*Q11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (P10-P00)*(Q00+Q01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (P01-P11)*(Q10+Q11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    return terms

def _strassen_block_decomposition():
    """
    Construct a rank-42 decomposition using Strassen's algorithm.
    Split A into two 2x2 blocks, B into 2x6 halves with 3 col blocks of width 2.
    Each (2x2)*(2x2) uses 7 Strassen terms. Total: 2*3*7 = 42.
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

def _verify_decomposition(result, n=2, m=4, p=6, trials=20):
    """Verify a decomposition with integer arithmetic."""
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    
    for trial in range(trials):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
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

def _schoolbook_decomposition():
    """Fallback: rank-48 schoolbook decomposition."""
    n, m, p = DIMENSIONS
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

def _try_rank_reduction(result, max_time=10.0):
    """
    Attempt to reduce the rank of a valid decomposition by identifying
    linearly dependent columns in U, V, W and removing them.
    This is a heuristic approach: if columns are linearly dependent,
    we can potentially merge them.
    """
    start_time = time.time()
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    
    if rank <= 1:
        return result
    
    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float).T  # (rank, n*m)
    V_np = np.array(V, dtype=float).T  # (rank, m*p)
    W_np = np.array(W, dtype=float).T  # (rank, p*n)
    
    # Try to find rank-deficient sets of columns
    # If U, V, W have rank r < rank, we can project down
    # But we need to maintain the tensor identity
    
    # Simple approach: try to remove one column at a time
    # Check if column r is a linear combination of others
    # This is expensive, so we only try a few
    
    for target_rank in range(rank - 1, 0, -1):
        if time.time() - start_time > max_time:
            break
        
        # Try to project to target_rank
        # Compute SVD of U, V, W and keep top target_rank components
        try:
            U_svd, U_s, U_vt = np.linalg.svd(U_np, full_matrices=False)
            V_svd, V_s, V_vt = np.linalg.svd(V_np, full_matrices=False)
            W_svd, W_s, W_vt = np.linalg.svd(W_np, full_matrices=False)
            
            # Keep top target_rank components
            U_reduced = U_svd[:, :target_rank] @ np.diag(U_s[:target_rank]) @ U_vt[:target_rank, :]
            V_reduced = V_svd[:, :target_rank] @ np.diag(V_s[:target_rank]) @ V_vt[:target_rank, :]
            W_reduced = W_svd[:, :target_rank] @ np.diag(W_s[:target_rank]) @ W_vt[:target_rank, :]
            
            # This is not guaranteed to work, so verify
            new_result = {
                "U": U_reduced.T.tolist(),
                "V": V_reduced.T.tolist(),
                "W": W_reduced.T.tolist()
            }
            
            if _verify_decomposition(new_result, 2, 4, 6, trials=5):
                return new_result
        except:
            pass
    
    return result

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2 by 4
    matrix by a 4 by 6 matrix.
    
    Strategy: Use Strassen-based rank-42 decomposition.
    The output factor follows the official trace convention W[k*n+i,r].
    """
    start_time = time.time()
    deadline = start_time + 120
    
    # Try Strassen decomposition first (rank 42)
    result = _strassen_block_decomposition()
    
    # Verify the decomposition
    if _verify_decomposition(result, 2, 4, 6, trials=20):
        # Try to reduce rank further
        if time.time() < deadline - 15:
            reduced = _try_rank_reduction(result, max_time=10.0)
            if len(reduced['U'][0]) < len(result['U'][0]):
                return reduced
        return result
    
    # Check deadline
    if time.time() > deadline - 10:
        return _schoolbook_decomposition()
    
    # Fallback to schoolbook (rank 48)
    schoolbook = _schoolbook_decomposition()
    if _verify_decomposition(schoolbook, 2, 4, 6, trials=20):
        return schoolbook
    
    return schoolbook
# EVOLVE-BLOCK-END
