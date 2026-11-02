import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def _strassen_factors():
    """Return (U1, V1, W1) for 2x2x2 Strassen, shapes (4,7),(4,7),(4,7)."""
    U1 = [[0] * 7 for _ in range(4)]
    V1 = [[0] * 7 for _ in range(4)]
    W1 = [[0] * 7 for _ in range(4)]

    # M1 = (A00+A11)(B00+B11)
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1
    W1[3][0] = 1

    # M2 = (A10+A11)B00
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1
    W1[3][1] = -1

    # M3 = A00(B01-B11)
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[2][2] = 1
    W1[3][2] = 1

    # M4 = A11(B10-B00)
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    W1[0][3] = 1
    W1[1][3] = 1

    # M5 = (A00+A01)B11
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1
    W1[2][4] = 1

    # M6 = (A10-A00)(B00+B01)
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1

    # M7 = (A01-A11)(B10+B11)
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1

    return U1, V1, W1

def _build_rank42():
    n, m, p = 2, 4, 6
    U1, V1, W1 = _strassen_factors()
    total_rank = 42
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    for j_block in range(3):
        for side in range(2):
            col_offset = (j_block * 2 + side) * 7
            for i in range(2):
                for j_local in range(2):
                    a = side * 2 + j_local
                    for s in range(7):
                        coeff = U1[i * 2 + j_local][s]
                        if coeff != 0:
                            U[i * m + a][col_offset + s] += coeff
            for j_local in range(2):
                a = side * 2 + j_local
                for k_local in range(2):
                    bcol = j_block * 2 + k_local
                    for s in range(7):
                        coeff = V1[j_local * 2 + k_local][s]
                        if coeff != 0:
                            V[a * p + bcol][col_offset + s] += coeff
            for i in range(2):
                for k_local in range(2):
                    bcol = j_block * 2 + k_local
                    for s in range(7):
                        coeff = W1[k_local * 2 + i][s]
                        if coeff != 0:
                            W[bcol * n + i][col_offset + s] += coeff
    return U, V, W

def _verify_exact(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for s in range(rank):
                    val += U[i * m + j][s] * V[j * p + k][s] * W[k * n + i][s]
                if val != 1:
                    return False
    return True

def _try_rank40():
    """Attempt to find a rank-40 decomposition by exploiting linear dependencies
    in the 6x Strassen block structure.
    
    The rank-42 decomposition has 6 blocks, each using 7 Strassen terms.
    If we can express 2 of the 42 rank-1 terms as linear combinations of others,
    we get rank 40.
    """
    n, m, p = 2, 4, 6
    U1, V1, W1 = _strassen_factors()
    
    # Build the full rank-42 structure first
    U42, V42, W42 = _build_rank42()
    
    # Convert to numpy for linear algebra
    U42_np = np.array(U42, dtype=float)
    V42_np = np.array(V42, dtype=float)
    W42_np = np.array(W42, dtype=float)
    
    # The tensor T[i,j,k] = sum_s U[i,j,s] * V[j,k,s] * W[k,i,s]
    # For rank reduction, we need to find if columns of the "flattened" 
    # representation are linearly dependent.
    
    # Alternative approach: Try to find a rank-40 by merging two blocks
    # that have some structure in common.
    
    # Let's try a different approach: use the fact that for 2x4x6,
    # we can potentially use a 2x2x2 Strassen for the 2x2 part and 
    # handle the 4x6 part more cleverly.
    
    # Actually, let's try to directly search for rank-40 by solving
    # a linear system with constraints.
    
    # For a rank-r decomposition: T[i,j,k] = sum_{s=1}^r u[i,j,s] v[j,k,s] w[k,i,s]
    # This is trilinear. We can fix U and V and solve for W linearly.
    
    # Strategy: Start from rank-42, try to set 2 columns to zero and 
    # see if we can adjust the remaining 40 to still satisfy all equations.
    
    # This is a nonlinear problem, but we can try a linearized approach.
    
    # Let's try a simpler heuristic: check if any two of the 42 rank-1 terms
    # are linearly dependent in some sense.
    
    # For each pair (s1, s2), check if we can write term s2 as a multiple 
    # of term s1 in the tensor space.
    
    # Tensor space dimension: n*m*p = 2*4*6 = 48
    # Each rank-1 term contributes a 48-dim vector (flattened tensor)
    
    def flatten_term(u_col, v_col, w_col):
        """Flatten a rank-1 term into a 48-dim vector."""
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = (i * m + j) * p + k
                    vec[idx] = u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
        return vec
    
    # Build the 48x42 matrix of rank-1 terms
    T_matrix = np.zeros((n * m * p, 42))
    for s in range(42):
        T_matrix[:, s] = flatten_term(U42_np[:, s], V42_np[:, s], W42_np[:, s])
    
    # Check rank of this matrix
    rank = np.linalg.matrix_rank(T_matrix, tol=1e-10)
    
    # If rank < 42, there are linear dependencies
    if rank < 42:
        # Find null space vectors
        # For each null space vector c, sum_s c[s] * term_s = 0
        # This means we can potentially eliminate some terms.
        
        # Get null space
        from scipy.linalg import null_space
        null_vecs = null_space(T_matrix, rcond=1e-10)
        
        # For each null space vector, identify which terms can be eliminated
        for nv in null_vecs:
            # Find the largest coefficient in magnitude
            max_idx = np.argmax(np.abs(nv))
            max_coeff = nv[max_idx]
            
            if abs(max_coeff) < 1e-10:
                continue
            
            # We can express term max_idx as a combination of others
            # This suggests we might be able to reduce rank
            
            # Try to construct a rank-41 decomposition by eliminating term max_idx
            # and adjusting others accordingly
            new_U = U42_np.copy()
            new_V = V42_np.copy()
            new_W = W42_np.copy()
            
            # Set the max_idx column to zero
            new_U[:, max_idx] = 0
            new_V[:, max_idx] = 0
            new_W[:, max_idx] = 0
            
            # Adjust other columns: for s != max_idx, add (nv[s]/nv[max_idx]) * term_max_idx
            # But this is not straightforward since we need to maintain the product form.
            
            # This approach is too complex. Let's try a different strategy.
            pass
    
    # Alternative: Try to find a rank-40 by using a different block structure.
    # For 2x4x6, we can view it as 2x(4x6) or (2x4)x6.
    
    # Let's try: use Strassen for the 2x2 block structure, but with 
    # optimized handling of the 4x6 part.
    
    # Actually, the current rank-42 is already quite good. Let's try to
    # verify if rank-40 is possible by checking the tensor rank bounds.
    
    # For 2x4x6 matrix multiplication, known lower bounds suggest rank >= 40.
    # Let's try to construct a rank-40 decomposition.
    
    # One approach: use the fact that 4 = 2+2 and 6 = 3+3, and use
    # different Strassen-like decompositions for each sub-block.
    
    # For now, return None to indicate no improvement found
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try to find a rank-40 decomposition
    rank40_result = _try_rank40()
    if rank40_result is not None:
        U, V, W = rank40_result
        if _verify_exact(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Return the known rank-42 decomposition
    U, V, W = _build_rank42()
    
    if _verify_exact(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
        
    # Fallback schoolbook
    schoolbook_rank = n * m * p
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U_fb[ii * m + jj][col] = 1
                V_fb[jj * p + kk][col] = 1
                W_fb[kk * n + ii][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
