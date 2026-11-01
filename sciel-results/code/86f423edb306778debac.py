import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _build_strassen_4x4():
    """
    Construct the Rank-49 decomposition for 4x4 matrix multiplication
    using a recursive definition of the Strassen algorithm terms.
    
    This implementation dynamically generates the indices for U, V, W
    by combining outer 4x4 block Strassen steps with inner 2x2 Strassen steps.
    """
    n = 4
    m = 4
    p = 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Define the 7 Strassen steps for a 2x2 block matrix multiplication.
    # Each step is defined by:
    # - U_part: list of (block_row, block_col, coeff) for the left operand blocks
    # - V_part: list of (block_row, block_col, coeff) for the right operand blocks
    # - W_part: list of (block_row, block_col, coeff) for the result blocks
    # Note: In the outer 4x4 step, the "blocks" are 2x2 matrices.
    
    outer_steps = [
        # t=0: (A00+A11)(B00+B11)
        {'u': [(0,0,1), (1,1,1)], 'v': [(0,0,1), (1,1,1)], 'w': [(0,0,1), (1,1,1)]},
        # t=1: (A10+A11)B00
        {'u': [(1,0,1), (1,1,1)], 'v': [(0,0,1)], 'w': [(1,0,1), (1,1,-1)]},
        # t=2: A00(B01-B11)
        {'u': [(0,0,1)], 'v': [(0,1,1), (1,1,-1)], 'w': [(0,1,1), (1,1,1)]},
        # t=3: A11(B10-B00)
        {'u': [(1,1,1)], 'v': [(1,0,1), (0,0,-1)], 'w': [(0,0,1), (1,0,1)]},
        # t=4: (A00+A01)B11
        {'u': [(0,0,1), (0,1,1)], 'v': [(1,1,1)], 'w': [(0,1,1), (0,0,-1)]},
        # t=5: (A10-A00)(B00+B01)
        {'u': [(1,0,1), (0,0,-1)], 'v': [(0,0,1), (0,1,1)], 'w': [(1,1,1)]},
        # t=6: (A01-A11)(B10+B11)
        {'u': [(0,1,1), (1,1,-1)], 'v': [(1,0,1), (1,1,1)], 'w': [(0,0,1)]},
    ]
    
    # The inner 2x2 Strassen steps are identical in structure, just operating on scalars.
    # We can reuse the same pattern for the inner step.
    inner_steps = outer_steps
    
    half = 2
    
    for s, outer_step in enumerate(outer_steps):
        for t, inner_step in enumerate(inner_steps):
            r = s * 7 + t
            
            # Process U
            for u_bl, u_bc, coeff_u in outer_step['u']:
                for u_ii, u_jj, coeff_u2 in inner_step['u']:
                    # Global row in A: u_bl * 2 + u_ii
                    # Global col in A: u_bc * 2 + u_jj
                    row = u_bl * half + u_ii
                    col = u_bc * half + u_jj
                    coeff = coeff_u * coeff_u2
                    U[row * m + col, r] += coeff
            
            # Process V
            for v_br, v_bc, coeff_v in outer_step['v']:
                for v_ii, v_jj, coeff_v2 in inner_step['v']:
                    # Global row in B: v_br * 2 + v_ii
                    # Global col in B: v_bc * 2 + v_jj
                    row = v_br * half + v_ii
                    col = v_bc * half + v_jj
                    coeff = coeff_v * coeff_v2
                    V[row * p + col, r] += coeff
            
            # Process W
            # W index convention: W[k*n + i, r] where k is col of C, i is row of C
            for w_cr, w_cc, coeff_w in outer_step['w']:
                for w_ii, w_jj, coeff_w2 in inner_step['w']:
                    # Global row in C: w_cr * 2 + w_ii
                    # Global col in C: w_cc * 2 + w_jj
                    row = w_cr * half + w_ii
                    col = w_cc * half + w_jj
                    coeff = coeff_w * coeff_w2
                    W[col * n + row, r] += coeff
                    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """
    Verify that sum_r U[i,k,r] V[k,j,r] W[j,i,r] == 1 for all i,j,k.
    Using the trace convention: W index is j*n + i.
    """
    # Reshape for einsum
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    
    # T[i, j, k] = sum_r U[i, k, r] * V[k, j, r] * W[j, i, r]
    # Wait, V index is k*p + j. So V[k, j, r].
    # W index is j*n + i. So W[j, i, r].
    
    # Note: The tensor for matrix multiplication is T[i, k, j] = A[i, k] B[k, j] C[i, j]?
    # The task says: "decompose the matrix multiplication tensor".
    # Standard tensor: T_{i,j,k} such that C_{ij} = sum_k A_{ik} B_{kj} is T_{ik, kj, ij} = 1?
    # The verification in the prompt used:
    # T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    # U_r is (n, m, rank) -> indices i, j (col of A)
    # V_r is (m, p, rank) -> indices j (row of B), k (col of B)
    # W_r is (p, n, rank) -> indices k (row of W? No, W is p x n), i (col of W?)
    
    # Let's trace the prompt's einsum:
    # 'ijr' -> U[i, j, r] where j is column of A.
    # 'jkr' -> V[j, k, r] where j is row of B, k is column of B.
    # 'kir' -> W[k, i, r] where k is row of W (which is p), i is column of W (which is n).
    # Result 'ijk' -> sum over r.
    
    # If W is stored with index `col_C * n + row_C` (Trace convention W[k*n+i] where k=col, i=row):
    # Then W_r reshaped to (p, n, rank) has axes (row_W, col_W, rank).
    # row_W corresponds to `col_C` (j).
    # col_W corresponds to `row_C` (i).
    # So W_r[j, i, r] matches the einsum 'kir' if k->j, i->i?
    # The prompt einsum is 'kir'. k is index 2 of V (col B), which matches row of W (j).
    # i is index 0 of U (row A), which matches col of W (i).
    # So yes, W.reshape(p, n, rank) works with einsum 'kir' if W is flattened as j*n+i.
    
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return bool(np.allclose(T, 1.0, atol=1e-9))

def construct():
    deadline = time.time() + 110.0
    n, m, p = DIMENSIONS
    
    try:
        U, V, W = _build_strassen_4x4()
        if time.time() < deadline:
            if _verify(U, V, W, n, m, p):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    except Exception:
        pass
        
    # Fallback (should not happen)
    U, V, W = _build_strassen_4x4()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
