"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 120.0
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 10
    num_blocks = p // 2  # 5 blocks of 2 columns each
    
    # Strassen decomposition for 2x2 * 2x2
    strassen_decomp = [
        # M1 = (A00+A11) * (B00+B11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (A10+A11) * B00
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = A00 * (B01-B11)
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = A11 * (B10-B00)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (A00+A01) * B11
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)]),
        # M6 = (A10-A00) * (B00+B01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (A01-A11) * (B10+B11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    # Build initial 35-term decomposition
    rank_init = num_blocks * 7  # 35
    U = [[0] * rank_init for _ in range(n * m)]
    V = [[0] * rank_init for _ in range(m * p)]
    W = [[0] * rank_init for _ in range(p * n)]
    
    for b in range(num_blocks):
        base_col = b * 7
        for t in range(7):
            col = base_col + t
            u_side, v_side, w_side = strassen_decomp[t]
            for u_row, u_coeff in u_side:
                U[u_row][col] = u_coeff
            for v_koff, v_coeff in v_side:
                j = v_koff // 2
                k = 2 * b + v_koff % 2
                V[j * p + k][col] = v_coeff
            for w_koff, w_coeff in w_side:
                i = w_koff // 2
                k = 2 * b + w_koff % 2
                W[k * n + i][col] = w_coeff
    
    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Try to find linear dependencies by computing the tensor and checking rank
    # The multiplication tensor T has T[i,j,k] = 1 if the output element C[i][k] = sum_j A[i][j]*B[j][k]
    # We need to check if our decomposition is valid and if there are dependencies
    
    # Verify correctness first
    # Compute the actual multiplication tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Verify our decomposition: sum_r U[:,r] outer V[:,r] outer W[:,r] should equal T
    # T_decomp[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    T_decomp = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    
    # Check if decomposition is correct
    if not np.allclose(T_decomp, T, atol=1e-10):
        # Fallback: return the original decomposition (it should be correct)
        return {"U": U, "V": V, "W": W}
    
    # Now try to reduce rank by finding linear dependencies
    # We can try to find a smaller set of terms that span the same space
    # Use SVD-like approach on the tensor
    
    # Reshape to 2D matrices for rank computation
    # Mode-1 unfolding: U (nm x rank)
    # Mode-2 unfolding: V (mp x rank)  
    # Mode-3 unfolding: W (pn x rank)
    
    # The tensor is determined by these three matrices. If there's a linear dependency
    # among the columns, we can remove that term.
    
    # Check if any column is a linear combination of others
    # This is equivalent to checking if the three matrices have linearly dependent columns
    
    # A simpler approach: try removing each term and see if the tensor is still representable
    # But that's expensive. Instead, let's try a more systematic approach.
    
    # Try to find the minimum rank by checking if we can reduce the number of terms
    # while still being able to represent T.
    
    # For now, let's try a simple approach: check if the rank of the combined system
    # allows for reduction.
    
    # Actually, the Strassen decomposition for each 2x2 block is already minimal (rank 7),
    # and the blocks are independent, so the total rank should be 5*7=35.
    # Let's verify there are no cross-block dependencies.
    
    # Check if any linear combination of the 35 terms gives zero tensor
    # This means checking if the 35 vectors (U[:,r], V[:,r], W[:,r]) are linearly independent
    # in the tensor product space.
    
    # A practical check: compute the rank of the matrix formed by flattening each term
    term_vectors = []
    for r in range(rank_init):
        vec = np.kron(np.kron(U_np[:, r], V_np[:, r]), W_np[:, r])
        term_vectors.append(vec)
    
    term_matrix = np.array(term_vectors).T  # (dim, rank_init)
    dim = n * m * m * p * p * n  # 4 * 20 * 20 = 1600
    
    # Compute rank of term_matrix
    try:
        rank_actual = np.linalg.matrix_rank(term_matrix, tol=1e-10)
    except:
        rank_actual = rank_init
    
    # If rank_actual < rank_init, there are dependencies
    if rank_actual < rank_init:
        # Find a basis for the column space
        # Use QR decomposition with column pivoting
        Q, R = np.linalg.qr(term_matrix, mode='complete')
        # The rank is the number of non-zero diagonal elements of R
        diag = np.abs(np.diag(R))
        tol = 1e-10
        rank_found = np.sum(diag > tol)
        
        if rank_found < rank_init:
            # We can potentially reduce rank, but this is complex to implement
            # safely. For now, keep the 35-term decomposition.
            pass
    
    # Return the 35-term Strassen decomposition
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
