import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Rank is 49 + 28 = 77
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen steps for 2x2x2
    strassen_steps_222 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    # Part 1: 4x4x4 part (cols 0-48)
    col_start_1 = 0
    
    for s in range(7):
        alpha_4, beta_4, gamma_4 = strassen_steps_222[s]
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
            
            col = col_start_1 + s * 7 + s2
            
            for (bi, bj), val_a4 in alpha_4.items():
                for (si, sj), val_a2 in alpha_2.items():
                    r = (bi * 2 + si) * m + (bj * 2 + sj)
                    U[r][col] += val_a4 * val_a2
                    
            for (bi, bj), val_b4 in beta_4.items():
                for (si, sj), val_b2 in beta_2.items():
                    r = (bi * 2 + si) * p + (bj * 2 + sj)
                    V[r][col] += val_b4 * val_b2
                    
            for (bi, bj), val_c4 in gamma_4.items():
                for (si, sj), val_c2 in gamma_2.items():
                    r = (bj * 2 + sj) * n + (bi * 2 + si)
                    W[r][col] += val_c4 * val_c2

    # Part 2: 4x4x2 part (cols 49-76)
    col_start_2 = 49
    
    mults = [
        ((0,0), (0,0)),
        ((0,1), (1,0)),
        ((1,0), (0,0)),
        ((1,1), (1,0))
    ]
    
    for m_idx, (a_block, b_block) in enumerate(mults):
        c_block = (a_block[0], 0)
        
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
            col = col_start_2 + m_idx * 7 + s2
            
            for (si, sj), val_a2 in alpha_2.items():
                r = (a_block[0] * 2 + si) * m + (a_block[1] * 2 + sj)
                U[r][col] += val_a2
                
            for (si, sj), val_b2 in beta_2.items():
                r = (b_block[0] * 2 + si) * p + (4 + sj)
                V[r][col] += val_b2
                
            for (si, sj), val_c2 in gamma_2.items():
                r = (4 + sj) * n + (c_block[0] * 2 + si)
                W[r][col] += val_c2

    # Try to reduce rank by finding linear dependencies
    # Build the tensor as a flattened 3D structure
    # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # If we can find coefficients c_r such that sum_r c_r * U[:,r]⊗V[:,r]⊗W[:,r] = 0,
    # we can eliminate those terms.
    
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    # Build matrix where each column r corresponds to the rank-1 tensor U[:,r]⊗V[:,r]⊗W[:,r]
    # Flatten each rank-1 tensor into a vector of length n*m*p
    # Then find the rank of the matrix of these vectors
    
    nm = n * m
    mp = m * p
    pn = p * n
    total_dim = nm * mp * pn  # 16*24*24 = 9216
    
    # This is too large for exact rational arithmetic in reasonable time
    # Instead, use a simpler approach: check if the 77 vectors are linearly independent
    # by computing the rank of the 3-mode flattening
    
    # Alternative: use the fact that the tensor is a sum of rank-1 terms.
    # Check for dependencies by solving a system.
    
    # Given time constraints, let's try a numerical approach first
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Build the tensor T of shape (n*m, m*p, p*n)
    # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # This can be written as: for each r, T += np.einsum('ir,jr,kr->ijk', U[:,r], V[:,r], W[:,r])
    
    # But we want to find if some columns are redundant.
    # A column r is redundant if there exist coefficients c_1,...,c_77 (not all zero) 
    # such that sum_s c_s * U[:,s]⊗V[:,s]⊗W[:,s] = 0.
    
    # This is equivalent to: the vectors v_r = U[:,r]⊗V[:,r]⊗W[:,r] in R^(nm*mp*pn) 
    # are linearly dependent.
    
    # Flatten each v_r into a vector of length nm*mp*pn = 16*24*24 = 9216
    # Then we have a 9216 x 77 matrix. Find its rank.
    
    if time.time() > deadline - 30:
        # Not enough time for rank reduction, return as-is
        return {"U": U, "V": V, "W": W}
    
    # Build the 9216 x 77 matrix
    # Each column r is the Kronecker product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # Actually, it's the vectorized form: for each (i,j,k), the entry is U[i,r]*V[j,r]*W[k,r]
    
    # We can build this more efficiently:
    # M[i,j,k,r] = U[i,r]*V[j,r]*W[k,r]
    # Flatten (i,j,k) to get a 9216 x 77 matrix
    
    # Use einsum to build the matrix
    # M_flat[row, r] where row = i*mp*pn + j*pn + k
    
    # This is memory intensive: 9216 * 77 * 8 bytes = ~5.7 MB, which is fine
    
    # Build column by column
    M = np.zeros((total_dim, rank), dtype=np.float64)
    for r in range(rank):
        # Vectorized outer product
        vec = np.einsum('i,j,k->ijk', U_np[:,r], V_np[:,r], W_np[:,r]).flatten()
        M[:,r] = vec
    
    # Find the rank of M using SVD
    try:
        U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
        # Rank is the number of singular values above tolerance
        tol = max(M.shape) * np.finfo(np.float64).eps * S[0] if len(S) > 0 else 0
        num_rank = np.sum(S > tol)
        
        if num_rank < rank:
            # We can reduce the rank
            # Keep only the first num_rank principal components
            # But we need to reconstruct the decomposition exactly...
            # This is tricky with floating point. Let's try a different approach.
            
            # Instead, find which columns are linearly dependent
            # and remove them, then verify the decomposition is still exact.
            
            # For now, if we find the rank is less, we can try to find a subset
            # of columns that spans the same space.
            
            # Use QR decomposition to find a basis
            Q, R = np.linalg.qr(M)
            # Find pivot columns
            diag_R = np.diag(R)
            pivot_cols = np.argsort(np.abs(diag_R))[::-1]
            
            # The first num_rank pivot columns form a basis
            # But we need to express the remaining columns as combinations...
            
            # This is getting complex. Let's just check if the rank is actually 77.
            # If it's less, we need a more sophisticated reduction.
            
            if num_rank < rank:
                # Attempt reduction: find a minimal subset
                # This requires solving a linear system, which may be slow
                pass
    except np.linalg.LinAlgError:
        pass
    
    # For now, return the original decomposition
    # The rank reduction requires more careful exact arithmetic
    return {"U": U, "V": V, "W": W}
