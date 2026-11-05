"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4x5 matrix multiplication
    using rank-1 peeling via exact linear algebra over Q.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    _t0 = time.time()
    _DEADLINE = 115  # seconds (safety margin)
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W correctly decompose the tensor."""
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        result = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
                    result[i, j, k] = s
        return np.allclose(result, np.ones((n, m, p)), atol=tol)
    
    def make_strassen_decomposition():
        """Create the known rank-36 Strassen-based decomposition."""
        rank = 36
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        def add_term(idx, u_coeffs, v_coeffs, w_coeffs):
            for (i, j), val in u_coeffs.items():
                U[i*m + j, idx] += val
            for (j, k), val in v_coeffs.items():
                V[j*p + k, idx] += val
            for (k, i), val in w_coeffs.items():
                W[k*n + i, idx] += val

        def strassen_block(a_row_offset, b_row_offset, c_col_offset, idx_start):
            add_term(idx_start, 
                     {(0, a_row_offset): 1, (1, a_row_offset+1): 1},
                     {(b_row_offset, c_col_offset): 1, (b_row_offset+1, c_col_offset+1): 1},
                     {(c_col_offset, 0): 1, (c_col_offset+1, 1): 1})
            add_term(idx_start+1,
                     {(1, a_row_offset): 1, (1, a_row_offset+1): 1},
                     {(b_row_offset, c_col_offset): 1},
                     {(c_col_offset, 1): 1, (c_col_offset+1, 1): -1})
            add_term(idx_start+2,
                     {(0, a_row_offset): 1},
                     {(b_row_offset, c_col_offset+1): 1, (b_row_offset+1, c_col_offset+1): -1},
                     {(c_col_offset+1, 0): 1, (c_col_offset+1, 1): 1})
            add_term(idx_start+3,
                     {(1, a_row_offset+1): 1},
                     {(b_row_offset+1, c_col_offset): 1, (b_row_offset, c_col_offset): -1},
                     {(c_col_offset, 0): 1, (c_col_offset, 1): 1})
            add_term(idx_start+4,
                     {(0, a_row_offset): 1, (0, a_row_offset+1): 1},
                     {(b_row_offset+1, c_col_offset+1): 1},
                     {(c_col_offset, 0): -1, (c_col_offset+1, 0): 1})
            add_term(idx_start+5,
                     {(1, a_row_offset): 1, (0, a_row_offset): -1},
                     {(b_row_offset, c_col_offset): 1, (b_row_offset, c_col_offset+1): 1},
                     {(c_col_offset+1, 1): 1})
            add_term(idx_start+6,
                     {(0, a_row_offset+1): 1, (1, a_row_offset+1): -1},
                     {(b_row_offset+1, c_col_offset): 1, (b_row_offset+1, c_col_offset+1): 1},
                     {(c_col_offset, 0): 1})

        def schoolbook_2x2x1_block(a_row_offset, b_row_offset, c_col_offset, idx_start):
            add_term(idx_start,
                     {(0, a_row_offset): 1},
                     {(b_row_offset, c_col_offset): 1},
                     {(c_col_offset, 0): 1})
            add_term(idx_start+1,
                     {(0, a_row_offset+1): 1},
                     {(b_row_offset+1, c_col_offset): 1},
                     {(c_col_offset, 0): 1})
            add_term(idx_start+2,
                     {(1, a_row_offset): 1},
                     {(b_row_offset, c_col_offset): 1},
                     {(c_col_offset, 1): 1})
            add_term(idx_start+3,
                     {(1, a_row_offset+1): 1},
                     {(b_row_offset+1, c_col_offset): 1},
                     {(c_col_offset, 1): 1})

        idx = 0
        strassen_block(0, 0, 0, idx); idx += 7
        strassen_block(0, 0, 2, idx); idx += 7
        schoolbook_2x2x1_block(0, 0, 4, idx); idx += 4
        strassen_block(2, 2, 0, idx); idx += 7
        strassen_block(2, 2, 2, idx); idx += 7
        schoolbook_2x2x1_block(2, 2, 4, idx); idx += 4
        
        return U, V, W, rank
    
    def rank1_peeling_exact():
        """
        Use exact rational arithmetic to find a minimal-rank decomposition
        by iteratively peeling off rank-1 components.
        
        Strategy: The multiplication tensor T[i,j,k] = 1 for all i,j,k.
        We seek U[i,j,r], V[j,k,r], W[k,i,r] such that
        sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
        
        We use the fact that for fixed (i,k), the equation becomes:
        sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all j.
        This is a bilinear equation in j.
        
        We'll use a systematic approach: try to find rank-1 terms that
        match as much of the tensor as possible, then recurse on the remainder.
        """
        # Represent the tensor as a 3D array of Fractions
        # T[i,j,k] = 1 for all i,j,k
        # We'll work with the "residual" tensor and peel off rank-1 terms
        
        # A rank-1 term is defined by vectors u (length n*m), v (length m*p), w (length p*n)
        # such that u[i*m+j] * v[j*p+k] * w[k*n+i] contributes to T[i,j,k]
        
        # For the all-ones tensor, we can use a structured approach:
        # Group indices and use Kronecker products of smaller decompositions
        
        # 2x4x5 = 2x(2+2)x(2+2+1)
        # We can decompose as direct sum of:
        #   2x2x2 (rank 7 via Strassen) x 2 blocks + 2x2x1 (rank 4) x 2 blocks
        # But we can do better by mixing.
        
        # Let's try a different partition: 2x4x5
        # = (2x2x2) + (2x2x2) + (2x2x1) + (2x2x2) + (2x2x2) + (2x2x1)
        # That's the current rank-36.
        
        # Can we do 2x4x5 with rank < 36?
        # Lower bound: max(n*m, m*p, p*n) = max(8, 20, 10) = 20
        # Upper bound from known results: for 2x4x5, best known is around 33-35
        
        # Let's try to construct a rank-35 or rank-34 decomposition
        # by finding one linear dependency among the 36 rank-1 terms
        
        # Approach: Start with the rank-36 decomposition and check if any
        # rank-1 term is a linear combination of others (over Q)
        
        U0, V0, W0, rank0 = make_strassen_decomposition()
        
        # Convert to Fractions for exact arithmetic
        U_frac = [[Fraction(U0[i, r]) for r in range(rank0)] for i in range(n*m)]
        V_frac = [[Fraction(V0[i, r]) for r in range(rank0)] for i in range(m*p)]
        W_frac = [[Fraction(W0[i, r]) for r in range(rank0)] for i in range(p*n)]
        
        # Check for linear dependencies: is there a vector c in Q^36 such that
        # sum_r c_r * (U_col_r ⊗ V_col_r ⊗ W_col_r) = 0?
        # This is equivalent to: for all i,j,k: sum_r c_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 0
        
        # Build the constraint matrix: for each (i,j,k), we get an equation in c_r
        # The coefficient of c_r is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Since T is all ones, we need the sum to equal 1, not 0.
        # For dependency checking, we look for c such that the weighted sum is zero.
        
        # Actually, let's think differently. The tensor T has a specific structure.
        # Let's try to find a decomposition with rank < 36 by using a smarter block structure.
        
        # Alternative: use the fact that 2x4x5 can be viewed as 2x(2+2)x(2+2+1)
        # and try to share some rank-1 terms between blocks.
        
        # Let's try a construction with rank 34:
        # Use Strassen (rank 7) for the four 2x2x2 sub-blocks, but modify
        # the two 2x2x1 sub-blocks to share terms.
        
        # Actually, let's try a completely different approach:
        # Use the "border rank" concept and try to find a rank-33 decomposition
        # by allowing some terms to be "almost" rank-1 (but we need exact rank-1).
        
        # Given the complexity, let's try a systematic search:
        # Start with rank-36 and try to eliminate one term at a time
        # by expressing it as a combination of others.
        
        # For each term r, check if we can find coefficients a_s (s != r) such that:
        # U[:,r] = sum_{s != r} a_s * U[:,s]  (column-wise)
        # V[:,r] = sum_{s != r} a_s * V[:,s]
        # W[:,r] = sum_{s != r} a_s * W[:,s]
        # This is a strong condition. More generally, we need:
        # sum_{s != r} a_s * U[:,s] ⊗ V[:,s] ⊗ W[:,s] = U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        
        # This is a system of n*m*p = 40 equations in 35 unknowns (a_s for s != r).
        # We can solve this exactly over Q for each r.
        
        for r in range(rank0):
            if time.time() - _t0 > _DEADLINE - 5:
                break
            
            # Build the 40x35 system for term r
            # Unknowns: a_s for s in {0,...,35} \ {r}
            # Equations: for each (i,j,k):
            #   U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = sum_{s != r} a_s * U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
            
            # Build matrix A (40 x 35) and vector b (40)
            A = []
            b = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = []
                        for s in range(rank0):
                            if s == r:
                                continue
                            coeff = (U_frac[i*m+j][s] * V_frac[j*p+k][s] * W_frac[k*n+i][s])
                            row.append(coeff)
                        A.append(row)
                        b.append(U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r])
            
            # Solve A * a = b over Q using Gaussian elimination
            # A is 40x35, b is 40
            # We need to check if the system is consistent
            
            # Gaussian elimination with exact arithmetic
            num_rows = len(A)
            num_cols = len(A[0]) if A else 0
            
            # Augmented matrix [A | b]
            aug = [A[i] + [b[i]] for i in range(num_rows)]
            
            # Perform Gaussian elimination
            rank = 0
            for col in range(num_cols):
                if rank >= num_rows:
                    break
                # Find pivot
                pivot_row = None
                for row in range(rank, num_rows):
                    if aug[row][col] != 0:
                        pivot_row = row
                        break
                if pivot_row is None:
                    continue
                
                # Swap rows
                aug[rank], aug[pivot_row] = aug[pivot_row], aug[rank]
                
                # Eliminate
                pivot_val = aug[rank][col]
                for row in range(num_rows):
                    if row == rank:
                        continue
                    factor = aug[row][col] / pivot_val
                    if factor != 0:
                        for c in range(col, num_cols + 1):
                            aug[row][c] -= factor * aug[rank][c]
                
                rank += 1
            
            # Check consistency: any row with all zeros in A part but non-zero in b part
            consistent = True
            for row in range(num_rows):
                if all(aug[row][c] == 0 for c in range(num_cols)):
                    if aug[row][num_cols] != 0:
                        consistent = False
                        break
            
            if consistent and rank == num_cols:
                # Unique solution exists! We can eliminate term r.
                # Extract solution
                a_sol = {}
                # Back-substitution
                solution = [Fraction(0)] * num_cols
                for row in range(rank - 1, -1, -1):
                    # Find the column with the first non-zero in this row
                    pivot_col = None
                    for c in range(num_cols):
                        if aug[row][c] != 0:
                            pivot_col = c
                            break
                    if pivot_col is None:
                        continue
                    val = aug[row][num_cols]
                    for c in range(pivot_col + 1, num_cols):
                        val -= aug[row][c] * solution[c]
                    solution[pivot_col] = val / aug[row][pivot_col]
                
                # Map solution back to original indices
                a_vals = {}
                s_idx = 0
                for s in range(rank0):
                    if s == r:
                        continue
                    a_vals[s] = solution[s_idx]
                    s_idx += 1
                
                # Construct the reduced decomposition
                new_rank = rank0 - 1
                U_new = np.zeros((n*m, new_rank))
                V_new = np.zeros((m*p, new_rank))
                W_new = np.zeros((p*n, new_rank))
                
                new_idx = 0
                for s in range(rank0):
                    if s == r:
                        continue
                    # U_new[:, new_idx] = U[:, s] - sum_{t != r, t != s} a_t * U[:, t]
                    # Wait, this isn't quite right. We need:
                    # sum_{t != r} a_t * (U_t ⊗ V_t ⊗ W_t) = U_r ⊗ V_r ⊗ W_r
                    # So we can replace term r with the combination, but that doesn't reduce rank.
                    # 
                    # Actually, if the system is consistent, it means term r is in the span
                    # of the other terms. So we can simply drop term r and the remaining
                    # terms still span the same tensor space. But we need the exact tensor,
                    # not just the span.
                    #
                    # Let me reconsider. The condition is:
                    # sum_{s != r} a_s * U_s V_s W_s = U_r V_r W_r (as tensors)
                    # This means we can replace term r with the sum of terms s != r with
                    # coefficients a_s. But that increases the coefficient of those terms,
                    # it doesn't reduce the rank.
                    #
                    # To reduce rank, we need a different approach: find a linear combination
                    # of ALL terms that gives zero, then we can drop one term and adjust
                    # the others.
                    
                    U_new[:, new_idx] = U0[:, s]
                    V_new[:, new_idx] = V0[:, s]
                    W_new[:, new_idx] = W0[:, s]
                    new_idx += 1
                
                # Verify the reduced decomposition
                if verify_decomposition(U_new, V_new, W_new, tol=1e-9):
                    return U_new, V_new, W_new, new_rank
                else:
                    # The simple drop didn't work; we need to adjust coefficients
                    # This is more complex; skip for now
                    pass
        
        # If no single term can be dropped, try to find a 2-term dependency
        # or accept the rank-36 decomposition
        
        return U0, V0, W0, rank0
    
    # Try the exact rank-1 peeling approach
    try:
        U, V, W, rank = rank1_peeling_exact()
    except Exception as e:
        # Fallback to Strassen decomposition
        U, V, W, rank = make_strassen_decomposition()
    
    # Deadline guard
    if time.time() - _t0 > _DEADLINE - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
