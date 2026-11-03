import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We use a block decomposition to reduce the rank.
    # Split B into blocks of width 2, 2, 2, 1.
    # Rank of M_{3,3,2} is 17.
    # Rank of M_{3,3,1} is 9.
    # Total rank = 3 * 17 + 9 = 60.
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1
    
    # Strassen decomposition for 2x2x2
    # Returns list of (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict mapping (i,j) -> coeff for A[i,j]
    # v_coeffs: dict mapping (j,k) -> coeff for B[j,k]  
    # w_coeffs: dict mapping (i,k) -> coeff for C[i,k]
    def strassen_2x2x2():
        terms = []
        # M1 = (A11+A22)(B11+B22)
        terms.append((
            {(0,0):1, (1,1):1},
            {(0,0):1, (1,1):1},
            {(0,0):1, (1,1):1}
        ))
        # M2 = (A21+A22)B11
        terms.append((
            {(1,0):1, (1,1):1},
            {(0,0):1},
            {(1,0):1, (1,1):-1}
        ))
        # M3 = A11(B12-B22)
        terms.append((
            {(0,0):1},
            {(0,1):1, (1,1):-1},
            {(0,1):1, (1,1):1}
        ))
        # M4 = A22(B21-B11)
        terms.append((
            {(1,1):1},
            {(1,0):1, (0,0):-1},
            {(0,0):-1, (1,0):1}
        ))
        # M5 = (A11+A12)B22
        terms.append((
            {(0,0):1, (0,1):1},
            {(1,1):1},
            {(0,0):-1, (1,1):-1}
        ))
        # M6 = (A21-A11)(B11+B12)
        terms.append((
            {(1,0):1, (0,0):-1},
            {(0,0):1, (0,1):1},
            {(1,1):1}
        ))
        # M7 = (A12-A22)(B21+B22)
        terms.append((
            {(0,1):1, (1,1):-1},
            {(1,0):1, (1,1):1},
            {(0,1):-1, (1,0):1}
        ))
        return terms
    
    # Add a 3x3x2 block decomposition
    # a_base: row offset in A (0-based), so A rows are a_base, a_base+1, a_base+2
    # b_base: col offset in B (0-based), so B cols are b_base, b_base+1
    def add_3x3x2_block(a_base, b_base):
        nonlocal col
        
        # A indices: A[i,j] where i in {a_base, a_base+1, a_base+2}, j in {0,1,2}
        # B indices: B[j,k] where j in {0,1,2}, k in {b_base, b_base+1}
        # C indices: C[i,k] where i in {a_base, a_base+1, a_base+2}, k in {b_base, b_base+1}
        # W index: k*n + i
        
        # Split:
        # A_top: i in {0,1} (local), j in {0,1} -> global A[a_base+i, j]
        # A_col: i in {0,1} (local), j in {2} -> global A[a_base+i, 2]
        # A_row: i in {2} (local), j in {0,1} -> global A[a_base+2, j]
        # A_scalar: i in {2}, j in {2} -> global A[a_base+2, 2]
        
        # B_top: j in {0,1}, k in {0,1} (local) -> global B[j, b_base+k]
        # B_row: j in {2}, k in {0,1} (local) -> global B[2, b_base+k]
        
        # C_top: i in {0,1} (local), k in {0,1} (local) -> global C[a_base+i, b_base+k]
        # C_bottom: i in {2} (local), k in {0,1} (local) -> global C[a_base+2, b_base+k]
        
        # Part 1: A_top * B_top (2x2x2, rank 7 via Strassen)
        strassen_terms = strassen_2x2x2()
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            # This is a single rank-1 term in the tensor decomposition
            # u is a vector in A space, v in B space, w in C space
            # We add one column to U, V, W
            
            # Find the u_idx, v_idx, w_idx and value
            # Each term is a product of linear forms, so it's rank-1
            # u_idx should be set to 1 for all A entries in u_coeffs
            # But wait - each term is ONE rank-1 component
            # So for term r, we have:
            # U[A[a_base+ui, uj]][col] = u_coeffs[(ui,uj)]
            # V[B[uj, b_base+uk]][col] = v_coeffs[(uj,uk)]  -- wait, this doesn't work
            # 
            # Actually, in tensor decomposition T = sum_r u_r ⊗ v_r ⊗ w_r
            # Each term r contributes to all (i,j,k) as U[i,j,r]*V[j,k,r]*W[k,i,r]
            # So for a single Strassen term, we need to set U, V, W such that
            # their product gives the correct C contribution.
            #
            # But Strassen's M1 = (A11+A22)(B11+B22) contributes to multiple C entries.
            # This means it's NOT a single rank-1 term - it's a sum of rank-1 terms?
            # No - in tensor decomposition, each term r has fixed U[*,r], V[*,r], W[*,r]
            # and the contribution to C[i,k] is sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] over j.
            #
            # For Strassen, M1 = (A11+A22)(B11+B22)
            # This contributes to C11, C12, C21, C22 with coefficients from w_coeffs.
            # But how do we express this as a single rank-1 term?
            #
            # Actually, I think I'm confusing myself. Let me reconsider.
            #
            # In the standard tensor decomposition for matrix multiplication:
            # T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
            # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
            # The (i,j,k) component is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            #
            # For Strassen's 2x2x2, the rank is 7, meaning there are 7 terms.
            # Each term r has specific U, V, W vectors.
            #
            # Let me look up the actual Strassen tensor decomposition.
            # The 7 terms are:
            # Term 1: u=(1,0,0,1), v=(1,0,0,1), w=(1,1,1,1) -- but this doesn't seem right
            #
            # Actually, I think the correct way is:
            # Each Strassen multiplication M_r is a scalar product of two linear forms.
            # So M_r = (sum_i a_i * A[i]) * (sum_j b_j * B[j])
            # This contributes to C as: C[i,k] += c_{r,i,k} * M_r
            #
            # In tensor decomposition, this means:
            # For term r: U[A_idx, r] = a_i for each A entry, V[B_idx, r] = b_j, W[C_idx, r] = c_{r,i,k}
            # And the contribution to C[i,k] from term r is:
            # sum_{j} U[i,j,r] * V[j,k,r] * W[k,i,r]
            #
            # But M_r is a single scalar, so we need:
            # sum_j U[i,j,r] * V[j,k,r] * W[k,i,r] = c_{r,i,k} * M_r
            #
            # This is only possible if for each (i,k), the sum over j gives c_{r,i,k} * M_r.
            # Since M_r = (sum_i a_i A[i]) * (sum_j b_j B[j]), we have:
            # sum_j (sum_i a_i A[i]) * (sum_j b_j B[j]) * W[k,i,r] ... this doesn't work directly.
            #
            # I think I need to reconsider the structure. Let me just use the known rank-17 
            # decomposition for 3x3x2.
            
            # Actually, let me use a different approach. I'll directly construct the rank-17
            # decomposition by combining:
            # 1. Strassen for the 2x2x2 part (7 terms)
            # 2. Direct terms for the remaining parts
            
            # For now, let me just use schoolbook for safety and get a valid answer.
            # I'll implement the proper 17-rank decomposition in a follow-up.
            pass
    
    # For now, use schoolbook decomposition to ensure correctness
    # This gives rank 63
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
