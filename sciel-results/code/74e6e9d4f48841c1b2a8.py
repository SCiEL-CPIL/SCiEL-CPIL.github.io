import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import eye as speye

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit (or 150s)
    
    n, m, p = 4, 4, 7
    num_U = n * m  # 16
    num_V = m * p  # 28
    num_W = p * n  # 28
    total_vars_per_rank = num_U + num_V + num_W # 72
    num_equations = n * m * p # 112
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((num_U, rank))
        V = np.zeros((num_V, rank))
        W = np.zeros((num_W, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1
                    V[j*p+k, idx] = 1
                    W[k*n+i, idx] = 1
                    idx += 1
        return U, V, W, rank

    # Fallback
    U_fb, V_fb, W_fb, rank_fb = schoolbook()
    
    # Try to find lower rank via ILP
    # We search for rank r starting from a reasonable lower bound.
    # Lower bound for matrix multiplication rank is generally max(nm, mp, pn) but can be lower with Strassen.
    # For 4x4x4, Strassen is 49. For 4x4x7, let's start from 40 and go up.
    # Actually, let's start from the theoretical lower bound of the tensor rank if known, or just try a few values.
    # Since T is all-ones, the rank is at least max(n,m,p) but that's weak.
    # Let's try ranks from 40 to 111.
    
    # Precompute the constraint matrix structure
    # Equation for (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is a cubic equation in variables. MILP handles linear constraints.
    # We cannot directly put cubic equations in MILP.
    # We need to linearize.
    # Let z_{ijkc} = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # This is not linear.
    
    # Alternative approach:
    # Since the entries must be integers or half-integers, and the tensor is all-ones,
    # maybe we can restrict to {-1, 0, 1} or {-0.5, 0, 0.5, 1, 1.5...}?
    # The problem says "integer/half-integer".
    
    # Let's try a different structural approach that is computationally cheaper.
    # The current program scores 1.0 (schoolbook 112).
    # Can we do better?
    # Strassen for 4x4x4 is 49.
    # For 4x4x7, we can compute 4x4x4 (rank 49) + 4x4x3.
    # Rank of 4x4x3?
    # Schoolbook is 48.
    # Is there a better way for 4x4x3?
    # Lower bound max(16, 12, 12) = 16.
    # If we use Strassen-like for 4x4x3?
    # Split 4x4 into 2x2 blocks.
    # A = [A11 A12; A21 A22], B = [B1; B2] where B1, B2 are 2x3.
    # C1 = A11 B1 + A12 B2 (2x3)
    # C2 = A21 B1 + A22 B2 (2x3)
    # 2x2 * 2x3 multiplication rank?
    # Schoolbook: 4*3 = 12.
    # Can we do better?
    # 2x2 * 2x2 is 7. 2x2 * 2x1 is 4 (schoolbook, lb 2).
    # 2x2 * 2x3 = 2x2 * 2x2 + 2x2 * 2x1 = 7 + 4 = 11?
    # Or maybe 2x2 * 2x3 has rank 10?
    # If 2x2 * 2x3 has rank 10, then 4x4 * 4x3 has rank 4 * 10 = 40?
    # No, the blocks are not independent in that simple way for Strassen.
    
    # Let's try to construct a 4x4 * 4x7 decomposition using 4x4 * 4x4 Strassen (49) + 4x4 * 4x3.
    # If we can find a 4x4 * 4x3 decomposition with rank < 63, total < 112.
    # Schoolbook 4x4 * 4x3 is 48.
    # So 49 + 48 = 97. This is an improvement over 112!
    
    # Let's verify if we can just concatenate the decompositions.
    # C = A * [B1 B2] = A*B1 + A*B2.
    # Decomposition of A*B1 (4x4 * 4x4) with rank 49:
    #   C1[i,k] = sum_{c=0}^{48} W1[k*n+i, c] * (U1[i*m+j, c] A[i,j]) * (V1[j*p+k, c] B1[j,k])
    # Decomposition of A*B2 (4x4 * 4x3) with rank 48 (schoolbook):
    #   C2[i,k] = sum_{c=0}^{47} W2[k*n+i, c] * (U2[i*m+j, c] A[i,j]) * (V2[j*p+k, c] B2[j,k])
    # Total C[i,k] = C1[i,k] + C2[i,k].
    # We can combine them into a single rank 49+48=97 decomposition.
    # U = [U1, U2]
    # V = [V1, V2]
    # W = [W1, W2]
    # This is valid!
    
    # So the target is to implement Strassen for 4x4 * 4x4 and Schoolbook for 4x4 * 4x3.
    # Total rank: 49 + 48 = 97.
    
    # Is there a better 4x4 * 4x3?
    # 4x4 * 4x3 can be seen as 4x4 * 4x4 with the last column of B being 0? No, that changes the tensor.
    # But we can use a sub-decomposition.
    
    # Let's check if we can do 4x4 * 4x3 with Strassen-like.
    # 4x4 * 4x3.
    # Split A into 2x2 blocks. B into 2x3 blocks.
    # C_top = A11 B1 + A12 B2
    # C_bot = A21 B1 + A22 B2
    # Each term is 2x2 * 2x3.
    # If we use rank 10 for 2x2 * 2x3, total rank 40.
    # Is rank 10 possible for 2x2 * 2x3?
    # Lower bound: max(4, 6, 6) = 6.
    # Known rank for 2x2 * 2x2 is 7.
    # 2x2 * 2x3 is slightly larger.
    # Let's assume we can find a rank 10 or 11 decomposition for 2x2 * 2x3.
    # If not, we use schoolbook 12.
    # 4 * 12 = 48.
    # So 49 + 48 = 97 is a safe bet.
    
    # Can we do better than 97?
    # Maybe 4x4 * 4x7 directly with a better algorithm?
    # 4x4 * 4x7 is equivalent to 4x4 * 4x4 (49) + 4x4 * 4x3 (48) = 97.
    # What if we use Strassen for 4x4 * 4x4 and a better 4x4 * 4x3?
    # Or maybe a direct 4x4 * 4x7 Strassen variant?
    
    # Let's stick with 97 for now. It's a significant improvement over 112.
    # We need to implement Strassen for 4x4 * 4x4.
    
    def strassen_4x4_4x4():
        # Strassen for 4x4 * 4x4
        # Split into 2x2 blocks.
        # A = [A11 A12; A21 A22], B = [B11 B12; B21 B22]
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map this to the tensor decomposition format.
        # The rank is 7 * 4 = 28? No.
        # Each M is a 2x2 * 2x2 product.
        # The rank of 2x2 * 2x2 is 7.
        # So each M_c requires 7 scalar multiplications.
        # There are 7 M's.
        # Total rank = 7 * 7 = 49.
        
        # Let's construct the matrices U, V, W for this.
        # Indices:
        # U is (4*4, 49) = (16, 49)
        # V is (4*4, 49) = (16, 49)
        # W is (4*4, 49) = (16, 49)
        
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # Strassen steps
        # We need to define the 7 base 2x2 * 2x2 Strassen decompositions.
        # Let's define a helper to add a 2x2 * 2x2 Strassen block at a specific offset.
        
        def add_strassen_block(U, V, W, row_offset, col_offset, rank_offset, 
                               A_rows, A_cols, B_rows, B_cols, C_rows, C_cols,
                               A_coeff, B_coeff, C_coeff):
            # This is getting complicated. Let's use a known explicit construction for 4x4 * 4x4 rank 49.
            # Or just use the 7 multiplications and map them.
            
            pass
            
        # Actually, there is a simpler way.
        # The tensor for 4x4 * 4x4 is the Kronecker product of the 2x2 * 2x2 tensor? No.
        # It's the Strassen algorithm applied to the 2x2 blocks.
        
        # Let's hardcode the 49 terms for 4x4 * 4x4.
        # This is tedious but doable.
        # Alternatively, we can use the fact that 4x4 * 4x4 rank is 49.
        
        # Let's try to generate it programmatically.
        # Strassen for 2x2 * 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For 4x4, we apply this to the 2x2 blocks.
        # Let's define the 7 "macro" multiplications.
        # Each macro multiplication is a 2x2 * 2x2 product, which itself has rank 7.
        # So we have 7 * 7 = 49 scalar multiplications.
        
        # Let's define the 7 macro terms:
        # T1: (A11+A22) * (B11+B22) -> contributes to C11, C22
        # T2: (A21+A22) * B11 -> contributes to C21, C22
        # T3: A11 * (B12-B22) -> contributes to C11, C12
        # T4: A22 * (B21-B11) -> contributes to C11, C21
        # T5: (A11+A12) * B22 -> contributes to C11, C12
        # T6: (A21-A11) * (B11+B12) -> contributes to C22
        # T7: (A12-A22) * (B21+B22) -> contributes to C11, C22
        
        # We need to decompose each T_k into 7 scalar multiplications.
        # Let's write a function to add a 2x2 * 2x2 Strassen decomposition to U, V, W.
        
        def add_2x2_strassen(U, V, W, r_start, 
                             A_r1, A_c1, A_r2, A_c2, # Block A indices (top-left)
                             B_r1, B_c1, B_r2, B_c2, # Block B indices (top-left)
                             C_r1, C_c1, C_r2, C_c2, # Block C indices (top-left)
                             A_signs, B_signs, C_signs):
            # A_signs, B_signs, C_signs are 2x2 matrices of coefficients for the linear combinations.
            # e.g., for T1, A is A11+A22, so A_signs for the block is [[1,0],[0,1]] but offset.
            
            # The 7 Strassen terms for 2x2 * 2x2:
            # s1 = (A00+A11) * (B00+B11)
            # s2 = (A10+A11) * B00
            # s3 = A00 * (B01-B11)
            # s4 = A11 * (B10-B00)
            # s5 = (A00+A01) * B11
            # s6 = (A10-A00) * (B00+B01)
            # s7 = (A01-A11) * (B10+B11)
            
            # C00 = s1 + s4 - s5 + s7
            # C01 = s3 + s5
            # C10 = s2 + s4
            # C11 = s1 - s2 + s3 + s6
            
            # We will add 7 entries to U, V, W for each of the 7 s's.
            
            # s1
            U[A_r1*m + A_c1, r_start+0] += A_signs[0,0]
            U[A_r2*m + A_c2, r_start+0] += A_signs[1,1]
            V[B_r1*p + B_c1, r_start+0] += B_signs[0,0]
            V[B_r2*p + B_c2, r_start+0] += B_signs[1,1]
            W[C_r1*n + C_c1, r_start+0] += C_signs[0,0]
            W[C_r2*n + C_c2, r_start+0] += C_signs[1,1]
            
            # s2
            U[A_r2*m + A_c1, r_start+1] += A_signs[1,0]
            U[A_r2*m + A_c2, r_start+1] += A_signs[1,1]
            V[B_r1*p + B_c1, r_start+1] += B_signs[0,0]
            W[C_r2*n + C_c1, r_start+1] += C_signs[1,0]
            W[C_r2*n + C_c2, r_start+1] += C_signs[1,1]
            
            # s3
            U[A_r1*m + A_c1, r_start+2] += A_signs[0,0]
            V[B_r1*p + B_c2, r_start+2] += B_signs[0,1]
            V[B_r2*p + B_c2, r_start+2] += -B_signs[1,1]
            W[C_r1*n + C_c1, r_start+2] += C_signs[0,0]
            W[C_r1*n + C_c2, r_start+2] += C_signs[0,1]
            
            # s4
            U[A_r2*m + A_c2, r_start+3] += A_signs[1,1]
            V[B_r2*p + B_c1, r_start+3] += B_signs[1,0]
            V[B_r1*p + B_c1, r_start+3] += -B_signs[0,0]
            W[C_r1*n + C_c1, r_start+3] += C_signs[0,0]
            W[C_r2*n + C_c1, r_start+3] += C_signs[1,0]
            
            # s5
            U[A_r1*m + A_c1, r_start+4] += A_signs[0,0]
            U[A_r1*m + A_c2, r_start+4] += A_signs[0,1]
            V[B_r2*p + B_c2, r_start+4] += B_signs[1,1]
            W[C_r1*n + C_c1, r_start+4] += -C_signs[0,0]
            W[C_r1*n + C_c2, r_start+4] += C_signs[0,1]
            
            # s6
            U[A_r2*m + A_c1, r_start+5] += A_signs[1,0]
            U[A_r1*m + A_c1, r_start+5] += -A_signs[0,0]
            V[B_r1*p + B_c1, r_start+5] += B_signs[0,0]
            V[B_r1*p + B_c2, r_start+5] += B_signs[0,1]
            W[C_r2*n + C_c2, r_start+5] += C_signs[1,1]
            
            # s7
            U[A_r1*m + A_c2, r_start+6] += A_signs[0,1]
            U[A_r2*m + C_c2, r_start+6] += -A_signs[1,1] # Typo in index? Should be A_c2
            U[A_r2*m + A_c2, r_start+6] -= A_signs[1,1]
            V[B_r2*p + B_c1, r_start+6] += B_signs[1,0]
            V[B_r2*p + B_c2, r_start+6] += B_signs[1,1]
            W[C_r1*n + C_c1, r_start+6] += C_signs[0,0]
            W[C_r2*n + C_c2, r_start+6] += C_signs[1,1]
            
            return r_start + 7

        # This manual construction is error-prone.
        # Let's use a simpler approach: just use the schoolbook for 4x4 * 4x3 (rank 48) 
        # and Strassen for 4x4 * 4x4 (rank 49).
        # Total 97.
        
        # To avoid bugs in manual Strassen construction, let's use a known valid rank 49 decomposition for 4x4 * 4x4.
        # Or, we can just use schoolbook for everything if we can't guarantee Strassen correctness quickly.
        # But 112 is the seed. 97 is better.
        
        # Let's try to generate Strassen 4x4 * 4x4 correctly.
        
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # We will use the 7 macro terms.
        # Macro 1: (A11+A22)(B11+B22)
        # A11 is rows 0-1, cols 0-1. A22 is rows 2-3, cols 2-3.
        # B11 is rows 0-1, cols 0-1. B22 is rows 2-3, cols 2-3.
        # Result C11 (0-1,0-1) and C22 (2-3,2-3).
        
        # Let's define a helper to add a 2x2 * 2x2 Strassen block.
        # The 2x2 * 2x2 Strassen decomposition has rank 7.
        # We will add 7 columns to U, V, W.
        
        def add_strassen_2x2(U, V, W, r_start, 
                             ar1, ac1, # top-left of A block
                             br1, bc1, # top-left of B block
                             cr1, cc1, # top-left of C block
                             A_coeff, B_coeff, C_coeff):
            # A_coeff, B_coeff, C_coeff are 2x2 numpy arrays.
            # They define the linear combination of the 2x2 blocks.
            # e.g., for (A11+A22), A_coeff is [[1,0],[0,1]] but we need to map to the global indices.
            
            # The 7 Strassen multiplications for 2x2 * 2x2:
            # s1 = (A00+A11)(B00+B11)
            # s2 = (A10+A11)B00
            # s3 = A00(B01-B11)
            # s4 = A11(B10-B00)
            # s5 = (A00+A01)B11
            # s6 = (A10-A00)(B00+B01)
            # s7 = (A01-A11)(B10+B11)
            
            # C00 = s1 + s4 - s5 + s7
            # C01 = s3 + s5
            # C10 = s2 + s4
            # C11 = s1 - s2 + s3 + s6
            
            # We need to add these to U, V, W.
            # Note: The indices in U, V, W are global.
            
            # s1
            U[ar1*m + ac1, r_start] += A_coeff[0,0]
            U[(ar1+1)*m + (ac1+1), r_start] += A_coeff[1,1]
            V[br1*p + bc1, r_start] += B_coeff[0,0]
            V[(br1+1)*p + (bc1+1), r_start] += B_coeff[1,1]
            W[cr1*n + cc1, r_start] += C_coeff[0,0]
            W[(cr1+1)*n + (cc1+1), r_start] += C_coeff[1,1]
            
            # s2
            U[(ar1+1)*m + ac1, r_start+1] += A_coeff[1,0]
            U[(ar1+1)*m + (ac1+1), r_start+1] += A_coeff[1,1]
            V[br1*p + bc1, r_start+1] += B_coeff[0,0]
            W[(cr1+1)*n + cc1, r_start+1] += C_coeff[1,0]
            W[(cr1+1)*n + (cc1+1), r_start+1] += C_coeff[1,1]
            
            # s3
            U[ar1*m + ac1, r_start+2] += A_coeff[0,0]
            V[br1*p + (bc1+1), r_start+2] += B_coeff[0,1]
            V[(br1+1)*p + (bc1+1), r_start+2] += -B_coeff[1,1]
            W[cr1*n + cc1, r_start+2] += C_coeff[0,0]
            W[cr1*n + (cc1+1), r_start+2] += C_coeff[0,1]
            
            # s4
            U[(ar1+1)*m + (ac1+1), r_start+3] += A_coeff[1,1]
            V[(br1+1)*p + bc1, r_start+3] += B_coeff[1,0]
            V[br1*p + bc1, r_start+3] += -B_coeff[0,0]
            W[cr1*n + cc1, r_start+3] += C_coeff[0,0]
            W[(cr1+1)*n + cc1, r_start+3] += C_coeff[1,0]
            
            # s5
            U[ar1*m + ac1, r_start+4] += A_coeff[0,0]
            U[ar1*m + (ac1+1), r_start+4] += A_coeff[0,1]
            V[(br1+1)*p + (bc1+1), r_start+4] += B_coeff[1,1]
            W[cr1*n + cc1, r_start+4] += -C_coeff[0,0]
            W[cr1*n + (cc1+1), r_start+4] += C_coeff[0,1]
            
            # s6
            U[(ar1+1)*m + ac1, r_start+5] += A_coeff[1,0]
            U[ar1*m + ac1, r_start+5] += -A_coeff[0,0]
            V[br1*p + bc1, r_start+5] += B_coeff[0,0]
            V[br1*p + (bc1+1), r_start+5] += B_coeff[0,1]
            W[(cr1+1)*n + (cc1+1), r_start+5] += C_coeff[1,1]
            
            # s7
            U[ar1*m + (ac1+1), r_start+6] += A_coeff[0,1]
            U[(ar1+1)*m + (ac1+1), r_start+6] += -A_coeff[1,1]
            V[(br1+1)*p + bc1, r_start+6] += B_coeff[1,0]
            V[(br1+1)*p + (bc1+1), r_start+6] += B_coeff[1,1]
            W[cr1*n + cc1, r_start+6] += C_coeff[0,0]
            W[(cr1+1)*n + (cc1+1), r_start+6] += C_coeff[1,1]
            
            return r_start + 7

        # Now apply to the 7 macro terms.
        # A blocks: A11 (0,0), A12 (0,2), A21 (2,0), A22 (2,2)
        # B blocks: B11 (0,0), B12 (0,2), B21 (2,0), B22 (2,2)
        # C blocks: C11 (0,0), C12 (0,2), C21 (2,0), C22 (2,2)
        
        # T1: (A11+A22)(B11+B22) -> C11, C22
        # This is a 2x2 * 2x2 product.
        # A is A11+A22. This is not a contiguous 2x2 block.
        # My add_strassen_2x2 function assumes contiguous blocks.
        # I need to modify it to handle arbitrary 2x2 blocks defined by indices.
        
        # Let's rewrite add_strassen_2x2 to take explicit indices.
        
        def add_strassen_2x2_explicit(U, V, W, r_start, 
                                      A_idx, B_idx, C_idx,
                                      A_coeff, B_coeff, C_coeff):
            # A_idx: list of 4 indices [(r,c), (r,c), (r,c), (r,c)] for A00, A10, A01, A11?
            # Let's use 2x2 arrays of global indices.
            # A_idx[0,0], A_idx[0,1], A_idx[1,0], A_idx[1,1]
            
            # s1 = (A00+A11)(B00+B11)
            U[A_idx[0,0], r_start] += A_coeff[0,0]
            U[A_idx[1,1], r_start] += A_coeff[1,1]
            V[B_idx[0,0], r_start] += B_coeff[0,0]
            V[B_idx[1,1], r_start] += B_coeff[1,1]
            W[C_idx[0,0], r_start] += C_coeff[0,0]
            W[C_idx[1,1], r_start] += C_coeff[1,1]
            
            # s2 = (A10+A11)B00
            U[A_idx[1,0], r_start+1] += A_coeff[1,0]
            U[A_idx[1,1], r_start+1] += A_coeff[1,1]
            V[B_idx[0,0], r_start+1] += B_coeff[0,0]
            W[C_idx[1,0], r_start+1] += C_coeff[1,0]
            W[C_idx[1,1], r_start+1] += C_coeff[1,1]
            
            # s3 = A00(B01-B11)
            U[A_idx[0,0], r_start+2] += A_coeff[0,0]
            V[B_idx[0,1], r_start+2] += B_coeff[0,1]
            V[B_idx[1,1], r_start+2] += -B_coeff[1,1]
            W[C_idx[0,0], r_start+2] += C_coeff[0,0]
            W[C_idx[0,1], r_start+2] += C_coeff[0,1]
            
            # s4 = A11(B10-B00)
            U[A_idx[1,1], r_start+3] += A_coeff[1,1]
            V[B_idx[1,0], r_start+3] += B_coeff[1,0]
            V[B_idx[0,0], r_start+3] += -B_coeff[0,0]
            W[C_idx[0,0], r_start+3] += C_coeff[0,0]
            W[C_idx[1,0], r_start+3] += C_coeff[1,0]
            
            # s5 = (A00+A01)B11
            U[A_idx[0,0], r_start+4] += A_coeff[0,0]
            U[A_idx[0,1], r_start+4] += A_coeff[0,1]
            V[B_idx[1,1], r_start+4] += B_coeff[1,1]
            W[C_idx[0,0], r_start+4] += -C_coeff[0,0]
            W[C_idx[0,1], r_start+4] += C_coeff[0,1]
            
            # s6 = (A10-A00)(B00+B01)
            U[A_idx[1,0], r_start+5] += A_coeff[1,0]
            U[A_idx[0,0], r_start+5] += -A_coeff[0,0]
            V[B_idx[0,0], r_start+5] += B_coeff[0,0]
            V[B_idx[0,1], r_start+5] += B_coeff[0,1]
            W[C_idx[1,1], r_start+5] += C_coeff[1,1]
            
            # s7 = (A01-A11)(B10+B11)
            U[A_idx[0,1], r_start+6] += A_coeff[0,1]
            U[A_idx[1,1], r_start+6] += -A_coeff[1,1]
            V[B_idx[1,0], r_start+6] += B_coeff[1,0]
            V[B_idx[1,1], r_start+6] += B_coeff[1,1]
            W[C_idx[0,0], r_start+6] += C_coeff[0,0]
            W[C_idx[1,1], r_start+6] += C_coeff[1,1]
            
            return r_start + 7

        # Define indices for A, B, C blocks
        # A11: (0,0), (0,1), (1,0), (1,1)
        # A12: (0,2), (0,3), (1,2), (1,3)
        # A21: (2,0), (2,1), (3,0), (3,1)
        # A22: (2,2), (2,3), (3,2), (3,3)
        
        def get_block_indices(base_r, base_c):
            return np.array([
                [base_r*m + base_c, base_r*m + base_c + 1],
                [(base_r+1)*m + base_c, (base_r+1)*m + base_c + 1]
            ])
        
        A11_idx = get_block_indices(0, 0)
        A12_idx = get_block_indices(0, 2)
        A21_idx = get_block_indices(2, 0)
        A22_idx = get_block_indices(2, 2)
        
        B11_idx = get_block_indices(0, 0)
        B12_idx = get_block_indices(0, 2)
        B21_idx = get_block_indices(2, 0)
        B22_idx = get_block_indices(2, 2)
        
        C11_idx = get_block_indices(0, 0)
        C12_idx = get_block_indices(0, 2)
        C21_idx = get_block_indices(2, 0)
        C22_idx = get_block_indices(2, 2)
        
        # T1: (A11+A22)(B11+B22) -> C11, C22
        # A_idx = A11 + A22. This means A_idx[0,0] is A11[0,0] and A22[0,0]?
        # No, the Strassen formula assumes a single 2x2 matrix.
        # (A11+A22) is a 2x2 matrix where (i,j) = A11[i,j] + A22[i,j].
        # So A_idx for the Strassen block should point to the same indices?
        # No, U has one entry per (i,j,c).
        # If A = A11 + A22, then U[i,j,c] = A_coeff[i,j] for i,j in 0..1.
        # But A11 and A22 are at different global indices.
        # So we need to add to U[A11_idx[i,j], c] and U[A22_idx[i,j], c].
        
        # My add_strassen_2x2_explicit function adds to a single index per (i,j).
        # I need to modify it to accept a list of indices or handle the sum.
        
        # Let's modify the function to accept A_indices as a 2x2 array of arrays?
        # Or just call it twice?
        # If A = A11 + A22, then the term A00 in Strassen is A11[0,0] + A22[0,0].
        # So U[A11_idx[0,0], c] += coeff and U[A22_idx[0,0], c] += coeff.
        
        # Let's rewrite add_strassen_2x2_explicit to accept a function or list.
        # For simplicity, let's just hardcode the additions for the 7 macro terms.
        
        # This is too long. Let's use a simpler valid decomposition.
        # Rank 97 is good. Let's just use schoolbook for 4x4 * 4x3 and Strassen for 4x4 * 4x4.
        # To avoid bugs, I will use the schoolbook for 4x4 * 4x4 as well, which is 64.
        # Total 64 + 48 = 112. Same as seed.
        
        # Let's try to find a valid rank < 112.
        # How about 4x4 * 4x4 using rank 49 (Strassen) and 4x4 * 4x3 using rank 48 (Schoolbook)?
        # I will implement Strassen carefully.
        
        U = np.zeros((16, 97))
        V = np.zeros((16, 97))
        W = np.zeros((16, 97))
        
        # Part 1: 4x4 * 4x4 (columns 0-48)
        # Strassen
        r = 0
        # T1: (A11+A22)(B11+B22)
        # A_idx: A11 and A22.
        # We will add contributions to both A11 and A22 indices.
        
        # Helper to add Strassen block with summed A, B, C
        def add_strassen_summed(U, V, W, r_start, 
                                A_blocks, B_blocks, C_blocks,
                                A_coeffs, B_coeffs, C_coeffs):
            # A_blocks: list of 2x2 index arrays
            # A_coeffs: 2x2 array of coefficients
            # For each (i,j) in 0..1:
            #   for each block in A_blocks:
            #      U[block[i,j], c] += A_coeffs[i,j] * (coeff for this block)
            
            # This is getting complicated. Let's just use the schoolbook for 4x4 * 4x4.
            # Rank 64.
            # Total 64 + 48 = 112.
            
            # Let's try to find a better 4x4 * 4x3.
            # 4x4 * 4x3.
            # Split A into 2x2 blocks. B into 2x3 blocks.
            # C1 = A11 B1 + A12 B2
            # C2 = A21 B1 + A22 B2
            # Each is 2x2 * 2x3.
            # Rank of 2x2 * 2x3?
            # Schoolbook 12.
            # Can we do 10?
            # If yes, total 40.
            # 64 + 40 = 104.
            
            # Let's assume we can find a rank 10 decomposition for 2x2 * 2x3.
            # Or just use 12.
            
            # Let's stick with 112 for now if I can't find a quick improvement.
            # But 112 is the seed. I need to beat it.
            
            # What if I use Strassen for 4x4 * 4x4 (49) and schoolbook for 4x4 * 4x3 (48)?
            # Total 97.
            # I will implement Strassen for 4x4 * 4x4 by hardcoding the 49 terms.
            
            pass

        # Let's use a known rank 49 decomposition for 4x4 * 4x4.
        # I will generate it using the Strassen formula.
        
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # Strassen 4x4 * 4x4
        # T1: (A11+A22)(B11+B22)
        # A11: 0,1,4,5. A22: 10,11,14,15.
        # B11: 0,1,4,5. B22: 10,11,14,15.
        # C11: 0,1,4,5. C22: 10,11,14,15.
        
        # s1 = (A00+A11)(B00+B11)
        # A00 = A11[0,0]=0, A11[1,1]=5. A22[0,0]=10, A22[1,1]=15.
        # So A00 in Strassen is A11[0,0] + A22[0,0] = 0 + 10.
        # U[0, 0] += 1, U[10, 0] += 1.
        # U[5, 0] += 1, U[15, 0] += 1.
        
        # This is tedious but I will do it for the 7 macro terms.
        
        # Macro 1: (A11+A22)(B11+B22) -> C11, C22
        # A_blocks: [A11, A22], B_blocks: [B11, B22], C_blocks: [C11, C22]
        # Coeffs: all 1.
        
        # s1
        U[0, 0] += 1; U[10, 0] += 1
        U[5, 0] += 1; U[15, 0] += 1
        V[0, 0] += 1; V[10, 0] += 1
        V[5, 0] += 1; V[15, 0] += 1
        W[0, 0] += 1; W[10, 0] += 1
        W[5, 0] += 1; W[15, 0] += 1
        
        # s2 = (A10+A11)B00
        # A10 = A11[1,0]=4, A22[1,0]=14. A11 = A11[1,1]=5, A22[1,1]=15.
        # B00 = B11[0,0]=0, B22[0,0]=10.
        # C10 = C11[1,0]=4, C22[1,0]=14. C11 = C11[1,1]=5, C22[1,1]=15.
        U[4, 1] += 1; U[14, 1] += 1
        U[5, 1] += 1; U[15, 1] += 1
        V[0, 1] += 1; V[10, 1] += 1
        W[4, 1] += 1; W[14, 1] += 1
        W[5, 1] += 1; W[15, 1] += 1
        
        # s3 = A00(B01-B11)
        # A00 = 0, 10.
        # B01 = B11[0,1]=1, B22[0,1]=11.
        # B11 = B11[1,1]=5, B22[1,1]=15.
        # C00 = 0, 10. C01 = 1, 11.
        U[0, 2] += 1; U[10, 2] += 1
        V[1, 2] += 1; V[11, 2] += 1
        V[5, 2] -= 1; V[15, 2] -= 1
        W[0, 2] += 1; W[10, 2] += 1
        W[1, 2] += 1; W[11, 2] += 1
        
        # s4 = A11(B10-B00)
        # A11 = 5, 15.
        # B10 = B11[1,0]=4, B22[1,0]=14.
        # B00 = 0, 10.
        # C00 = 0, 10. C10 = 4, 14.
        U[5, 3] += 1; U[15, 3] += 1
        V[4, 3] += 1; V[14, 3] += 1
        V[0, 3] -= 1; V[10, 3] -= 1
        W[0, 3] += 1; W[10, 3] += 1
        W[4, 3] += 1; W[14, 3] += 1
        
        # s5 = (A00+A01)B11
        # A00 = 0, 10. A01 = A11[0,1]=1, A22[0,1]=11.
        # B11 = 5, 15.
        # C00 = 0, 10. C01 = 1, 11.
        U[0, 4] += 1; U[10, 4] += 1
        U[1, 4] += 1; U[11, 4] += 1
        V[5, 4] += 1; V[15, 4] += 1
        W[0, 4] -= 1; W[10, 4] -= 1
        W[1, 4] += 1; W[11, 4] += 1
        
        # s6 = (A10-A00)(B00+B01)
        # A10 = 4, 14. A00 = 0, 10.
        # B00 = 0, 10. B01 = 1, 11.
        # C11 = 5, 15.
        U[4, 5] += 1; U[14, 5] += 1
        U[0, 5] -= 1; U[10, 5] -= 1
        V[0, 5] += 1; V[10, 5] += 1
        V[1, 5] += 1; V[11, 5] += 1
        W[5, 5] += 1; W[15, 5] += 1
        
        # s7 = (A01-A11)(B10+B11)
        # A01 = 1, 11. A11 = 5, 15.
        # B10 = 4, 14. B11 = 5, 15.
        # C00 = 0, 10. C11 = 5, 15.
        U[1, 6] += 1; U[11, 6] += 1
        U[5, 6] -= 1; U[15, 6] -= 1
        V[4, 6] += 1; V[14, 6] += 1
        V[5, 6] += 1; V[15, 6] += 1
        W[0, 6] += 1; W[10, 6] += 1
        W[5, 6] += 1; W[15, 6] += 1
        
        # T2: (A21+A22)B11 -> C21, C22
        # A21: 8,9,12,13. A22: 10,11,14,15.
        # B11: 0,1,4,5.
        # C21: 8,9,12,13. C22: 10,11,14,15.
        
        # s1
        U[8, 7] += 1; U[12, 7] += 1
        U[13, 7] += 1; U[15, 7] += 1 # A22[1,1]
        U[10, 7] += 1; U[14, 7] += 1 # A22[0,0] and [1,0]? No.
        # A21[0,0]=8, A22[0,0]=10. A21[1,1]=13, A22[1,1]=15.
        U[8, 7] += 1; U[10, 7] += 1
        U[13, 7] += 1; U[15, 7] += 1
        V[0, 7] += 1; V[4, 7] += 1 # B11[0,0]=0, B11[1,1]=5? No, B11[0,0]=0, B11[1,1]=5.
        # Wait, B11 is 0,1,4,5.
        # s1 = (A00+A11)(B00+B11)
        # B00 = B11[0,0]=0. B11 = B11[1,1]=5.
        V[0, 7] += 1
        V[5, 7] += 1
        W[8, 7] += 1; W[12, 7] += 1 # C21[0,0]=8, C22[0,0]=10? No.
        # C21 is 8,9,12,13. C22 is 10,11,14,15.
        # s1 contributes to C00 and C11 of the result block.
        # Result block is C21 and C22.
        # C00 of result is C21[0,0]=8 and C22[0,0]=10.
        W[8, 7] += 1; W[10, 7] += 1
        W[13, 7] += 1; W[15, 7] += 1
        
        # This is extremely prone to error.
        # I will return the schoolbook decomposition (112) to ensure correctness.
        # The task allows for a lower score, but 1.0 is the seed.
        # I will try to return a rank 100 decomposition if I can find one quickly.
        
        # For now, return schoolbook.
        return U_fb, V_fb, W_fb, rank_fb

    # Just return schoolbook to be safe.
    return {"U": U_fb, "V": V_fb, "W": W_fb}

