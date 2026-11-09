"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # We use a block decomposition to reduce the rank below the schoolbook 100.
    # A is 4x5, B is 5x5.
    # Split A into A1 (4x4) and A2 (4x1).
    # Split B into B11 (4x4), B12 (4x1), B21 (1x4), B22 (1x1).
    # C = A1*B11 + A2*B21 + A1*B12 + A2*B22.
    #
    # The multiplication tensor for C is the sum of the tensors for these four terms.
    # Since the row/column indices for the sub-matrices are disjoint, the sub-tensors
    # live in orthogonal subspaces of the tensor product space.
    # Thus, the rank of the total multiplication is the sum of the ranks of the parts.
    #
    # We use Strassen's algorithm for the 4x4x4 multiplication (rank 49).
    # We use schoolbook for the other parts:
    # A1*B12: 4x4x1 -> rank 16
    # A2*B21: 4x1x4 -> rank 16
    # A2*B22: 4x1x1 -> rank 4
    # Total rank = 49 + 16 + 16 + 4 = 85.
    
    n, m, p = DIMENSIONS
    
    # Helper to construct the Strassen decomposition for 4x4x4
    def strassen_4x4x4(offset_u, offset_v, offset_w):
        # Returns (U, V, W) lists of length 49 for the 4x4x4 block.
        # Indices:
        # U index: i*4 + j (i in 0..3, j in 0..3) relative to offset_u
        # V index: j*4 + k (j in 0..3, k in 0..3) relative to offset_v
        # W index: k*4 + i (k in 0..3, i in 0..3) relative to offset_w
        # Note: The global indices are:
        # U_global[i*5 + j] = U_local[i*4 + j]
        # V_global[j*5 + k] = V_local[j*4 + k]
        # W_global[k*4 + i] = W_local[k*4 + i]
        
        # Strassen's 7 steps for 2x2x2 blocks (each block is 2x2).
        # We split the 4x4 into four 2x2 blocks.
        # A11, A12, A21, A22 (2x2 each)
        # B11, B12, B21, B22 (2x2 each)
        
        # The 7 intermediate matrices are sums/diffs of 2x2 blocks.
        # We need to express the entries of U, V, W as 0, 1, or -1.
        
        # Let's define the Strassen steps in terms of the 16 variables of A and B.
        # Actually, it's easier to just use the known Strassen rank 7 decomposition for 2x2.
        # And combine them.
        
        # However, implementing Strassen for 4x4 from scratch is error-prone.
        # Let's stick to a known valid decomposition if possible, or verify carefully.
        
        # Alternative: Use the known rank 49 decomposition for 4x4x4?
        # Actually, the rank of 4x4x4 is known to be 49 (Strassen's bound is tight for 4x4? No, 4x4 rank is 49 is the best known upper bound? 
        # Wait, the rank of 4x4x4 is 49?
        # Strassen gives 4^3 * (7/8) = 64 * 0.875 = 56? No.
        # Strassen: T(n) = 7 T(n/2) + ...
        # T(4) = 7 * T(2) = 7 * 7 = 49. Yes, Strassen gives rank 49 for 4x4x4.
        
        # Let's construct the 4x4x4 Strassen decomposition.
        # We partition A into 4 2x2 blocks.
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        
        # Strassen steps for each 2x2 block multiplication:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M5
        
        # We need to map this to the tensor decomposition.
        # The tensor T satisfies C_{ik} = sum_j A_{ij} B_{jk}.
        # In the decomposition, C = sum_r U_r A V_r^T B? No.
        # The standard form is: T = sum_{r=1}^R U_r \otimes V_r \otimes W_r.
        # Where U_r is a vector in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}.
        # And the product is sum_r (U_r^T \vec{A}) (V_r^T \vec{B}) \vec{C} \text{ (reshaped)}.
        
        # For a single term (i,j,k), U has 1 at i*m+j, V has 1 at j*p+k, W has 1 at k*n+i.
        
        # For Strassen, each "step" M_s is a sum of products of linear combinations.
        # (sum_{ij} u_{ij} A_{ij}) (sum_{jk} v_{jk} B_{jk})
        # This contributes to multiple C_{ik} entries.
        
        # This is getting complex. Let's use a simpler, verified decomposition if rank 85 is too hard to get right quickly.
        # But schoolbook is 100. We need to beat 100.
        # Can we do better than 85 with a simpler scheme?
        # What if we just split 5 into 2+3?
        # A (4x5) = A1 (4x2) + A2 (4x3).
        # B (5x5) = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3.
        
        # C = A1 B11 + A1 B12 + A2 B21 + A2 B22.
        
        # Ranks:
        # A1 B11: 4x2x2 -> schoolbook rank 16.
        # A1 B12: 4x2x3 -> schoolbook rank 24.
        # A2 B21: 4x3x2 -> schoolbook rank 24.
        # A2 B22: 4x3x3 -> schoolbook rank 36.
        # Total: 16 + 24 + 24 + 36 = 100. No improvement.
        
        # What about 1+4 split?
        # A1 (4x1), A2 (4x4).
        # B11 (1x1), B12 (1x4), B21 (4x1), B22 (4x4).
        
        # C = A1 B11 + A1 B12 + A2 B21 + A2 B22.
        
        # Ranks:
        # A1 B11: 4x1x1 -> 4.
        # A1 B12: 4x1x4 -> 16.
        # A2 B21: 4x4x1 -> 16.
        # A2 B22: 4x4x4 -> 49 (Strassen).
        # Total: 4 + 16 + 16 + 49 = 85.
        
        # So we just need to implement Strassen for 4x4x4 correctly.
        
        # Let's build the Strassen 4x4x4 decomposition explicitly.
        # We have 16 variables for A (indices i,j in 0..3) and 16 for B (indices j,k in 0..3).
        # We partition them into four 2x2 blocks.
        # Block (b1, b2) where b1, b2 in {0, 1}.
        # A_{b1*2 + i, b2*2 + j} for i,j in 0..1.
        
        # The 7 Strassen steps M1..M7 are:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # Each Ms is a product of a linear form in A and a linear form in B.
        # Let L_A_s be the vector of coefficients for A, L_B_s for B.
        # The contribution to the tensor is L_A_s \otimes L_B_s \otimes L_C_s.
        # Where L_C_s is the vector of coefficients for C.
        
        # We need to determine L_C_s.
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M5
        
        # Note: C00, C01, C10, C11 are 2x2 blocks.
        # So C is 4x4.
        # C_{i,k} is determined by which block it is in and the position within the block.
        # Since the block multiplication is uniform, the coefficients L_C_s are the same for all 4 positions in the 2x2 block.
        
        # Let's map the global indices.
        # i_global = b1*2 + i_local
        # j_global = b2*2 + j_local
        # k_global = b3*2 + k_local
        
        # For a given Strassen step s:
        # L_A_s is a 16-dim vector (or 20-dim in global, but only 4 non-zeros per block).
        # L_B_s is a 16-dim vector.
        # L_C_s is a 16-dim vector.
        
        # Let's define the coefficients for A and B for each step.
        # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1).
        # B blocks: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1).
        
        # Step 1:
        # A: A00 (1), A11 (1).
        # B: B00 (1), B11 (1).
        # C: C00 (1), C11 (1).
        
        # Step 2:
        # A: A10 (1), A11 (1).
        # B: B00 (1).
        # C: C10 (1), C11 (-1).
        
        # Step 3:
        # A: A00 (1).
        # B: B01 (1), B11 (-1).
        # C: C01 (1), C11 (1).
        
        # Step 4:
        # A: A11 (1).
        # B: B10 (1), B00 (-1).
        # C: C00 (-1), C10 (1).
        
        # Step 5:
        # A: A00 (1), A01 (1).
        # B: B11 (1).
        # C: C01 (1), C11 (1).
        
        # Step 6:
        # A: A10 (1), A00 (-1).
        # B: B00 (1), B01 (1).
        # C: C00 (1).
        
        # Step 7:
        # A: A01 (1), A11 (-1).
        # B: B10 (1), B11 (1).
        # C: C00 (-1).
        
        # Now we need to expand these into the full 4x4 indices.
        # For A00: i in 0..1, j in 0..1. Global i = i, j = j.
        # For A01: i in 0..1, j in 0..1. Global i = i, j = j+2.
        # For A10: i in 0..1, j in 0..1. Global i = i+2, j = j.
        # For A11: i in 0..1, j in 0..1. Global i = i+2, j = j+2.
        
        # Similarly for B and C.
        
        # We will create 7 sets of (U, V, W) for the 4x4x4 block.
        # Each set has 4 entries (for the 2x2 size).
        # So total 7 * 4 = 28? No.
        # Each Strassen step corresponds to 4 tensor terms (one for each element in the 2x2 output block).
        # Wait.
        # The product (sum A) (sum B) gives a scalar.
        # This scalar is added to multiple C entries.
        # In the tensor decomposition, each term is a rank-1 tensor.
        # So for each Strassen step s, we have 4 rank-1 tensors (for the 4 positions in the 2x2 block).
        # Total 7 * 4 = 28?
        # No, that's not right.
        # The rank of 2x2x2 is 7.
        # The rank of 4x4x4 is 49.
        # So we need 49 terms.
        
        # Why 49?
        # Because we have 4 independent 2x2 multiplications? No.
        # Strassen reduces 8 to 7.
        # For 4x4, we have 4 blocks of 2x2.
        # But the Strassen steps mix the blocks.
        # Actually, the standard Strassen algorithm for 4x4 uses 49 multiplications.
        # How?
        # It treats the 4x4 as four 2x2 blocks.
        # The 7 steps are applied to the 2x2 blocks.
        # But each "block" is 2x2, so each "multiplication" is 4 scalar multiplications.
        # So 7 * 4 = 28?
        # This contradicts the known rank 49.
        
        # Let's re-verify.
        # Rank of 4x4x4 is 49?
        # Strassen's bound: T(4) = 7 T(2). T(2)=7. So T(4)=49.
        # Yes.
        # So each Strassen step for 2x2 blocks actually consists of 4 scalar multiplications.
        # So we have 7 steps * 4 scalars = 28?
        # Wait.
        # If T(2)=7, then T(4) = 7 * T(2) = 49.
        # So each of the 7 "steps" in the 4x4 case corresponds to 7 scalar multiplications?
        # No.
        # The recursion is T(n) = 7 T(n/2).
        # T(4) = 7 T(2) = 7 * 7 = 49.
        # So we have 7 "sub-problems" of size 2x2.
        # Each sub-problem has rank 7.
        # So total rank 49.
        
        # This means we need to decompose each of the 7 Strassen steps (which are 2x2 multiplications) into 7 rank-1 tensors.
        # So we need the rank-7 decomposition of 2x2x2.
        
        # Let's get the rank-7 decomposition of 2x2x2.
        # It's the standard Strassen decomposition.
        # It has 7 terms.
        
        # So, for the 4x4x4 case:
        # We have 7 Strassen steps.
        # Each step is a 2x2x2 multiplication.
        # Each 2x2x2 multiplication has a rank-7 decomposition.
        # So we have 7 * 7 = 49 rank-1 tensors.
        
        # This is manageable.
        
        # First, define the rank-7 decomposition of 2x2x2.
        # Let's call the 2x2 matrix indices (i,j) for A, (j,k) for B, (k,i) for C.
        # i,j,k in {0,1}.
        
        # The 7 terms are:
        # 1. (A00+A11)(B00+B11) -> C00, C11
        # 2. (A10+A11)B00 -> C10, C11
        # 3. A00(B01-B11) -> C01, C11
        # 4. A11(B10-B00) -> C00, C10
        # 5. (A00+A01)B11 -> C01, C11
        # 6. (A10-A00)(B00+B01) -> C00
        # 7. (A01-A11)(B10+B11) -> C00
        
        # We need to list the 7 terms explicitly as (u, v, w) vectors of length 4.
        
        # Term 1:
        # u = [1, 0, 1, 1] (A00, A01, A10, A11) -> indices 0,1,2,3.
        # Actually, u[i*m+j]. For 2x2, m=2.
        # u[0*2+0]=1, u[0*2+1]=0, u[1*2+0]=0, u[1*2+1]=1.
        # u = [1, 0, 0, 1].
        # v[0*2+0]=1, v[0*2+1]=0, v[1*2+0]=0, v[1*2+1]=1.
        # v = [1, 0, 0, 1].
        # w[0*2+0]=1, w[0*2+1]=1, w[1*2+0]=0, w[1*2+1]=0.
        # w = [1, 1, 0, 0].
        
        # Term 2:
        # u = [0, 0, 1, 1].
        # v = [1, 0, 0, 0].
        # w = [0, 0, 1, 1].
        
        # Term 3:
        # u = [1, 0, 0, 0].
        # v = [0, 1, 0, -1].
        # w = [0, 1, 0, 1].
        
        # Term 4:
        # u = [0, 0, 0, 1].
        # v = [0, 0, 1, -1].
        # w = [1, 0, 1, 0].
        
        # Term 5:
        # u = [1, 1, 0, 0].
        # v = [0, 0, 0, 1].
        # w = [0, 1, 0, 1].
        
        # Term 6:
        # u = [ -1, 0, 1, 0].
        # v = [1, 1, 0, 0].
        # w = [1, 0, 0, 0].
        
        # Term 7:
        # u = [0, 1, 0, -1].
        # v = [0, 0, 1, 1].
        # w = [1, 0, 0, 0].
        
        # Let's verify Term 6: (A10-A00)(B00+B01) -> C00.
        # u = A10 - A00. Indices: A10 is (1,0) -> idx 2. A00 is (0,0) -> idx 0.
        # u = [-1, 0, 1, 0]. Correct.
        # v = B00 + B01. Indices: B00 is (0,0) -> idx 0. B01 is (0,1) -> idx 1.
        # v = [1, 1, 0, 0]. Correct.
        # w = C00. Index (0,0) -> idx 0.
        # w = [1, 0, 0, 0]. Correct.
        
        # Okay, we have the 2x2x2 decomposition.
        
        # Now, for the 4x4x4 case, we apply this to each of the 7 Strassen steps.
        # The 7 Strassen steps are linear combinations of the 2x2 blocks.
        # We need to "expand" the 2x2 indices to 4x4 indices.
        
        # For a given Strassen step s (0..6), and a given 2x2 decomposition term t (0..6):
        # We have a pair of linear forms (L_A, L_B) for the 2x2 blocks.
        # We need to map these to the 4x4 vectors.
        
        # This is quite complex to implement correctly without errors.
        # Given the time constraint, maybe I should stick to the 85 rank idea but implement it carefully.
        
        # Actually, there is a simpler way.
        # The rank of 4x4x4 is 49.
        # I can just hardcode the 49 terms if I can find them or derive them.
        
        # But wait, do I really need Strassen?
        # What if I use the 1+4 split and just use schoolbook for the 4x4x4 part?
        # Rank 4x4x4 schoolbook is 64.
        # Total rank = 4 + 16 + 16 + 64 = 100.
        # No improvement.
        
        # So I must use Strassen for the 4x4x4 part to get below 100.
        
        # Let's try to implement the Strassen 4x4x4 decomposition by "lifting" the 2x2x2 decomposition.
        
        # For each of the 7 Strassen steps (let's call them S1..S7):
        # Each Sk is a product of a linear combination of A-blocks and a linear combination of B-blocks.
        # Let the A-linear form be sum_{ij} a_{ij} A_{ij} (where A_{ij} is a 2x2 block).
        # Let the B-linear form be sum_{jk} b_{jk} B_{jk}.
        
        # The product is (sum a_{ij} A_{ij}) (sum b_{jk} B_{jk}).
        # This is a 2x2 matrix product? No, it's a scalar?
        # No, A_{ij} and B_{jk} are 2x2 matrices.
        # The product of two 2x2 matrices is a 2x2 matrix.
        # But Strassen's algorithm treats them as scalars in the "block" level?
        # No, Strassen's algorithm works on matrices.
        # The "multiplications" are matrix multiplications.
        # So each Strassen step involves a 2x2 matrix multiplication.
        # And we decompose that 2x2 matrix multiplication into 7 rank-1 tensors.
        
        # So, for each Strassen step s:
        # We have a 2x2 matrix A_s and a 2x2 matrix B_s.
        # We compute C_s = A_s B_s.
        # We decompose C_s = sum_{t=1}^7 U_{s,t} A_s V_{s,t}^T B_s W_{s,t}.
        
        # The overall C is a linear combination of C_s.
        
        # This seems correct.
        
        # Now, how to map to global indices?
        # A_s is a linear combination of the four 2x2 blocks of A.
        # So the vector U_{s,t} (which is a 4-vector for the 2x2 A_s) needs to be expanded to a 16-vector for the 4x4 A.
        
        # If A_s = sum_{ij} alpha_{ij} A_{block_{ij}}, then the global U vector has alpha_{ij} * U_{local} in the block_{ij} positions.
        
        # This is doable.
        
        # Let's define the 7 Strassen steps in terms of block coefficients.
        
        # Block indices: (0,0), (0,1), (1,0), (1,1).
        
        # S1: A = A00 + A11, B = B00 + B11.
        # S2: A = A10 + A11, B = B00.
        # S3: A = A00, B = B01 - B11.
        # S4: A = A11, B = B10 - B00.
        # S5: A = A00 + A01, B = B11.
        # S6: A = A10 - A00, B = B00 + B01.
        # S7: A = A01 - A11, B = B10 + B11.
        
        # And the C contributions:
        # C00 = S1 - S4 + S6 - S7
        # C01 = S3 + S5
        # C10 = S2 + S4
        # C11 = S1 - S2 + S3 + S5
        
        # Note: C00, etc. are 2x2 blocks.
        
        # So, for each step s, and each term t in the 2x2 decomposition:
        # We have local vectors u_t, v_t, w_t (length 4).
        # We expand them to global vectors U, V, W (length 16).
        
        # Expansion:
        # For A:
        # If the step s has block coefficients alpha_{ij}, then
        # U_global[ (i*2+ii)*4 + (j*2+jj) ] = alpha_{ij} * u_t[ii*2+jj].
        
        # Similarly for B and C.
        
        # This will give 7 * 7 = 49 terms.
        
        # Let's implement this.
        
        # Define the 2x2x2 decomposition terms.
        # u, v, w are lists of 4 floats.
        
        def strassen_2x2_terms():
            terms = []
            # Term 1
            terms.append(([1,0,0,1], [1,0,0,1], [1,1,0,0]))
            # Term 2
            terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,1]))
            # Term 3
            terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,1]))
            # Term 4
            terms.append(([0,0,0,1], [0,0,1,-1], [1,0,1,0]))
            # Term 5
            terms.append(([1,1,0,0], [0,0,0,1], [0,1,0,1]))
            # Term 6
            terms.append(([-1,0,1,0], [1,1,0,0], [1,0,0,0]))
            # Term 7
            terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
            return terms
        
        terms_2x2 = strassen_2x2_terms()
        
        # Define the Strassen steps for 4x4.
        # Each step is a dict: 'A': {(i,j): coeff}, 'B': {(i,j): coeff}, 'C': {(i,j): coeff}
        
        steps = []
        
        # S1
        steps.append({'A': {(0,0):1, (1,1):1}, 'B': {(0,0):1, (1,1):1}, 'C': {(0,0):1, (1,1):1}})
        # S2
        steps.append({'A': {(1,0):1, (1,1):1}, 'B': {(0,0):1}, 'C': {(1,0):1, (1,1):-1}})
        # S3
        steps.append({'A': {(0,0):1}, 'B': {(0,1):1, (1,1):-1}, 'C': {(0,1):1, (1,1):1}})
        # S4
        steps.append({'A': {(1,1):1}, 'B': {(1,0):1, (0,0):-1}, 'C': {(0,0):-1, (1,0):1}})
        # S5
        steps.append({'A': {(0,0):1, (0,1):1}, 'B': {(1,1):1}, 'C': {(0,1):1, (1,1):1}})
        # S6
        steps.append({'A': {(1,0):1, (0,0):-1}, 'B': {(0,0):1, (0,1):1}, 'C': {(0,0):1}})
        # S7
        steps.append({'A': {(0,1):1, (1,1):-1}, 'B': {(1,0):1, (1,1):1}, 'C': {(0,0):-1}})
        
        # Now generate the 49 terms for 4x4x4.
        U_4x4 = []
        V_4x4 = []
        W_4x4 = []
        
        for s in steps:
            for t in range(7):
                u_loc, v_loc, w_loc = terms_2x2[t]
                
                U_glob = [0]*16
                V_glob = [0]*16
                W_glob = [0]*16
                
                # Expand A
                for (bi, bj), coeff in s['A'].items():
                    for ii in range(2):
                        for jj in range(2):
                            gi = bi*2 + ii
                            gj = bj*2 + jj
                            U_glob[gi*4 + gj] += coeff * u_loc[ii*2 + jj]
                            
                # Expand B
                for (bi, bj), coeff in s['B'].items():
                    for ii in range(2):
                        for jj in range(2):
                            gi = bi*2 + ii
                            gj = bj*2 + jj
                            V_glob[gi*4 + gj] += coeff * v_loc[ii*2 + jj]
                            
                # Expand C
                for (bi, bj), coeff in s['C'].items():
                    for ii in range(2):
                        for jj in range(2):
                            gi = bi*2 + ii
                            gj = bj*2 + jj
                            W_glob[gi*4 + gj] += coeff * w_loc[ii*2 + jj]
                            
                U_4x4.append(U_glob)
                V_4x4.append(V_glob)
                W_4x4.append(W_glob)
                
        # Now we have the 49 terms for the 4x4x4 block (A2*B22).
        # We need to place them into the global U, V, W matrices for the 4x5x5 problem.
        
        # Global dimensions:
        # U: 4x5 (20 rows)
        # V: 5x5 (25 rows)
        # W: 5x4 (20 rows)
        
        # The 4x4x4 block corresponds to:
        # A2: rows 0..3, cols 4..7 (but A is 4x5, so cols 0..4. A2 is cols 1..4? No.)
        
        # Recall the split:
        # A (4x5) = A1 (4x1) + A2 (4x4).
        # A1 is col 0. A2 is cols 1..4.
        
        # B (5x5) = [[B11, B12], [B21, B22]].
        # B11 is row 0, col 0.
        # B12 is row 0, cols 1..4.
        # B21 is rows 1..4, col 0.
        # B22 is rows 1..4, cols 1..4.
        
        # C (4x5).
        # C = A1 B11 + A1 B12 + A2 B21 + A2 B22.
        
        # The term A2 B22 corresponds to:
        # A indices: i in 0..3, j in 1..4.
        # B indices: j in 1..4, k in 1..4.
        # C indices: k in 1..4, i in 0..3.
        
        # So in the global matrices:
        # U row index: i*5 + j.
        # V row index: j*5 + k.
        # W row index: k*4 + i.
        
        # We need to map the local 4x4 indices (0..3) to the global indices.
        # A2 local (i, j_loc) -> global (i, j_loc + 1).
        # B22 local (j_loc, k_loc) -> global (j_loc + 1, k_loc + 1).
        # C22 local (k_loc, i) -> global (k_loc + 1, i).
        
        # Let's construct the global U, V, W.
        
        U = [[0]*85 for _ in range(20)]
        V = [[0]*85 for _ in range(25)]
        W = [[0]*85 for _ in range(20)]
        
        rank = 0
        
        # Add the 49 terms for A2 B22
        for r in range(49):
            u_loc = U_4x4[r]
            v_loc = V_4x4[r]
            w_loc = W_4x4[r]
            
            for i in range(4):
                for j_loc in range(4):
                    val = u_loc[i*4 + j_loc]
                    if val != 0:
                        j_global = j_loc + 1
                        U[i*5 + j_global][rank] = val
                        
            for j_loc in range(4):
                for k_loc in range(4):
                    val = v_loc[j_loc*4 + k_loc]
                    if val != 0:
                        j_global = j_loc + 1
                        k_global = k_loc + 1
                        V[j_global*5 + k_global][rank] = val
                        
            for k_loc in range(4):
                for i in range(4):
                    val = w_loc[k_loc*4 + i]
                    if val != 0:
                        k_global = k_loc + 1
                        W[k_global*4 + i][rank] = val
                        
            rank += 1
            
        # Now add the other 36 terms (schoolbook).
        
        # Term A1 B11: 4x1x1. Rank 4.
        # A1: col 0. B11: row 0, col 0.
        # C: row 0, col 0.
        for i in range(4):
            U[i*5 + 0][rank] = 1
            V[0*5 + 0][rank] = 1
            W[0*4 + i][rank] = 1
            rank += 1
            
        # Term A1 B12: 4x1x4. Rank 16.
        # A1: col 0. B12: row 0, cols 1..4.
        # C: rows 1..4, col 0.
        for i in range(4):
            for k in range(1, 5):
                U[i*5 + 0][rank] = 1
                V[0*5 + k][rank] = 1
                W[k*4 + i][rank] = 1
                rank += 1
                
        # Term A2 B21: 4x4x1. Rank 16.
        # A2: cols 1..4. B21: rows 1..4, col 0.
        # C: row 0, cols 1..4.
        for i in range(4):
            for j in range(1, 5):
                U[i*5 + j][rank] = 1
                V[j*5 + 0][rank] = 1
                W[0*4 + i][rank] = 1
                rank += 1
                
        # Total rank should be 49 + 4 + 16 + 16 = 85.
        
        assert rank == 85, f"Rank is {rank}, expected 85"
        
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

