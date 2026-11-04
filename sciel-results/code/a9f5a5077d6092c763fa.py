import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank calculated as 37
    rank = 37
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal current_rank
        # Strassen 2x2x2 decomposition
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Coefficients for U, V, W for each of the 7 terms
        # U indices: (i, r) where i in {0,1}, r in {0..6}
        # V indices: (j, r) where j in {0,1}, r in {0..6}
        # W indices: (k, r) where k in {0,1}, r in {0..6}
        
        # We need to define U[i, r], V[j, r], W[k, r] such that
        # sum_r U[i,r] V[j,r] W[k,r] = C[i,k]
        
        # Let's use the standard Strassen coefficients.
        # The terms are linear combinations of products.
        # It is easier to define the decomposition of the tensor T[i,j,k] = A[i,j] B[j,k].
        
        # Term 1: (A00+A11)(B00+B11)
        # A00: i=0,j=0. A11: i=1,j=1.
        # B00: j=0,k=0. B11: j=1,k=1.
        # This term contributes to C00 and C11.
        # C00 gets +1, C11 gets +1.
        # C01, C10 get 0.
        
        # We can define U, V, W for each term r.
        # For term r, U[i,r], V[j,r], W[k,r] are constants.
        
        # Let's hardcode the 7x2x2x2 structure.
        # r=0: M1
        # U[0,0]=1, U[1,0]=1
        # V[0,0]=1, V[1,0]=1
        # W[0,0]=1, W[1,0]=1
        # Check: sum U[i,0]V[j,0]W[k,0] = (i==0?1:1)*(j==0?1:1)*(k==0?1:1) = 1.
        # This adds 1 to all C[i,k].
        # But M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # The tensor product A[i,j]B[j,k] sums to C[i,k].
        # M1 is not a single tensor term, it's a sum of 4 tensor terms.
        # Wait, Strassen works by computing 7 products of sums.
        # The decomposition of the multiplication tensor itself is different.
        
        # The rank of M_2,2,2 is 7.
        # The decomposition is:
        # T[i,j,k] = sum_{r=0}^6 U[i,r] V[j,r] W[k,r]
        
        # Standard Strassen decomposition coefficients:
        # r=0: U=[1,1], V=[1,1], W=[1,1] -> Sum = 1 for all i,j,k? No.
        # U[i,0] V[j,0] W[k,0].
        # If U=[1,1], V=[1,1], W=[1,1], then term is 1.
        # This corresponds to (A00+A11+A10+A01)(B00+B11+B10+B01)? No.
        
        # Let's look up the explicit U,V,W for Strassen.
        # Reference: "Rank of matrix multiplication"
        
        # One known decomposition:
        # r=0: U=(1,1), V=(1,1), W=(1,1)
        # r=1: U=(0,1), V=(1,0), W=(1,0)
        # r=2: U=(1,0), V=(0,1), W=(1,0)
        # r=3: U=(1,1), V=(0,1), W=(0,1)
        # r=4: U=(1,0), V=(1,1), W=(0,1)
        # r=5: U=(0,1), V=(1,1), W=(1,0)
        # r=6: U=(1,0), V=(1,0), W=(1,1)
        
        # Let's verify r=0:
        # sum U[i,0]V[j,0]W[k,0] = 1*1*1 = 1.
        # This adds 1 to T[i,j,k] for all i,j,k.
        # T[i,j,k] is 1 if we are computing A[i,j]B[j,k] for A=I, B=I?
        # If A=I, B=I, C=I.
        # T[i,j,k] = delta_ij delta_jk = delta_ik.
        # Sum of all 7 terms should be delta_ik.
        
        # Let's check the sum for i=0, k=0 (should be 1).
        # r=0: 1*1*1 = 1
        # r=1: 0*1*1 = 0
        # r=2: 1*0*1 = 0
        # r=3: 1*0*0 = 0
        # r=4: 1*1*0 = 0
        # r=5: 0*1*1 = 0
        # r=6: 1*1*1 = 1
        # Sum = 2. Should be 1.
        
        # The coefficients above might be for a specific basis or scaled.
        # Let's use a verified set.
        
        # Verified Strassen U,V,W (from literature):
        # r=0: U=(1,1), V=(1,1), W=(1,1)
        # r=1: U=(0,1), V=(1,0), W=(1,0)
        # r=2: U=(1,0), V=(0,1), W=(1,0)
        # r=3: U=(1,1), V=(0,1), W=(0,1)
        # r=4: U=(1,0), V=(1,1), W=(0,1)
        # r=5: U=(0,1), V=(1,1), W=(1,0)
        # r=6: U=(1,0), V=(1,0), W=(1,1)
        
        # Let's re-evaluate sum for i=0, j=0, k=0.
        # r=0: 1*1*1 = 1
        # r=1: 0*1*1 = 0
        # r=2: 1*0*1 = 0
        # r=3: 1*0*0 = 0
        # r=4: 1*1*0 = 0
        # r=5: 0*1*1 = 0
        # r=6: 1*1*1 = 1
        # Sum = 2.
        
        # It seems these coefficients yield 2*delta_ik?
        # If so, we can just divide by 2? Or the decomposition is for 2*T?
        # If the sum is 2*delta_ik, then we can scale W by 0.5.
        
        # Let's check i=0, j=1, k=1. (Should be 0)
        # r=0: 1*1*1 = 1
        # r=1: 0*0*1 = 0
        # r=2: 1*1*1 = 1
        # r=3: 1*1*0 = 0
        # r=4: 1*1*0 = 0
        # r=5: 0*1*1 = 0
        # r=6: 1*0*1 = 0
        # Sum = 2. Should be 0.
        
        # So this set is NOT correct for T[i,j,k] = delta_ik.
        
        # Correct Strassen decomposition for T[i,j,k]:
        # T = M1 - M2 + M3 + M4 ...
        
        # Let's use the explicit linear combination approach.
        # We know the 7 products P0..P6.
        # We can define U,V,W such that the sum reconstructs the products.
        
        # Actually, it is safer to just use the naive decomposition for 2x2x2 (rank 8) if I can't recall the exact coefficients.
        # But rank 7 is the point.
        
        # Let's use this known valid set (from Wikipedia "Strassen algorithm" tensor decomposition):
        # The 7 terms are:
        # 1. (A00+A11)(B00+B11)
        # 2. (A10+A11)B00
        # 3. A00(B10-B11)
        # 4. A11(B01-B00)
        # 5. (A00+A10)B11
        # 6. (A10-A00)(B00+B01)
        # 7. (A01-A11)(B10+B11)
        
        # We can implement this by adding 7 terms to U,V,W.
        # For each term, we define U,V,W vectors.
        
        # Term 1: (A00+A11)(B00+B11)
        # This is not a single rank-1 tensor. It is a sum of 4.
        # Strassen's trick is that the 7 terms are rank-1 tensors in the *product* space,
        # but we are decomposing the *multiplication* tensor.
        
        # The multiplication tensor T[i,j,k] has rank 7.
        # The decomposition is:
        # T[i,j,k] = sum_{r=0}^6 U[i,r] V[j,r] W[k,r]
        
        # I will use the coefficients from:
        # https://en.wikipedia.org/wiki/Strassen_algorithm#Tensor_decomposition
        # U = [ [1, 0, 1, 1, 1, 0, 1],
        #       [1, 1, 0, 1, 0, 1, 0] ]
        # V = [ [1, 1, 0, 1, 1, 1, 0],
        #       [1, 0, 1, 1, 0, 0, 1] ]
        # W = [ [1, 1, 1, 0, 0, 1, 1],
        #       [1, 0, 0, 1, 1, 0, 1] ]
        
        # Let's verify this set.
        # i=0, j=0, k=0:
        # r=0: 1*1*1 = 1
        # r=1: 0*1*1 = 0
        # r=2: 1*0*1 = 0
        # r=3: 1*1*0 = 0
        # r=4: 1*1*0 = 0
        # r=5: 0*1*1 = 0
        # r=6: 1*0*1 = 0
        # Sum = 1. Correct.
        
        # i=0, j=0, k=1:
        # r=0: 1*1*1 = 1
        # r=1: 0*1*0 = 0
        # r=2: 1*0*0 = 0
        # r=3: 1*1*1 = 1
        # r=4: 1*1*1 = 1
        # r=5: 0*1*0 = 0
        # r=6: 1*0*1 = 0
        # Sum = 3. Should be 0.
        
        # This set is also wrong.
        
        # I will fall back to the naive 2x2x2 decomposition (rank 8) to ensure correctness, 
        # but use the optimized splitting for larger blocks.
        # Rank for 2x2x2 = 8.
        # Part 1 Top: 4 blocks * 8 = 32.
        # Part 1 Bottom: 4.
        # Part 2: 5.
        # Total: 41.
        
        # This is still an improvement over 60.
        
        # Naive 2x2x2:
        # r=0: i=0,j=0,k=0
        # r=1: i=0,j=1,k=0
        # r=2: i=0,j=0,k=1
        # r=3: i=0,j=1,k=1
        # r=4: i=1,j=0,k=0
        # r=5: i=1,j=1,k=0
        # r=6: i=1,j=0,k=1
        # r=7: i=1,j=1,k=1
        
        for r in range(8):
            i_loc = (r // 4)
            j_loc = (r % 4) // 2
            k_loc = (r % 4) % 2
            
            # Map local to global
            i_g = i_off + i_loc
            j_g = j_off + j_loc
            k_g = k_off + k_loc
            
            U[i_g * m + j_g][current_rank] = 1
            V[j_g * p + k_g][current_rank] = 1
            W[k_g * n + i_g][current_rank] = 1
            current_rank += 1

    def add_1_n_1(i_off, j_off, k_off, n_rows, m_cols):
        # 1 x m x 1
        # Rank 1
        nonlocal current_rank
        i_g = i_off
        k_g = k_off
        # Sum_j A[i,j] B[j,k]
        # U[i, r] V[j, r] W[k, r]
        # r=0
        U[i_g * m + j_off][current_rank] = 0 # Placeholder
        # We need sum_j A[i,j] B[j,k] = U[i,r] V[j,r] W[k,r]
        # Set W[k,r] = 1.
        # U[i,r] = 1.
        # V[j,r] = A[i,j] B[j,k]? No, V depends on j, not on A.
        # Wait, the decomposition must work for ALL A, B.
        # So U, V, W cannot depend on A, B values.
        # My previous derivation for 1xm1 was:
        # U[i,r] = A[i,j]? No, U is a matrix of coefficients.
        
        # Re-evaluating 1xm1 decomposition.
        # T[i,j,k] = A[i,j] B[j,k].
        # We need T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r].
        # Here i=0, k=0.
        # T[0,j,0] = A[0,j] B[j,0].
        # If we use rank 1:
        # A[0,j] B[j,0] = U[0,0] V[j,0] W[0,0].
        # This implies A[0,j] B[j,0] is a rank-1 matrix in j.
        # It is NOT generally rank 1.
        # So 1xm1 has rank > 1.
        
        # Rank of M_{1,m,1} is m?
        # Yes, the rank is m.
        # Decomposition:
        # r = j.
        # U[0, j] = 1.
        # V[j, j] = 1. (Identity)
        # W[0, j] = 1.
        # Sum_j U[0,j] V[j',j] W[0,j] = sum_j 1 * delta_{j',j} * 1 = 1 if j'=j?
        # No, the sum is over r.
        # T[i,j',k] = sum_r U[i,r] V[j',r] W[k,r].
        # We want T[0,j',0] = A[0,j'] B[j',0].
        # If we set r=j'.
        # U[0,r] = 1.
        # V[j',r] = delta_{j',r}.
        # W[0,r] = 1.
        # Sum_r 1 * delta_{j',r} * 1 = 1.
        # This gives 1, not A[0,j']B[j',0].
        
        # The coefficients U,V,W are FIXED.
        # So we CANNOT use A and B values in U,V,W.
        
        # Therefore, the rank of M_{1,m,1} is NOT m in the tensor decomposition sense?
        # Wait. The tensor T[i,j,k] is the identity tensor for the multiplication.
        # T[i,j,k] = 1 if the indices match the multiplication pattern?
        # No, T[i,j,k] is the coefficient of A[i,j] B[j,k] in C[i,k].
        # T[i,j,k] = 1 for all valid i,j,k.
        
        # So for 1xm1:
        # T[0,j,0] = 1.
        # We need sum_r U[0,r] V[j,r] W[0,r] = 1.
        # This is easy.
        # Set U[0,0]=1, W[0,0]=1, V[j,0]=1 for all j.
        # Sum = 1.
        # Rank 1!
        
        # So 1xm1 has rank 1.
        
        # My previous confusion was mixing up the values of A,B with the tensor coefficients.
        # The tensor coefficients are all 1.
        
        # So:
        # 2x2x2 rank 7 (Strassen) or 8 (Naive).
        # 1xm1 rank 1.
        
        # Let's use Naive 2x2x2 (rank 8) to be safe.
        
        # Re-calculate rank with Naive 2x2x2:
        # Part 1 Top: 4 * 8 = 32.
        # Part 1 Bottom:
        # 1x4x4 split to 1x4x2 + 1x4x2.
        # 1x4x2 split to 1x4x1 + 1x4x1.
        # Each 1x4x1 has rank 1.
        # So 1x4x2 has rank 2.
        # 1x4x4 has rank 4.
        # Part 2:
        # 3x4x1 split to 2x4x1 + 1x4x1.
        # 2x4x1 split to 2x2x1 + 2x2x1.
        # 2x2x1:
        # T[i,j,0] = 1.
        # sum_r U[i,r] V[j,r] W[0,r] = 1.
        # Set r=0. U[i,0]=1, V[j,0]=1, W[0,0]=1.
        # Rank 1.
        # So 2x4x1 has rank 2.
        # 1x4x1 has rank 1.
        # Part 2 total rank 3.
        
        # Total Rank: 32 (Top) + 4 (Bottom) + 3 (Part 2) = 39.
        
        # This is still better than 60.
        
        # Implementation:
        
        # For 1xm1 (or 1xmxp):
        # T[i,j,k] = 1.
        # We can use rank 1 if p=1?
        # If p>1, T[i,j,k] = 1 for all k.
        # sum_r U[i,r] V[j,r] W[k,r] = 1.
        # If we set W[k,r] = 1 for all k, and U[i,0]=1, V[j,0]=1, then sum = 1.
        # So 1xmxp has rank 1?
        # Yes, if we don't care about which k it contributes to?
        # No, W[k,r] depends on k.
        # If we set W[k,0] = 1 for all k, then the term is 1 for all k.
        # This works.
        
        # So any 1xmxp block has rank 1.
        
        # And any nx1x1 block has rank 1.
        # And any nx1xp block?
        # T[i,0,k] = 1.
        # sum_r U[i,r] V[0,r] W[k,r] = 1.
        # Set V[0,0]=1, U[i,0]=1, W[k,0]=1.
        # Rank 1.
        
        # So any block with a dimension of 1 has rank 1.
        
        # Only 2x2x2 blocks need rank 7 or 8.
        
        # Final Rank Calculation:
        # Part 1 Top: 4 blocks of 2x2x2. Rank 4*8 = 32.
        # Part 1 Bottom: 1x4x4.
        # Split to 1x4x2 + 1x4x2.
        # 1x4x2 has rank 1.
        # So 1x4x4 has rank 2.
        # Part 2: 3x4x1.
        # Split to 2x4x1 + 1x4x1.
        # 2x4x1 has rank 1.
        # 1x4x1 has rank 1.
        # Part 2 total rank 2.
        
        # Total Rank: 32 + 2 + 2 = 36.
        
        # Let's verify 1x4x2 rank 1.
        # T[0,j,k] = 1.
        # sum_r U[0,r] V[j,r] W[k,r] = 1.
        # Set r=0. U[0,0]=1, V[j,0]=1, W[k,0]=1.
        # Sum = 1.
        # Yes.
        
        # So total rank is 36.
        
        # I will implement this.
        
        pass

    # Part 1 Top: 4 blocks of 2x2x2
    add_strassen(0, 0, 0)
    add_strassen(0, 2, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 2)
    
    # Part 1 Bottom: 1x4x4
    # Split k into 0,1 and 2,3
    # 1x4x2 (k=0,1) -> Rank 1
    # 1x4x2 (k=2,3) -> Rank 1
    # Implement 1xmxp rank 1
    def add_1_m_p(i_off, j_off, k_off, m_len, p_len):
        nonlocal current_rank
        i_g = i_off
        U[i_g * m + j_off][current_rank] = 1
        for j in range(m_len):
            V[(j_off + j) * p + k_off][current_rank] = 1
        for k in range(p_len):
            W[(k_off + k) * n + i_g][current_rank] = 1
        current_rank += 1
        
    add_1_m_p(2, 0, 0, 4, 2)
    add_1_m_p(2, 0, 2, 4, 2)
    
    # Part 2: 3x4x1
    # Split i into 0,1 and 2
    # 2x4x1 -> Rank 1
    # 1x4x1 -> Rank 1
    def add_n_1_1(i_off, j_off, k_off, n_len):
        nonlocal current_rank
        k_g = k_off
        for i in range(n_len):
            U[(i_off + i) * m + j_off][current_rank] = 1
        for j in range(4):
            V[j * p + k_g][current_rank] = 1
        for i in range(n_len):
            W[k_g * n + (i_off + i)][current_rank] = 1
        current_rank += 1
        
    add_n_1_1(0, 0, 4, 2)
    add_n_1_1(2, 0, 4, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
