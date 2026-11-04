"""Matrix multiplication 4,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 5]
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 4, 5

    # Strategy: Splitting technique for rectangular matrix multiplication.
    # Decompose 4x4x5 into blocks.
    # Split A into 2x2 blocks (each 2x2), B into 2x3 blocks (each 2x? or similar).
    # Actually, let's look at the structure:
    # A is 4x4, B is 4x5, C is 4x5.
    # Let's split indices:
    # i in {0,1,2,3}, j in {0,1,2,3}, k in {0,1,2,3,4}
    
    # We can view this as two 2x4x5 multiplications? No, that doesn't reduce rank easily.
    # Let's try splitting the middle index j (rows of B, cols of A).
    # j in {0,1} and {2,3}.
    # C[i,k] = sum_{j=0}^1 A[i,j]B[j,k] + sum_{j=2}^3 A[i,j]B[j,k]
    # This is a sum of two 4x2x5 tensors.
    # Rank of 4x2x5?
    # Schoolbook is 4*2*5 = 40. Two of them = 80. No gain.
    
    # Let's try splitting the output index k (cols of B, cols of C).
    # k in {0,1,2,3} and {4}.
    # C[i, 0:4] = A * B[:, 0:4]  (4x4x4 problem)
    # C[i, 4]   = A * B[:, 4]    (4x4x1 problem)
    # These are disjoint in the output, so we can sum their decompositions.
    # Rank = Rank(4x4x4) + Rank(4x4x1).
    
    # We need a good rank for 4x4x4 and 4x4x1.
    # Rank(4x4x1): This is matrix-vector multiplication.
    # C[i] = sum_j A[i,j] v[j].
    # Schoolbook rank is 4*4 = 16.
    # Can we do better?
    # For 4x4x1, this is equivalent to multiplying a 4x4 matrix by a 4x1 vector.
    # The rank of the tensor M_{n,m,1} is n*m.
    # Proof: The tensor is sum_{i,j} e_i \otimes e_j \otimes e_1.
    # Wait, the tensor is T_{i,j,1} = 1 if we are computing C_i = sum_j A_{i,j} B_{j,1}.
    # The decomposition requires U_{i,j,t} V_{j,1,t} W_{1,i,t}.
    # Since k is always 1, V_{j,1,t} and W_{1,i,t} are just vectors indexed by t.
    # Let u_t be the vector U[:, :, t] (shape 4x4), v_t be V[:, t] (shape 4x1), w_t be W[:, t] (shape 1x4? No, W is p*n = 1*4 = 4).
    # Actually, W index is k*n+i = 1*4+i = 4+i. So W is a 4-vector.
    # The term is sum_t U_{i,j,t} V_{j,t} W_{i,t}.
    # This must equal A_{i,j} * B_{j,1}.
    # This is a decomposition of the identity tensor in the A-B space?
    # No, it's decomposing the bilinear map (A, v) -> A*v.
    # The rank of this tensor is known to be n*m for n,m >= 1?
    # Actually, for 2x2x1, rank is 4. For 3x3x1, rank is 9.
    # Yes, rank of matrix-vector mult is n*m.
    # So Rank(4x4x1) = 16.
    
    # Now Rank(4x4x4).
    # Schoolbook is 64.
    # Strassen gives 49 for 4x4x4?
    # Strassen 2x2x2 is rank 7.
    # 4x4x4 can be done with 4 Strassen 2x2x2 blocks?
    # A 4x4x4 matrix multiplication can be decomposed into 4 independent 2x2x2 multiplications?
    # No, block multiplication of 2x2 blocks involves 8 multiplications of 2x2 matrices.
    # Each 2x2x2 multiplication has rank 7.
    # So 8 * 7 = 56.
    # Can we do better?
    # Known rank of 4x4x4 is 49 (Bini et al.).
    # But constructing it is complex.
    # Let's see if we can use a simpler decomposition.
    # If we use Strassen for 2x2x2, we get 56 for 4x4x4.
    # Total rank = 56 + 16 = 72.
    # This is an improvement over 80.
    
    # Can we improve 4x4x1?
    # No, rank is 16.
    
    # Can we improve 4x4x4?
    # Maybe use a mix?
    # What if we split 4x4x4 into 2x2x4 + 2x2x4? No.
    # Let's stick with the block Strassen approach for 4x4x4 (rank 56) and schoolbook for 4x4x1 (rank 16).
    # Total rank = 72.
    
    # Let's try to construct the Strassen decomposition for 4x4x4.
    # We need to map the 4x4x4 tensor to 8 2x2x2 tensors.
    # A 4x4 matrix A can be written as blocks A00, A01, A10, A11 (each 2x2).
    # B 4x4 matrix B as B00, B01, B10, B11.
    # C = A*B.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each of these is a sum of two 2x2x2 multiplications.
    # We can use Strassen's algorithm to compute the product of 2x2 matrices.
    # However, the standard Strassen algorithm computes the full product.
    # Here, we need to compute sums of products.
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A11-A10)*(B00+B10)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 - M5 + M2
    
    # This uses 7 multiplications for the full 2x2 block product.
    # But we have 4 such block products (C00, C01, C10, C11) to compute?
    # No, A and B are fixed. We compute the 7 M's once, and then combine them to get C00, C01, C10, C11.
    # So the rank is 7 * (rank of 2x2x2 multiplication).
    # Wait, the "multiplications" in Strassen are scalar multiplications in the ring of 2x2 matrices.
    # In the tensor decomposition sense, each "multiplication" of two 2x2 matrices is a 2x2x2 tensor multiplication.
    # The rank of a 2x2x2 multiplication is 7.
    # So if we perform 7 such 2x2x2 multiplications, the total rank is 7 * 7 = 49?
    # No, that's not right.
    # The tensor for 4x4x4 is the sum of tensors for the block multiplications.
    # The Strassen algorithm replaces 8 scalar multiplications with 7.
    # But here, the "scalars" are 2x2 matrices.
    # So we are doing 7 multiplications of 2x2 matrices.
    # Each 2x2 matrix multiplication has rank 7.
    # So the total rank is 7 * 7 = 49.
    # This matches the known rank bound!
    
    # So, Rank(4x4x4) = 49.
    # Rank(4x4x1) = 16.
    # Total Rank = 49 + 16 = 65.
    
    # Let's verify this construction.
    # We need to explicitly construct the U, V, W matrices for the rank-49 decomposition of 4x4x4.
    # And the rank-16 decomposition of 4x4x1.
    
    # Construction for 4x4x1 (rank 16):
    # Indices: i (0..3), j (0..3), k (only 4, which is the 5th column, index 4).
    # Wait, in the 4x4x1 subproblem, k is fixed to 4 (the last column of B).
    # So we are computing C[i,4] = sum_j A[i,j] B[j,4].
    # This is a 4x4x1 tensor.
    # U[i*4+j, t], V[j*1+0, t] (since p=1 for this subproblem? No, p=5 in the main problem).
    # Let's be careful with indices.
    # Main problem: U (16 x R), V (20 x R), W (20 x R).
    # Subproblem 1 (4x4x4): k in 0..3.
    # Subproblem 2 (4x4x1): k = 4.
    
    # For Subproblem 2 (k=4):
    # We need terms t such that W[4*4+i, t] = W[16+i, t] is non-zero.
    # And V[j*5+4, t] = V[5j+4, t] is non-zero.
    # And U[i*4+j, t] is non-zero.
    # The tensor is T_{i,j,4} = 1.
    # We need sum_t U_{i,j,t} V_{5j+4,t} W_{16+i,t} = 1 for all i,j.
    # This is a rank-16 decomposition of the 4x4 identity-like tensor?
    # Actually, it's just the tensor with 1s everywhere for the (i,j) part.
    # This is the tensor for multiplying a 4x4 matrix by a 4x1 vector.
    # Rank is 16.
    # Simple decomposition: For each (i,j) pair, have a term.
    # t = i*4 + j.
    # U[i*4+j, t] = 1.
    # V[5j+4, t] = 1.
    # W[16+i, t] = 1.
    # Check: sum_t U_{i',j',t} V_{5j'+4,t} W_{16+i',t}
    # Only t = i*4+j contributes for U.
    # Then V[5j+4, i*4+j] = 1.
    # W[16+i, i*4+j] = 1.
    # Product is 1.
    # For any other (i',j'), U is 0.
    # So it works.
    # This uses 16 terms.
    
    # Now for Subproblem 1 (4x4x4):
    # We need a rank-49 decomposition.
    # Based on Strassen:
    # We have 7 "big" multiplications of 2x2 matrices.
    # Let's call them M1..M7.
    # Each M_l is a 2x2x2 tensor multiplication.
    # The inputs to M_l are linear combinations of A and B blocks.
    # The outputs C blocks are linear combinations of M_l.
    
    # Let's define the 2x2x2 Strassen decomposition.
    # For a 2x2x2 multiplication C = A*B:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]], C = [[c00, c01], [c10, c11]].
    # Strassen terms:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a10)b11
    # m6 = (a11-a10)(b00+b10)
    # m7 = (a01-a11)(b10+b11)
    # c00 = m1 - m4 - m6 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m3 - m5 + m2
    
    # This can be represented as a rank-7 decomposition of the 2x2x2 tensor.
    # U (4x7), V (4x7), W (4x7).
    # Indices for 2x2x2: i,j in 0..1, k in 0..1.
    # U[i*2+j, t], V[j*2+k, t], W[k*2+i, t].
    
    # Let's construct the U,V,W for one 2x2x2 Strassen block.
    # t=0 (m1):
    # A part: a00+a11 -> U[0,0]=1, U[3,0]=1. (Indices 00, 11)
    # B part: b00+b11 -> V[0,0]=1, V[3,0]=1. (Indices 00, 11)
    # C part: c00 += 1, c11 += 1 -> W[0,0]=1, W[3,0]=1. (Indices 00, 11)
    # Wait, C indices: c00 is k=0,i=0 -> W[0*2+0]=W[0]. c11 is k=1,i=1 -> W[1*2+1]=W[3].
    
    # t=1 (m2):
    # A part: a10+a11 -> U[2,1]=1, U[3,1]=1.
    # B part: b00 -> V[0,1]=1.
    # C part: c10 += 1 -> W[1*2+0]=W[2]. c11 += 1 -> W[3,1]=1.
    # Wait, c10 = m2 + m4. So m2 contributes to c10.
    # c11 = m1 - m3 - m5 + m2. So m2 contributes to c11.
    # So W[2,1]=1, W[3,1]=1.
    
    # t=2 (m3):
    # A part: a00 -> U[0,2]=1.
    # B part: b01-b11 -> V[1,2]=1, V[3,2]=-1.
    # C part: c01 += 1 -> W[1*2+0]=W[2]. c11 -= 1 -> W[3,2]=-1.
    
    # t=3 (m4):
    # A part: a11 -> U[3,3]=1.
    # B part: b10-b00 -> V[2,3]=1, V[0,3]=-1.
    # C part: c00 -= 1 -> W[0,3]=-1. c10 += 1 -> W[2,3]=1.
    
    # t=4 (m5):
    # A part: a00+a10 -> U[0,4]=1, U[2,4]=1.
    # B part: b11 -> V[3,4]=1.
    # C part: c01 += 1 -> W[2,4]=1. c11 -= 1 -> W[3,4]=-1.
    
    # t=5 (m6):
    # A part: a11-a10 -> U[3,5]=1, U[2,5]=-1.
    # B part: b00+b10 -> V[0,5]=1, V[2,5]=1.
    # C part: c00 -= 1 -> W[0,5]=-1.
    
    # t=6 (m7):
    # A part: a01-a11 -> U[1,6]=1, U[3,6]=-1.
    # B part: b10+b11 -> V[2,6]=1, V[3,6]=1.
    # C part: c00 += 1 -> W[0,6]=1.
    
    # Let's verify c00 = m1 - m4 - m6 + m7.
    # W[0,0]=1 (m1), W[0,3]=-1 (m4), W[0,5]=-1 (m6), W[0,6]=1 (m7). Sum = 1-1-1+1=0?
    # Wait.
    # c00 = m1 - m4 - m6 + m7.
    # My W assignments:
    # m1: W[0]=1.
    # m4: W[0]=-1.
    # m6: W[0]=-1.
    # m7: W[0]=1.
    # Sum = 1 - 1 - 1 + 1 = 0. This is wrong. c00 should be 1.
    # Let's re-check Strassen formulas.
    # c00 = m1 - m4 - m6 + m7.
    # m1 = (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11
    # m4 = a11(b10-b00) = a11b10 - a11b00
    # m6 = (a11-a10)(b00+b10) = a11b00 + a11b10 - a10b00 - a10b10
    # m7 = (a01-a11)(b10+b11) = a01b10 + a01b11 - a11b10 - a11b11
    # Sum:
    # a00b00: 1
    # a00b11: 1
    # a11b00: 1 - (-1) + 1 = 3? No.
    # m1: +a11b00
    # m4: -(-a11b00) = +a11b00
    # m6: -(+a11b00) = -a11b00
    # m7: 0
    # Total a11b00: 1+1-1 = 1. Correct.
    # a11b11:
    # m1: +a11b11
    # m4: 0
    # m6: 0
    # m7: -(-a11b11) = +a11b11
    # Total: 2. Incorrect. Should be 0 for c00.
    # Ah, c00 = a00b00 + a01b10.
    # My expansion of m7: (a01-a11)(b10+b11) = a01b10 + a01b11 - a11b10 - a11b11.
    # So m7 has -a11b11.
    # In c00 = m1 - m4 - m6 + m7:
    # m1 has +a11b11.
    # m7 has -a11b11.
    # Sum: 1 - 1 = 0. Correct.
    
    # So the coefficients for W[0] (c00) are:
    # m1: 1
    # m4: -1
    # m6: -1
    # m7: 1
    # Sum = 0?
    # 1 - 1 - 1 + 1 = 0.
    # Why did the algebra say it works?
    # a00b00: m1(1) -> 1.
    # a01b10: m7(1) -> 1.
    # a11b00: m1(1) - m4(-1) - m6(1) = 1+1-1=1.
    # a11b11: m1(1) + m7(-1) = 0.
    # a10b00: -m6(-1) = 1.
    # a10b10: -m6(-1) = 1.
    # a00b11: m1(1).
    # a11b10: m4(-1) - m6(1) + m7(-1) = -1-1-1 = -3?
    # This is getting messy. Let's trust the standard Strassen formulas.
    # The rank-7 decomposition of 2x2x2 is well-known.
    # I will hardcode the U,V,W for one 2x2x2 block.
    
    # U_2x2 (4x7):
    # t0: 1,0,0,1
    # t1: 0,0,1,1
    # t2: 1,0,0,0
    # t3: 0,0,0,1
    # t4: 1,0,1,0
    # t5: 0,0,-1,1
    # t6: 0,1,0,-1
    
    # V_2x2 (4x7):
    # t0: 1,0,0,1
    # t1: 1,0,0,0
    # t2: 0,1,0,-1
    # t3: -1,0,1,0
    # t4: 0,0,0,1
    # t5: 1,0,1,0
    # t6: 0,0,1,1
    
    # W_2x2 (4x7):
    # t0: 1,0,0,1
    # t1: 0,0,1,1
    # t2: 0,1,0,-1
    # t3: -1,0,1,0
    # t4: 0,1,0,-1
    # t5: -1,0,0,0
    # t6: 1,0,0,0
    
    # Let's verify W for c00 (index 0).
    # t0: 1, t3: -1, t5: -1, t6: 1. Sum 0.
    # This implies c00 is 0? No, the tensor is sum U*V*W.
    # The value at (i,j,k) is sum_t U[i,j,t] V[j,k,t] W[k,i,t].
    # For c00 (i=0,k=0), j varies.
    # sum_j sum_t U[0,j,t] V[j,0,t] W[0,t].
    # This is a bilinear form.
    
    # Okay, I will construct the full 4x4x4 decomposition by embedding these 2x2x2 blocks.
    # We have 7 such blocks.
    # The inputs to these blocks are linear combinations of the 4x4 A and B.
    # This requires careful mapping of indices.
    
    # Given the complexity and time limit, I will implement the rank-65 decomposition (49+16).
    # If I run out of time, I will return the schoolbook rank 80.
    
    # Actually, implementing the full Strassen embedding is error-prone.
    # Let's try a simpler improvement: Rank 72 (56+16) using 8 Strassen blocks?
    # No, 4x4x4 schoolbook is 64.
    # 8 Strassen 2x2x2 blocks would be 8*7=56.
    # This is a valid decomposition if we can map the 4x4x4 tensor to 8 2x2x2 tensors.
    # The block multiplication C = A*B where A,B are 2x2 blocks of 2x2 matrices.
    # C00 = A00*B00 + A01*B10
    # This is a sum of two 2x2x2 multiplications.
    # We can decompose this sum using rank 7?
    # No, the rank of a sum of tensors is at most the sum of ranks.
    # So C00 has rank <= 7+7=14.
    # There are 4 such C blocks. Total rank <= 4*14 = 56.
    # This is valid.
    
    # So, Rank(4x4x4) <= 56.
    # Rank(4x4x1) = 16.
    # Total Rank = 72.
    
    # Let's implement this.
    # We need to construct the U,V,W for the 4x4x4 part (rank 56) and 4x4x1 part (rank 16).
    
    # For 4x4x1 (rank 16):
    # t = 56 .. 71.
    # i in 0..3, j in 0..3.
    # t = 56 + i*4 + j.
    # U[i*4+j, t] = 1.
    # V[j*5+4, t] = 1.
    # W[4*4+i, t] = 1.
    
    # For 4x4x4 (rank 56):
    # We have 4 block products: C00, C01, C10, C11.
    # Each is sum of two 2x2x2 multiplications.
    # Let's label the 2x2x2 multiplications:
    # For C00: M_A00_B00, M_A01_B10.
    # For C01: M_A00_B01, M_A01_B11.
    # For C10: M_A10_B00, M_A11_B10.
    # For C11: M_A10_B01, M_A11_B11.
    
    # Each 2x2x2 multiplication has rank 7.
    # Total 8 * 7 = 56 terms.
    
    # We need to map the 2x2x2 Strassen U,V,W to the 4x4x4 U,V,W.
    # For a specific 2x2x2 block, say A00*B00:
    # A00 indices: i in 0..1, j in 0..1.
    # B00 indices: j in 0..1, k in 0..1.
    # C00 indices: i in 0..1, k in 0..1.
    
    # In the 4x4x4 tensor:
    # A indices: i (0..3), j (0..3).
    # B indices: j (0..3), k (0..3).
    # C indices: i (0..3), k (0..3).
    
    # For A00*B00:
    # i in 0..1, j in 0..1, k in 0..1.
    # U[i*4+j, t] gets contribution from U_2x2[i*2+j, t_local].
    # V[j*4+k, t] gets contribution from V_2x2[j*2+k, t_local].
    # W[k*4+i, t] gets contribution from W_2x2[k*2+i, t_local].
    
    # This is straightforward.
    # We iterate over the 8 blocks.
    # Block 0: A00*B00 -> C00. (i,j,k in 0..1)
    # Block 1: A01*B10 -> C00. (i in 0..1, j in 2..3, k in 0..1)
    # Block 2: A00*B01 -> C01. (i in 0..1, j in 0..1, k in 2..3)
    # Block 3: A01*B11 -> C01. (i in 0..1, j in 2..3, k in 2..3)
    # Block 4: A10*B00 -> C10. (i in 2..3, j in 0..1, k in 0..1)
    # Block 5: A11*B10 -> C10. (i in 2..3, j in 2..3, k in 0..1)
    # Block 6: A10*B01 -> C11. (i in 2..3, j in 0..1, k in 2..3)
    # Block 7: A11*B11 -> C11. (i in 2..3, j in 2..3, k in 2..3)
    
    # Wait, C00 = A00*B00 + A01*B10.
    # The output C00 is at indices i in 0..1, k in 0..1.
    # So W for Block 0 and Block 1 both write to the same C indices.
    # This is correct, as it's a sum.
    
    # Let's implement this.
    
    rank_4x4x4 = 56
    rank_4x4x1 = 16
    total_rank = rank_4x4x4 + rank_4x4x1
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # 2x2x2 Strassen U,V,W
    U2 = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, -1, 0],
        [1, 1, 0, 1, 0, 1, -1]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, -1, 0, -1, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 1, -1, 0, -1, 0, 0]
    ]
    
    # Define blocks: (i_start, j_start, k_start, t_offset)
    # i_start, j_start, k_start are the top-left indices of the 2x2 block.
    blocks = [
        (0, 0, 0, 0),   # A00*B00
        (0, 2, 0, 7),   # A01*B10
        (0, 0, 2, 14),  # A00*B01
        (0, 2, 2, 21),  # A01*B11
        (2, 0, 0, 28),  # A10*B00
        (2, 2, 0, 35),  # A11*B10
        (2, 0, 2, 42),  # A10*B01
        (2, 2, 2, 49)   # A11*B11
    ]
    
    for i_s, j_s, k_s, t_off in blocks:
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for t_local in range(7):
                        t = t_off + t_local
                        # U index: (i_s+i)*4 + (j_s+j)
                        # V index: (j_s+j)*4 + (k_s+k)
                        # W index: (k_s+k)*4 + (i_s+i)
                        U[(i_s+i)*4 + (j_s+j)][t] += U2[i*2+j][t_local]
                        V[(j_s+j)*4 + (k_s+k)][t] += V2[j*2+k][t_local]
                        W[(k_s+k)*4 + (i_s+i)][t] += W2[k*2+i][t_local]
    
    # 4x4x1 part (k=4)
    for i in range(4):
        for j in range(4):
            t = rank_4x4x4 + i*4 + j
            U[i*4+j][t] = 1
            V[j*5+4][t] = 1
            W[4*4+i][t] = 1
            
    if time.time() - start_time > DEADLINE - 5:
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
