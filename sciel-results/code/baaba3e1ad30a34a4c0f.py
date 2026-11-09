"""Optimized tensor decomposition for 2x3 * 3x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (27, rank)
    # W shape: (p*n, rank) = (18, rank)
    # Indexing: U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    # Tensor entry: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want T[i,j,k] = delta[i,j,k] (the standard matrix multiplication tensor)
    # i.e., T[i,j,k] = 1 if i,j,k are consistent with matrix multiplication, else 0
    # Actually, the matrix multiplication tensor T_{ijk} has T_{ijk} = 1 for all valid (i,j,k)
    # Wait, let me reconsider. The tensor for matrix multiplication C = A * B where
    # A is n x m, B is m x p, C is n x p is:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # So the tensor is T_{ijk} with C_{ik} = sum_j A_{ij} B_{jk}
    # In the factorization: C_{ik} = sum_j sum_c U_{ij,c} V_{jk,c} W_{ki,c}
    # where W is indexed as W[k*n + i, c] per the trace convention W[k*n+i, r]
    # Wait, the problem says: W[k*n+i, r] - so W index is k*n + i, where k is output col, i is output row
    
    # Let me re-read: "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # The tensor is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, we need: sum_j A[i,j] * B[j,k] = C[i,k]
    # So the "tensor" T[i,j,k] should be such that summing over j gives C[i,k]
    # Actually, the standard formulation: the matrix multiplication tensor M_{ijk} where
    # M_{ijk} = 1 if the indices are "compatible" in the multiplication sense.
    # More precisely, we want: for any matrices A (n x m) and B (m x p),
    # (A * B)[i,k] = sum_j A[i,j] B[j,k]
    # This is equivalent to: (A * B)[i,k] = sum_{j,c} A[i,j] B[j,k] * delta_{i,k} ... no.
    
    # Let me think again. The factorization is:
    # C[i,k] = sum_j sum_c U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k] * W[k*n+i, c]
    # Wait, that's not right either. Let me re-read the contract.
    
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries that exactly decompose the matrix multiplication tensor"
    
    # The standard way: We have three factors U, V, W such that for all i, j, k:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where T[i,j,k] is the matrix multiplication tensor, which is 1 for all i,j,k (since every combination of indices participates in the sum).
    
    # Actually no. The matrix multiplication tensor is defined such that:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # In terms of a trilinear form: (A*B)[i,k] = sum_{j,c} A[i,j] * B[j,k] * M[i,j,k]
    # where M[i,j,k] = 1 for all i, j, k? No, that would give sum_j A[i,j] B[j,k] which is correct.
    # So M[i,j,k] = 1 for all i, j, k.
    
    # Wait, but then the factorization would be:
    # sum_j A[i,j] B[j,k] = sum_{j,c} A[i,j] B[j,k] * (sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c])
    # So we need sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i, j, k.
    
    # Hmm, that doesn't seem right for a tensor decomposition. Let me look at the schoolbook example:
    # In the schoolbook: for each (i,j,k), we have one term where U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # So T[i,j,k] = 1 for each (i,j,k) pair used.
    # And indeed, sum_j A[i,j] B[j,k] = sum_{j} A[i,j] B[j,k] * 1
    # So the tensor M[i,j,k] = 1 for all valid i, j, k.
    
    # OK so the goal is: find rank-r U, V, W such that for all i in [0,n), j in [0,m), k in [0,p):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a rank decomposition of the all-ones tensor of size n x m x p.
    # The schoolbook solution has rank n*m*p with each term being a rank-1 tensor e_i ⊗ e_j ⊗ e_k.
    
    # To reduce rank, we need to find a lower-rank decomposition of the all-ones tensor.
    # But wait - is the matrix multiplication tensor really the all-ones tensor?
    
    # Let me reconsider. In standard tensor decomposition for matrix multiplication:
    # The matrix multiplication tensor M is a 3-tensor of size (n*p) x (n*m) x (m*p) where
    # M[(i,k),(i,j),(j,k)] = 1 and 0 otherwise.
    
    # In the factorization format given here, we have:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The product is a 3-tensor of size (n*m) x (m*p) x (p*n)
    # Entry (a, b, c) = sum_r U[a,r] * V[b,r] * W[c,r]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    # For this to be the matrix multiplication tensor, we need:
    # Entry (i*m+j, j*p+k, k*n+i) = 1 for all i,j,k
    # and 0 for all other combinations.
    
    # But since a = i*m+j uniquely determines i and j, b = j*p+k uniquely determines j and k,
    # and c = k*n+i uniquely determines k and i, the only non-zero entries are those where
    # the indices are consistent: j from a matches j from b, k from b matches k from c,
    # i from c matches i from a.
    
    # So effectively, we need: for all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # and for inconsistent index triples (which don't exist as valid (a,b,c) combinations since
    # a, b, c each encode unique index pairs), the value is irrelevant.
    
    # So the problem reduces to: find the minimum rank r such that there exist U, V, W with
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summing to 1 for all i,j,k.
    
    # This is equivalent to finding a rank-r decomposition of the "multiplication tensor"
    # where the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # Now, for the 2x3x9 case:
    # n=2, m=3, p=9
    # We can exploit the structure. Note that p=9 = 3*3.
    # We can think of B as a 3x(3*3) matrix, or equivalently, as three 3x3 blocks.
    
    # Strategy: Decompose the 3x9 multiplication into three independent 3x3 multiplications.
    # For each block b in {0,1,2}, we multiply A (2x3) by B_b (3x3) to get C_b (2x3).
    # Each 2x3x3 multiplication has a known rank.
    
    # For 2x3x3: schoolbook rank is 2*3*3 = 18.
    # But we can do better. Using Strassen-like methods:
    # Split A into A1 (1x3) and A2 (1x3).
    # A1 * B = a1*b1 + a2*b2 + ... no, A1 is 1x3, B is 3x3, so A1*B is 1x3.
    # (A1*B)[k] = sum_j A1[j] * B[j,k]
    # This is a 1x3x3 multiplication. Rank of 1x3x3: schoolbook is 9, but can we do better?
    # For 1x3x3: C[k] = sum_j A[j] B[j,k]. This is a 3x3 matrix-vector product on the left.
    # The tensor is T[j,k,l]... wait, indices are j (from A), k (from B col), and output is just one vector of length 3.
    # Actually for 1x3x3: A is 1x3, B is 3x3, C is 1x3.
    # C[k] = sum_j A[j] B[j,k]
    # Tensor: T[j,k] with output C[k] = sum_j A[j] B[j,k] T[j,k]
    # We need T[j,k] such that sum_j A[j] B[j,k] T[j,k] = sum_j A[j] B[j,k]
    # So T[j,k] = 1 for all j,k. This is a 3x3 tensor, rank 9 in general.
    
    # Alternatively, think of it as: the multiplication is (1x3) * (3x3) = (1x3)
    # This is the same as a 3x3 matrix acting on a row vector.
    # The tensor has rank equal to the "multiplication rank" of 1x3x3.
    # I believe this is 9 (schoolbook is optimal for 1xn x nxn when n is prime? Not sure).
    
    # Let me try a different approach. Use the known rank for 2x3x3.
    # Actually, let me just use a block decomposition with Strassen for the 3x3 parts.
    
    # Better idea: Decompose 2x3 * 3x9 as:
    # Reshape B (3x9) into three 3x3 blocks: B0, B1, B2
    # C = A * B = A * [B0, B1, B2] = [A*B0, A*B1, A*B2]
    # Each A*Bb is a 2x3 * 3x3 = 2x3 multiplication.
    # So total rank = 3 * rank(2x3x3)
    
    # What is rank(2x3x3)?
    # Schoolbook: 2*3*3 = 18
    # Can we do better? 
    # Using Strassen for the 2x2 part:
    # A = [A_top; A_bot] where A_top is 1x3, A_bot is 1x3
    # A * B = [A_top * B; A_bot * B]
    # A_top * B is 1x3 * 3x3 = 1x3, which has tensor rank 9 (I think).
    # So total would be 9 + 9 = 18, same as schoolbook.
    
    # Alternative: Use a more sophisticated decomposition.
    # Actually, for 2x3x3, I recall the rank might be 13 or so using Strassen-like methods.
    # Let me just implement the schoolbook for now (rank 18 per block, total 54) and see if I can improve.
    
    # Wait, the current program has rank 54. If I can get to 39 or lower, that's an improvement.
    
    # Let me try: for each 2x3x3 block, use a decomposition of rank 13.
    # 13 * 3 = 39 total rank.
    
    # How to get rank 13 for 2x3x3?
    # Split the 3 columns of B into two parts: first 2 columns and last 1 column.
    # A (2x3) * B (3x3):
    # Let B = [B_left | B_right] where B_left is 3x2, B_right is 3x1
    # C = [A * B_left | A * B_right]
    # A * B_left: 2x3 * 3x2 = 2x2. Rank of 2x2x2 is 7 (Strassen).
    # A * B_right: 2x3 * 3x1 = 2x1. Rank of 2x3x1: 
    #   C[i] = sum_j A[i,j] B[j] for i=0,1
    #   This is two independent 3x1 vector multiplications.
    #   Tensor: T[i,j] with C[i] = sum_j A[i,j] B[j] T[i,j]
    #   We need T[i,j] = 1 for all i,j. This is a 2x3 tensor, rank 6 (schoolbook).
    #   Can we do better? For 2x3x1, the rank is 6 (I believe this is optimal since 
    #   it's equivalent to two 3-dimensional vectors being multiplied).
    # Total: 7 + 6 = 13. Yes!
    
    # So for each 2x3x3 block, rank 13. Total rank = 3 * 13 = 39.
    
    # Let me implement this.
    
    # First, let me define the Strassen decomposition for 2x2x2.
    # A (2x2) * B (2x2) = C (2x2)
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # For the tensor decomposition, we need to express each C entry as a linear combination of the m's.
    # The U, V, W factors for Strassen 2x2x2:
    # For each of the 7 multiplications, we have a rank-1 term.
    
    # Let me define the Strassen factors for 2x2x2:
    # U (4x7): rows indexed by (i,j) for A, V (4x7): rows indexed by (j,k) for B, W (4x7): rows indexed by (k,i) for C
    # Using 0-based indexing: A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    # B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    # C[0,0]=C11, C[0,1]=C12, C[1,0]=C21, C[1,1]=C22
    
    # m1 = (a+d)(e+h): 
    #   U: A[0,0]=1, A[1,1]=1; V: B[0,0]=1, B[1,1]=1; W: C[0,0]+=1, C[1,1]+=1... 
    #   Wait, I need to be careful with the sign and accumulation.
    
    # Let me just build the full 2x3x3 decomposition directly.
    
    # For a 2x3 * 3x3 multiplication:
    # A (2x3), B (3x3), C (2x3)
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # Split B into B_left (3x2) and B_right (3x1)
    # C_left = A * B_left (2x2), C_right = A * B_right (2x1)
    
    # For C_left (2x2 = 2x3 * 3x2):
    # A is 2x3, B_left is 3x2
    # C_left[i,k] = sum_j A[i,j] B_left[j,k] for k=0,1
    # This is a 2x3x2 multiplication.
    # Split A into A_col0 (2x1) and A_col1 (2x1) and A_col2 (2x1)? 
    # No, better: think of it as (2x3) * (3x2) = 2x2
    # We can use Strassen by padding or by direct decomposition.
    
    # Actually, let me use a different split. For 2x3x3:
    # Use the following decomposition:
    # m1 = (A[0,0] + A[1,0]) * (B[0,0] + B[1,0])  # for C[0,0] and C[1,0]
    # This is getting complicated. Let me just implement the rank-13 decomposition explicitly.
    
    # Rank 13 for 2x3x3:
    # 7 terms for the 2x2 part (first two columns of B)
    # 6 terms for the 2x1 part (last column of B)
    
    # For the 2x2 part: A (2x3) * B_left (3x2) = C_left (2x2)
    # C_left[i,k] = sum_{j=0}^{2} A[i,j] B_left[j,k]
    # Let me define:
    # a0 = A[0,0], a1 = A[0,1], a2 = A[0,2]
    # b0 = A[1,0], b1 = A[1,1], b2 = A[1,2]
    # e0 = B_left[0,0], f0 = B_left[0,1]
    # g0 = B_left[1,0], h0 = B_left[1,1]
    # i0 = B_left[2,0], j0 = B_left[2,1]
    
    # C_left[0,0] = a0*e0 + a1*g0 + a2*i0
    # C_left[0,1] = a0*f0 + a1*h0 + a2*j0
    # C_left[1,0] = b0*e0 + b1*g0 + b2*i0
    # C_left[1,1] = b0*f0 + b1*h0 + b2*j0
    
    # Using Strassen-like:
    # Let me use a 2x2 Strassen on a reduced 2x2 problem.
    # Define:
    # A' = [[a0+a2, a1], [b0+b2, b1]]  # 2x2, combining col 0 and 2
    # B' = [[e0, f0], [g0, h0]]  # 2x2 (first two rows of B_left)
    # Then A' * B' gives:
    # [ (a0+a2)*e0 + a1*g0, (a0+a2)*f0 + a1*h0 ]
    # [ (b0+b2)*e0 + b1*g0, (b0+b2)*f0 + b1*h0 ]
    # We need to add back a2*i0, b2*i0 for the first column, and a2*j0, b2*j0 for the second.
    
    # This is getting messy. Let me just use a known rank-13 decomposition.
    
    # Actually, let me try a simpler approach: just use schoolbook for the whole thing but with some optimizations.
    # Schoolbook rank for 2x3x9 is 54.
    # If I can get to 39, that's a big improvement.
    
    # Let me just implement the 3 blocks of 2x3x3 with rank 13 each.
    # For each block, I'll use:
    # - 7 Strassen terms for the 2x2 core (using first 2 columns of the 3x3 block)
    # - 6 schoolbook terms for the 3rd column
    
    # Strassen for 2x2 (first 2 columns of B block, all 3 rows of A... wait, A is 2x3, B block is 3x3)
    # C_block[i,k] = sum_{j=0}^{2} A[i,j] B_block[j,k]
    # For k=0,1 (first two columns):
    # C_block[i,0] = A[i,0]*B_block[0,0] + A[i,1]*B_block[1,0] + A[i,2]*B_block[2,0]
    # C_block[i,1] = A[i,0]*B_block[0,1] + A[i,1]*B_block[1,1] + A[i,2]*B_block[2,1]
    
    # Let me use the following 7-term decomposition for the 2x3x2 part:
    # m1 = (A[0,0]+A[1,0])*(B[0,0]+B[1,0])
    # m2 = (A[1,0]+A[1,1])*B[0,0]
    # m3 = A[0,0]*(B[0,1]-B[1,1])
    # m4 = A[1,1]*(B[1,0]-B[0,0])
    # m5 = (A[0,0]+A[0,1])*B[1,1]
    # m6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # m7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    # Plus terms for the 3rd row of A:
    # m8 = A[0,2]*B[2,0]
    # m9 = A[0,2]*B[2,1]
    # m10 = A[1,2]*B[2,0]
    # m11 = A[1,2]*B[2,1]
    # That's 11 terms for the 2x3x2 part, plus 6 for the 3rd column = 17. Too many.
    
    # I need a cleaner approach. Let me just use the schoolbook for each 2x3x3 block (rank 18)
    # and see if the total is 54. Then try to optimize.
    
    # Actually, let me just implement the direct schoolbook for 2x3x9 (rank 54) as a baseline,
    # and then try to improve within the time limit.
    
    # For now, let me implement a rank-39 solution using the 7+6 decomposition per block.
    
    # Here's my plan for each 2x3x3 block (rank 13):
    # Indices: i in {0,1} (rows of A and C), j in {0,1,2} (cols of A, rows of B), k in {0,1,2} (cols of B and C)
    # 
    # Terms 1-7: Strassen for the 2x2 part using A[:,0:2] and B[0:2,:]
    #   A_sub = A[:, 0:2] (2x2), B_sub = B[0:2, :] (2x3)... no, B_sub should be 2x2 for Strassen.
    #   
    # Let me use: A_sub = A[:, 0:2] (2x2), B_sub = B[0:2, 0:2] (2x2)
    # C_sub = A_sub * B_sub (2x2)
    # C_sub[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C_sub[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C_sub[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C_sub[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # 
    # Strassen for C_sub:
    # m1 = (A[0,0]+A[1,0])*(B[0,0]+B[0,1])  # Wait, I need to be consistent.
    # Standard Strassen for [[a,b],[c,d]] * [[e,f],[g,h]]:
    # m1 = (a+d)(e+h), m2 = (c+d)e, m3 = a(f-h), m4 = d(g-e), m5 = (a+b)h, m6 = (c-a)(e+f), m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7, C12 = m3 + m5, C21 = m2 + m4, C22 = m1 - m2 + m3 + m6
    # 
    # In our case: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1], e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # 
    # After computing C_sub, we need to add the contributions from the 3rd row of B and 3rd col of A:
    # C[0,0] += A[0,2]*B[2,0]
    # C[0,1] += A[0,2]*B[2,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,0] += A[1,2]*B[2,0]
    # C[1,1] += A[1,2]*B[2,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    
    # So additional terms:
    # m8 = A[0,2]*B[2,0] -> contributes to C[0,0]
    # m9 = A[0,2]*B[2,1] -> contributes to C[0,1]
    # m10 = A[1,2]*B[2,0] -> contributes to C[1,0]
    # m11 = A[1,2]*B[2,1] -> contributes to C[1,1]
    # m12 = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2] -> C[0,2]
    # m13 = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2] -> C[1,2]
    # 
    # But m12 and m13 are sums, not single products. Each needs to be decomposed.
    # m12 = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2] -> 3 terms
    # m13 = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2] -> 3 terms
    # 
    # Total: 7 + 4 + 3 + 3 = 17. Still too many.
    
    # I think I need to accept a higher rank or find a better decomposition.
    # For now, let me just implement the schoolbook (rank 54) and ensure it works.
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
