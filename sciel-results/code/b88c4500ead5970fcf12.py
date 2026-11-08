import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7
    deadline = start_time + 100.0  # safety margin before 120s kill

    # Strategy: decompose 4x4x7 as direct sum of:
    # 1) 4x4x4 block using Strassen (rank 13)
    # 2) 4x4x3 block using schoolbook (rank 12)
    # Total rank = 13 + 12 = 25
    
    # But we can do better: use a smarter decomposition for the 4x4x3 part.
    # Actually, let's think about this more carefully.
    
    # The tensor for (4x4)*(4x7) has entries T[i,j,k] where:
    # i in [0,3] (row of A), j in [0,3] (col of A / row of B), k in [0,6] (col of B)
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So the structure is: for each j, we have a "fiber" connecting i-dimension and k-dimension
    
    # Direct sum approach:
    # Split k into two groups: k in {0,1,2,3} (size 4) and k in {4,5,6} (size 3)
    # For k in {0,1,2,3}: this is like 4x4x4 matrix mult, use Strassen rank 13
    # For k in {4,5,6}: this is like 4x4x3, use schoolbook rank 12
    # Total: 13 + 12 = 25
    
    # Let's build Strassen's 13-term decomposition for 4x4x4.
    # Strassen for 2x2x2 has rank 7. For 4x4x4, we apply Strassen recursively.
    # 4x4 = 2x(2x2), so we apply Strassen twice: rank = 7^2 = 49? No wait.
    # Actually, the standard Strassen decomposition for 4x4x4 using block structure:
    # Split 4x4 into 2x2 blocks, apply Strassen: gives 7 terms, each a 2x2 mult
    # Then each 2x2 mult can be done with Strassen (7 terms) or schoolbook (4 terms)
    # If we use Strassen at both levels: 7 * 7 = 49 (too high)
    # If we use Strassen at top, schoolbook at bottom: 7 * 4 = 28
    # But we know there's a rank-13 decomposition for 4x4x4.
    
    # Actually, the minimal rank of 4x4x4 is known to be 13.
    # Let me look up the explicit rank-13 decomposition.
    
    # The rank-13 decomposition of the 4x4x4 matrix multiplication tensor:
    # This is the "Strassen-Winograd" decomposition or similar.
    
    # Let me use a known explicit rank-13 decomposition.
    # One way: use the fact that M(4,4,4) can be embedded from smaller tensors.
    
    # Actually, let me just implement a correct decomposition, even if not minimal,
    # and aim for a reasonable rank. The schoolbook gives 4*4*7=112.
    # The hybrid gives 13+12=25. Let's verify the Strassen part works.
    
    # Strassen's 7-term formula for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5-M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M7
    
    # For 4x4x4, I'll use the block approach: split into 2x2 blocks of 2x2.
    # This gives 7 Strassen multiplications of 2x2, each with schoolbook 4 terms = 28 total.
    # Or use Strassen for inner: 7*7=49.
    # 28 is better. Let's use that.
    
    # Actually, for simplicity and correctness, let me just use a known working decomposition.
    # Let me implement the block Strassen for 4x4x4 with rank 28, and schoolbook for 4x4x3 with rank 12.
    # Total rank = 40. That's still much better than 112.
    
    # But wait - can I do even better? Let me think about a direct approach.
    
    # Alternative: treat the whole 4x4x7 tensor directly.
    # The tensor is T[i,j,k] = delta structure where for fixed j, we have identity in (i,k) space.
    # Actually no, T[i,j,k] = 1 for all i,j,k? No.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k], which is 1.
    # So T[i,j,k] = 1 for all i in [0,3], j in [0,3], k in [0,6].
    
    # Wait, that means the tensor is just the all-ones tensor? No, that's not right either.
    # The matrix multiplication tensor T_{i,j,k} has T_{i,j,k} = 1 if the term A_{i,j} * B_{j,k} 
    # contributes to C_{i,k}. In standard matrix mult, it always does, so T_{i,j,k}=1.
    # But that can't be right because then the tensor would have a very low rank.
    
    # Let me re-read the problem. The tensor is defined by:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T_{i,j,k}
    # For matrix mult: T_{i,j,k} = 1 (since A_{i,j}*B_{j,k} always contributes to C_{i,k})
    # 
    # Hmm, but that would make the problem trivial with very low rank.
    # Let me reconsider. Maybe T_{i,j,k} is not all ones.
    
    # Actually, I think I'm confusing myself. Let me re-read the interface:
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # 
    # For matrix multiplication C = A*B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T_{i,j,k} represents the bilinear structure. 
    # T_{i,j,k} = 1 means the product A[i,j]*B[j,k] appears in C[i,k].
    # In standard matrix mult, yes, T_{i,j,k} = 1 for all valid i,j,k.
    
    # So the tensor is indeed the all-ones tensor of shape (4,4,7)?
    # That would have rank 1 (just take U=V=W=1 vector)!
    # That can't be the intended problem.
    
    # I must be misunderstanding the tensor convention. Let me re-read.
    # "W[k*n+i,r]" - this suggests W is indexed by (k*n+i, r)
    # So W[k*n+i] corresponds to the output C[i,k] at position k*n+i.
    
    # Maybe the tensor is not all ones. Perhaps it's:
    # T_{i,j,k} = 1 if i == k (some diagonal structure)?
    # No, that doesn't make sense for matrix mult.
    
    # I think the standard matrix multiplication tensor is:
    # T_{i,j,k} = delta_{something}
    # Actually, I believe the tensor for matrix multiplication is:
    # T_{i,j,k} = 1 for all i,j,k (it's the "identity" tensor in the middle index)
    
    # Let me just go with the schoolbook to be safe, since I'm confused about the exact tensor structure.
    # Schoolbook: rank = n*m*p = 4*4*7 = 112
    
    # Actually wait. Let me reconsider. If T_{i,j,k}=1 for all i,j,k, then:
    # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
    # This is a very different constraint than matrix multiplication.
    
    # I think the issue is that the tensor IS the matrix multiplication tensor, which means:
    # For the bilinear map (A,B) -> C where C[i,k] = sum_j A[i,j]*B[j,k],
    # the tensor T has T[i,j,k] = 1.
    
    # But if that's the case, the rank is 1! Just use constant functions.
    # Unless... the evaluator is checking something else.
    
    # Let me just implement the schoolbook decomposition which is definitely correct
    # for the standard interpretation, and see what happens.
    
    rank = n * m * p
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
