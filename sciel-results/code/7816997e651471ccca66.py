import numpy as np
import time
import random

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 40
    
    # Build trivial (schoolbook) decomposition as fallback
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Strassen-based decomposition for 2x4x5
    # Split A (2x4) into blocks: A = [[A11, A12], [A21, A22]] where each is 2x2
    # Split B (4x5) into blocks: B = [[B11, B12], [B21, B22]] where B11,B21 are 2x2 and B12,B22 are 2x3
    # C = A*B = [[C11, C12], [C21, C22]] where C11,C21 are 2x2 and C12,C22 are 2x3
    
    # For each 2x2x2 block multiplication, use Strassen (rank 7 instead of 8)
    # We have 4 such 2x2x2 multiplications (for C11, C12, C21, C22)
    # But C12 and C22 involve 2x2 * 2x3, which we can decompose differently
    
    # Strategy: Use Strassen for the 2x2x2 parts
    # C11 = A11*B11 + A12*B21 (two 2x2x2 multiplications)
    # C12 = A11*B12 + A12*B22 (two 2x2x2x3 multiplications)  
    # C21 = A21*B11 + A22*B21 (two 2x2x2 multiplications)
    # C22 = A21*B12 + A22*B22 (two 2x2x2x3 multiplications)
    
    # For 2x2x2: Strassen uses 7 multiplications
    # For 2x2x3: We can treat as 2x2x2 + 2x2x1, or use a specialized approach
    
    # Let's build a rank-36 decomposition using Strassen for the 2x2x2 blocks
    # and handle the 2x2x3 blocks efficiently
    
    # Actually, let's think more carefully:
    # Total rank = 4 * 7 (for the four 2x2x2 Strassen decompositions) = 28
    # But we need to handle the 2x2x3 blocks too
    
    # Better approach: Use the fact that 2x2x3 can be done with rank 5 (known result)
    # So total rank = 2*7 + 2*5 = 24? Let me verify this is achievable.
    
    # For now, let's implement a clean Strassen-based approach with rank 36
    # by using Strassen for each 2x2x2 sub-problem and schoolbook for the rest
    
    # Define Strassen formulas for 2x2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For our 2x4 * 4x5 case:
    # A is 2x4: A = [[A11, A12], [A21, A22]] each 2x2
    # B is 4x5: B = [[B11, B12], [B21, B22]] where B11,B21 are 2x2, B12,B22 are 2x3
    
    # C11 = A11*B11 + A12*B21 (both 2x2x2) -> use Strassen twice = 14 rank
    # C12 = A11*B12 + A12*B22 (2x2x3 each) -> need efficient decomposition
    # C21 = A21*B11 + A22*B21 (both 2x2x2) -> use Strassen twice = 14 rank
    # C22 = A21*B12 + A22*B22 (2x2x3 each) -> need efficient decomposition
    
    # For 2x2x3: C = A*B where A is 2x2, B is 2x3
    # We can write B = [B_col1, B_col2, B_col3] where each is 2x1
    # C = A * B = [A*B_col1, A*B_col2, A*B_col3]
    # Each A*B_col is 2x1, which is just a vector multiplication (rank 2 each)
    # So 2x2x3 has rank at most 6 (2*3)
    # But we can do better: treat as 2x2x2 + 2x2x1
    # 2x2x2 has rank 7 (Strassen), 2x2x1 has rank 2 (just dot products)
    # So 2x2x3 has rank at most 9? That's not helpful.
    
    # Actually, for 2x2x3, the minimum rank is 5 (known result)
    # Let me use a known rank-5 decomposition for 2x2x3
    
    # For now, let's use a simpler approach: 
    # Use Strassen for the 2x2x2 parts and schoolbook for the 2x2x3 parts
    # This gives rank = 2*7 + 2*6 = 26
    
    # But we can do better. Let's try rank 36 first with a clean implementation.
    
    # Alternative: Use the full tensor structure
    # The matmul tensor for 2x4x5 has rank at most 36 (from known results)
    
    # Let me construct a rank-36 decomposition using Strassen-like ideas
    
    # Index convention:
    # U[i*m+j, r] for i in [0,2), j in [0,4)
    # V[j*p+k, r] for j in [0,4), k in [0,5)
    # W[k*n+i, r] for k in [0,5), i in [0,2)
    
    # Split indices:
    # i: 0 or 1
    # j: 0,1,2,3 -> j_group = j//2 (0 or 1), j_local = j%2
    # k: 0,1,2,3,4 -> k_group = k//2 (0,1,2), k_local = k%2
    
    # A[i,j]: i in {0,1}, j in {0,1,2,3}
    #   A[i, j_group*2 + j_local]
    # B[j,k]: j in {0,1,2,3}, k in {0,1,2,3,4}
    #   B[j_group*2 + j_local, k_group*2 + k_local]
    # C[i,k]: i in {0,1}, k in {0,1,2,3,4}
    #   C[i, k_group*2 + k_local]
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #        = sum_{j_group, j_local} A[i, j_group*2+j_local] * B[j_group*2+j_local, k]
    
    # For fixed i, k_group, k_local:
    # C[i, k_group*2+k_local] = sum_{j_group} sum_{j_local} A[i, j_group*2+j_local] * B[j_group*2+j_local, k_group*2+k_local]
    
    # Let's group by (i, j_group, k_group, k_local):
    # For each combination, we have a 2x2 matrix multiplication (over j_local)
    # A_sub = A[i, j_group*2 : j_group*2+2]  (1x2 vector)
    # B_sub = B[j_group*2 : j_group*2+2, k_group*2+k_local]  (2x1 vector)
    # C_val = dot(A_sub, B_sub)
    
    # This is getting complex. Let me just implement a verified rank-36 decomposition.
    
    # Based on known results, the rank of 2x4x5 matmul is 36.
    # Let me construct it using Strassen on 2x2x2 blocks.
    
    # For each (i, k_group, k_local) where i in {0,1}, k_group in {0,1,2}, k_local in {0,1}:
    # We need to compute C[i, k_group*2+k_local] = sum_{j} A[i,j] * B[j, k_group*2+k_local]
    
    # Split j into j_group in {0,1} and j_local in {0,1}:
    # C[i, k_group*2+k_local] = sum_{j_group in {0,1}} sum_{j_local in {0,1}} A[i, j_group*2+j_local] * B[j_group*2+j_local, k_group*2+k_local]
    
    # For fixed (i, j_group, k_group, k_local), we have:
    # sum_{j_local} A[i, j_group*2+j_local] * B[j_group*2+j_local, k_group*2+k_local]
    # This is a dot product of two 2-vectors.
    
    # Number of such terms: 2 (i) * 2 (j_group) * 3 (k_group) * 2 (k_local) = 24
    # Each dot product has rank 2 (for the 2-dimensional j_local sum)
    # Total rank = 24 * 2 = 48? That's worse than schoolbook.
    
    # I'm overcomplicating this. Let me just use a known rank-36 construction.
    
    # Actually, the simplest approach: use Strassen for the 2x2x2 subproblems.
    # There are 4 "main" 2x2x2 multiplications if we think of it as:
    # C = [[A11, A12], [A21, A22]] * [[B11, B12], [B21, B22]]
    # where A11,A12,A21,A22 are 2x2, B11,B21 are 2x2, B12,B22 are 2x3
    
    # C11 = A11*B11 + A12*B21: two 2x2x2 mults, each rank 7 -> 14
    # C21 = A21*B11 + A22*B21: two 2x2x2 mults, each rank 7 -> 14
    # C12 = A11*B12 + A12*B22: two 2x2x3 mults
    # C22 = A21*B12 + A22*B22: two 2x2x3 mults
    
    # For 2x2x3, minimum rank is 5. So C12 and C22 each need 2*5 = 10 rank
    # Total = 14 + 14 + 10 + 10 = 48? Still not 36.
    
    # I think the 36 rank comes from a more clever decomposition that shares some terms.
    
    # For now, let's just return the schoolbook rank 40 decomposition, which is correct.
    # The random search in the original code was ineffective.
    
    return best_result
# EVOLVE-BLOCK-END
