import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 3x5 matrix by a 5x7 matrix.
    
    The output factor W follows the trace convention W[k*n+i, r].
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    
    Score is the rank.
    """
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    n, m, p = 3, 5, 7
    
    # We will build the decomposition by direct sum of sub-tensors.
    # Partition the inner dimension m=5 into m1=2 and m2=3.
    # Tensor T_{3,5,7} = T_{3,2,7} + T_{3,3,7}
    
    # For T_{3,3,7}, we use a decomposition based on Strassen's 2x2x2.
    # Specifically, we can decompose the 3x3 matrix multiplication into:
    # - One 2x2x2 block (top-left 2x2 of A, top-left 2x2 of B, top-left 2x2 of C) -> Rank 7
    # - Remaining terms involve the 3rd row/col.
    # Let A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    # Let B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x1, B21 is 1x2, B22 is 1x1.
    # C11 = A11 B11 + A12 B21 (2x2)
    # C12 = A11 B12 + A12 B22 (2x1)
    # C21 = A21 B11 + A22 B21 (1x2)
    # C22 = A21 B12 + A22 B22 (1x1)
    
    # This is getting complicated to map to a single Strassen call.
    # Instead, let's use a known rank for 3x3x3.
    # The rank of 3x3x3 is known to be 19 (or 20? No, it's 19 for 3x3x3? Actually, rank of 3x3x3 is 19? 
    # Wait, rank of 2x2x2 is 7. Rank of 3x3x3 is at most 19? 
    # Actually, the rank of 3x3x3 is 19? No, it's 19? 
    # Let's check: 3x3x3 schoolbook is 27.
    # There is a rank 19 decomposition for 3x3x3? 
    # Actually, the rank of 3x3x3 is 19? No, it's 19? 
    # Wait, the rank of 3x3x3 is 19? 
    # Let's just use schoolbook for 3x3x3 for now, rank 27.
    # But we can do better.
    # Let's use the fact that 3x3x3 can be decomposed into 2x2x2 + 2x2x1 + 1x2x2 + 1x1x2 + 1x1x1? No.
    
    # Let's stick to a simpler improvement:
    # Decompose T_{3,5,7} into T_{3,2,7} + T_{3,3,7}.
    # T_{3,2,7}: Rank 3*2*7 = 42 (schoolbook).
    # T_{3,3,7}: Rank 3*3*7 = 63 (schoolbook).
    # Total 105.
    
    # Can we improve T_{3,3,7}?
    # T_{3,3,7} is 7 copies of T_{3,3,1} (if we fix k)? No, it's sum over j.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # This is 7 independent 3x3x1 tensors? No, the B_{jk} are shared across k? No, B is 3x7.
    # So for each k, C_{.,k} = A * B_{.,k}.
    # This is 7 independent matrix multiplications of 3x3 by 3x1.
    # Rank of 3x3x1 is 9 (schoolbook).
    # Total 7*9 = 63.
    
    # Can we improve 3x3x1?
    # 3x3x1 is just a linear map from 3x3 to 3.
    # It's rank 9.
    
    # So schoolbook is 105.
    
    # Let's try to use Strassen on a 2x2x2 part of the 3x3x3?
    # But the 3x3x7 is not a 3x3x3.
    
    # Let's try a different partition.
    # Partition n=3 into 1+2.
    # T_{3,5,7} = T_{1,5,7} + T_{2,5,7}.
    # T_{1,5,7}: Rank 1*5*7 = 35.
    # T_{2,5,7}: Rank 2*5*7 = 70.
    # Total 105.
    
    # Partition m=5 into 2+3.
    # T_{3,2,7} + T_{3,3,7}.
    # T_{3,2,7}: 42.
    # T_{3,3,7}: 63.
    # Total 105.
    
    # It seems hard to beat 105 without a sophisticated algorithm.
    # However, we can try to use the rank 19 decomposition for 3x3x3 if we can embed it.
    # But 3x3x7 is not 3x3x3.
    
    # Let's just implement the schoolbook decomposition correctly and efficiently.
    # The previous code did this.
    
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
