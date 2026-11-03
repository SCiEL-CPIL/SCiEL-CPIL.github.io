import numpy as np
import time

DIMENSIONS = [3, 3, 7]
_START_TIME = time.time()
_DEADLINE = 120.0

def _check_time():
    global _START_TIME
    return time.time() - _START_TIME < _DEADLINE - 5.0

def construct():
    global _START_TIME
    _START_TIME = time.time()
    n, m, p = DIMENSIONS
    
    # The trivial rank is n*m*p = 63
    # We want to find a rank-minimal exact tensor decomposition.
    # The tensor for matrix multiplication of n x m by m x p matrices is a tensor of order 3
    # with dimensions (n*m, m*p, p*n).
    
    # The trivial decomposition has rank n*m*p where each rank-1 term corresponds to a single
    # entry of the product.
    
    # For 3x3x7, the trivial rank is 63.
    # We can potentially do better by exploiting structure.
    
    # One approach: use a Strassen-like algorithm for the 3x3x3 part and standard for the 3x3x4 part.
    # C = A * B where A is 3x3 and B is 3x7.
    # We can write B = [B1 | B2] where B1 is 3x3 and B2 is 3x4.
    # C = A*B1 + A*B2.
    
    # For A*B1 (3x3x3), the rank is at most 7 (using a rank-7 algorithm for 3x3 matrix multiplication).
    # For A*B2 (3x3x4), the rank is at most 12 (trivial: 3*3*4 = 36? No, 3*3*4 = 36... wait)
    # Actually, for multiplying a 3x3 by a 3x4, the trivial rank is 3*3*4 = 36.
    # But we can do better by viewing it as 4 separate 3x3x1 multiplications, each of rank 9, total 36.
    # Or as one 3x3x4 multiplication.
    
    # Actually, let's think about this differently.
    # The tensor for C[i,k] = sum_j A[i,j] * B[j,k]
    # has a natural decomposition of rank n*m*p where each term is:
    # U[i*m+j, term] = A[i,j], V[j*p+k, term] = B[j,k], W[k*n+i, term] = 1
    # But this is the trivial decomposition with rank = number of (i,j,k) triples = n*m*p.
    
    # For 3x3x7, that's 63.
    
    # Can we do better? Let's try to use a known low-rank algorithm.
    # For 3x3x3, the best known is rank 7 (but this is for square matrices).
    # For 3x3x7, we can potentially use a combination.
    
    # Let's try a different approach: use linear algebra to find a minimal rank decomposition.
    # The matrix multiplication tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1.
    # We want to decompose T as sum_{r=1}^{rank} u_r * v_r * w_r.
    
    # This is equivalent to finding the tensor rank of T.
    
    # For small dimensions, we can try to find a low-rank decomposition by solving a system of equations.
    # But this is computationally expensive for rank > 20 or so.
    
    # Let's try a pragmatic approach:
    # 1. Start with the trivial rank-63 decomposition.
    # 2. Try to reduce the rank by finding linear dependencies.
    
    # Actually, let's try to construct a rank-28 decomposition or similar.
    # For 3x3x7, we can write:
    # C = A * B
    # where A is 3x3 and B is 3x7.
    
    # We can split B into two parts: B = [B1, B2] where B1 is 3x3 and B2 is 3x4.
    # C = A*B1 + A*B2.
    
    # For A*B1 (3x3x3), use a rank-7 algorithm (if available) or rank-9 trivial.
    # For A*B2 (3x3x4), use rank-12 (3*4, treating each column separately) or rank-36 trivial.
    
    # Actually, for 3x3x4, we can do:
    # C2 = A * B2
    # C2[i,k] = sum_j A[i,j] * B2[j,k]
    # This is a 3x4 matrix, so we need 12 entries.
    # The trivial decomposition has rank 3*3*4 = 36.
    # But we can do better by using the structure.
    
    # Let's try to construct a decomposition with rank 7 + 12 = 19.
    # But we need to be careful: the 7-rank decomposition for 3x3x3 may not exist.
    # The best known for 3x3x3 is actually rank 19 (or is it 7?).
    
    # Wait, for 2x2x2, Strassen gives rank 7.
    # For 3x3x3, the best known upper bound is 19 (Landsberg, 1994) or maybe lower.
    # Actually, I think the best known for 3x3x3 is rank 19.
    
    # So using rank 19 for 3x3x3 and rank 12 for 3x3x4 gives total rank 31.
    # That's better than 63.
    
    # But I don't have a concrete rank-19 decomposition for 3x3x3.
    
    # Let's try a different approach: use a greedy algorithm to find a low-rank decomposition.
    
    # Actually, let's just return the trivial decomposition for now, since I can't quickly
    # construct a better one.
    
    # Wait, let me try to think about this more carefully.
    
    # The tensor T for 3x3x7 matrix multiplication has:
    # - First index: i*m+j where i in [0,3), j in [0,3) -> 9 values
    # - Second index: j*p+k where j in [0,3), k in [0,7) -> 21 values
    # - Third index: k*n+i where k in [0,7), i in [0,3) -> 21 values
    
    # T[a,b,c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' and j = j', i = i', k = k'
    # Otherwise T[a,b,c] = 0.
    
    # So T is a 9x21x21 tensor.
    
    # The trivial decomposition has 63 rank-1 terms.
    
    # Can we find a decomposition with fewer terms?
    
    # One approach: use the fact that the tensor has a lot of zeros.
    # We can try to find a set of rank-1 tensors that sum to T.
    
    # Let's try a simple approach: use the fact that for each fixed j, the terms
    # T[i*m+j, j*p+k, k*n+i] form a 3x7x3x7 tensor (if we fix j).
    
    # Actually, let's try to use a Kruskal-Candecards type approach.
    
    # For now, let's just return the trivial decomposition.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
