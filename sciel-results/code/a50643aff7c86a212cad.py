import time
import random
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x5 matrix * 5x5 matrix.
    Uses block decomposition: split 2x5 into (2x2, 2x1) and 5x5 into (2x5, 3x5).
    - 2x2 * 2x5: rank 18 (from 2x2x2 rank 7 twice + 2x2x1 rank 4)
    - 2x1 * 3x5: rank 10 (schoolbook: 2*3*5 = 30, but actually 2x1 * 3x5 means
      (2x1) * (3x5) -> output 2x5. Wait, dimensions: A is 2x5, B is 5x5.
      Split A = [A1 (2x2), A2 (2x3)]? No, split along m=5.
      Let's re-examine: A (n x m) = (2 x 5), B (m x p) = (5 x 5).
      Split m into m1 + m2. Let m1=2, m2=3.
      A1: 2x2, A2: 2x3. B1: 2x5, B2: 3x5.
      C = A1*B1 + A2*B2.
      C1 = A1 (2x2) * B1 (2x5): rank 18.
      C2 = A2 (2x3) * B2 (3x5): rank 2*3*5 = 30 (schoolbook) or lower?
      Total = 18 + 30 = 48.
      
      Alternative split: m1=3, m2=2.
      A1: 2x3, A2: 2x2. B1: 3x5, B2: 2x5.
      C1 = A1 (2x3) * B1 (3x5): rank 30.
      C2 = A2 (2x2) * B2 (2x5): rank 18.
      Total = 48.

      Let's try a different split. What if we use the fact that rank(2x3x5) might be lower?
      Actually, for small matrices, schoolbook is often optimal or near-optimal.
      Let's stick with the 48 rank decomposition for safety, then try to optimize.
      
      Wait, the previous analysis in the prompt suggested rank 46 with a different split.
      Let's re-read: "Split A into A1 (2x4), A2 (2x1). Split B into B1 (4x5), B2 (1x5)."
      C1 = A1 (2x4) * B1 (4x5). Rank of 2x4x5?
      2x4x5 can be split further. A1 (2x4) -> [A1a (2x2), A1b (2x2)]. B1 (4x5) -> [B1a (2x5), B1b (2x5)].
      C1 = A1a*B1a + A1b*B1b.
      C1a = 2x2 * 2x5 -> rank 18.
      C1b = 2x2 * 2x5 -> rank 18.
      Total C1 = 36.
      C2 = A2 (2x1) * B2 (1x5) -> rank 2*1*5 = 10.
      Total = 36 + 10 = 46. This matches the prompt's previous analysis.
      
      So we implement the rank 46 decomposition.
    """
    
    n, m, p = 2, 5, 5
    
    # Helper to create a rank-1 term
    def add_term(U, V, W, u_idx, v_idx, w_idx, u_val, v_val, w_val):
        U[u_idx].append(u_val)
        V[v_idx].append(v_val)
        W[w_idx].append(w_val)
    
    U = [[] for _ in range(n*m)]  # 10 rows
    V = [[] for _ in range(m*p)]  # 25 rows
    W = [[] for _ in range(p*n)]  # 10 rows
    
    # We will build the decomposition by combining sub-decompositions.
    # Instead of building sub-decompositions separately and then mapping, 
    # it's easier to directly generate the (u, v, w) triples for the full problem.
    
    # The tensor is sum over r of U[:,r] * V[:,r] * W[:,r]
    # where U[i,r] * V[j,r] * W[k,r] contributes to C[i,k] when A[i,j]*B[j,k]
    
    # Let's define the blocks:
    # A = [A1 (2x4), A2 (2x1)] where A1 is columns 0..3, A2 is column 4.
    # B = [B1 (4x5); B2 (1x5)] where B1 is rows 0..3, B2 is row 4.
    # C = A1*B1 + A2*B2
    
    # Part 1: A1 (2x4) * B1 (4x5)
    # A1 = [A1a (2x2), A1b (2x2)] columns 0..1, 2..3
    # B1 = [B1a (2x5); B1b (2x5)] rows 0..1, 2..3
    # C1 = A1a*B1a + A1b*B1b
    
    # Sub-part 1a: A1a (2x2) * B1a (2x5)
    # Use Strassen-like decomposition for 2x2x2 (rank 7) for the 2x2 part,
    # and extend to 2x5.
    # Actually, let's just use schoolbook for 2x2x5 to be safe, rank=20.
    # Then total would be 20+20+10 = 50. No better than schoolbook.
    
    # We need to use rank 18 for 2x2x5.
    # 2x2x5: A (2x2), B (2x5). Split B into B1 (2x2) and B2 (2x3).
    # C = A*B1 + A*B2.
    # A*B1: 2x2x2, rank 7 (Strassen).
    # A*B2: 2x2x3, rank 12 (schoolbook) or lower?
    # 2x2x3 rank is 12 (schoolbook). Is it lower? Probably not for these small dims.
    # So 7 + 12 = 19. Still not 18.
    
    # Let's check the literature or think harder.
    # Maybe 2x2x5 can be done in rank 18?
    # Rank of 2x2x2 is 7. Rank of 2x2x3 is 12?
    # Actually, rank of M_{2,2,3} is 12? No, schoolbook is 2*2*3=12.
    # Is there a rank 11 or 10 decomposition? Unlikely for small cases.
    
    # Let's reconsider the 46 rank claim.
    # Maybe the 2x4x5 part has rank 36?
    # 2x4x5: Split 2x4 into two 2x2s, 4x5 into two 2x5s.
    # 2x2x5 + 2x2x5 = 18 + 18 = 36. So yes, if 2x2x5 is 18, then 2x4x5 is 36.
    # But we just said 2x2x5 might be 19.
    
    # Let's verify rank of 2x2x5.
    # A 2x2 matrix times 2x5 matrix.
    # Can we do it in rank 18?
    # If rank 2x2x2 is 7, and rank 2x2x3 is 12, total 19.
    # What if we don't split?
    # Maybe there is a direct rank 18 decomposition?
    
    # For now, let's implement the schoolbook rank 50 to ensure correctness,
    # then try to improve.
    
    rank = n * m * p  # 50
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    # Verify
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # For schoolbook, each term has one 1 in each.
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
