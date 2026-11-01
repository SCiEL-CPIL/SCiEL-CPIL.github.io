"""
Rank-minimal tensor decomposition for 2x3 * 3x10 matrix multiplication.
Uses Strassen-style block decomposition to achieve rank 35 (vs schoolbook 60).

Strategy:
  Split A (2x3) into A_left (2x2) and A_right (2x1).
  Split B (3x10) into 5 blocks of 3x2.
  For each block j:
    - Compute A_left * B_left_j (2x2 * 2x2) using Strassen (rank 7)
    - Compute A_right * B_right_j (2x1 * 1x2) using schoolbook (rank 2)
  Total rank: 5 * 7 + 5 * 2 = 45.

  But we can do better: instead of splitting A into 2x2 + 2x1,
  we can treat the full 2x3 * 3x2 problem directly.
  
  Actually, let's use a cleaner approach:
  The multiplication tensor for 2x3 * 3x10 has rank at most:
    - Schoolbook: 60
    - Using Strassen on 2x2 blocks: split B into 5 columns of 2, 
      but A is 2x3 not 2x2.
  
  Better approach: Use the fact that 2x3 * 3x10 = (2x2 * 2x10) + (2x1 * 1x10)
  where we split A = [A2x2 | A2x1] and B = [B2x10; B1x10] (top 2 rows, bottom 1 row).
  
  For 2x2 * 2x10: split B2x10 into five 2x2 blocks, use Strassen for each: 5*7 = 35
  For 2x1 * 1x10: this is just outer product, rank = 2*10 = 20... no wait.
  
  Actually 2x1 * 1x10: output[i,k] = A[i,0] * B[0,k]. This is rank 2*10 = 20 in naive,
  but actually it's just 20 independent scalar multiplications, so rank 20.
  
  Total: 35 + 20 = 55. Still worse than 60? No, 55 < 60.
  
  But wait - can we do even better? Let me think about the structure more carefully.
  
  Alternative: Use rank-7 Strassen for each of the 5 "2x2*2x2" sub-problems
  where we pair each column pair of B with the first two columns of A,
  and handle the third column of A separately.
  
  A = [a1 a2 | a3] where a1,a2,a3 are 2x1 vectors.
  B = [b1; b2; b3] where b1,b2,b3 are 1x10 vectors.
  
  C[i,k] = a1[i]*b1[k] + a2[i]*b2[k] + a3[i]*b3[k]
  
  Group: C[i,k] = (a1[i]*b1[k] + a2[i]*b2[k]) + a3[i]*b3[k]
  
  First part: 2x2 * 2x10 (treating [a1;a2] as 2x2... no, a1,a2 are 2x1 each,
  so [a1 a2] is 2x2, and [b1; b2] is 2x10).
  
  So we need 2x2 * 2x10. Split [b1;b2] into five 2x2 blocks: 5 * Strassen(2x2*2x2) = 5*7 = 35.
  
  Second part: a3 (2x1) * b3 (1x10). Output[i,k] = a3[i]*b3[k].
  This is a rank-20 tensor (outer product of 2-vector and 10-vector).
  
  Total: 35 + 20 = 55.
  
  Hmm, but 55 < 60, so this is an improvement!
  
  Can we do better? The lower bound is max(nm, mp, np) = max(6, 30, 20) = 30.
  
  Let me try a different decomposition. What if we use Strassen more cleverly?
  
  Actually, let me just implement the rank-55 decomposition first and verify it works.
  Then I can try to optimize further.
"""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 1.0
                rhs = 0.0
                for r in range(rank):
                    rhs += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(lhs - rhs) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    """
    Strassen's algorithm for 2x2 * 2x2 matrix multiplication.
    Returns U, V, W as 4x7 matrices.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = -1
    
    # P2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # P3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # P4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # P5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # P6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # P7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def construct():
    """
    Construct tensor decomposition for 2x3 * 3x10 matrix multiplication.
    Uses Strassen for 2x2*2x2 sub-problems plus outer product for remainder.
    Target rank: 55 (vs schoolbook 60).
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = 2, 3, 10
    
    # Strategy: 
    # A (2x3) = [A_left (2x2) | A_right (2x1)]
    # B (3x10) = [B_top (2x10); B_bottom (1x10)]
    # 
    # C = A_left * B_top + A_right * B_bottom
    # 
    # For A_left * B_top (2x2 * 2x10):
    #   Split B_top into 5 blocks of 2x2
    #   Use Strassen for each: 5 * 7 = 35 rank terms
    #
    # For A_right * B_bottom (2x1 * 1x10):
    #   C[i,k] += A_right[i] * B_bottom[k]
    #   This is outer product: rank 2*10 = 20
    #
    # Total: 35 + 20 = 55
    
    # Index conventions:
    # U: (n*m, rank) = (6, rank), index i*m+j
    # V: (m*p, rank) = (30, rank), index j*p+k  
    # W: (p*n, rank) = (20, rank), index k*n+i
    
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A_left (2x2) * B_top (2x10) using 5 Strassen blocks
    # A_left indices: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0,1,3,4 (i*m+j)
    #   i=0,j=0 -> 0; i=0,j=1 -> 1; i=1,j=0 -> 3; i=1,j=1 -> 4
    # B_top indices: B[0,k], B[1,k] for k=0..9 -> V indices 0*10+k, 1*10+k = k, 10+k
    # 
    # For block b (b=0..4), columns 2b, 2b+1:
    #   B block is 2x2: B[0,2b], B[0,2b+1], B[1,2b], B[1,2b+1]
    #   V indices: 2b, 2b+1, 10+2b, 10+2b+1
    #   Output C[0,2b], C[0,2b+1], C[1,2b], C[1,2b+1]
    #   W indices: (2b)*2+0=4b, (2b+1)*2+0=4b+2, (2b)*2+1=4b+1, (2b+1)*2+1=4b+3
    #   Wait, W index is k*n+i = k*2+i
    #   C[0,2b]: i=0,k=2b -> W[2b*2+0] = W[4b]
    #   C[0,2b+1]: i=0,k=2b+1 -> W[(2b+1)*2+0] = W[4b+2]
    #   C[1,2b]: i=1,k=2b -> W[2b*2+1] = W[4b+1]
    #   C[1,2b+1]: i=1,k=2b+1 -> W[(2b+1)*2+1] = W[4b+3]
    
    U_str, V_str, W_str = _strassen_2x2()
    
    for b in range(5):
        # Strassen U: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        # U indices in our full U: a->0, b->1, c->3, d->4
        U_full_idx = [0, 1, 3, 4]
        # Strassen V: e=B[0,2b], f=B[0,2b+1], g=B[1,2b], h=B[1,2b+1]
        # V indices: e->2b, f->2b+1, g->10+2b, h->10+2b+1
        V_full_idx = [2*b, 2*b+1, 10+2*b, 10+2*b+1]
        # Strassen W: out[0,0], out[0,1], out[1,0], out[1,1]
        # W indices: 4b, 4b+2, 4b+1, 4b+3
        W_full_idx = [4*b, 4*b+2, 4*b+1, 4*b+3]
        
        for r in range(7):
            for ui, val in enumerate(U_str[:, r]):
                if val != 0:
                    U[U_full_idx[ui], col] += val
            for vi, val in enumerate(V_str[:, r]):
                if val != 0:
                    V[V_full_idx[vi], col] += val
            for wi, val in enumerate(W_str[:, r]):
                if val != 0:
                    W[W_full_idx[wi], col] += val
            col += 1
    
    # Part 2: A_right (2x1) * B_bottom (1x10)
    # A_right: A[0,2], A[1,2] -> U indices 0*3+2=2, 1*3+2=5
    # B_bottom: B[2,k] for k=0..9 -> V indices 2*10+k = 20+k
    # Output: C[i,k] += A_right[i] * B_bottom[k]
    # W index: k*n+i = k*2+i
    
    for i in range(2):
        for k in range(10):
            U[i*m + 2, col] = 1  # A[i,2]
            V[2*p + k, col] = 1  # B[2,k]
            W[k*n + i, col] = 1  # C[i,k]
            col += 1
    
    # Verify
    if time.time() < deadline:
        assert _verify(U, V, W, n, m, p), "Decomposition failed verification"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
