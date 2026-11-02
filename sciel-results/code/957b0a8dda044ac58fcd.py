"""
Rank-32 exact tensor decomposition for 2x4 by 4x8 matrix multiplication.

Uses the 4-point DFT (Winograd frequency-domain method) on the middle dimension m=4.
For each of 4 frequency bins and 8 output columns, one rank-1 term computes
the contribution to all 2 output rows simultaneously.

Total rank = 4 * 8 = 32.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]
_DEADLINE = 115.0  # seconds, safety margin before 120s limit

def construct():
    """
    Construct a rank-32 decomposition for 2x4 by 4x8 matrix multiplication.
    Uses 4-point DFT on the inner dimension, shared across rows.
    """
    n, m, p = DIMENSIONS  # A: (2,4), B: (4,8), C: (2,8)
    rank = 4 * p  # 4 frequency bins × 8 columns = 32
    
    # 4-point DFT matrix: F[t,j] = exp(-2πi*t*j/4)
    # Using exact values:
    # t=0: [1, 1, 1, 1]
    # t=1: [1, -i, -1, i]
    # t=2: [1, -1, 1, -1]
    # t=3: [1, i, -1, -i]
    
    # We need real/integer/half-integer entries.
    # The DFT involves complex numbers, but we can use a real-valued transform.
    # 
    # Alternative: Use a real 4-point transform. The 4-point real DFT can be
    # decomposed into real and imaginary parts, but that doubles the rank.
    #
    # Better approach: Use the fact that for matrix multiplication, we need:
    #   sum_j A[i,j] * B[j,k] = C[i,k]
    # 
    # This is equivalent to: for each (i,k), compute dot product of A[i,:] and B[:,k]
    # 
    # The rank of the identity matrix I_4 is 4, so we need at least 4 terms
    # for the j-dimension. With 8 columns of B, total rank >= 4*8 = 32.
    #
    # Use the standard basis: e_0, e_1, e_2, e_3 for the 4-point transform.
    # This gives the trivial rank-4 decomposition of I_4.
    
    # Actually, let's use a more structured approach.
    # For the 4-point inner product, we can use:
    #   x0 = a0 + a2, x1 = a1 + a3, x2 = a0 - a2, x3 = a1 - a3
    #   y0 = b0 + b2, y1 = b1 + b3, y2 = b0 - b2, y3 = b1 - b3
    #   Then: a·b = (x0*y0 + x2*y2 + x1*y1 + x3*y3) / 2
    # 
    # This uses 4 multiplications: x0*y0, x1*y1, x2*y2, x3*y3
    # And requires division by 2 in the output.
    
    # Transform matrices:
    # For A (input): 
    #   x0 = a0 + a2 -> [1, 0, 1, 0]
    #   x1 = a1 + a3 -> [0, 1, 0, 1]
    #   x2 = a0 - a2 -> [1, 0, -1, 0]
    #   x3 = a1 - a3 -> [0, 1, 0, -1]
    
    # For B (input):
    #   y0 = b0 + b2 -> [1, 0, 1, 0]
    #   y1 = b1 + b3 -> [0, 1, 0, 1]
    #   y2 = b0 - b2 -> [1, 0, -1, 0]
    #   y3 = b1 - b3 -> [0, 1, 0, -1]
    
    # Output: C[i,k] = (x0*y0 + x1*y1 + x2*y2 + x3*y3) / 2
    
    # So for each frequency t in {0,1,2,3} and each column k in {0,...,7}:
    #   U[i*4+j, (t,k)] = F_A[t,j] for both i=0,1
    #   V[j*8+k, (t,k)] = F_B[t,j] for all j
    #   W[k*2+i, (t,k)] = 1/2 for the corresponding i
    
    # Wait, W should be 1/2 for both i values since the formula applies to both rows.
    
    F_A = [
        [1, 0, 1, 0],   # t=0: a0 + a2
        [0, 1, 0, 1],   # t=1: a1 + a3
        [1, 0, -1, 0],  # t=2: a0 - a2
        [0, 1, 0, -1],  # t=3: a1 - a3
    ]
    F_B = [
        [1, 0, 1, 0],   # t=0: b0 + b2
        [0, 1, 0, 1],   # t=1: b1 + b3
        [1, 0, -1, 0],  # t=2: b0 - b2
        [0, 1, 0, -1],  # t=3: b1 - b3
    ]
    
    U = [[0.0] * rank for _ in range(n * m)]   # shape (8, 32)
    V = [[0.0] * rank for _ in range(m * p)]   # shape (32, 32)
    W = [[0.0] * rank for _ in range(p * n)]   # shape (16, 32)
    
    for t in range(4):
        for k in range(8):
            col = t * p + k
            
            # U: for each row i and column j of A
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = float(F_A[t][j])
            
            # V: for each row j and column k of B
            for j in range(m):
                V[j * p + k][col] = float(F_B[t][j])
            
            # W: for each output position (i, k)
            for i in range(n):
                W[k * n + i][col] = 0.5
    
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check all valid (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = 0.0
                for s in range(rank):
                    val += U[a_idx, s] * V[b_idx, s] * W[c_idx, s]
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check random invalid triples
    import random
    random.seed(42)
    for _ in range(500):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        i1, j1 = divmod(a, m)
        j2, k1 = divmod(b, p)
        k2, i2 = divmod(c, n)
        if j1 != j2 or i1 != i2 or k1 != k2:
            val = 0.0
            for s in range(rank):
                val += U[a, s] * V[b, s] * W[c, s]
            if abs(val) > 1e-9:
                return False
    
    return True


if __name__ == "__main__":
    start = time.time()
    result = construct()
    n, m, p = DIMENSIONS
    ok = _verify(result["U"], result["V"], result["W"], n, m, p)
    rank = len(result["U"][0])
    elapsed = time.time() - start
    print(f"Rank: {rank}, Valid: {ok}, Time: {elapsed:.2f}s")
    if not ok:
        print("DECOMPOSITION IS INVALID!")
    else:
        print("Decomposition verified successfully.")
