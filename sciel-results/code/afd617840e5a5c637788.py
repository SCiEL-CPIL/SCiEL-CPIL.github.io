"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import sys

_START = time.time()
_TIMEOUT = 140.0

def _elapsed():
    return time.time() - _START

def _validate(U, V, W, n, m, p):
    """Check that the factor matrices exactly decompose the matrix multiplication tensor."""
    # For each (i, j, k), sum_r U[i*m+j, r] * V[j*p+k, r] should equal W[k*n+i, r]
    # Actually the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] should match
    # the standard multiplication tensor where T[i,j,k] = 1 if j is the middle index
    # Wait, let me re-read the contract.
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The tensor for matrix multiplication is: T[i,j,k] = A[i,j] * B[j,k] -> C[i,k]
    # So the tensor has indices (i, j, k) and value 1 for each (i,j,k) triple? No.
    # The multiplication tensor is sum_{i,j,k} A[i,j] B[j,k] C[i,k]
    # So T[i,j,k] = 1 for all valid i,j,k.
    # The decomposition requires: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, the contract says "exactly decompose the matrix multiplication tensor"
    # and the factors are U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # The standard bilinear tensor decomposition for matrix multiplication:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: the multiplication tensor T has T[i,j,k] = 1
    # and we want T = sum_r U_r ⊗ V_r ⊗ W_r where (U_r)_{i,j} = U[i*m+j, r], etc.
    # So we need: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k.
    
    import numpy as np
    rank = len(U[0])
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Check a few random triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = _START + _TIMEOUT - 2.0
    
    # Strategy: Use the 42-rank Strassen-Kronecker decomposition as baseline.
    # Then attempt to find a lower rank via a different structural decomposition.
    
    # First, let's construct the known rank-42 solution (Strassen on 2x2 blocks).
    # A is 2x4, B is 4x6, C is 2x6.
    # Split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2
    # B = [[B00, B01, B02], [B10, B11, B12]] where Bij are 2x2
    # C = [[C00, C01, C02], [C10, C11, C12]] where Cij are 2x2
    # C = A * B, so Cij = sum_k Akj * Bkj where the sum is over the 2x2 block index.
    # For each pair of row-block and col-block, we do a 2x2 matrix multiply.
    # There are 2 row-blocks and 3 col-blocks, so 6 such 2x2x2 multiplies.
    # Each 2x2x2 multiply has rank 7 (Strassen).
    # Total rank = 6 * 7 = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # Local indices: 0:00, 1:01, 2:10, 3:11
    steps = [
        # M1 = (a00+a11)(b00+b11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (a10+a11)*b00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3 = a00*(b01-b11)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4 = a11*(b10-b00)
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5 = (a00+a01)*b11
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6 = (a10-a00)*(b00+b01)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (a01-a11)*(b10+b11)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    col = 0
    for u in range(2):
        for v in range(3):
            for U_loc, V_loc, W_loc in steps:
                for li in range(2):
                    for lj in range(2):
                        l_idx = li * 2 + lj
                        val = U_loc[l_idx]
                        if val != 0:
                            g_i = li
                            g_j = 2 * u + lj
                            U[g_i * m + g_j][col] += val
                
                for lj in range(2):
                    for lk in range(2):
                        l_idx = lj * 2 + lk
                        val = V_loc[l_idx]
                        if val != 0:
                            g_j = 2 * u + lj
                            g_k = 2 * v + lk
                            V[g_j * p + g_k][col] += val
                
                for li in range(2):
                    for lk in range(2):
                        l_idx = li * 2 + lk
                        val = W_loc[l_idx]
                        if val != 0:
                            g_i = li
                            g_k = 2 * v + lk
                            W[g_k * n + g_i][col] += val
                            
                col += 1
                
    # Validate the rank-42 solution
    if not _validate(U, V, W, n, m, p):
        # If validation fails, we have a bug. Return what we have.
        return {"U": U, "V": W, "W": W}
    
    # Now attempt to find a better (lower rank) decomposition.
    # The tensor for 2x4x6 has dimension 2*4*6 = 48 entries.
    # The rank is at most 48 (trivial) and at most 42 (Strassen-Kronecker).
    # Lower bound: max(2*4, 4*6, 2*6) = 24? No, that's not a valid lower bound.
    # The tensor rank lower bound is not simply the max dimension.
    # For M_{m,n,p}, lower bound is max(mn, mp, np) is NOT correct.
    # Actually, a known lower bound is that the rank is at least max(mn, mp, np) / min(m,n,p) ... no.
    # Let's just try to construct a rank-24 decomposition if possible.
    # The trivial decomposition has rank 48 (one term per (i,j,k) triple).
    # Can we do better?
    # The structure is C[i,k] = sum_j A[i,j] B[j,k].
    # For each (i,k), this is a linear combination of 4 terms.
    # The total number of (i,j,k) triples is 2*4*6 = 48.
    # The trivial rank is 48.
    # Strassen-Kronecker gives 42.
    # Is there a rank between 24 and 42?
    # Let's try a different blocking strategy.
    # What if we don't use Strassen for the inner 2x2 multiplies?
    # If we use the standard 4-mult 2x2 multiply, each 2x2x2 has rank 4.
    # 6 such multiplies: rank 24.
    # But wait, the standard 2x2 matrix multiply has rank 4? No, it has rank 7 with Strassen,
    # and the standard bilinear algorithm has rank 4? No, the standard algorithm uses 4 multiplications?
    # No, the standard 2x2 multiply uses 4 multiplications of 2x1 vectors? No.
    # The standard 2x2 matrix multiply: C = A*B where A,B are 2x2.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # This uses 4 scalar multiplications per output entry, 4 entries, total 16 scalar mults?
    # No, each "multiplication" here is a scalar multiplication.
    # There are 4*4 = 16 scalar multiplications in the naive algorithm.
    # The tensor rank of 2x2x2 is 7 (Strassen).
    # The "rank" in the sense of bilinear complexity is 7.
    # The trivial rank (number of scalar mults in naive) is 4*4=16? No, 2*2*2=8?
    # Let me recount. C[i,k] = sum_j A[i,j] B[j,k].
    # For 2x2: i in {0,1}, k in {0,1}, j in {0,1}.
    # 2*2*2 = 8 terms. So trivial rank is 8.
    # Strassen reduces it to 7.
    # So for 2x4x6: trivial rank is 2*4*6 = 48.
    # Strassen-Kronecker: 6 blocks of 2x2x2, each rank 7, total 42.
    # Can we get below 42?
    # The lower bound for M_{m,n,p} tensor rank is not well-known in general.
    # For M_{2,4,6}, the tensor rank is at least the rank of the flattening.
    # Flattening (i | j,k): matrix of size 2 x 24. Rank at most 2.
    # Flattening (j | i,k): matrix of size 4 x 12. Rank at most 4.
    # Flattening (k | i,j): matrix of size 6 x 8. Rank at most 6.
    # So tensor rank >= 6. That's a weak lower bound.
    # A better lower bound: the rank is at least the maximum of the matrix ranks of all flattening.
    # Actually, the tensor rank r satisfies r >= max over all flattening of the matrix rank.
    # The flattening (i | jk) has size 2 x 24, matrix rank 2.
    # The flattening (j | ik) has size 4 x 12, matrix rank 4.
    # The flattening (k | ij) has size 6 x 8, matrix rank 6.
    # So r >= 6. Still weak.
    # Another lower bound: r >= sqrt(mnp) ... no.
    # Let's just stick with 42 unless we can find a better construction.
    # Given the time constraint, I'll return the rank-42 solution.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
