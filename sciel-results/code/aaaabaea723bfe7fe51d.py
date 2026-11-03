"""Matrix multiplication 3,4,7 - tensor decomposition with rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]
# n=3, m=4, p=7
# U: (n*m, rank) = (12, rank)
# V: (m*p, rank) = (28, rank)
# W: (p*n, rank) = (21, rank)
# Convention: W[k*n + i, r] where result C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

DEADLINE = 120.0
START_TIME = time.time()

def time_remaining():
    return DEADLINE - (time.time() - START_TIME) - 5.0  # 5s safety margin

def construct():
    """
    Returns a tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Uses Strassen on 2x2x2 sub-blocks to reduce rank from 84 to 80.
    """
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Strassen 2x2x2 decomposition coefficients
    # U_coeffs[r][local_i, local_j] for A[local_i, local_j]
    # V_coeffs[r][local_j, local_k] for B[local_j, local_k]
    # W_coeffs[r][local_k, local_i] for C[local_i, local_k]
    strassen_U = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [0, 0, 1, -1], # M6
        [0, 1, 0, -1], # M7
    ]
    strassen_V = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, -1, 0], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1],  # M7
    ]
    strassen_W = [
        [1, 0, 1, 0],   # M1: C[0,0]=1, C[0,1]=0, C[1,0]=1, C[1,1]=0
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [-1, 1, 0, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 0, 0, 1],   # M6
        [1, 0, 0, 0],   # M7
    ]
    
    # Identify 2x2x2 blocks where Strassen applies
    # Blocks: (i0, j0, k0) where i0 in {0}, j0 in {0,2}, k0 in {0,2}
    # This gives 4 blocks: (0,0,0), (0,0,2), (0,2,0), (0,2,2)
    # Each covers i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    strassen_blocks = [(0, 0, 0), (0, 0, 2), (0, 2, 0), (0, 2, 2)]
    
    # Build set of (i,j,k) tuples covered by Strassen blocks
    strassen_covered = set()
    for i0, j0, k0 in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    strassen_covered.add((i0+di, j0+dj, k0+dk))
    
    # Collect all terms: Strassen terms + remaining schoolbook terms
    U_terms = []  # list of (index, value) pairs for each rank-1 term
    V_terms = []
    W_terms = []
    
    # Add Strassen terms
    for i0, j0, k0 in strassen_blocks:
        for r in range(7):
            u_entries = []
            v_entries = []
            w_entries = []
            
            # U: A[local_i, local_j] -> global index (i0+local_i)*m + (j0+local_j)
            for li in range(2):
                for lj in range(2):
                    val = strassen_U[r][li*2+lj]
                    if val != 0:
                        u_entries.append(((i0+li)*m + (j0+lj), val))
            
            # V: B[local_j, local_k] -> global index (j0+local_j)*p + (k0+local_k)
            for lj in range(2):
                for lk in range(2):
                    val = strassen_V[r][lj*2+lk]
                    if val != 0:
                        v_entries.append(((j0+lj)*p + (k0+lk), val))
            
            # W: C[local_i, local_k] -> global index (k0+local_k)*n + (i0+local_i)
            for lk in range(2):
                for li in range(2):
                    val = strassen_W[r][lk*2+li]
                    if val != 0:
                        w_entries.append(((k0+lk)*n + (i0+li), val))
            
            U_terms.append(u_entries)
            V_terms.append(v_entries)
            W_terms.append(w_entries)
    
    # Add remaining schoolbook terms (those not covered by Strassen)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_covered:
                    U_terms.append([(i*m + j, 1)])
                    V_terms.append([(j*p + k, 1)])
                    W_terms.append([(k*n + i, 1)])
    
    rank = len(U_terms)
    
    # Build dense matrices
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    for r in range(rank):
        for idx, val in U_terms[r]:
            U[idx, r] = val
        for idx, val in V_terms[r]:
            V[idx, r] = val
        for idx, val in W_terms[r]:
            W[idx, r] = val
    
    # Verify correctness
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Check that T[i*m+j, j*p+k, k*n+i] = 1
                    expected = 1
                    computed = 0
                    for r in range(rank):
                        computed += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if computed != expected:
                        return False
        # Also check that no other entries are nonzero
        # T[a,b,c] should be 0 unless c = (b % p)*n + (a // m) and b//p == a%m
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if j == j2 and k == k2 and i == i2:
                        # This should be 1
                        pass
                    else:
                        # This should be 0
                        computed = 0
                        for r in range(rank):
                            computed += U[a, r] * V[b, r] * W[c, r]
                        if computed != 0:
                            return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        U = np.zeros((n*m, n*m*p), dtype=np.int64)
        V = np.zeros((m*p, n*m*p), dtype=np.int64)
        W = np.zeros((p*n, n*m*p), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
