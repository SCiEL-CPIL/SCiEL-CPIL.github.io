"""Matrix multiplication 3,4,7 - tensor decomposition with rank minimization."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

DEADLINE = 150.0
START_TIME = time.time()

def time_remaining():
    return DEADLINE - (time.time() - START_TIME) - 5.0

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Strassen 2x2x2 decomposition coefficients
    # Each coefficient is a 2x2 matrix flattened to 4 elements [00, 01, 10, 11]
    # U_coeffs[r][li*2+lj] for A[li, lj]
    # V_coeffs[r][lj*2+lk] for B[lj, lk]
    # W_coeffs[r][lk*2+li] for C[li, lk]
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
    # We cover i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # This is a 2x4x4 block.
    # We can split j into {0,1} and {2,3} to make two 2x2x4 blocks?
    # No, Strassen is 2x2x2.
    # We can split k into {0,1}, {2,3}.
    # So we have blocks:
    # (i in {0,1}, j in {0,1}, k in {0,1})
    # (i in {0,1}, j in {0,1}, k in {2,3})
    # (i in {0,1}, j in {2,3}, k in {0,1})
    # (i in {0,1}, j in {2,3}, k in {2,3})
    # This covers i in {0,1}, j in {0..3}, k in {0..3}.
    # Total elements: 2*4*4 = 32.
    # Strassen rank per 2x2x2 block: 7.
    # 4 blocks * 7 = 28 terms.
    # Schoolbook rank for this region: 32.
    # Savings: 4 terms.
    
    strassen_blocks = [
        (0, 0, 0), # i0=0, j0=0, k0=0
        (0, 0, 2), # i0=0, j0=0, k0=2
        (0, 2, 0), # i0=0, j0=2, k0=0
        (0, 2, 2), # i0=0, j0=2, k0=2
    ]
    
    # Collect all terms
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Add Strassen terms
    for i0, j0, k0 in strassen_blocks:
        for r in range(7):
            u_entries = []
            v_entries = []
            w_entries = []
            
            for li in range(2):
                for lj in range(2):
                    val = strassen_U[r][li*2+lj]
                    if val != 0:
                        u_entries.append(((i0+li)*m + (j0+lj), val))
            
            for lj in range(2):
                for lk in range(2):
                    val = strassen_V[r][lj*2+lk]
                    if val != 0:
                        v_entries.append(((j0+lj)*p + (k0+lk), val))
            
            for lk in range(2):
                for li in range(2):
                    val = strassen_W[r][lk*2+li]
                    if val != 0:
                        w_entries.append(((k0+lk)*n + (i0+li), val))
            
            U_terms.append(u_entries)
            V_terms.append(v_entries)
            W_terms.append(w_entries)
    
    # Define the set of (i,j,k) covered by Strassen
    strassen_covered = set()
    for i0, j0, k0 in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    strassen_covered.add((i0+di, j0+dj, k0+dk))
    
    # Add remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_covered:
                    U_terms.append([(i*m + j, 1)])
                    V_terms.append([(j*p + k, 1)])
                    W_terms.append([(k*n + i, 1)])
    
    rank = len(U_terms)
    
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
                    expected = 1
                    computed = 0
                    for r in range(rank):
                        computed += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if computed != expected:
                        return False
        # Check zeros
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
                        pass
                    else:
                        computed = 0
                        for r in range(rank):
                            computed += U[a, r] * V[b, r] * W[c, r]
                        if computed != 0:
                            return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
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
