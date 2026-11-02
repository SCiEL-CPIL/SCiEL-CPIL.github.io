import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    n, m, p = 3, 5, 7

    # Schoolbook decomposition
    def schoolbook(n, m, p):
        r = n * m * p
        U = np.zeros((n * m, r), dtype=np.int64)
        V = np.zeros((m * p, r), dtype=np.int64)
        W = np.zeros((p * n, r), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U, 'V': V, 'W': W}

    # Strassen decomposition for 2x2x2
    def strassen_222():
        # Returns U, V, W for M_{2,2,2} with rank 7
        # Indices: U[0..3, 0..6], V[0..3, 0..6], W[0..3, 0..6]
        # U corresponds to A (2x2), V to B (2x2), W to C (2x2)
        # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        U = np.zeros((4, 7), dtype=np.int64)
        V = np.zeros((4, 7), dtype=np.int64)
        W = np.zeros((4, 7), dtype=np.int64)
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = -1 # C00 += M1, C11 -= M1 (Wait, C11 = M1 - M2 + M3 + M6. So W[3,0]=1? No, let's check signs)
        # Let's use the standard Strassen formulas carefully.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1: A00, A11, B00, B11
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1  # C00
        W[3, 0] = 1  # C11

        # M2: A10, A11, B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1  # C10
        W[3, 1] = -1 # C11

        # M3: A00, B01, B11
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1  # C01
        W[3, 2] = 1  # C11

        # M4: A11, B10, B00
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1  # C00
        W[2, 3] = 1  # C10

        # M5: A00, A01, B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1 # C00
        W[1, 4] = 1  # C01

        # M6: A10, A00, B00, B01
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1  # C11

        # M7: A01, A11, B10, B11
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1  # C00

        return {'U': U, 'V': V, 'W': W}

    def combine_decompositions(decs):
        """Combine a list of decompositions (dicts with U, V, W) into one."""
        if not decs:
            return None
        total_rank = sum(d['U'].shape[1] for d in decs)
        # We need to know the dimensions n, m, p for the combined result
        # This function assumes all decs have the same n, m, p
        # But in our recursive splitting, we split dimensions.
        # So we need a different combine: block diagonal or similar.
        # Actually, we are building the decomposition for a specific n, m, p.
        # The recursive function will return U, V, W for that specific n, m, p.
        pass

    # Recursive decomposition
    def decompose(n, m, p):
        # Base cases
        if n == 1 and m == 1 and p == 1:
            U = np.ones((1, 1), dtype=np.int64)
            V = np.ones((1, 1), dtype=np.int64)
            W = np.ones((1, 1), dtype=np.int64)
            return {'U': U, 'V': V, 'W': W}
        
        if n == 2 and m == 2 and p == 2:
            return strassen_222()
            
        # If any dimension is 1, use schoolbook (no reduction possible)
        if n == 1 or m == 1 or p == 1:
            return schoolbook(n, m, p)
            
        # Try to split the largest dimension
        # Split n
        if n >= 2:
            n1, n2 = 1, n - 1
            # M_{n1,m,p} + M_{n2,m,p}
            d1 = decompose(n1, m, p)
            d2 = decompose(n2, m, p)
            # Combine d1 and d2
            # d1 is for n1 x m x p, d2 is for n2 x m x p
            # U: (n1*m + n2*m, rank1+rank2)
            # V: (m*p, rank1+rank2) -> Same V rows? No, V is (m*p, rank). 
            # In the combined decomposition, the rank terms are disjoint.
            # So V for d1 is padded with zeros for d2's rank, and vice versa.
            r1 = d1['U'].shape[1]
            r2 = d2['U'].shape[1]
            r_total = r1 + r2
            
            U = np.zeros((n*m, r_total), dtype=np.int64)
            V = np.zeros((m*p, r_total), dtype=np.int64)
            W = np.zeros((p*n, r_total), dtype=np.int64)
            
            # Place d1
            U[:n1*m, :r1] = d1['U']
            V[:, :r1] = d1['V']
            W[:, :r1] = d1['W']
            
            # Place d2
            U[n1*m:, r1:] = d2['U']
            V[:, r1:] = d2['V']
            W[:, r1:] = d2['W']
            
            return {'U': U, 'V': V, 'W': W}
            
        # Split m
        if m >= 2:
            m1, m2 = 1, m - 1
            d1 = decompose(n, m1, p)
            d2 = decompose(n, m2, p)
            r1 = d1['U'].shape[1]
            r2 = d2['U'].shape[1]
            r_total = r1 + r2
            
            U = np.zeros((n*m, r_total), dtype=np.int64)
            V = np.zeros((m*p, r_total), dtype=np.int64)
            W = np.zeros((p*n, r_total), dtype=np.int64)
            
            # U: n*m1 + n*m2
            U[:n*m1, :r1] = d1['U']
            U[n*m1:, r1:] = d2['U']
            
            # V: m1*p + m2*p
            V[:m1*p, :r1] = d1['V']
            V[m1*p:, r1:] = d2['V']
            
            # W: same
            W[:, :r1] = d1['W']
            W[:, r1:] = d2['W']
            
            return {'U': U, 'V': V, 'W': W}
            
        # Split p
        if p >= 2:
            p1, p2 = 1, p - 1
            d1 = decompose(n, m, p1)
            d2 = decompose(n, m, p2)
            r1 = d1['U'].shape[1]
            r2 = d2['U'].shape[1]
            r_total = r1 + r2
            
            U = np.zeros((n*m, r_total), dtype=np.int64)
            V = np.zeros((m*p, r_total), dtype=np.int64)
            W = np.zeros((p*n, r_total), dtype=np.int64)
            
            # U: same
            U[:, :r1] = d1['U']
            U[:, r1:] = d2['U']
            
            # V: m*p1 + m*p2
            V[:, :m*p1] = d1['V'][:, :m*p1] # Wait, V shape is (m*p, rank).
            # d1 V is (m*p1, r1). d2 V is (m*p2, r2).
            V[:m*p1, :r1] = d1['V']
            V[m*p1:, r1:] = d2['V']
            
            # W: p1*n + p2*n
            W[:p1*n, :r1] = d1['W']
            W[p1*n:, r1:] = d2['W']
            
            return {'U': U, 'V': V, 'W': W}
            
        # Fallback
        return schoolbook(n, m, p)

    # Get the decomposition
    result = decompose(n, m, p)
    
    # Verify
    U = np.array(result['U'], dtype=np.int64)
    V = np.array(result['V'], dtype=np.int64)
    W = np.array(result['W'], dtype=np.int64)
    rank = U.shape[1]
    
    # Simple verification for a few random entries
    # T[a,b,c] = sum_r U[a,r]V[b,r]W[c,r]
    # Expected 1 if a=i*m+j, b=j*p+k, c=k*n+i
    
    def verify(U, V, W):
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        for a in range(n * m):
            i = a // m
            j = a % m
            for b in range(m * p):
                j2 = b // p
                k = b % p
                for c in range(p * n):
                    k2 = c // n
                    i2 = c % n
                    expected = 1 if (i == i2 and j == j2 and k == k2) else 0
                    val = np.dot(U[a], V[b] * W[c])
                    if val != expected:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails
        return schoolbook(n, m, p)
        
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
# EVOLVE-BLOCK-END
