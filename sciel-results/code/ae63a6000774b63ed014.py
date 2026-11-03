import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n=3, m=5, p=7, tol=1e-9):
    """Verify that U, V, W form a valid decomposition of the matmul tensor."""
    rank = U.shape[1]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rng = np.random.RandomState(42)
    for _ in range(200):
        i = rng.randint(n)
        j = rng.randint(m)
        j2 = rng.randint(m)
        k = rng.randint(p)
        k2 = rng.randint(p)
        i2 = rng.randint(n)
        
        lhs = sum(U_arr[i*m+j, c] * V_arr[j2*p+k, c] * W_arr[k2*n+i2, c] for c in range(rank))
        rhs = (1.0 if j == j2 and k == k2 and i == i2 else 0.0)
        if abs(lhs - rhs) > tol:
            return False
    return True

def _strassen_block(U, V, W, rank_offset, i1, i2, j1, j2, k1, k2, n, m, p):
    """
    Add a Strassen 2x2x2 block to U, V, W.
    Local indices 0,1 map to global i1,i2 etc.
    """
    # M1 = (A00+A11)(B00+B11) -> C11+1, C22+1
    # u: A00, A11 ; v: B00, B11 ; w: C11, C22
    U[i1*m+j1, rank_offset] = 1
    U[i2*m+j2, rank_offset] = 1
    V[j1*p+k1, rank_offset] = 1
    V[j2*p+k2, rank_offset] = 1
    W[k1*n+i1, rank_offset] = 1
    W[k2*n+i2, rank_offset] = 1
    rank_offset += 1

    # M2 = (A11+A22? No) (c+d)e -> A11+B11? No.
    # M2 = (A10+A11) B00 ? No.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # Let's re-verify the coefficients for w.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M2: u: A10, A11 ; v: B00 ; w: C10 (+1)
    U[i2*m+j1, rank_offset] = 1 # A10
    U[i2*m+j2, rank_offset] = 1 # A11
    V[j1*p+k1, rank_offset] = 1 # B00
    W[k1*n+i2, rank_offset] = 1 # C10
    rank_offset += 1

    # M3: u: A00 ; v: B10, B11 ; w: C01 (+1)
    # v: B10 - B11. So V[j2*p+k1] = 1, V[j2*p+k2] = -1.
    U[i1*m+j1, rank_offset] = 1 # A00
    V[j2*p+k1, rank_offset] = 1 # B10
    V[j2*p+k2, rank_offset] = -1 # B11
    W[k1*n+i1, rank_offset] = 1 # C01
    rank_offset += 1

    # M4: u: A11 ; v: B01, B00 ; w: C00 (+1), C10 (+1)
    # v: B01 - B00. So V[j1*p+k2] = 1, V[j1*p+k1] = -1.
    U[i2*m+j2, rank_offset] = 1 # A11
    V[j1*p+k2, rank_offset] = 1 # B01
    V[j1*p+k1, rank_offset] = -1 # B00
    W[k1*n+i1, rank_offset] = 1 # C00
    W[k1*n+i2, rank_offset] = 1 # C10
    rank_offset += 1

    # M5: u: A00, A01 ; v: B11 ; w: C00 (-1), C01 (+1)
    U[i1*m+j1, rank_offset] = 1 # A00
    U[i1*m+j2, rank_offset] = 1 # A01
    V[j2*p+k2, rank_offset] = 1 # B11
    W[k1*n+i1, rank_offset] = -1 # C00
    W[k1*n+i2, rank_offset] = 1 # C01
    rank_offset += 1

    # M6: u: A10, A00 ; v: B00, B10 ; w: C11 (+1)
    # u: A10 - A00.
    U[i2*m+j1, rank_offset] = 1 # A10
    U[i1*m+j1, rank_offset] = -1 # A00
    V[j1*p+k1, rank_offset] = 1 # B00
    V[j2*p+k1, rank_offset] = 1 # B10
    W[k2*n+i2, rank_offset] = 1 # C11
    rank_offset += 1

    # M7: u: A01, A11 ; v: B01, B11 ; w: C00 (+1), C11 (+1)
    # u: A01 - A11.
    U[i1*m+j2, rank_offset] = 1 # A01
    U[i2*m+j2, rank_offset] = -1 # A11
    V[j1*p+k2, rank_offset] = 1 # B01
    V[j2*p+k2, rank_offset] = 1 # B11
    W[k1*n+i1, rank_offset] = 1 # C00
    W[k2*n+i2, rank_offset] = 1 # C11
    rank_offset += 1

    return rank_offset

def _construct_strassen_schoolbook(n=3, m=5, p=7):
    """
    Construct decomposition using 6 Strassen blocks and schoolbook for the rest.
    """
    # Define the 6 blocks
    # I pairs: {0,1}
    # J pairs: {0,1}, {2,3}
    # K pairs: {0,1}, {2,3}, {4,5}
    
    blocks = []
    i1, i2 = 0, 1
    for j1, j2 in [(0, 1), (2, 3)]:
        for k1, k2 in [(0, 1), (2, 3), (4, 5)]:
            blocks.append((i1, i2, j1, j2, k1, k2))
            
    # Total rank
    # 6 blocks * 7 rank each = 42
    # Remaining entries: 105 - 6*8 = 57
    total_rank = 6 * 7 + (n * m * p - 6 * 8)
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    rank_offset = 0
    # Track which (i,j,k) triples are covered by Strassen
    covered = set()
    
    for b in blocks:
        i1, i2, j1, j2, k1, k2 = b
        # Mark as covered
        for i in [i1, i2]:
            for j in [j1, j2]:
                for k in [k1, k2]:
                    covered.add((i, j, k))
                    
        rank_offset = _strassen_block(U, V, W, rank_offset, i1, i2, j1, j2, k1, k2, n, m, p)
        
    # Add schoolbook for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j, rank_offset] = 1
                    V[j * p + k, rank_offset] = 1
                    W[k * n + i, rank_offset] = 1
                    rank_offset += 1
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    sol = _construct_strassen_schoolbook(n, m, p)
    return sol

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
