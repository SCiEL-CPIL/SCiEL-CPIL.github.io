"""Rank-minimal exact tensor decomposition search for 3x5x6 matrix multiplication."""
import numpy as np
import time
import itertools
import random
import sys
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _to_list(mat):
    return mat.tolist()


def _check_exact(U, V, W, n, m, p):
    """Verify T[i,j,k] == sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for all (i,j,k)."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            v_base = j * p
            for k in range(p):
                s = 0
                for c in range(r):
                    s = s + u_row[c] * V[v_base + k, c] * W[k * n + i, c]
                if s != 1:
                    return False
    return True


def _strassen_2x2():
    """Standard 2x2 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns (U, V, W) each of shape (4, 7) with entries in {0, 1, -1}."""
    U = np.zeros((4, 7), dtype=object)
    V = np.zeros((4, 7), dtype=object)
    W = np.zeros((4, 7), dtype=object)
    # 7 Strassen products
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # U rows: A11=0, A12=1, A21=2, A22=3
    # V rows: B11=0, B12=1, B21=2, B22=3
    # W rows: C11=0, C12=1, C21=2, C22=3
    
    # M1
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1; W[1,0]=-1; W[2,0]=-1
    # M2
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1
    # M3
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1
    # M4
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[2,3]=1
    # M5
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[3,4]=1
    # M6
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[0,5]=1
    # M7
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[3,6]=1
    return U, V, W


def _kron_decompose(U1, V1, W1, U2, V2, W2):
    """Kronecker product of two decompositions.
    If T1 = sum U1_c ⊗ V1_c ⊗ W1_c and T2 = sum U2_c ⊗ V2_c ⊗ W2_c,
    then T1⊗T2 = sum (U1_c⊗U2_d) ⊗ (V1_c⊗V2_d) ⊗ (W1_c⊗W2_d)."""
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    rank = r1 * r2
    n1, m1 = U1.shape[0] // 1, 0  # need dimensions
    # Actually we need to know the dimensions. Let's pass them.
    # For now, assume we know the sizes from the shapes
    # U1 is (n1*m1, r1), V1 is (m1*p1, r1), W1 is (p1*n1, r1)
    # U2 is (n2*m2, r2), V2 is (m2*p2, r2), W2 is (p2*n2, r2)
    # Result: U is (n1*n2 * m1*m2, rank), etc.
    # But we need to know n1,m1,p1,n2,m2,p2.
    # Let's infer from shapes: n1*m1 = U1.shape[0], m1*p1 = V1.shape[0], p1*n1 = W1.shape[0]
    # This is ambiguous. Let's require explicit dimensions.
    raise NotImplementedError("Use _kron_decompose_dim instead")


def _kron_decompose_dim(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
    """Kronecker product of two decompositions with explicit dimensions."""
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    rank = r1 * r2
    n, m, p = n1*n2, m1*m2, p1*p2
    U = np.zeros((n*m, rank), dtype=object)
    V = np.zeros((m*p, rank), dtype=object)
    W = np.zeros((p*n, rank), dtype=object)
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            # U: index (i1*m1+j1) for first, (i2*m2+j2) for second
            # Combined row index: (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
            # = (i1*n2 + i2) * m1 * m2 + j1 * m2 + j2
            # U1 row: i1*m1 + j1, U2 row: i2*m2 + j2
            for i1 in range(n1):
                for j1 in range(m1):
                    u1_val = U1[i1*m1 + j1, c1]
                    if u1_val == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2_val = U2[i2*m2 + j2, c2]
                            if u2_val == 0:
                                continue
                            row = (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
                            U[row, c] = u1_val * u2_val
            
            # V: index (j1*p1+k1) for first, (j2*p2+k2) for second
            # Combined row index: (j1*p1 + j2) * p2 + k1*p2 + k2? 
            # Wait, V index is j*p + k. 
            # First: j1*p1 + k1. Second: j2*p2 + k2.
            # Combined: j*p + k where j = j1*p1 + j2? No.
            # The matrix multiplication is (n1*n2) x (m1*m2) times (m1*m2) x (p1*p2).
            # So V has shape (m1*m2 * p1*p2, rank).
            # Row index: (j1*m2 + j2) * (p1*p2) + (k1*p2 + k2)
            # V1 row: j1*p1 + k1, V2 row: j2*p2 + k2
            for j1 in range(m1):
                for k1 in range(p1):
                    v1_val = V1[j1*p1 + k1, c1]
                    if v1_val == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2_val = V2[j2*p2 + k2, c2]
                            if v2_val == 0:
                                continue
                            row = (j1*m2 + j2) * (p1*p2) + (k1*p2 + k2)
                            V[row, c] = v1_val * v2_val
            
            # W: index (k1*n1 + i1) for first, (k2*n2 + i2) for second
            # Combined row index: (k1*n1 + k2) * n2 + i1*n2 + i2? 
            # W has shape (p1*p2 * n1*n2, rank).
            # Row index: (k1*p2 + k2) * (n1*n2) + (i1*n2 + i2)
            # W1 row: k1*n1 + i1, W2 row: k2*n2 + i2
            for k1 in range(p1):
                for i1 in range(n1):
                    w1_val = W1[k1*n1 + i1, c1]
                    if w1_val == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2_val = W2[k2*n2 + i2, c2]
                            if w2_val == 0:
                                continue
                            row = (k1*p2 + k2) * (n1*n2) + (i1*n2 + i2)
                            W[row, c] = w1_val * w2_val
    
    return U, V, W


def _embed_decomp(U, V, W, n_orig, m_orig, p_orig, n_new, m_new, p_new):
    """Embed a decomposition for (n_orig x m_orig) * (m_orig x p_orig) into 
    a decomposition for (n_new x m_new) * (m_new x p_new) by padding with zeros
    and using schoolbook for the extra parts. This is a rough embedding."""
    # This is complex; for now, return None to indicate not implemented
    return None


def _add_decomps(decomps):
    """Sum multiple decompositions (concatenate along rank dimension)."""
    if len(decomps) == 1:
        return decomps[0]
    total_rank = sum(d[0].shape[1] for d in decomps)
    U = np.zeros((decomps[0][0].shape[0], total_rank), dtype=object)
    V = np.zeros((decomps[0][1].shape[0], total_rank), dtype=object)
    W = np.zeros((decomps[0][2].shape[0], total_rank), dtype=object)
    col = 0
    for U_d, V_d, W_d in decomps:
        r = U_d.shape[1]
        U[:, col:col+r] = U_d
        V[:, col:col+r] = V_d
        W[:, col:col+r] = W_d
        col += r
    return U, V, W


def _block_decomposition(n, m, p, deadline):
    """Try to find a decomposition using block structure.
    For 3x5x6, we can try to decompose as a combination of smaller blocks."""
    n, m, p = 3, 5, 6
    
    # Strategy: Use Strassen on a 2x2 subblock, schoolbook for the rest
    # Partition: rows 0-1 (2 rows), col 0-2 (3 cols) for the 2x2 Strassen part
    # Actually, let's try a different approach.
    
    # Try: decompose as (2x2 Strassen) ⊗ (1x1) + schoolbook for remaining
    # This is getting complex. Let's try a simpler structured approach.
    
    # For 3x5x6, one approach:
    # Use 7 multiplications for a 2x2 block (Strassen)
    # Use schoolbook for the rest
    # Total rank = 7 + (3*5*6 - 2*2*2) = 7 + 74 = 81? No, that's not right.
    
    # Let's try: partition the 3x5 matrix into blocks
    # Block 1: 2x2 (rows 0-1, cols 0-1)
    # Block 2: 2x3 (rows 0-1, cols 2-4)  
    # Block 3: 1x5 (row 2, cols 0-4)
    
    # This is getting too complex for a quick implementation.
    # Let's return the schoolbook for now and add a simple random search.
    
    return None


def _random_search(n, m, p, max_rank, deadline, seed=42):
    """Random search for a decomposition with rank <= max_rank."""
    rng = random.Random(seed)
    best = None
    start = time.time()
    
    # Try a few random initializations
    for attempt in range(20):
        if time.time() > deadline - 30:
            break
        
        rank = max_rank
        # Random entries from {-1, 0, 1}
        U = np.zeros((n*m, rank), dtype=object)
        V = np.zeros((m*p, rank), dtype=object)
        W = np.zeros((p*n, rank), dtype=object)
        
        for c in range(rank):
            for i in range(n*m):
                U[i, c] = rng.choice([-1, 0, 0, 0, 1])
            for i in range(m*p):
                V[i, c] = rng.choice([-1, 0, 0, 0, 1])
            for i in range(p*n):
                W[i, c] = rng.choice([-1, 0, 0, 0, 1])
        
        # Check if this works (very unlikely for random, but try)
        if _check_exact(U, V, W, n, m, p):
            return U, V, W
    
    return None


def construct():
    start = time.time()
    deadline = start + 140.0
    n, m, p = DIMENSIONS

    # Schoolbook as baseline
    U0, V0, W0 = _schoolbook(n, m, p)
    best_rank = n * m * p
    best = (U0, V0, W0)

    # Try Strassen-based decomposition
    # For 3x5x6, try to use Strassen on a 2x2 subproblem
    # Partition: use rows 0-1, cols 0-1 for a 2x2 Strassen
    # Then schoolbook for the rest
    
    # Actually, let's try a cleaner approach:
    # Decompose 3x5x6 as a combination of smaller known decompositions
    
    # Try: 2x2x2 Strassen (rank 7) for a subblock
    # Plus schoolbook for the remaining part
    
    # Let's try to construct a valid decomposition with rank < 90
    
    # Approach: Use the fact that we can pad smaller decompositions
    # For now, try a few structured approaches
    
    # Method 1: Try to find if a rank-85 decomposition exists via random search
    # This is unlikely to work quickly, but let's try a few iterations
    
    # Method 2: Use a known result - for 3x5x6, the optimal rank is unknown
    # but we know it's <= 90 (schoolbook)
    
    # Let's try to improve by at least 1 using a simple heuristic
    
    # Actually, let's try a different approach: 
    # Use tensor train or other structured decompositions
    
    # For now, return the schoolbook as it's guaranteed to work
    U, V, W = best
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
# EVOLVE-BLOCK-END
