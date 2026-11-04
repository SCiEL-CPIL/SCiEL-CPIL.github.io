import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 9]

def _strassen_2x2():
    """Return (U, V, W) for 2x2 x 2x2 -> 2x2 Strassen, rank 7.
    U: (4,7) indexed [i*2+k], V: (4,7) indexed [k*2+j], W: (4,7) indexed [j*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[1,1]=1; W[3,1]=-1
    # M3 = A00(B01-B11)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[2,2]=1; W[3,2]=1
    # M4 = A11(B10-B00)
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[1,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[2,4]=1
    # M6 = (A10-A00)(B00+B01)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    # M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1
    
    return U, V, W

def _build_4x4x4_strassen():
    """4x4 x 4x4 -> 4x4 via Strassen, rank 49.
    U: (16,49) [i*4+k], V: (16,49) [k*4+j], W: (16,49) [j*4+i]
    """
    U2, V2, W2 = _strassen_2x2()
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    big_mults = [
        {'A': [((0,0),1),((1,1),1)], 'B': [((0,0),1),((1,1),1)], 'C': [((0,0),1),((1,1),1)]},
        {'A': [((1,0),1),((1,1),1)], 'B': [((0,0),1)], 'C': [((0,1),1),((1,1),-1)]},
        {'A': [((0,0),1)], 'B': [((0,1),1),((1,1),-1)], 'C': [((1,0),1),((1,1),1)]},
        {'A': [((1,1),1)], 'B': [((1,0),1),((0,0),-1)], 'C': [((0,0),1),((0,1),1)]},
        {'A': [((0,0),1),((0,1),1)], 'B': [((1,1),1)], 'C': [((0,0),-1),((1,0),1)]},
        {'A': [((1,0),1),((0,0),-1)], 'B': [((0,0),1),((0,1),1)], 'C': [((1,1),1)]},
        {'A': [((0,1),1),((1,1),-1)], 'B': [((1,0),1),((1,1),1)], 'C': [((0,0),1)]},
    ]
    
    for bi, big in enumerate(big_mults):
        base = bi * 7
        for si in range(7):
            r = base + si
            for (bI, bK), sA in big['A']:
                for i2 in range(2):
                    for k2 in range(2):
                        i = 2*bI + i2
                        k = 2*bK + k2
                        U[i*4+k, r] += sA * U2[i2*2+k2, si]
            for (bK, bJ), sB in big['B']:
                for k2 in range(2):
                    for j2 in range(2):
                        k = 2*bK + k2
                        j = 2*bJ + j2
                        V[k*4+j, r] += sB * V2[k2*2+j2, si]
            for (bJ, bI), sC in big['C']:
                for j2 in range(2):
                    for i2 in range(2):
                        j = 2*bJ + j2
                        i = 2*bI + i2
                        W[j*4+i, r] += sC * W2[j2*2+i2, si]
    return U, V, W

def _verify(U, V, W, rank, n=4, m=4, p=9):
    """Verify decomposition with random test."""
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    a_vecs = np.einsum('ikr,ik->ir', Ur, A)
    b_vecs = np.einsum('kjr,kj->jr', Vr, B)
    C_comp = np.einsum('jir,ir,jr->ij', Wr, a_vecs, b_vecs)
    return np.allclose(C_comp, C_expected, atol=1e-10)

def _build_4x4x9_v1():
    """Two 4x4 Strassen blocks + 1 direct column. Rank = 49+49+16 = 114."""
    U44, V44, W44 = _build_4x4x4_strassen()
    R = 114
    U = np.zeros((16, R))
    V = np.zeros((36, R))
    W = np.zeros((36, R))
    
    # Block 1: cols 0-3
    U[:, 0:49] = U44
    for k in range(4):
        for j in range(4):
            V[k*9+j, 0:49] = V44[k*4+j, :]
    for j in range(4):
        for i in range(4):
            W[j*4+i, 0:49] = W44[j*4+i, :]
    
    # Block 2: cols 4-7
    U[:, 49:98] = U44
    for k in range(4):
        for j in range(4):
            V[k*9+(j+4), 49:98] = V44[k*4+j, :]
    for j in range(4):
        for i in range(4):
            W[(j+4)*4+i, 49:98] = W44[j*4+i, :]
    
    # Block 3: col 8 direct
    for i in range(4):
        for k in range(4):
            r = 98 + i*4 + k
            U[i*4+k, r] = 1
            V[k*9+8, r] = 1
            W[8*4+i, r] = 1
    
    return U, V, W, R

def _build_4x4x9_v2():
    """One 4x4 Strassen + one 4x4 Strassen + one 4x5 with 4x4 Strassen + 1 direct.
    Actually: split 9 cols as 4+4+1. Same as v1. Let me try 4+5 split.
    4x5: use 4x4 Strassen for first 4 cols + 1 direct = 49+16 = 65
    Total: 49 + 65 = 114. Same rank.
    Let me try a different approach: 4x4 Strassen for cols 0-3,
    and for cols 4-8 (5 cols), use a custom 4x5 decomposition.
    """
    # For 4x5: split into 4x4 + 4x1
    # 4x4 Strassen: rank 49
    # 4x1 direct: rank 16
    # Total for 4x5: 65
    # Grand total: 49 + 65 = 114
    # Same as v1. No improvement.
    # Try: 4x4 Strassen for cols 0-3, 4x4 Strassen for cols 4-7, 4x1 for col 8.
    # That's v1.
    # Alternative: treat all 9 cols together with a block approach.
    # Split A into 2x2 blocks of 2x2, B into 2x(9) blocks... complex.
    # Let's just use v1.
    return _build_4x4x9_v1()

def _build_4x4x9_v3():
    """Try to use a 4x4x9 decomposition that shares structure.
    Idea: The 4x4x9 tensor can be seen as 9 copies of 4x4x1 tensors.
    Each 4x4x1 has rank at most 16 (direct) or could be lower.
    Actually, 4x4x1 is just a single column of the product, which is a
    linear combination of 4 vectors (columns of A) with coefficients from
    the corresponding column of B. The tensor T[i,k,j] for fixed j is
    just A[i,k] * B[k,j], which has rank 4 (as a 4x4x1 tensor).
    So 9 columns * rank 4 each = 36? No, that's not how it works.
    
    Actually, the full 4x4x9 tensor T[i,k,j] = delta (it's the identity in k
    for the multiplication). The rank of the 4x4x9 multiplication tensor is
    what we're trying to minimize.
    
    Known: rank(4x4x4) = 49 (Strassen).
    For 4x4x9, we can use:
    - 4x4x4 Strassen for first 4 cols: rank 49
    - 4x4x4 Strassen for next 4 cols: rank 49
    - 4x4x1 direct for last col: rank 16
    Total: 114
    
    Can we do better? What if we use a 4x4x5 decomposition?
    4x4x5 = 4x4x4 + 4x4x1 = 49 + 16 = 65
    4x4x9 = 4x4x4 + 4x4x5 = 49 + 65 = 114. Same.
    
    Or 4x4x9 = 4x4x4 + 4x4x4 + 4x4x1 = 49+49+16 = 114.
    
    What about using a non-block approach? The rank of the 4x4x9 tensor
    might be less than 114. Let's think...
    
    Actually, I recall that for 4x4xn, the rank is known for small n.
    For n=1: rank = 16 (direct, since 4x4x1 is just a matrix-vector product
    which has rank 16 in the tensor sense... wait, no.
    
    Let me reconsider. The multiplication tensor for m x n * n x p has
    entries T[i,k,j] which is 1 if the indices match the multiplication
    pattern. For 4x4x9, T is a 4x4x9 tensor where T[i,k,j] = 1 for all
    valid (i,k,j) in the multiplication.
    
    The rank of this tensor is the minimum number of rank-1 tensors needed.
    A rank-1 tensor is u_i * v_k * w_j.
    
    For 4x4x4, Strassen gives rank 49.
    For 4x4x9, the trivial bound is 4*4*9 = 144.
    Using block decomposition: 49 + 49 + 16 = 114.
    
    Can we do 4x4x9 in less than 114? 
    One approach: use 4x4x4 Strassen for the first 8 columns (two blocks)
    and handle the 9th column. That's what we have.
    
    Another approach: maybe we can find a decomposition that shares
    some rank-1 terms between blocks. But Strassen's decomposition is
    already optimal for 4x4x4, so sharing is unlikely to help.
    
    Let me just go with 114 and make sure it's correct.
    """
    return _build_4x4x9_v1()

def _direct_decomposition(n=4, m=4, p=9):
    """Direct (schoolbook) decomposition. Rank = n*m*p = 144."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    idx = 0
    for i in range(n):
        for j in range(p):
            for k in range(m):
                U[i*m+k, idx] = 1
                V[k*p+j, idx] = 1
                W[j*n+i, idx] = 1
                idx += 1
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    
    # Build the 114-rank decomposition
    U, V, W, R = _build_4x4x9_v1()
    
    # Verify with multiple random trials for robustness
    np.random.seed(42)
    verified = True
    for trial in range(3):
        if not _verify(U, V, W, R, n, m, p):
            verified = False
            break
    
    if verified and (time.time() - start_time) < deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to direct
    U, V, W, R = _direct_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
