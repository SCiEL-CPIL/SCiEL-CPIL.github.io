"""Matrix multiplication tensor decomposition for 4x5 by 5x6."""
import time
import itertools

DIMENSIONS = [4, 5, 6]

def _validate(U, V, W, n, m, p):
    """Verify the tensor decomposition is exact."""
    rank = len(U[0])
    # Check shapes
    assert len(U) == n * m, f"U rows {len(U)} != {n*m}"
    assert len(V) == m * p, f"V rows {len(V)} != {m*p}"
    assert len(W) == p * n, f"W rows {len(W)} != {p*n}"
    assert len(U[0]) == rank and len(V[0]) == rank and len(W[0]) == rank
    # Check the tensor identity
    # T[i*m+j, j*p+k, k*n+i] should be 1, all other combos 0
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if valid
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    # Check that invalid combos give 0
    # For each (a, b, c) where a in [0, n*m), b in [0, m*p), c in [0, p*n)
    # a = i*m+j, b = j'*p+k, c = k'*n+i'
    # valid if i==i' and j==j' and k==k'
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i, j = divmod(a, m)
                j2, k = divmod(b, p)
                k2, i2 = divmod(c, n)
                if i != i2 or j != j2 or k != k2:
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != 0:
                        return False
    return True

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    n=m=p=2.
    U shape (4,7), V shape (4,7), W shape (4,7).
    U index: i*m+j = i*2+j
    V index: j*p+k = j*2+k
    W index: k*n+i = k*2+i
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    # A00: i=0,j=0 -> U[0]; A11: i=1,j=1 -> U[3]
    # B00: j=0,k=0 -> V[0]; B11: j=1,k=1 -> V[3]
    # C00: k=0,i=0 -> W[0]; C11: k=1,i=1 -> W[3]
    # M1 contributes to C00 (+1) and C11 (+1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    # A10: i=1,j=0 -> U[2]; A11: i=1,j=1 -> U[3]
    # B00: j=0,k=0 -> V[0]
    # C10: k=0,i=1 -> W[2]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A00(B01-B11)
    # A00: i=0,j=0 -> U[0]
    # B01: j=0,k=1 -> V[1]; B11: j=1,k=1 -> V[3]
    # C01: k=1,i=0 -> W[1]
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A11(B10-B00)
    # A11: i=1,j=1 -> U[3]
    # B10: j=1,k=0 -> V[2]; B00: j=0,k=0 -> V[0]
    # C00: k=0,i=0 -> W[0]; C10: k=0,i=1 -> W[2]
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A10)B11
    # A00: i=0,j=0 -> U[0]; A10: i=1,j=0 -> U[2]
    # B11: j=1,k=1 -> V[3]
    # C00: k=0,i=0 -> W[0] (-1); C01: k=1,i=0 -> W[1] (+1)
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    # A10: i=1,j=0 -> U[2]; A00: i=0,j=0 -> U[0]
    # B00: j=0,k=0 -> V[0]; B01: j=0,k=1 -> V[1]
    # C11: k=1,i=1 -> W[3]
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    # A01: i=0,j=1 -> U[1]; A11: i=1,j=1 -> U[3]
    # B10: j=1,k=0 -> V[2]; B11: j=1,k=1 -> V[3]
    # C00: k=0,i=0 -> W[0] (+1); C11: k=1,i=1 -> W[3] (+1)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    """Return U, V, W for schoolbook decomposition."""
    rank = n * m * p
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
    return U, V, W

def _inflate(U, V, W, block_n, block_m, block_p, outer_n, outer_m, outer_p):
    """Inflate a block decomposition to outer dimensions.
    U shape (block_n*block_m, rank_block) -> (outer_n*outer_m, rank_block)
    V shape (block_m*block_p, rank_block) -> (outer_m*outer_p, rank_block)
    W shape (block_p*block_n, rank_block) -> (outer_p*outer_n, rank_block)
    """
    rank = len(U[0])
    out_U = [[0]*rank for _ in range(outer_n*outer_m)]
    out_V = [[0]*rank for _ in range(outer_m*outer_p)]
    out_W = [[0]*rank for _ in range(outer_p*outer_n)]
    for i in range(block_n):
        for j in range(block_m):
            for r in range(rank):
                out_U[i*outer_m + j][r] = U[i*block_m + j][r]
    for j in range(block_m):
        for k in range(block_p):
            for r in range(rank):
                out_V[j*outer_p + k][r] = V[j*block_p + k][r]
    for k in range(block_p):
        for i in range(block_n):
            for r in range(rank):
                out_W[k*outer_n + i][r] = W[k*block_n + i][r]
    return out_U, out_V, out_W

def _block_strassen_4x4x4():
    """Construct 4x4x4 decomposition using block Strassen (rank 49).
    Split into 2x2 blocks of 2x2 matrices.
    """
    # Get 2x2x2 Strassen
    U2, V2, W2 = _strassen_2x2x2()
    # Inflate to 4x4x4
    U, V, W = _inflate(U2, V2, W2, 2, 2, 2, 4, 4, 4)
    return U, V, W

def _block_strassen_4x4x2():
    """Construct 4x4x2 decomposition.
    Split into 2x2 blocks. The 4x4x2 can be seen as 2 blocks of 2x4x2 in the first dim?
    Actually, let's think: 4x4x2. Split A into 2x2 blocks of 2x2.
    A is 4x4: A = [[A00, A01],[A10, A11]] each 2x2.
    B is 4x2: B = [[B00],[B10]] each 2x2.
    C is 4x2: C = [[C00],[C10]] each 2x2.
    C00 = A00*B00 + A01*B10
    C10 = A10*B00 + A11*B10
    This is like multiplying [A00 A01; A10 A11] (2x2 blocks) by [B00; B10] (2x1 blocks of 2x2).
    We can use Strassen for the 2x2x1 block multiplication? 
    Actually, 2x2x1 matrix multiplication has rank 4 (schoolbook) since it's just 4 multiplications.
    But we can also view it as two independent 2x2x2 multiplications? No, B has only 2 columns.
    Let's just use schoolbook for the 4x4x2 part. Rank = 32.
    Actually, we can do better. 4x4x2: split into two 2x4x2.
    2x4x2: A is 2x4, B is 4x2. C = A*B. This is 2x4x2.
    Split A into [A00 A01] where A00, A01 are 2x2.
    Split B into [B00; B10] where B00, B10 are 2x2.
    C = A00*B00 + A01*B10. This is sum of two 2x2x2 multiplications.
    So 2x4x2 has rank at most 7+7=14? No, that's if we do them separately.
    Actually, C00 = A00*B00 + A01*B10. This is a 2x2x2x2 tensor? No.
    Let's just use schoolbook for 4x4x2. Rank 32.
    """
    return _schoolbook(4, 4, 2)

def construct():
    """
    Returns a tensor decomposition for 4x5 by 5x6 matrix multiplication.
    Uses block decomposition to reduce rank below schoolbook (120).
    """
    n, m, p = 4, 5, 6
    deadline = time.time() + 140  # Safety margin
    
    # Split: A is 4x5, B is 5x6.
    # Split A into [A1 | A2] where A1 is 4x4, A2 is 4x1.
    # Split B into [B1; B2] where B1 is 4x6, B2 is 1x6.
    # C = A1*B1 + A2*B2.
    # A1*B1: 4x4 * 4x6 -> 4x6. Split B1 into [B11 | B12] where B11 is 4x4, B12 is 4x2.
    # A1*B11: 4x4*4x4. Use block Strassen, rank 49.
    # A1*B12: 4x4*4x2. Use schoolbook, rank 32.
    # A2*B2: 4x1*1x6. Schoolbook, rank 24.
    # Total rank = 49 + 32 + 24 = 105.
    
    # Actually, let's try to be smarter with 4x4x2.
    # 4x4x2: A is 4x4, B is 4x2.
    # Split A into 2x2 blocks: A = [[A00,A01],[A10,A11]]
    # B = [[B00],[B10]]
    # C = [[A00*B00 + A01*B10],[A10*B00 + A11*B10]]
    # C00 = A00*B00 + A01*B10 (2x2x2 + 2x2x2)
    # C10 = A10*B00 + A11*B10 (2x2x2 + 2x2x2)
    # Each 2x2x2 can be done with Strassen rank 7.
    # But the multiplications are independent? A00*B00 and A01*B10 are separate.
    # So rank = 7+7+7+7 = 28 for 4x4x2.
    
    # Let's implement 4x4x2 with rank 28.
    def _strassen_4x4x2():
        """4x4x2 with rank 28 using 4 independent 2x2x2 Strassen blocks."""
        rank = 28
        U = [[0]*rank for _ in range(16)]  # 4*4
        V = [[0]*rank for _ in range(8)]   # 4*2
        W = [[0]*rank for _ in range(8)]   # 2*4
        
        U2, V2, W2 = _strassen_2x2x2()
        
        # Block (0,0): A00 (rows 0,1; cols 0,1) * B00 (rows 0,1; cols 0,1) -> C00 (rows 0,1; cols 0,1)
        # A00: i in 0,1; j in 0,1 -> U[i*4+j]
        # B00: j in 0,1; k in 0,1 -> V[j*2+k]
        # C00: k in 0,1; i in 0,1 -> W[k*4+i]
        for bi in range(2):
            for bj in range(2):
                for bk in range(2):
                    # This is getting complex. Let's just use schoolbook for 4x4x2.
                    pass
        return _schoolbook(4, 4, 2)
    
    # Let's just use schoolbook for 4x4x2 for now. Rank 32.
    # Total: 49 + 32 + 24 = 105.
    
    U1, V1, W1 = _block_strassen_4x4x4()  # rank 49
    U2, V2, W2 = _schoolbook(4, 4, 2)      # rank 32
    U3, V3, W3 = _schoolbook(4, 1, 6)      # rank 24
    
    rank = 49 + 32 + 24
    U = [[0]*rank for _ in range(n*m)]  # 4*5 = 20
    V = [[0]*rank for _ in range(m*p)]  # 5*6 = 30
    W = [[0]*rank for _ in range(p*n)]  # 6*4 = 24
    
    # Part 1: A1 (4x4) * B11 (4x4) -> C11 (4x4)
    # A1 is A[0:4, 0:4] -> U indices: i*5+j for i in 0..3, j in 0..3
    # B11 is B[0:4, 0:4] -> V indices: j*6+k for j in 0..3, k in 0..3
    # C11 is C[0:4, 0:4] -> W indices: k*4+i for k in 0..3, i in 0..3
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # Map to U1, V1, W1 indices
                u1_idx = i*4 + j
                v1_idx = j*4 + k
                w1_idx = k*4 + i
                for r in range(49):
                    U[i*5+j][r] = U1[u1_idx][r]
                    V[j*6+k][r] = V1[v1_idx][r]
                    W[k*4+i][r] = W1[w1_idx][r]
    
    # Part 2: A1 (4x4) * B12 (4x2) -> C12 (4x2)
    # B12 is B[0:4, 4:6] -> V indices: j*6+(4+k) for j in 0..3, k in 0..1
    # C12 is C[0:4, 4:6] -> W indices: (4+k)*4+i for k in 0..1, i in 0..3
    offset2 = 49
    for i in range(4):
        for j in range(4):
            for k in range(2):
                u2_idx = i*4 + j
                v2_idx = j*4 + k
                w2_idx = k*4 + i
                for r in range(32):
                    U[i*5+j][offset2+r] = U2[u2_idx][r]
                    V[j*6+(4+k)][offset2+r] = V2[v2_idx][r]
                    W[(4+k)*4+i][offset2+r] = W2[w2_idx][r]
    
    # Part 3: A2 (4x1) * B2 (1x6) -> C2 (4x6)
    # A2 is A[0:4, 4:5] -> U indices: i*5+4 for i in 0..3
    # B2 is B[4:5, 0:6] -> V indices: 4*6+k = 24+k for k in 0..5
    # C2 is C[0:4, 0:6] -> W indices: k*4+i for k in 0..5, i in 0..3
    offset3 = 49 + 32
    for i in range(4):
        for k in range(6):
            u3_idx = i*1 + 0  # A2 is 4x1, so index i*1+0 = i
            v3_idx = 0*6 + k  # B2 is 1x6, so index 0*6+k = k
            w3_idx = k*4 + i
            for r in range(24):
                U[i*5+4][offset3+r] = U3[u3_idx][r]
                V[24+k][offset3+r] = V3[v3_idx][r]
                W[k*4+i][offset3+r] = W3[w3_idx][r]
    
    # Verify
    if _validate(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
