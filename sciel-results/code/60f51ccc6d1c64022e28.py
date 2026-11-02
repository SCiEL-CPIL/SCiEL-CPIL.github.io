"""Block-structured low-rank decomposition for 4x5x6 matrix multiplication."""
import time
import random

DIMENSIONS = [4, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0

def remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W, "rank": rank}

def _strassen_4x4x4_terms():
    """
    Rank-27 decomposition for 4x4x4 matrix multiplication.
    Uses Strassen's method: 4x4 = 2x2 blocks, 7 multiplications of 2x2x2.
    Each 2x2x2 uses Winograd rank 7, total 7*7 = 49? No.
    
    Actually, Strassen for 4x4x4: partition into 2x2 blocks in each dimension.
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B21-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each M is a 2x2x2 product, rank 7. Total rank = 7*7 = 49.
    But we can do better by sharing. Let's just use 49 for now and see.
    """
    # Winograd 2x2x2 terms
    winograd_222 = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        ([0,0,0,1], [-1,0,1,0], [-1,1,1,0]),
        ([1,1,0,0], [0,0,1,0], [1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,-1,1]),
        ([0,1,0,-1], [0,1,0,1], [-1,1,1,0]),
    ]
    
    # Strassen combinations for 4x4x4
    # A is 4x4: A11=A[0:2,0:2], A12=A[0:2,2:4], A21=A[2:4,0:2], A22=A[2:4,2:4]
    # B is 4x4: B11=B[0:2,0:2], B12=B[0:2,2:4], B21=B[2:4,0:2], B22=B[2:4,2:4]
    # C is 4x4: C11=C[0:2,0:2], etc.
    
    # For each Strassen multiplication, we need to:
    # 1. Form the linear combinations of A and B blocks
    # 2. Multiply (2x2x2, rank 7)
    # 3. Combine results into C blocks
    
    # U vector: 16 entries (4x4 A), V vector: 16 entries (4x4 B), W vector: 16 entries (4x4 C)
    # Index: A[i*4+j], B[j*4+k], C[k*4+i]
    
    terms = []
    
    # Strassen M1 = (A11+A22)*(B11+B22)
    # A11+A22: indices (0,0),(0,1),(1,0),(1,1) and (2,2),(2,3),(3,2),(3,3)
    # B11+B22: indices (0,0),(0,1),(1,0),(1,1) and (2,2),(2,3),(3,2),(3,3)
    # Result goes to C11 and C22
    for (u, v, w) in winograd_222:
        ufull = [0]*16
        vfull = [0]*16
        wfull = [0]*16
        # u: A11+A22
        ufull[0*4+0] = u[0]  # A[0,0]
        ufull[0*4+1] = u[1]  # A[0,1]
        ufull[1*4+0] = u[2]  # A[1,0]
        ufull[1*4+1] = u[3]  # A[1,1]
        ufull[2*4+2] = u[0]  # A[2,2]
        ufull[2*4+3] = u[1]  # A[2,3]
        ufull[3*4+2] = u[2]  # A[3,2]
        ufull[3*4+3] = u[3]  # A[3,3]
        # v: B11+B22
        vfull[0*4+0] = v[0]
        vfull[0*4+1] = v[1]
        vfull[1*4+0] = v[2]
        vfull[1*4+1] = v[3]
        vfull[2*4+2] = v[0]
        vfull[2*4+3] = v[1]
        vfull[3*4+2] = v[2]
        vfull[3*4+3] = v[3]
        # w: C11 and C22
        # C11 = w[0:4] mapped to C[0:2,0:2], C22 = w[4:8]... no, w is 4 entries for 2x2
        # w[0]=C11[0,0], w[1]=C11[0,1], w[2]=C11[1,0], w[3]=C11[1,1]
        wfull[0*4+0] = w[0]  # C[0,0]
        wfull[0*4+1] = w[1]  # C[0,1]
        wfull[1*4+0] = w[2]  # C[1,0]
        wfull[1*4+1] = w[3]  # C[1,1]
        wfull[2*4+2] = w[0]  # C[2,2]
        wfull[2*4+3] = w[1]  # C[2,3]
        wfull[3*4+2] = w[2]  # C[3,2]
        wfull[3*4+3] = w[3]  # C[3,3]
        terms.append((ufull, vfull, wfull))
    
    # M2 = (A21+A22)*B11
    # A21+A22: A[2,0],A[2,1],A[3,0],A[3,1] + A[2,2],A[2,3],A[3,2],A[3,3]
    # B11: B[0,0],B[0,1],B[1,0],B[1,1]
    # Result to C21
    for (u, v, w) in winograd_222:
        ufull = [0]*16
        vfull = [0]*16
        wfull = [0]*16
        ufull[2*4+0] = u[0]
        ufull[2*4+1] = u[1]
        ufull[3*4+0] = u[2]
        ufull[3*4+1] = u[3]
        ufull[2*4+2] = u[0]
        ufull[2*4+3] = u[1]
        ufull[3*4+2] = u[2]
        ufull[3*4+3] = u[3]
        vfull[0*4+0] = v[0]
        vfull[0*4+1] = v[1]
        vfull[1*4+0] = v[2]
        vfull[1*4+1] = v[3]
        wfull[2*4+0] = w[0]
        wfull[2*4+1] = w[1]
        wfull[3*4+0] = w[2]
        wfull[3*4+1] = w[3]
        terms.append((ufull, vfull, wfull))
    
    # M3 = A11*(B12-B22)
    for (u, v, w) in winograd_222:
        ufull = [0]*16
        vfull = [0]*16
        wfull = [0]*16
        ufull[0*4+0] = u[0]
        ufull[0*4+1] = u[1]
        ufull[1*4+0] = u[2]
        ufull[1*4+1] = u[3]
        vfull[0*4+2] = v[0]
        vfull[0*4+3] = v[1]
        vfull[1*4+2] = v[2]
        vfull[1*4+3] = v[3]
        vfull[2*4+2] = -v[0]
        vfull[2*4+3] = -v[1]
        vfull[3*4+2] = -v[2]
        vfull[3*4+3] = -v[3]
        wfull[0*4+2] = w[0]
        wfull[0*4+3] = w[1]
        wfull[1*4+2] = w[2]
        wfull[1*4+3] = w[3]
        terms.append((ufull, vfull, wfull))
    
    # M4 = A22*(B21-B11)
    for (u, v, w) in winograd_222:
        ufull = [0]*16
        vfull = [0]*16
        wfull = [0]*16
        ufull[2*4+2] = u[0]
        ufull[2*4+3] = u[1]
        ufull[3*4+2] = u[2]
        ufull[3*4+3] = u[3]
        vfull[2*4+0] = v[0]
        vfull[2*4+1] = v[1]
        vfull[3*4+0] = v[2]
        vfull[3*4+1] = v[3]
        vfull[0*4+0] = -v[0]
        vfull[0*4+1] = -v[1]
        vfull[1*4+0] = -v[2]
        vfull[1*4+1] = -v[3]
        wfull[2*4+0] = w[0]
        wfull[2*4+1] = w[1]
        wfull[3*4+0] = w[2]
        wfull[3*4+1] = w[3]
        terms.append((ufull, vfull, wfull))
    
    # M5 = (A11+A12)*B22
    for (u, v, w) in winograd_222:
        ufull = [0]*16
        vfull = [0]*16
        wfull = [0]*16
        ufull[0*4+0] = u[0]
        ufull[0*4+1] = u[1]
        ufull[1*4+0] = u[2]
        ufull[1*4+1] = u[3]
        ufull[0*4+2] = u[0]
        ufull[0*4+3] = u[1]
        ufull[1*4+2] = u[2]
        ufull[1*4+3] = u[3]
        vfull[2*4+2] = v[0]
        vfull[2*4+3] = v[1]
        vfull[3*4+2] = v[2]
        vfull[3*4+3] = v[3]
        wfull[0*4+2] = -w[0]
        wfull[0*4+3] = -w[1]
        wfull[1*4+2] = -w[2]
        wfull[1*4+3] = -w[3]
        terms.append((ufull, vfull, wfull))
    
    # M6 = (A21-A11)*(B11+B12)
    for (u, v, w) in winograd_222:
        ufull = [0]*16
        vfull = [0]*16
        wfull = [0]*16
        ufull[2*4+0] = u[0]
        ufull[2*4+1] = u[1]
        ufull[3*4+0] = u[2]
        ufull[3*4+1] = u[3]
        ufull[0*4+0] = -u[0]
        ufull[0*4+1] = -u[1]
        ufull[1*4+0] = -u[2]
        ufull[1*4+1] = -u[3]
        vfull[0*4+0] = v[0]
        vfull[0*4+1] = v[1]
        vfull[1*4+0] = v[2]
        vfull[1*4+1] = v[3]
        vfull[0*4+2] = v[0]
        vfull[0*4+3] = v[1]
        vfull[1*4+2] = v[2]
        vfull[1*4+3] = v[3]
        wfull[2*4+2] = w[0]
        wfull[2*4+3] = w[1]
        wfull[3*4+2] = w[2]
        wfull[3*4+3] = w[3]
        terms.append((ufull, vfull, wfull))
    
    # M7 = (A12-A22)*(B21+B22)
    for (u, v, w) in winograd_222:
        ufull = [0]*16
        vfull = [0]*16
        wfull = [0]*16
        ufull[0*4+2] = u[0]
        ufull[0*4+3] = u[1]
        ufull[1*4+2] = u[2]
        ufull[1*4+3] = u[3]
        ufull[2*4+2] = -u[0]
        ufull[2*4+3] = -u[1]
        ufull[3*4+2] = -u[2]
        ufull[3*4+3] = -u[3]
        vfull[2*4+0] = v[0]
        vfull[2*4+1] = v[1]
        vfull[3*4+0] = v[2]
        vfull[3*4+1] = v[3]
        vfull[2*4+2] = v[0]
        vfull[2*4+3] = v[1]
        vfull[3*4+2] = v[2]
        vfull[3*4+3] = v[3]
        wfull[0*4+0] = w[0]
        wfull[0*4+1] = w[1]
        wfull[1*4+0] = w[2]
        wfull[1*4+1] = w[3]
        terms.append((ufull, vfull, wfull))
    
    return terms

def _verify(U, V, W, rank):
    """Verify exact decomposition using random integer matrices."""
    n, m, p = DIMENSIONS
    for _ in range(5):
        if remaining() <= 0:
            return False
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        for i in range(n):
            for k in range(p):
                acc = 0
                for j in range(m):
                    for c in range(rank):
                        u_val = U[i * m + j][c]
                        if u_val == 0:
                            continue
                        acc += u_val * W[k * n + i][c] * V[j * p + k][c]
                if acc != C[i][k]:
                    return False
    return True

def _embed_4x4x4(terms, i_base, j_base, k_base, n, m, p):
    """Embed 4x4x4 terms into full-sized vectors."""
    full_terms = []
    for (u, v, w) in terms:
        ufull = [0] * (n * m)
        vfull = [0] * (m * p)
        wfull = [0] * (p * n)
        
        for a in range(4):
            for b in range(4):
                idx = (i_base + a) * m + (j_base + b)
                ufull[idx] = u[a * 4 + b]
        
        for b in range(4):
            for c in range(4):
                idx = (j_base + b) * p + (k_base + c)
                vfull[idx] = v[b * 4 + c]
        
        for c in range(4):
            for a in range(4):
                idx = (k_base + c) * n + (i_base + a)
                wfull[idx] = w[c * 4 + a]
        
        full_terms.append((ufull, vfull, wfull))
    return full_terms

def _winograd_2x2x2():
    """Returns 7 rank-1 terms for 2x2x2 matmul."""
    terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        ([0,0,0,1], [-1,0,1,0], [-1,1,1,0]),
        ([1,1,0,0], [0,0,1,0], [1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,-1,1]),
        ([0,1,0,-1], [0,1,0,1], [-1,1,1,0]),
    ]
    return terms

def _embed_block(terms, i_base, j_base, k_base, i_size, j_size, k_size, n, m, p):
    """Embed a block's terms into full-sized vectors."""
    full_terms = []
    for (u, v, w) in terms:
        ufull = [0] * (n * m)
        vfull = [0] * (m * p)
        wfull = [0] * (p * n)
        
        for a in range(i_size):
            for b in range(j_size):
                idx = (i_base + a) * m + (j_base + b)
                ufull[idx] = u[a * j_size + b]
        
        for b in range(j_size):
            for c in range(k_size):
                idx = (j_base + b) * p + (k_base + c)
                vfull[idx] = v[b * k_size + c]
        
        for c in range(k_size):
            for a in range(i_size):
                idx = (k_base + c) * n + (i_base + a)
                wfull[idx] = w[c * i_size + a]
        
        full_terms.append((ufull, vfull, wfull))
    return full_terms

def _build_from_terms(terms_U, terms_V, terms_W, n, m, p):
    """Build U, V, W matrices from lists of term vectors."""
    rank = len(terms_U)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(zip(terms_U, terms_V, terms_W)):
        for i in range(n * m):
            U[i][c] = u[i]
        for j in range(m * p):
            V[j][c] = v[j]
        for k in range(p * n):
            W[k][c] = w[k]
    
    return {"U": U, "V": V, "W": W, "rank": rank}

def _strassen_2x2x2():
    """
    Strassen's 7-term algorithm for 2x2x2 matrix multiplication.
    A is 2x2, B is 2x2, C = A*B is 2x2.
    Indexing: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
              B[0]=B00, B[1]=B01, B[2]=B10, B[3]=B11
              C[0]=C00, C[1]=C01, C[2]=C10, C[3]=C11
    """
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    terms = []
    
    # M1: U=(1,0,0,1), V=(1,0,0,1), W contributes to C00(+1), C11(+1)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M2: U=(0,0,1,1), V=(1,0,0,0), W contributes to C10(+1), C11(-1)
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    
    # M3: U=(1,0,0,0), V=(0,1,0,-1), W contributes to C01(+1)
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    
    # M4: U=(0,0,0,1), V=(-1,0,1,0), W contributes to C00(+1), C10(+1), C11(-1)... 
    # Actually C00=M1+M4-M5+M7, C10=M2+M4, C11=M1-M2+M3+M6
    # M4 contributes: C00(+1), C10(+1)
    terms.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
    
    # M5: U=(1,1,0,0), V=(0,0,1,0), W contributes to C00(-1), C01(+1)
    terms.append(([1,1,0,0], [0,0,1,0], [-1,1,0,0]))
    
    # M6: U=(-1,0,1,0), V=(1,1,0,0), W contributes to C11(+1)
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # M7: U=(0,1,0,-1), V=(0,1,0,1), W contributes to C00(+1)
    terms.append(([0,1,0,-1], [0,1,0,1], [1,0,0,0]))
    
    return terms

def _verify_strassen_222(terms):
    """Verify Strassen 2x2x2 terms."""
    import random
    for _ in range(10):
        A = [random.randint(-2, 2) for _ in range(4)]
        B = [random.randint(-2, 2) for _ in range(4)]
        C = [0, 0, 0, 0]
        C[0] = A[0]*B[0] + A[1]*B[2]
        C[1] = A[0]*B[1] + A[1]*B[3]
        C[2] = A[2]*B[0] + A[3]*B[2]
        C[3] = A[2]*B[1] + A[3]*B[3]
        for (u, v, w) in terms:
            m1 = sum(u[i]*A[i] for i in range(4))
            m2 = sum(v[i]*B[i] for i in range(4))
            for i in range(4):
                C[i] += u[i] * v[i] * w[i] * 0  # This is wrong, let me fix
        # Actually need to compute sum over terms of U_term[i]*V_term[j]*W_term[k]
        C = [0, 0, 0, 0]
        for (u, v, w) in terms:
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        C[k] += u[i] * v[j] * w[k] * A[i] * B[j]
        if C != [C[0], C[1], C[2], C[3]]:
            pass
        expected = [A[0]*B[0] + A[1]*B[2], A[0]*B[1] + A[1]*B[3], A[2]*B[0] + A[3]*B[2], A[2]*B[1] + A[3]*B[3]]
        if C != expected:
            return False
    return True

def _strategy_blockwise_strassen():
    """
    Block-wise decomposition:
    - Partition 4x5 into 2x2 blocks in i,j: 2x2 blocks of (2x2)
    - Partition 5x6 into 2x2 blocks in j,k: 2x3 blocks of (2x2)
    - Use Strassen for 2x2x2 sub-problems
    
    Strategy:
    For i-blocks 0,1 (each size 2) and j-blocks 0,1 (each size 2):
      For k-blocks 0,1,2 (each size 2):
        Use Strassen 2x2x2: rank 7
    Total: 2 * 2 * 3 * 7 = 84
    
    But we can do better by using the full 4x4x4 Strassen (rank 49) for the
    4x4x4 sub-problem, then handle the remaining j=4 column and k=4,5 columns.
    
    Alternative: Use 4x4x4 Strassen (rank 49) for i=0..3, j=0..3, k=0..3
    Then for j=4: need 4x1x6 sub-problem
    Then for j=0..3, k=4,5: need 4x4x2 sub-problem
    
    Let's try a different approach:
    - Use 2x2x2 Strassen blocks
    - i: 0-1, 2-3 (2 blocks)
    - j: 0-1, 2-3, 4 (3 blocks, last one size 1)
    - k: 0-1, 2-3, 4-5 (3 blocks)
    
    For 2x2x2 blocks: 2*2*3*7 = 84
    For 2x1x2 blocks (j=4): 2*1*3*? 
    For 2x2x1 blocks: not applicable since k is always even-sized
    
    Actually let me think differently. The key insight is:
    - 4x5x6 = (2+2) x (2+2+1) x (2+2+2)
    - We can use Strassen on 2x2x2 blocks
    - For the 1-wide j-block, we need a 2x1x2 decomposition
    
    2x1x2 matrix multiplication: C[i,k] = A[i,0]*B[0,k]
    This is just rank 4 (schoolbook) since it's a rank-1 outer product structure.
    Actually no: C[i,k] = sum_j A[i,j]*B[j,k], with j only 0 (size 1).
    So C[i,k] = A[i,0]*B[0,k], which is rank 4 (one term per (i,k) pair).
    
    Wait, that's just 2*2 = 4 terms for each 2x1x2 block.
    
    Total: 2*2*3*7 + 2*1*3*4 = 84 + 24 = 108. Same as hybrid.
    
    Let me try the 4x4x4 Strassen approach more carefully.
    """
    n, m, p = DIMENSIONS
    
    # Use 4x4x4 Strassen for i=0..3, j=0..3, k=0..3 (rank 49)
    strassen_terms = _strassen_4x4x4_terms()
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    # Embed 4x4x4 Strassen
    embedded = _embed_4x4x4(strassen_terms, 0, 0, 0, n, m, p)
    terms_U.extend([t[0] for t in embedded])
    terms_V.extend([t[1] for t in embedded])
    terms_W.extend([t[2] for t in embedded])
    
    # Now handle remaining:
    # j=4, k=0..5: 4x1x6 sub-problem, rank 4*6 = 24
    # j=0..3, k=4,5: 4x4x2 sub-problem
    
    # For 4x4x2, we can use Strassen-like approach:
    # Split into 2x2x2 blocks in i,j and handle k=4,5 separately
    # Actually, 4x4x2: C[i,k] = sum_j A[i,j]*B[j,k] for k in {4,5}
    # This is like computing two 4x4x1 products.
    # Each 4x4x1: C[i] = sum_j A[i,j]*B[j], rank 16 (schoolbook)
    # Two of them: rank 32
    # But we can share: for each j, B[j,4] and B[j,5] are separate
    # Actually we can use: for each (i,j) pair, we have A[i,j] * (B[j,4], B[j,5])
    # This gives C[i,4] += A[i,j]*B[j,4] and C[i,5] += A[i,j]*B[j,5]
    # That's 16 terms, each contributing to both C[i,4] and C[i,5]
    # Wait no, each term is rank-1: U[i*m+j]=1, V[j*p+4]=B[j,4]... no, V is fixed.
    
    # Actually in the tensor decomposition, each term has fixed U, V, W vectors.
    # For 4x4x2 sub-problem, schoolbook gives 4*4*2 = 32 terms.
    # Can we do better? 
    # C[i,4] = sum_j A[i,j]*B[j,4]
    # C[i,5] = sum_j A[i,j]*B[j,5]
    # We can combine: for each j, define D[j] = (B[j,4], B[j,5])
    # Then C[i,:] = sum_j A[i,j] * D[j]
    # This is a 4x4 matrix times 4x2 matrix, which is rank 16 if we allow
    # W to have 2 entries per term. But in our format, each term has one W vector.
    # So we need separate terms for k=4 and k=5, giving 32 terms.
    # Unless we use a Strassen-like trick...
    
    # For now, use schoolbook for the remaining parts:
    # j=4, k=0..5: 4*6 = 24 terms
    for i in range(n):
        for k in range(p):
            ufull = [0] * (n * m)
            vfull = [0] * (m * p)
            wfull = [0] * (p * n)
            ufull[i * m + 4] = 1
            vfull[4 * p + k] = 1
            wfull[k * n + i] = 1
            terms_U.append(ufull)
            terms_V.append(vfull)
            terms_W.append(wfull)
    
    # j=0..3, k=4,5: 4*4*2 = 32 terms
    for i in range(n):
        for j in range(4):
            for k in range(4, p):
                ufull = [0] * (n * m)
                vfull = [0] * (m * p)
                wfull = [0] * (p * n)
                ufull[i * m + j] = 1
                vfull[j * p + k] = 1
                wfull[k * n + i] = 1
                terms_U.append(ufull)
                terms_V.append(vfull)
                terms_W.append(wfull)
    
    return _build_from_terms(terms_U, terms_V, terms_W, n, m, p)

def _strategy_4x4x2_strassen():
    """
    Use 4x4x4 Strassen (rank 49) for i=0..3, j=0..3, k=0..3
    Use a rank-28 decomposition for the 4x4x2 sub-problem (j=0..3, k=4,5)
    Use schoolbook for j=4 (rank 24)
    Total: 49 + 28 + 24 = 101
    
    For 4x4x2: C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k] for k in {4,5}
    We can view this as: for each i, C[i,:] = A[i,:] * B[:,{4,5}]
    where A[i,:] is 1x4 and B[:,{4,5}] is 4x2.
    
    Using Strassen on the 4x4 part:
    Split A into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    Split B into 2x2 blocks: B11, B12 (each 2x2, but B is 4x2 so blocks are 2x2)
    
    Actually B is 4x2, so we can split into B_top (2x2) and B_bot (2x2).
    C_top = A_top * B_top + A_bot * B_bot... no, that's not right.
    
    C[i,k] for i in 0,1: C[i,k] = sum_j A[i,j]*B[j,k]
    = A[i,0:2] * B[0:2,k] + A[i,2:4] * B[2:4,k]
    
    So C_top (2x2) = A_top (2x4) * B_top (4x2)
    C_bot (2x2) = A_bot (2x4) * B_bot (4x2)
    
    A_top is 2x4, B_top is 4x2. This is a 2x4x2 sub-problem.
    Rank of 2x4x2: schoolbook is 2*4*2 = 16.
    Can we do better? 
    C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    We can use Strassen on the 2x2x2 sub-blocks:
    Split A[i,0:2] and A[i,2:4], B[0:2,k] and B[2:4,k]
    C[i,k] = A[i,0:2]*B[0:2,k] + A[i,2:4]*B[2:4,k]
    
    Each 2x2x2 uses rank 7, so total 14 for the 2x4x2.
    Two of these (for C_top and C_bot): 28.
    
    So 4x4x2 has rank 28 using Strassen on 2x2x2 sub-blocks.
    """
    n, m, p = DIMENSIONS
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    # 4x4x4 Strassen for i=0..3, j=0..3, k=0..3 (rank 49)
    strassen_terms = _strassen_4x4x4_terms()
    embedded = _embed_4x4x4(strassen_terms, 0, 0, 0, n, m, p)
    terms_U.extend([t[0] for t in embedded])
    terms_V.extend([t[1] for t in embedded])
    terms_W.extend([t[2] for t in embedded])
    
    # 4x4x2 sub-problem for j=0..3, k=4,5 using Strassen (rank 28)
    # C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k] for k in {4,5}
    # Split into: C[i,k] = A[i,0:2]*B[0:2,k] + A[i,2:4]*B[2:4,k]
    # Each 2x2x2 uses Strassen rank 7
    
    strassen_222 = _strassen_2x2x2()
    
    # For i=0,1 (top), j=0,1 (left), k=4,5: 2x2x2 Strassen
    # U: A[i,j] for i in 0,1 and j in 0,1
    # V: B[j,k] for j in 0,1 and k in 4,5
    # W: C[i,k] for i in 0,1 and k in 4,5
    for (u, v, w) in strassen_222:
        ufull = [0] * (n * m)
        vfull = [0] * (m * p)
        wfull = [0] * (p * n)
        for a in range(2):
            for b in range(2):
                ufull[a * m + b] = u[a * 2 + b]
        for b in range(2):
            for c in range(2):
                vfull[(b) * p + (4 + c)] = v[b * 2 + c]
        for c in range(2):
            for a in range(2):
                wfull[(4 + c) * n + a] = w[c * 2 + a]
        terms_U.append(ufull)
        terms_V.append(vfull)
        terms_W.append(wfull)
    
    # For i=0,1 (top), j=2,3 (right), k=4,5: 2x2x2 Strassen
    for (u, v, w) in strassen_222:
        ufull = [0] * (n * m)
        vfull = [0] * (m * p)
        wfull = [0] * (p * n)
        for a in range(2):
            for b in range(2):
                ufull[a * m + (2 + b)] = u[a * 2 + b]
        for b in range(2):
            for c in range(2):
                vfull[(2 + b) * p + (4 + c)] = v[b * 2 + c]
        for c in range(2):
            for a in range(2):
                wfull[(4 + c) * n + a] = w[c * 2 + a]
        terms_U.append(ufull)
        terms_V.append(vfull)
        terms_W.append(wfull)
    
    # For i=2,3 (bottom), j=0,1 (left), k=4,5: 2x2x2 Strassen
    for (u, v, w) in strassen_222:
        ufull = [0] * (n * m)
        vfull = [0] * (m * p)
        wfull = [0] * (p * n)
        for a in range(2):
            for b in range(2):
                ufull[(2 + a) * m + b] = u[a * 2 + b]
        for b in range(2):
            for c in range(2):
                vfull[(b) * p + (4 + c)] = v[b * 2 + c]
        for c in range(2):
            for a in range(2):
                wfull[(4 + c) * n + (2 + a)] = w[c * 2 + a]
        terms_U.append(ufull)
        terms_V.append(vfull)
        terms_W.append(wfull)
    
    # For i=2,3 (bottom), j=2,3 (right), k=4,5: 2x2x2 Strassen
    for (u, v, w) in strassen_222:
        ufull = [0] * (n * m)
        vfull = [0] * (m * p)
        wfull = [0] * (p * n)
        for a in range(2):
            for b in range(2):
                ufull[(2 + a) * m + (2 + b)] = u[a * 2 + b]
        for b in range(2):
            for c in range(2):
                vfull[(2 + b) * p + (4 + c)] = v[b * 2 + c]
        for c in range(2):
            for a in range(2):
                wfull[(4 + c) * n + (2 + a)] = w[c * 2 + a]
        terms_U.append(ufull)
        terms_V.append(vfull)
        terms_W.append(wfull)
    
    # Schoolbook for j=4, k=0..5 (rank 24)
    for i in range(n):
        for k in range(p):
            ufull = [0] * (n * m)
            vfull = [0] * (m * p)
            wfull = [0] * (p * n)
            ufull[i * m + 4] = 1
            vfull[4 * p + k] = 1
            wfull[k * n + i] = 1
            terms_U.append(ufull)
            terms_V.append(vfull)
            terms_W.append(wfull)
    
    return _build_from_terms(terms_U, terms_V, terms_W, n, m, p)

def _strategy_hybrid():
    """
    Use 2x2x2 Winograd blocks for i:0-3, j:0-3, k:0-5,
    then schoolbook for j=4 (all i,k).
    Total: 12 blocks * 7 + 24 = 84 + 24 = 108
    """
    n, m, p = DIMENSIONS
    winograd_222 = _winograd_2x2x2()
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    for ib in range(2):
        for jb in range(2):
            for kb in range(3):
                i_base = 2 * ib
                j_base = 2 * jb
                k_base = 2 * kb
                
                embedded = _embed_block(winograd_222, i_base, j_base, k_base, 2, 2, 2, n, m, p)
                terms_U.extend([t[0] for t in embedded])
                terms_V.extend([t[1] for t in embedded])
                terms_W.extend([t[2] for t in embedded])
    
    for i in range(n):
        for k in range(p):
            ufull = [0] * (n * m)
            vfull = [0] * (m * p)
            wfull = [0] * (p * n)
            ufull[i * m + 4] = 1
            vfull[4 * p + k] = 1
            wfull[k * n + i] = 1
            terms_U.append(ufull)
            terms_V.append(vfull)
            terms_W.append(wfull)
    
    return _build_from_terms(terms_U, terms_V, terms_W, n, m, p)

def construct():
    n, m, p = DIMENSIONS
    
    candidates = []
    
    if remaining() > 30:
        try:
            candidates.append(_strategy_4x4x2_strassen())
        except Exception as e:
            pass
    
    if remaining() > 30:
        try:
            candidates.append(_strategy_strassen_444())
        except Exception as e:
            pass
    
    if remaining() > 10:
        try:
            candidates.append(_strategy_hybrid())
        except Exception as e:
            pass
    
    candidates.append(_schoolbook())
    
    best = None
    for cand in candidates:
        if remaining() <= 0:
            break
        rank = cand["rank"]
        U, V, W = cand["U"], cand["V"], cand["W"]
        if _verify(U, V, W, rank):
            if best is None or rank < best["rank"]:
                best = cand
    
    if best is None:
        best = candidates[-1]
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
