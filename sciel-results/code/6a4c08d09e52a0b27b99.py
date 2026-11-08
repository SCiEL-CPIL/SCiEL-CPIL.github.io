import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U shape: (4, 7)  # indices i*2+j
    V shape: (4, 7)  # indices j*2+k
    W shape: (4, 7)  # indices k*2+i
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1=(A11-A10)(B00+B01)
    U[3][0]=1; U[2][0]=-1; V[0][0]=1; V[1][0]=1; W[0][0]=1; W[2][0]=-1; W[3][0]=1
    # M2=(A10+A11)(B10+B11)
    U[2][1]=1; U[3][1]=1; V[2][1]=1; V[3][1]=1; W[2][1]=1; W[3][1]=1
    # M3=(A00+A10)(B00)
    U[0][2]=1; U[2][2]=1; V[0][2]=1; W[2][2]=1; W[3][2]=1
    # M4=(A11+A01)(B11)
    U[3][3]=1; U[1][3]=1; V[3][3]=1; W[0][3]=1; W[2][3]=-1
    # M5=(A00-A10)(B01+B11)
    U[0][4]=1; U[2][4]=-1; V[1][4]=1; V[3][4]=1; W[0][4]=-1; W[2][4]=1
    # M6=(A01+A11)(B10-B00)
    U[1][5]=1; U[3][5]=1; V[2][5]=1; V[0][5]=-1; W[2][5]=-1; W[3][5]=-1
    # M7=(A00+A01)(B00-B10)
    U[0][6]=1; U[1][6]=1; V[0][6]=1; V[2][6]=-1; W[0][6]=1; W[2][6]=-1
    
    return U, V, W

def _schoolbook_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 4 (schoolbook).
    U shape: (4, 4), V shape: (4, 4), W shape: (4, 4)
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(4)]
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                r = i*2 + j*1 + k  # 0..7? No, rank 4
                # Actually schoolbook for 2x2x2 has rank 4
                # Let me re-index: r = i*2 + j + k? No.
                # Standard: for each (i,j,k), one rank-1 term
                # But 2x2x2 has 8 terms in schoolbook... 
                # Wait, schoolbook for n×n×n is n^3 terms. For 2x2x2 that's 8.
                # But we want rank 4? That's not schoolbook.
                # Let me just do rank 8 schoolbook.
                pass
    
    # Actually, let me just do rank 8 schoolbook for 2x2x2
    U = [[0]*8 for _ in range(4)]
    V = [[0]*8 for _ in range(4)]
    W = [[0]*8 for _ in range(4)]
    
    r = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j][r] = 1
                V[j*2+k][r] = 1
                W[k*2+i][r] = 1
                r += 1
    
    return U, V, W

def _kronecker_4x4x4():
    """
    Construct 4x4x4 using Kronecker product of 2x2x2 (Strassen rank 7) and 2x2x2 (schoolbook rank 8).
    Rank = 7 * 8 = 56. That's not good.
    
    Let me think differently. We want rank 20 for 4x4x4.
    
    Actually, the known best for 4x4x4 is 49 (Strassen recursive) or possibly lower.
    Let me just use Strassen recursive: 4x4x4 = 2*(4x2x2) type decomposition.
    
    Alternative: Use the fact that 4x4x4 can be done with 49 multiplications via Strassen.
    Let me implement that.
    """
    # 4x4x4 via Strassen: split into four 2x2 blocks
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2
    # C = [[C00, C01], [C10, C11]] where each is 2x2
    # C00 = (A00+A11)(B00+B11) - A11*B11 - A00*B00
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = (A10+A11)(B10+B11) - A11*B11 - A10*B10
    # This uses 7 multiplications per 2x2 block, but they're entangled.
    # Total: 4*7 = 28? No, Strassen for 4x4x4 is 49.
    # Actually Strassen for n×n×n where n=2^k uses 7^(k+1) multiplications.
    # For n=4 (k=1): 7^2 = 49. Yes.
    
    # Let me implement 4x4x4 with rank 49 using recursive Strassen.
    # This is complex. Let me use a simpler approach:
    # Just use the Kronecker product of two 2x2x2 Strassen decompositions.
    # Rank = 7*7 = 49.
    
    U1, V1, W1 = _strassen_2x2x2()
    
    total_rank = 49
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(16)]
    W = [[0]*total_rank for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            
            # U: index (i1*2+i0, j1*2+j0) -> (i1*4+j1)*2 + (i0*2+j0)? 
            # A is 4x4, indexed by (i, j) where i,j in 0..3
            # Split: i = i1*2 + i0, j = j1*2 + j0
            # U index: i*4 + j = (i1*2+i0)*4 + (j1*2+j0) = i1*8 + i0*4 + j1*2 + j0
            for i1 in range(2):
                for i0 in range(2):
                    for j1 in range(2):
                        for j0 in range(2):
                            a = i1*8 + i0*4 + j1*2 + j0
                            a1 = i1*2 + j1
                            a0 = i0*2 + j0
                            U[a][col] = U1[a1][r1] * U1[a0][r2]
                            
            # V: index (j, k) where j,k in 0..3
            # j = j1*2+j0, k = k1*2+k0
            # V index: j*4+k = j1*8+j0*4+k1*2+k0
            for j1 in range(2):
                for j0 in range(2):
                    for k1 in range(2):
                        for k0 in range(2):
                            b = j1*8 + j0*4 + k1*2 + k0
                            b1 = j1*2 + k1
                            b0 = j0*2 + k0
                            V[b][col] = V1[b1][r1] * V1[b0][r2]
                            
            # W: index (k, i) where k,i in 0..3
            # k = k1*2+k0, i = i1*2+i0
            # W index: k*4+i = k1*8+k0*4+i1*2+i0
            for k1 in range(2):
                for k0 in range(2):
                    for i1 in range(2):
                        for i0 in range(2):
                            c = k1*8 + k0*4 + i1*2 + i0
                            c1 = k1*2 + i1
                            c0 = k0*2 + i0
                            W[c][col] = W1[c1][r1] * W1[c0][r2]
                            
    return U, V, W

def _build_4x4x8():
    """
    Construct 4x4x8 by splitting into two 4x4x4 sub-problems.
    A is 4x4, B is 4x8, C is 4x8.
    Split B into B_left (4x4) and B_right (4x4).
    C_left = A * B_left, C_right = A * B_right.
    Use 4x4x4 decomposition (rank 49) for each.
    Total rank = 49 + 49 = 98.
    
    Hmm, that's the same as before. Let me try a different split.
    
    Actually, let me just try to get a lower rank. The key insight is that 
    4x4x8 might have a lower rank than 2*49=98.
    
    For now, let me just implement the 4x4x4 rank 49 and use it twice.
    """
    U444, V444, W444 = _kronecker_4x4x4()
    
    rank444 = 49
    total_rank = 2 * rank444  # 98
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(32)]
    W = [[0]*total_rank for _ in range(32)]
    
    # Left half: B columns 0-3, C columns 0-3
    for r in range(rank444):
        for i in range(16):
            U[i][r] = U444[i][r]
        for j in range(4):
            for k in range(4):
                V[j*4+k][r] = V444[j*4+k][r]
        for k in range(4):
            for i in range(4):
                W[k*4+i][r] = W444[k*4+i][r]
                
    # Right half: B columns 4-7, C columns 4-7
    for r in range(rank444):
        rr = r + rank444
        for i in range(16):
            U[i][rr] = U444[i][r]
        for j in range(4):
            for k in range(4):
                V[j*4+(k+4)][rr] = V444[j*4+k][r]
        for k in range(4):
            for i in range(4):
                W[(k+4)*4+i][rr] = W444[k*4+i][r]
                
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, rank):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check diagonal
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                actual = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(actual - 1.0) > 1e-9:
                    return False
                    
    # Check off-diagonal with random sampling
    np.random.seed(42)
    num_checks = 5000
    for _ in range(num_checks):
        a = np.random.randint(n * m)
        b = np.random.randint(m * p)
        c = np.random.randint(p * n)
        actual = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
        
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
        
        if abs(actual - expected) > 1e-9:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Build 4x4x8 using 4x4x4 blocks
    try:
        U, V, W = _build_4x4x8()
        rank = len(U[0])
        
        if time.time() < deadline:
            if _verify_decomposition(U, V, W, n, m, p, rank):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook
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
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
