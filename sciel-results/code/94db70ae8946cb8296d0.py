import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 4]
TIME_LIMIT = 140  # Leave 10s margin for the 150s limit

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k.
    Uses integer arithmetic to avoid floating point issues.
    """
    r = len(U[0])
    # Convert to numpy arrays for fast computation
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    
    # Reshape
    U_r = U_arr.reshape(n, m, r)
    V_r = V_arr.reshape(m, p, r)
    W_r = W_arr.reshape(p, n, r)
    
    # Compute reconstruction
    # Result[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
    try:
        reconstructed = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    except Exception:
        return False
    
    return np.all(reconstructed == 1)

def _verify_decomposition_float(U, V, W, n, m, p):
    """
    Fast float verification. Returns True if likely valid (needs exact check).
    """
    r = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_r = U_arr.reshape(n, m, r)
    V_r = V_arr.reshape(m, p, r)
    W_r = W_arr.reshape(p, n, r)
    
    try:
        reconstructed = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    except Exception:
        return False
    
    # Check if close to 1
    return np.allclose(reconstructed, 1.0, atol=1e-6)

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2.
    Returns (U, V, W) where each is a list of 7 rank-1 terms.
    U: 4x7 (2x2 matrix elements), V: 4x7, W: 4x7
    Indexing: U[row, col] where row = i*2+j for 2x2 matrix [[i=0,j=0],[i=0,j=1],[i=1,j=0],[i=1,j=1]]
    """
    # Strassen formulas:
    # m1 = (x1+x4)(y1+y4)
    # m2 = (x3+x4)y1
    # m3 = x1(y2-y4)
    # m4 = x4(y3-y1)
    # m5 = (x1+x2)y4
    # m6 = (x3-x1)(y1+y2)
    # m7 = (x2-x4)(y3+y4)
    #
    # z1 = m1 + m4 - m5 + m7
    # z2 = m3 + m5
    # z3 = m2 + m4
    # z4 = m1 - m2 + m3 + m6
    #
    # x = [[x1, x2], [x3, x4]] -> indices: x1=0, x2=1, x3=2, x4=3
    # y = [[y1, y2], [y3, y4]] -> indices: y1=0, y2=1, y3=2, y4=3
    # z = [[z1, z2], [z3, z4]] -> indices: z1=0, z2=1, z3=2, z4=3
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # m1 = (x1+x4)(y1+y4): U[0]+=1, U[3]+=1, V[0]+=1, V[3]+=1, W[0]+=1, W[3]+=1
    # Wait, W contributes to z, not to the product. Let me re-think.
    # In tensor decomposition: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    # For 2x2*2x2: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta (where delta is the product)
    # Actually, the tensor is T[i,j,k] = 1 if the (i,j) element of X times (j,k) element of Y contributes to (i,k) element of Z
    # Wait no. The matrix multiplication tensor has T[i,j,k] = 1 where i indexes rows of X and Z, j indexes cols of X and rows of Y, k indexes cols of Y and Z.
    # So T[i,j,k] = 1 for all valid i,j,k (it's the identity tensor for bilinear map)
    # Actually no, the tensor is: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all i,j,k in range.
    # The rank-1 decomposition is: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c[i,j] * v_c[j,k] * w_c[k,i]... 
    
    # Let me use the standard formulation:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i,j,k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k.
    
    # Rank-1: T = sum_c u_c ⊗ v_c ⊗ w_c
    # u_c is vector over (i,j), v_c over (j,k), w_c over (i,k)
    # But in our problem, W is indexed by (k*n+i) which is (k,i) for n=2... 
    
    # Let me just use the verification function to check.
    # For 2x2*2x2: n=2, m=2, p=2
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (4, rank)  
    # W has shape (p*n, rank) = (4, rank)
    # Verification: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    
    # With n=m=p=2:
    # sum_c U[2i+j, c] * V[2j+k, c] * W[2k+i, c] == 1
    
    # Strassen:
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    # Each m_c is a product of a linear form in A and a linear form in B.
    # m1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # So for m1: U[0,0]=1, U[3,3]=1 (A[0,0] and A[1,1]), V[0,0]=1, V[3,3]=1 (B[0,0] and B[1,1])
    # Wait, U[2i+j, c] corresponds to A[i,j]. So U[0,c] = A[0,0], U[1,c] = A[0,1], U[2,c] = A[1,0], U[3,c] = A[1,1]
    # V[2j+k, c] corresponds to B[j,k]. So V[0,c] = B[0,0], V[1,c] = B[0,1], V[2,c] = B[1,0], V[3,c] = B[1,1]
    # W[2k+i, c] corresponds to C[i,k]... wait, W[2k+i, c] where k is col of B/row of C? 
    
    # Actually W[k*n+i, c] with n=2: W[2k+i, c]. This indexes the (i,k) element of C.
    # So W[0,c] = C[0,0] coefficient, W[1,c] = C[1,0] coefficient, W[2,c] = C[0,1] coefficient, W[3,c] = C[1,1] coefficient
    
    # For m1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]):
    # U[0,0]=1, U[3,0]=1, V[0,0]=1, V[3,0]=1
    # This contributes to all C[i,k] where the product appears. But in Strassen, m1 contributes to C[0,0] and C[1,1].
    # So W[0,0]=1 (C[0,0]), W[3,0]=1 (C[1,1])
    
    # Let me build each m_c:
    # m1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # m2 = (A10+A11)(B00) -> C10 += 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1  # C[1,0]
    
    # m3 = A00*(B01-B11) -> C01 += 1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1  # C[0,1]
    
    # m4 = A11*(B10-B00) -> C00 += 1, C10 += 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1  # C[0,0] and C[1,0]
    
    # m5 = (A00+A01)*B11 -> C00 -= 1, C01 += 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1  # C[0,0] -= 1, C[0,1] += 1
    
    # m6 = (A10-A00)*(B00+B01) -> C11 += 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C[1,1]
    
    # m7 = (A01-A11)*(B10+B11) -> C00 += 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C[0,0]
    
    return U, V, W

def _strassen_2x4_4x4():
    """
    Construct a rank-28 decomposition for 2x4 * 4x4 using four Strassen 2x2*2x2 decompositions.
    
    A (2x4) = [A_L | A_R] where A_L, A_R are 2x2
    B (4x4) = [[B11, B12], [B21, B22]] where each is 2x2
    C (2x4) = [A_L*B11 + A_R*B21, A_L*B12 + A_R*B22]
    
    We use 4 Strassen decompositions:
    - M1 = A_L * B11 (7 terms)
    - M2 = A_R * B21 (7 terms)
    - M3 = A_L * B12 (7 terms)
    - M4 = A_R * B22 (7 terms)
    
    C[:, 0:2] = M1 + M2
    C[:, 2:4] = M3 + M4
    """
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0]*rank for _ in range(n*m)]  # 8x28
    V = [[0]*rank for _ in range(m*p)]  # 16x28
    W = [[0]*rank for _ in range(p*n)]  # 8x28
    
    # Get Strassen 2x2*2x2 decomposition
    S_U, S_V, S_W = _strassen_2x2()
    
    # A_L elements: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[2], A[1,1]=U[3]
    # A_R elements: A[0,2]=U[4], A[0,3]=U[5], A[1,2]=U[6], A[1,3]=U[7]
    # 
    # B11 elements: B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[2], B[1,1]=V[3]
    # B12 elements: B[0,2]=V[4], B[0,3]=V[5], B[1,2]=V[6], B[1,3]=V[7]
    # B21 elements: B[2,0]=V[8], B[2,1]=V[9], B[3,0]=V[10], B[3,1]=V[11]
    # B22 elements: B[2,2]=V[12], B[2,3]=V[13], B[3,2]=V[14], B[3,3]=V[15]
    #
    # C elements: C[0,0]=W[0], C[1,0]=W[1], C[0,1]=W[2], C[1,1]=W[3],
    #            C[0,2]=W[4], C[1,2]=W[5], C[0,3]=W[6], C[1,3]=W[7]
    #
    # W indexing: W[k*n+i] = W[2k+i]
    # C[i,k] -> W[2k+i]
    # C[0,0]=W[0], C[1,0]=W[1], C[0,1]=W[2], C[1,1]=W[3]
    # C[0,2]=W[4], C[1,2]=W[5], C[0,3]=W[6], C[1,3]=W[7]
    
    # For M1 = A_L * B11:
    # A_L[i,j] corresponds to U[i*2+j] for i=0,1; j=0,1 -> U[0], U[1], U[2], U[3]
    # B11[j,k] corresponds to V[(j*2+k)] for j=0,1; k=0,1 -> V[0], V[1], V[2], V[3]
    # Result C[i,k] for i=0,1; k=0,1 -> W[2k+i]
    
    # M1 contributes to C[:, 0:2], so W indices 0-3
    for c in range(7):
        base = 0  # M1 uses rank indices 0-6
        # U: A_L elements
        for row in range(4):
            U[row][base + c] += S_U[row][c]
        # V: B11 elements
        for row in range(4):
            V[row][base + c] += S_V[row][c]
        # W: C[:, 0:2] elements
        for row in range(4):
            W[row][base + c] += S_W[row][c]
    
    # For M2 = A_R * B21:
    # A_R[i,j] corresponds to U[i*2+2+j] for i=0,1; j=0,1 -> U[4], U[5], U[6], U[7]
    # B21[j,k] corresponds to V[(2+j)*2+k] for j=0,1; k=0,1 -> V[8], V[9], V[10], V[11]
    # Result C[i,k] for i=0,1; k=0,1 -> W[2k+i]
    
    for c in range(7):
        base = 7  # M2 uses rank indices 7-13
        # U: A_R elements
        for row in range(4):
            U[4 + row][base + c] += S_U[row][c]
        # V: B21 elements
        for row in range(4):
            V[8 + row][base + c] += S_V[row][c]
        # W: C[:, 0:2] elements
        for row in range(4):
            W[row][base + c] += S_W[row][c]
    
    # For M3 = A_L * B12:
    # A_L[i,j] -> U[0..3]
    # B12[j,k] -> V[4..7]
    # Result C[i,k] for i=0,1; k=2,3 -> W[2*(k-2)+i] = W[4..7]
    
    for c in range(7):
        base = 14  # M3 uses rank indices 14-20
        # U: A_L elements
        for row in range(4):
            U[row][base + c] += S_U[row][c]
        # V: B12 elements
        for row in range(4):
            V[4 + row][base + c] += S_V[row][c]
        # W: C[:, 2:4] elements
        for row in range(4):
            W[4 + row][base + c] += S_W[row][c]
    
    # For M4 = A_R * B22:
    # A_R[i,j] -> U[4..7]
    # B22[j,k] -> V[12..15]
    # Result C[i,k] for i=0,1; k=2,3 -> W[4..7]
    
    for c in range(7):
        base = 21  # M4 uses rank indices 21-27
        # U: A_R elements
        for row in range(4):
            U[4 + row][base + c] += S_U[row][c]
        # V: B22 elements
        for row in range(4):
            V[12 + row][base + c] += S_V[row][c]
        # W: C[:, 2:4] elements
        for row in range(4):
            W[4 + row][base + c] += S_W[row][c]
    
    return {"U": U, "V": V, "W": W}

def _try_rank_decomposition(r, n, m, p, time_limit, start_time):
    """
    Try to find a rank-r decomposition using random search and local refinement.
    """
    best_result = None
    
    # Number of random trials
    trials = 200
    
    for trial in range(trials):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization
        U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p * n)]
        
        # Quick float check
        if not _verify_decomposition_float(U, V, W, n, m, p):
            continue
        
        # Exact check
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Try a different random strategy: use {-1, 1} only (no zeros)
    for trial in range(100):
        if time.time() - start_time > time_limit:
            break
        
        U = [[random.choice([-1, 1]) for _ in range(r)] for _ in range(n * m)]
        V = [[random.choice([-1, 1]) for _ in range(r)] for _ in range(m * p)]
        W = [[random.choice([-1, 1]) for _ in range(r)] for _ in range(p * n)]
        
        if not _verify_decomposition_float(U, V, W, n, m, p):
            continue
        
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # First try the structured Strassen-based decomposition
    start_time = time.time()
    
    # Try rank 28 (four Strassen 2x2*2x2)
    result_28 = _strassen_2x4_4x4()
    if _verify_decomposition(result_28["U"], result_28["V"], result_28["W"], n, m, p):
        # Check if we can do better
        best_rank = 28
        best_result = result_28
    else:
        # Fallback to schoolbook
        best_result = _schoolbook()
        best_rank = rank_schoolbook
    
    # Try to find even lower ranks using random search
    for r in range(16, best_rank):
        if time.time() - start_time > TIME_LIMIT:
            break
        
        result = _try_rank_decomposition(r, n, m, p, TIME_LIMIT, start_time)
        
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    return best_result
