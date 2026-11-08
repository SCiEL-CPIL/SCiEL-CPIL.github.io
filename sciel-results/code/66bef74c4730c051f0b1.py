import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 4]
TIME_LIMIT = 130  # Leave 20s margin for the 150s limit

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

def _try_rank_decomposition(r, n, m, p, time_limit, start_time):
    """
    Try to find a rank-r decomposition using random search and local refinement.
    """
    # We use a combination of:
    # 1. Random initialization with {-1, 0, 1}
    # 2. Local search / hill climbing
    # 3. ALS-style refinement
    
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
    
    # If random search failed, try structured approach
    # Use the fact that matrix multiplication can be decomposed using
    # Kronecker products and known low-rank decompositions
    
    # For 2x4 * 4x4, we can think of it as:
    # A (2x4) * B (4x4) = C (2x4)
    # We can split A into two 2x2 blocks: A = [A1, A2] where A1, A2 are 2x2
    # B can be split into 2x2 blocks: B = [[B11, B12], [B21, B22]]
    # C = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    # Each 2x2 * 2x2 multiplication has rank 7 (Strassen)
    # We need to compute:
    # C1 = A1*B11 + A2*B21  (2x2 matrix)
    # C2 = A1*B12 + A2*B22  (2x2 matrix)
    
    # Using Strassen for each 2x2*2x2:
    # C1 needs 7 multiplications, C2 needs 7 multiplications
    # But we can share some? Not directly obvious.
    
    # Total rank would be 14? That seems too low. Let me reconsider.
    
    # Actually, the tensor for 2x4 * 4x4 has dimension 2*4*4 = 32.
    # The schoolbook rank is 32.
    # Can we do better?
    
    # Let's try a different approach: use the known rank for 2x2 matrix mult (7)
    # and combine them.
    
    # A is 2x4, B is 4x4.
    # Write A = [A_left, A_right] where A_left, A_right are 2x2.
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2.
    
    # C = A * B = [A_left * B11 + A_right * B21, A_left * B12 + A_right * B22]
    
    # Let M1 = A_left * B11, M2 = A_right * B21, M3 = A_left * B12, M4 = A_right * B22
    # C = [M1 + M2, M3 + M4]
    
    # Each Mi is a 2x2 matrix product. Using Strassen, each takes 7 rank-1 terms.
    # So total 28 terms for the four products.
    # Then we add M1+M2 and M3+M4, which are linear combinations, so no extra rank.
    
    # So rank 28 might be achievable!
    
    # Let's try to construct this explicitly.
    
    # Strassen decomposition for 2x2 * 2x2:
    # Given X = [[x1, x2], [x3, x4]], Y = [[y1, y2], [y3, y4]]
    # Z = X * Y = [[z1, z2], [z3, z4]]
    
    # Strassen:
    # m1 = (x1 + x4)(y1 + y4)
    # m2 = (x3 + x4)y1
    # m3 = x1(y2 - y4)
    # m4 = x4(y3 - y1)
    # m5 = (x1 + x2)y4
    # m6 = (x3 - x1)(y1 + y2)
    # m7 = (x2 - x4)(y3 + y4)
    
    # z1 = m1 + m4 - m5 + m7
    # z2 = m3 + m5
    # z3 = m2 + m4
    # z4 = m1 - m2 + m3 + m6
    
    # For our problem:
    # A_left = [[a11, a12], [a21, a22]]
    # A_right = [[a13, a14], [a23, a24]]
    # B11 = [[b11, b12], [b21, b22]]
    # B12 = [[b13, b14], [b23, b24]]
    # B21 = [[b31, b32], [b41, b42]]
    # B22 = [[b33, b34], [b43, b44]]
    
    # C = A * B
    # C = [[c11, c12, c13, c14], [c21, c22, c23, c24]]
    
    # C[:, 0:2] = A_left * B11 + A_right * B21
    # C[:, 2:4] = A_left * B12 + A_right * B22
    
    # Let's compute each using Strassen.
    
    # This is getting complex. Let me just implement the random search more aggressively.
    
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

def _strassen_2x2_decomposition():
    """
    Returns the Strassen decomposition for 2x2 matrix multiplication.
    Each entry is a list of (index, coefficient) pairs representing the rank-1 terms.
    
    For X = [[x0, x1], [x2, x3]], Y = [[y0, y1], [y2, y3]], Z = X*Y = [[z0, z1], [z2, z3]]
    
    Strassen's algorithm uses 7 multiplications:
    m0 = (x0 + x3)(y0 + y3)
    m1 = (x2 + x3)y0
    m2 = x0(y1 - y3)
    m3 = x3(y2 - y0)
    m4 = (x0 + x1)y3
    m5 = (x2 - x0)(y0 + y1)
    m6 = (x1 - x3)(y2 + y3)
    
    z0 = m0 + m3 - m4 + m6
    z1 = m2 + m4
    z2 = m1 + m3
    z3 = m0 - m1 + m2 + m5
    """
    # Return the decomposition as U, V, W matrices for 2x2 * 2x2
    # U: 2x2 (4 entries) x 7 (rank)
    # V: 2x2 (4 entries) x 7 (rank)
    # W: 2x2 (4 entries) x 7 (rank)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # m0 = (x0 + x3)(y0 + y3)
    # U contributions: x0 -> +1, x3 -> +1
    U[0][0] = 1; U[3][0] = 1
    # V contributions: y0 -> +1, y3 -> +1
    V[0][0] = 1; V[3][0] = 1
    # W contributions: z0 -> +1
    W[0][0] = 1
    
    # m1 = (x2 + x3)y0
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # z2 = m1 + m3
    W[2][1] = 1
    
    # m2 = x0(y1 - y3)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # z1 = m2 + m4, z3 = m0 - m1 + m2 + m5
    W[1][2] = 1; W[3][2] = 1
    
    # m3 = x3(y2 - y0)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # z0 = m0 + m3 - m4 + m6, z2 = m1 + m3
    W[0][3] = 1; W[2][3] = 1
    
    # m4 = (x0 + x1)y3
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # z0 = m0 + m3 - m4 + m6, z1 = m2 + m4
    W[0][4] = -1; W[1][4] = 1
    
    # m5 = (x2 - x0)(y0 + y1)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # z3 = m0 - m1 + m2 + m5
    W[3][5] = 1
    
    # m6 = (x1 - x3)(y2 + y3)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # z0 = m0 + m3 - m4 + m6
    W[0][6] = 1
    
    return U, V, W

def _construct_strassen_based_decomposition():
    """
    Construct a rank-28 decomposition using Strassen's algorithm for 2x2 matrix multiplication.
    
    For 2x4 * 4x4:
    A = [[a00, a01, a02, a03],
         [a10, a11, a12, a13]]
    
    B = [[b00, b01, b02, b03],
         [b10, b11, b12, b13],
         [b20, b21, b22, b23],
         [b30, b31, b32, b33]]
    
    C = A * B (2x4)
    
    We can write:
    A_left = [[a00, a01], [a10, a11]]
    A_right = [[a02, a03], [a12, a13]]
    
    B = [[B11, B12],
         [B21, B22]]
    where
    B11 = [[b00, b01], [b10, b11]]
    B12 = [[b02, b03], [b12, b13]]
    B21 = [[b20, b21], [b30, b31]]
    B22 = [[b22, b23], [b32, b33]]
    
    C_left = A_left * B11 + A_right * B21
    C_right = A_left * B12 + A_right * B22
    
    Each of these is a sum of two 2x2 matrix products, which we can compute using Strassen.
    """
    # Get Strassen decomposition for 2x2
    U_s, V_s, W_s = _strassen_2x2_decomposition()
    
    # We need to compute:
    # C_left = A_left * B11 + A_right * B21
    # C_right = A_left * B12 + A_right * B22
    
    # Using Strassen, each 2x2*2x2 product uses 7 rank-1 terms.
    # So we need 7 + 7 + 7 + 7 = 28 rank-1 terms total.
    
    # The U matrix will have shape (8, 28) where 8 = 2*4 (rows of A)
    # The V matrix will have shape (16, 28) where 16 = 4*4 (entries of B)
    # The W matrix will have shape (8, 28) where 8 = 4*2 (entries of C)
    
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Mapping for A (2x4):
    # A[0,0] = a00 -> index 0
    # A[0,1] = a01 -> index 1
    # A[0,2] = a02 -> index 2
    # A[0,3] = a03 -> index 3
    # A[1,0] = a10 -> index 4
    # A[1,1] = a11 -> index 5
    # A[1,2] = a12 -> index 6
    # A[1,3] = a13 -> index 7
    
    # Mapping for B (4x4):
    # B[0,0] = b00 -> index 0
    # B[0,1] = b01 -> index 1
    # B[0,2] = b02 -> index 2
    # B[0,3] = b03 -> index 3
    # B[1,0] = b10 -> index 4
    # B[1,1] = b11 -> index 5
    # B[1,2] = b12 -> index 6
    # B[1,3] = b13 -> index 7
    # B[2,0] = b20 -> index 8
    # B[2,1] = b21 -> index 9
    # B[2,2] = b22 -> index 10
    # B[2,3] = b23 -> index 11
    # B[3,0] = b30 -> index 12
    # B[3,1] = b31 -> index 13
    # B[3,2] = b32 -> index 14
    # B[3,3] = b33 -> index 15
    
    # Mapping for C (2x4) with W[k*n+i, r] convention:
    # C[0,0] = c00 -> k=0, i=0 -> index 0*2+0 = 0
    # C[0,1] = c01 -> k=1, i=0 -> index 1*2+0 = 2
    # C[0,2] = c02 -> k=2, i=0 -> index 2*2+0 = 4
    # C[0,3] = c03 -> k=3, i=0 -> index 3*2+0 = 6
    # C[1,0] = c10 -> k=0, i=1 -> index 0*2+1 = 1
    # C[1,1] = c11 -> k=1, i=1 -> index 1*2+1 = 3
    # C[1,2] = c12 -> k=2, i=1 -> index 2*2+1 = 5
    # C[1,3] = c13 -> k=3, i=1 -> index 3*2+1 = 7
    
    # For A_left * B11:
    # A_left = [[a00, a01], [a10, a11]]
    # B11 = [[b00, b01], [b10, b11]]
    # Result = [[c00, c01], [c10, c11]]
    
    # For A_right * B21:
    # A_right = [[a02, a03], [a12, a13]]
    # B21 = [[b20, b21], [b30, b31]]
    # Result = [[c00', c01'], [c10', c11']]
    
    # C_left = A_left * B11 + A_right * B21
    # So c00 = c00_from_left + c00_from_right, etc.
    
    # Let's use Strassen for each product.
    
    # Strassen indices: 0-6 for first product, 7-13 for second product, etc.
    
    # Product 1: A_left * B11 (terms 0-6)
    for t in range(7):
        # U contributions from A_left
        # A_left[0,0] = a00 -> U[0]
        # A_left[0,1] = a01 -> U[1]
        # A_left[1,0] = a10 -> U[4]
        # A_left[1,1] = a11 -> U[5]
        
        # In Strassen, X[0]=x0, X[1]=x1, X[2]=x2, X[3]=x3
        # So x0=a00, x1=a01, x2=a10, x3=a11
        
        # U_s[0][t] contributes to x0 (a00)
        if U_s[0][t] != 0:
            U[0][t] += U_s[0][t]
        # U_s[1][t] contributes to x1 (a01)
        if U_s[1][t] != 0:
            U[1][t] += U_s[1][t]
        # U_s[2][t] contributes to x2 (a10)
        if U_s[2][t] != 0:
            U[4][t] += U_s[2][t]
        # U_s[3][t] contributes to x3 (a11)
        if U_s[3][t] != 0:
            U[5][t] += U_s[3][t]
        
        # V contributions from B11
        # B11[0,0] = b00 -> V[0]
        # B11[0,1] = b01 -> V[1]
        # B11[1,0] = b10 -> V[4]
        # B11[1,1] = b11 -> V[5]
        
        # y0=b00, y1=b01, y2=b10, y3=b11
        
        if V_s[0][t] != 0:
            V[0][t] += V_s[0][t]
        if V_s[1][t] != 0:
            V[1][t] += V_s[1][t]
        if V_s[2][t] != 0:
            V[4][t] += V_s[2][t]
        if V_s[3][t] != 0:
            V[5][t] += V_s[3][t]
        
        # W contributions to result
        # Result[0,0] = c00 -> W[0]
        # Result[0,1] = c01 -> W[2]
        # Result[1,0] = c10 -> W[1]
        # Result[1,1] = c11 -> W[3]
        
        # z0=c00, z1=c01, z2=c10, z3=c11
        
        if W_s[0][t] != 0:
            W[0][t] += W_s[0][t]
        if W_s[1][t] != 0:
            W[2][t] += W_s[1][t]
        if W_s[2][t] != 0:
            W[1][t] += W_s[2][t]
        if W_s[3][t] != 0:
            W[3][t] += W_s[3][t]
    
    # Product 2: A_right * B21 (terms 7-13)
    for t in range(7):
        term = t + 7
        
        # A_right[0,0] = a02 -> U[2]
        # A_right[0,1] = a03 -> U[3]
        # A_right[1,0] = a12 -> U[6]
        # A_right[1,1] = a13 -> U[7]
        
        if U_s[0][t] != 0:
            U[2][term] += U_s[0][t]
        if U_s[1][t] != 0:
            U[3][term] += U_s[1][t]
        if U_s[2][t] != 0:
            U[6][term] += U_s[2][t]
        if U_s[3][t] != 0:
            U[7][term] += U_s[3][t]
        
        # B21[0,0] = b20 -> V[8]
        # B21[0,1] = b21 -> V[9]
        # B21[1,0] = b30 -> V[12]
        # B21[1,1] = b31 -> V[13]
        
        if V_s[0][t] != 0:
            V[8][term] += V_s[0][t]
        if V_s[1][t] != 0:
            V[9][term] += V_s[1][t]
        if V_s[2][t] != 0:
            V[12][term] += V_s[2][t]
        if V_s[3][t] != 0:
            V[13][term] += V_s[3][t]
        
        # Result adds to C_left, so same W indices as before
        if W_s[0][t] != 0:
            W[0][term] += W_s[0][t]
        if W_s[1][t] != 0:
            W[2][term] += W_s[1][t]
        if W_s[2][t] != 0:
            W[1][term] += W_s[2][t]
        if W_s[3][t] != 0:
            W[3][term] += W_s[3][t]
    
    # Product 3: A_left * B12 (terms 14-20)
    for t in range(7):
        term = t + 14
        
        # A_left same as before
        if U_s[0][t] != 0:
            U[0][term] += U_s[0][t]
        if U_s[1][t] != 0:
            U[1][term] += U_s[1][t]
        if U_s[2][t] != 0:
            U[4][term] += U_s[2][t]
        if U_s[3][t] != 0:
            U[5][term] += U_s[3][t]
        
        # B12[0,0] = b02 -> V[2]
        # B12[0,1] = b03 -> V[3]
        # B12[1,0] = b12 -> V[6]
        # B12[1,1] = b13 -> V[7]
        
        if V_s[0][t] != 0:
            V[2][term] += V_s[0][t]
        if V_s[1][t] != 0:
            V[3][term] += V_s[1][t]
        if V_s[2][t] != 0:
            V[6][term] += V_s[2][t]
        if V_s[3][t] != 0:
            V[7][term] += V_s[3][t]
        
        # Result goes to C_right
        # C_right[0,0] = c02 -> W[4]
        # C_right[0,1] = c03 -> W[6]
        # C_right[1,0] = c12 -> W[5]
        # C_right[1,1] = c13 -> W[7]
        
        if W_s[0][t] != 0:
            W[4][term] += W_s[0][t]
        if W_s[1][t] != 0:
            W[6][term] += W_s[1][t]
        if W_s[2][t] != 0:
            W[5][term] += W_s[2][t]
        if W_s[3][t] != 0:
            W[7][term] += W_s[3][t]
    
    # Product 4: A_right * B22 (terms 21-27)
    for t in range(7):
        term = t + 21
        
        # A_right same as before
        if U_s[0][t] != 0:
            U[2][term] += U_s[0][t]
        if U_s[1][t] != 0:
            U[3][term] += U_s[1][t]
        if U_s[2][t] != 0:
            U[6][term] += U_s[2][t]
        if U_s[3][t] != 0:
            U[7][term] += U_s[3][t]
        
        # B22[0,0] = b22 -> V[10]
        # B22[0,1] = b23 -> V[11]
        # B22[1,0] = b32 -> V[14]
        # B22[1,1] = b33 -> V[15]
        
        if V_s[0][t] != 0:
            V[10][term] += V_s[0][t]
        if V_s[1][t] != 0:
            V[11][term] += V_s[1][t]
        if V_s[2][t] != 0:
            V[14][term] += V_s[2][t]
        if V_s[3][t] != 0:
            V[15][term] += V_s[3][t]
        
        # Result adds to C_right
        if W_s[0][t] != 0:
            W[4][term] += W_s[0][t]
        if W_s[1][t] != 0:
            W[6][term] += W_s[1][t]
        if W_s[2][t] != 0:
            W[5][term] += W_s[2][t]
        if W_s[3][t] != 0:
            W[7][term] += W_s[3][t]
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    best_result = _schoolbook()
    best_rank = rank_schoolbook
    
    start_time = time.time()
    
    # First, try the Strassen-based decomposition which should give rank 28
    strassen_result = _construct_strassen_based_decomposition()
    if _verify_decomposition(strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p):
        best_rank = 28
        best_result = strassen_result
    
    # Search for lower ranks
    for r in range(16, best_rank):
        if time.time() - start_time > TIME_LIMIT:
            break
        
        result = _try_rank_decomposition(r, n, m, p, TIME_LIMIT, start_time)
        
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    return best_result
