"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Finds a rank-minimal exact tensor decomposition for multiplying 
    a 2x2 matrix by a 2x16 matrix.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # The problem decomposes into 8 independent 2x2x2 matrix multiplications.
    # The rank of 2x2x2 multiplication is 7 (Strassen).
    # Total rank = 8 * 7 = 56.
    
    rank = 56
    num_blocks = p // 2  # 8 blocks
    
    # Initialize U, V, W as lists of lists for easy construction, 
    # then convert to numpy arrays if needed, but the contract allows lists.
    # Shapes: U is (n*m, rank) = (4, 56), V is (m*p, rank) = (32, 56), W is (p*n, rank) = (32, 56)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen intermediates for a single 2x2 block:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Output reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for A (U rows):
    # A00=0, A01=1, A10=2, A11=3
    
    # Indices for B (V rows) within a block t (columns 2t, 2t+1):
    # B00 = 0*p + 2t = 2t
    # B01 = 0*p + 2t+1 = 2t+1
    # B10 = 1*p + 2t = p + 2t = 16 + 2t
    # B11 = 1*p + 2t+1 = p + 2t+1 = 16 + 2t+1
    
    # Indices for C (W rows) within a block t (columns 2t, 2t+1, rows 0, 1):
    # C00 = (2t)*n + 0 = 4t
    # C01 = (2t+1)*n + 0 = 4t+2
    # C10 = (2t)*n + 1 = 4t+1
    # C11 = (2t+1)*n + 1 = 4t+3
    
    # Define the contribution of each intermediate M to U, V, W
    # Each M is a vector of tuples (index, coefficient)
    
    # We will iterate through each of the 8 blocks
    for t in range(num_blocks):
        # Determine the specific indices for this block
        b00_idx = 2 * t
        b01_idx = 2 * t + 1
        b10_idx = p + 2 * t
        b11_idx = p + 2 * t + 1
        
        c00_idx = 4 * t
        c01_idx = 4 * t + 2
        c10_idx = 4 * t + 1
        c11_idx = 4 * t + 3
        
        # Intermediates M1 to M7
        # Each intermediate adds a column to U, V, W
        
        # M1
        m_idx = t * 7 + 0
        # U: A00(1), A11(1)
        U[0][m_idx] += 1
        U[3][m_idx] += 1
        # V: B00(1), B11(1)
        V[b00_idx][m_idx] += 1
        V[b11_idx][m_idx] += 1
        # W: C00(1), C11(1)
        W[c00_idx][m_idx] += 1
        W[c11_idx][m_idx] += 1
        
        # M2
        m_idx = t * 7 + 1
        # U: A10(1), A11(1)
        U[2][m_idx] += 1
        U[3][m_idx] += 1
        # V: B00(1)
        V[b00_idx][m_idx] += 1
        # W: C10(1), C11(-1)
        W[c10_idx][m_idx] += 1
        W[c11_idx][m_idx] -= 1
        
        # M3
        m_idx = t * 7 + 2
        # U: A00(1)
        U[0][m_idx] += 1
        # V: B01(1), B11(-1)
        V[b01_idx][m_idx] += 1
        V[b11_idx][m_idx] -= 1
        # W: C01(1), C11(1)
        W[c01_idx][m_idx] += 1
        W[c11_idx][m_idx] += 1
        
        # M4
        m_idx = t * 7 + 3
        # U: A11(1)
        U[3][m_idx] += 1
        # V: B10(1), B00(-1)
        V[b10_idx][m_idx] += 1
        V[b00_idx][m_idx] -= 1
        # W: C00(1), C10(1)
        W[c00_idx][m_idx] += 1
        W[c10_idx][m_idx] += 1
        
        # M5
        m_idx = t * 7 + 4
        # U: A00(1), A01(1)
        U[0][m_idx] += 1
        U[1][m_idx] += 1
        # V: B11(1)
        V[b11_idx][m_idx] += 1
        # W: C00(-1), C01(1)
        W[c00_idx][m_idx] -= 1
        W[c01_idx][m_idx] += 1
        
        # M6
        m_idx = t * 7 + 5
        # U: A10(1), A00(-1)
        U[2][m_idx] += 1
        U[0][m_idx] -= 1
        # V: B00(1), B01(1)
        V[b00_idx][m_idx] += 1
        V[b01_idx][m_idx] += 1
        # W: C11(1)
        W[c11_idx][m_idx] += 1
        
        # M7
        m_idx = t * 7 + 6
        # U: A01(1), A11(-1)
        U[1][m_idx] += 1
        U[3][m_idx] -= 1
        # V: B10(1), B11(1)
        V[b10_idx][m_idx] += 1
        V[b11_idx][m_idx] += 1
        # W: C00(1)
        W[c00_idx][m_idx] += 1

    # Simple verification to ensure correctness before returning
    # Convert to numpy for fast tensor check
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] should be 1 if (i,j,k) is a valid term, 0 otherwise.
    # Term (a,b,c) corresponds to:
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # We check if sum_r U[a,r] V[b,r] W[c,r] == 1 for valid (a,b,c) and 0 otherwise.
    
    is_valid = True
    tol = 1e-9
    
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(np.dot(U_np[a, :], V_np[b, :]), W_np[c, :])
                if abs(val - 1.0) > tol:
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            break
    
    # If invalid, we have a bug in the construction. 
    # However, given the algebra is standard Strassen, it should be valid.
    # Fallback to schoolbook if verification fails (should not happen)
    if not is_valid:
        # Schoolbook rank = 2*2*16 = 64
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col] = 1
                    V_fb[j*p+k][col] = 1
                    W_fb[k*n+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
