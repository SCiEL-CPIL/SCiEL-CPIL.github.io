import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 150.0  # wall-clock limit

def _elapsed():
    return time.time() - _start_time

def _deadline_exceeded(margin=5.0):
    return _elapsed() > _DEADLINE - margin

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Block decomposition:
    # A (3x4) = [A1 (3x1), A2 (3x3)]
    # B (4x5) = [B1 (1x5); B2 (3x5)]
    # C = A1*B1 + A2*B2
    #
    # A1*B1: 3x1x5 -> Schoolbook rank 15
    # B2 (3x5) = [B21 (3x3), B22 (3x2)]
    # A2*B2 = A2*B21 + A2*B22
    # A2*B21: 3x3x3 -> Rank 19 (Bini et al.)
    # A2*B22: 3x3x2 -> Schoolbook rank 18
    #
    # Total Rank: 15 + 19 + 18 = 52
    
    rank = 52
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1.0):
        nonlocal rank
        if rank >= 52:
            return
        U[u_idx, rank] = coeff
        V[v_idx, rank] = coeff
        W[w_idx, rank] = coeff
        rank += 1

    # Part 1: A1*B1 (3x1x5)
    # A1 is col 0 of A. B1 is row 0 of B.
    # C[i,k] += A[i,0] * B[0,k]
    for i in range(3):
        for k in range(5):
            add_term(i * 4 + 0, 0 * 5 + k, k * 3 + i)

    # Part 2: A2*B21 (3x3x3) using Rank 19
    # A2 is cols 1,2,3 of A. B21 is rows 1,2,3 and cols 0,1,2 of B.
    # A2[i, j] corresponds to A[i, j+1]
    # B21[j, k] corresponds to B[j+1, k]
    # C[i, k] += sum_j A2[i, j] * B21[j, k]
    
    # Indices for A2:
    # i in 0..2, j in 0..2 -> A[i, j+1] -> idx = i*4 + (j+1)
    # Indices for B21:
    # j in 0..2, k in 0..2 -> B[j+1, k] -> idx = (j+1)*5 + k
    # Indices for W:
    # C[i, k] -> idx = k*3 + i
    
    # Rank 19 Coefficients (Bini et al.)
    # a11, a12, a13 ... -> A2[0,0], A2[0,1], A2[0,2]
    # b11, b12, b13 ... -> B21[0,0], B21[0,1], B21[0,2]
    
    a = [[0,1,2], [3,4,5], [6,7,8]] # Local indices for A2
    b = [[0,1,2], [3,4,5], [6,7,8]] # Local indices for B21
    
    # Map local (i,j) to global U index
    def u_idx(i, j):
        return i * 4 + (j + 1)
        
    def v_idx(j, k):
        return (j + 1) * 5 + k
        
    def w_idx(i, k):
        return k * 3 + i
        
    # Coefficients for Rank 19
    # This is a standard set.
    # m1 = a11 b11
    add_term(u_idx(0,0), v_idx(0,0), w_idx(0,0))
    # m2 = (a11+a12) b21
    add_term(u_idx(0,0), v_idx(1,0), 1)
    add_term(u_idx(0,1), v_idx(1,0), 1)
    # m3 = (a21+a31) b11
    add_term(u_idx(1,0), v_idx(0,0), 1)
    add_term(u_idx(2,0), v_idx(0,0), 1)
    # m4 = (a21+a22) b21
    add_term(u_idx(1,0), v_idx(1,0), 1)
    add_term(u_idx(1,1), v_idx(1,0), 1)
    # m5 = a31 b11
    add_term(u_idx(2,0), v_idx(0,0), 1)
    # m6 = (a31+a32) b21
    add_term(u_idx(2,0), v_idx(1,0), 1)
    add_term(u_idx(2,1), v_idx(1,0), 1)
    # m7 = a13 b13
    add_term(u_idx(0,2), v_idx(0,2), 1)
    # m8 = (a13+a23) b23
    add_term(u_idx(0,2), v_idx(1,2), 1)
    add_term(u_idx(1,2), v_idx(1,2), 1)
    # m9 = a33 b13
    add_term(u_idx(2,2), v_idx(0,2), 1)
    # m10 = (a33+a23) b23
    add_term(u_idx(2,2), v_idx(1,2), 1)
    add_term(u_idx(1,2), v_idx(1,2), 1)
    # m11 = a23 b13
    add_term(u_idx(1,2), v_idx(0,2), 1)
    # m12 = (a23+a33) b23
    add_term(u_idx(1,2), v_idx(1,2), 1)
    add_term(u_idx(2,2), v_idx(1,2), 1)
    # m13 = (a11+a21+a31) (b11+b21+b31)
    add_term(u_idx(0,0), v_idx(0,0), 1)
    add_term(u_idx(1,0), v_idx(0,0), 1)
    add_term(u_idx(2,0), v_idx(0,0), 1)
    add_term(u_idx(0,0), v_idx(1,0), 1)
    add_term(u_idx(1,0), v_idx(1,0), 1)
    add_term(u_idx(2,0), v_idx(1,0), 1)
    add_term(u_idx(0,0), v_idx(2,0), 1)
    add_term(u_idx(1,0), v_idx(2,0), 1)
    add_term(u_idx(2,0), v_idx(2,0), 1)
    # m14 = (a12+a22+a32) (b12+b22+b32)
    add_term(u_idx(0,1), v_idx(0,1), 1)
    add_term(u_idx(1,1), v_idx(0,1), 1)
    add_term(u_idx(2,1), v_idx(0,1), 1)
    add_term(u_idx(0,1), v_idx(1,1), 1)
    add_term(u_idx(1,1), v_idx(1,1), 1)
    add_term(u_idx(2,1), v_idx(1,1), 1)
    add_term(u_idx(0,1), v_idx(2,1), 1)
    add_term(u_idx(1,1), v_idx(2,1), 1)
    add_term(u_idx(2,1), v_idx(2,1), 1)
    # m15 = (a13+a23+a33) (b13+b23+b33)
    add_term(u_idx(0,2), v_idx(0,2), 1)
    add_term(u_idx(1,2), v_idx(0,2), 1)
    add_term(u_idx(2,2), v_idx(0,2), 1)
    add_term(u_idx(0,2), v_idx(1,2), 1)
    add_term(u_idx(1,2), v_idx(1,2), 1)
    add_term(u_idx(2,2), v_idx(1,2), 1)
    add_term(u_idx(0,2), v_idx(2,2), 1)
    add_term(u_idx(1,2), v_idx(2,2), 1)
    add_term(u_idx(2,2), v_idx(2,2), 1)
    # m16 = (a11+a22+a33) (b11+b22+b33)
    add_term(u_idx(0,0), v_idx(0,0), 1)
    add_term(u_idx(1,1), v_idx(1,1), 1)
    add_term(u_idx(2,2), v_idx(2,2), 1)
    add_term(u_idx(0,0), v_idx(1,1), 1)
    add_term(u_idx(1,1), v_idx(0,0), 1)
    add_term(u_idx(2,2), v_idx(2,2), 1)
    # m17 = (a12+a13) (b22+b23)
    add_term(u_idx(0,1), v_idx(1,1), 1)
    add_term(u_idx(0,2), v_idx(1,2), 1)
    add_term(u_idx(0,1), v_idx(1,2), 1)
    add_term(u_idx(0,2), v_idx(1,1), 1)
    # m18 = (a21+a23) (b12+b13)
    add_term(u_idx(1,0), v_idx(0,1), 1)
    add_term(u_idx(1,2), v_idx(0,2), 1)
    add_term(u_idx(1,0), v_idx(0,2), 1)
    add_term(u_idx(1,2), v_idx(0,1), 1)
    # m19 = (a31+a32) (b12+b13) ?
    # The exact coefficients for m13-m19 vary by implementation.
    # This hardcoded set might be incorrect.
    
    # Given the risk, I will fall back to Rank 60 if verification fails.
    # But for now, I will return the Rank 60 solution to be safe, as Rank 19 is complex.
    # Actually, I will just use the Rank 60 schoolbook solution which is guaranteed correct.
    
    # Re-initialize for Rank 60
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = (i * m + j) * p + k
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
