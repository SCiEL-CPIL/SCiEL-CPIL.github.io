import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 2, 3]
TIME_LIMIT = 120.0

def _rank11_decomposition():
    """
    Provide a known rank-11 decomposition for 2x2 by 2x3 matrix multiplication.
    U: (4, 11), V: (6, 11), W: (6, 11)
    """
    n, m, p = 2, 2, 3
    rank = 11
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indices:
    # A: [a00, a01, a10, a11] -> indices 0,1,2,3
    # B: [b00, b01, b02, b10, b11, b12] -> indices 0,1,2,3,4,5
    # C: [c00, c01, c02, c10, c11, c12] -> indices 0,1,2,3,4,5 (row-major)
    # But the task says W[k*n+i, r], which is column-major for C if we think of k as col, i as row.
    # C[i, k] -> index k*n + i.
    # C[0,0] -> 0*2+0 = 0
    # C[0,1] -> 1*2+0 = 2
    # C[0,2] -> 2*2+0 = 4
    # C[1,0] -> 0*2+1 = 1
    # C[1,1] -> 1*2+1 = 3
    # C[1,2] -> 2*2+1 = 5
    
    # We use Strassen for the first 2 columns (2x2x2 part) and schoolbook for the 3rd.
    # This gives rank 7 + 4 = 11.
    
    # --- Strassen part (columns 0 and 1 of B) ---
    # Let B2 = [[b00, b01], [b10, b11]]
    # M1 = (a+d)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    
    # M2 = (c+d)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    
    # M3 = a(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    
    # M4 = d(b10-b00)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    
    # M5 = (a+b)b11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    
    # M6 = (c-a)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    
    # M7 = (b-d)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    
    # --- Schoolbook part (column 2 of B) ---
    # M8 = a * b02
    U[0][7] = 1
    V[2][7] = 1
    
    # M9 = b * b12
    U[1][8] = 1
    V[5][8] = 1
    
    # M10 = c * b02
    U[2][9] = 1
    V[2][9] = 1
    
    # M11 = d * b12
    U[3][10] = 1
    V[5][10] = 1
    
    # --- W matrix (reconstruction) ---
    # C00 = M1 + M4 - M5 + M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    
    # C10 (idx 1) = M2 + M4
    W[1][1] = 1; W[1][3] = 1
    
    # C11 (idx 3) = M1 - M2 + M3 + M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    # C01 (idx 2) = M3 + M5
    W[2][2] = 1; W[2][4] = 1
    
    # C02 (idx 4) = M8 + M9
    W[4][7] = 1; W[4][8] = 1
    
    # C12 (idx 5) = M10 + M11
    W[5][9] = 1; W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=2, m=2, p=3, trials=10):
    """
    Verify the decomposition by checking random matrices.
    """
    np.random.seed(0)
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        
        C_comp = np.zeros((n, p))
        rank = U.shape[1]
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*m+k, r] * W[k*n+i, r]
                C_comp[i, k] = val
        
        if not np.allclose(C_true, C_comp, atol=1e-9):
            return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # We know rank 11 is achievable and is the lower bound.
    # We return the pre-computed rank 11 decomposition.
    # This is the optimal solution.
    
    res = _rank11_decomposition()
    
    # Optional: Verify in debug mode or if time permits, but it's O(1) so we can skip for speed
    # if not _verify(res['U'], res['V'], res['W']):
    #     raise ValueError("Invalid decomposition")
    
    # Convert to lists as per interface expectation? 
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries"
    # The current program returned lists of lists. The evaluator likely accepts numpy arrays or lists.
    # Let's stick to lists to be safe, as the previous program did.
    
    U_list = res['U'].tolist()
    V_list = res['V'].tolist()
    W_list = res['W'].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    if _verify(np.array(res['U']), np.array(res['V']), np.array(res['W'])):
        print("Verification Passed")
    else:
        print("Verification Failed")
