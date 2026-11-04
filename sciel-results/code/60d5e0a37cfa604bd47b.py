import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Rank 55 Decomposition
    # Strategy:
    # 1. Partition A (2x3) into A1 (2x2) and A2 (2x1)
    # 2. Partition B (3x10) into B1 (2x10) and B2 (1x10)
    # 3. C = A1 @ B1 + A2 @ B2
    # 4. Compute A1 @ B1 (2x2 @ 2x10) using 5 Strassen steps (Rank 35)
    # 5. Compute A2 @ B2 (2x1 @ 1x10) using 10 Schoolbook steps (Rank 20)
    # Total Rank: 55
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    def add_term(u_indices, v_indices, w_indices, coeff=1):
        """
        Adds a single multiplicative term to the decomposition.
        u_indices: list of (row, coeff) or just row if coeff is 1
        v_indices: list of (row, coeff) or just row if coeff is 1
        w_indices: list of (row, coeff) or just row if coeff is 1
        """
        nonlocal r
        # Helper to handle mixed coeff
        def get_val(idx, coeff_map):
             if isinstance(idx, tuple):
                 return idx[1]
             return coeff_map.get(idx, 1)
        
        # Simpler: just pass lists of (index, value)
        # But the signature above is loose. Let's just use direct assignment in the main loop for clarity and speed, 
        # as abstracting too much might introduce bugs or slow down execution.
        pass

    r = 0
    
    # --- Part 1: A1 (2x2) @ B1 (2x10) ---
    # A1 indices in U:
    # A[0,0] -> 0
    # A[0,1] -> 1
    # A[1,0] -> 3
    # A[1,1] -> 4
    # B1 indices in V:
    # B[0,k] -> k
    # B[1,k] -> 10+k
    
    # Strassen for 2x2 matrices:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 - M6
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M7
    
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Term 1: M1
        U[0][r] += 1; U[4][r] += 1
        V[k0][r] += 1; V[10+k1][r] += 1
        W[k0*2+0][r] += 1; W[k1*2+1][r] += 1
        r += 1
        
        # Term 2: M2
        U[3][r] += 1; U[4][r] += 1
        V[k0][r] += 1
        W[k0*2+1][r] += 1; W[k1*2+1][r] -= 1
        r += 1
        
        # Term 3: M3
        U[0][r] += 1
        V[k1][r] += 1; V[10+k1][r] -= 1
        W[k1*2+0][r] += 1; W[k1*2+1][r] += 1
        r += 1
        
        # Term 4: M4
        U[4][r] += 1
        V[10+k0][r] += 1; V[k0][r] -= 1
        W[k0*2+0][r] += 1; W[k0*2+1][r] += 1
        r += 1
        
        # Term 5: M5
        U[0][r] += 1; U[1][r] += 1
        V[10+k1][r] += 1
        W[k0*2+0][r] -= 1; W[k1*2+0][r] += 1
        r += 1
        
        # Term 6: M6
        U[3][r] += 1; U[0][r] -= 1
        V[k0][r] += 1; V[k1][r] += 1
        W[k1*2+1][r] += 1
        r += 1
        
        # Term 7: M7
        U[1][r] += 1; U[4][r] -= 1
        V[10+k0][r] += 1; V[10+k1][r] += 1
        W[k0*2+0][r] += 1
        r += 1

    # --- Part 2: A2 (2x1) @ B2 (1x10) ---
    # A2 indices in U:
    # A[0,2] -> 2
    # A[1,2] -> 5
    # B2 indices in V:
    # B[2,k] -> 20+k
    
    for k in range(10):
        # Term for C[0,k]
        U[2][r] += 1
        V[20+k][r] += 1
        W[k*2+0][r] += 1
        r += 1
        
        # Term for C[1,k]
        U[5][r] += 1
        V[20+k][r] += 1
        W[k*2+1][r] += 1
        r += 1

    if r != rank:
        return _trivial_decomposition(n, m, p)

    return {"U": U, "V": V, "W": W}

def _trivial_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][r] = 1
                V[j*p+k][r] = 1
                W[k*n+i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
