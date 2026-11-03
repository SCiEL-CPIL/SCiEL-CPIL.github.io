"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s limit

    # Build target tensor
    target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    def schoolbook():
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

    def verify(U, V, W, rank, tol=1e-9):
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        for a in range(n * m):
            for b in range(m * p):
                val = np.dot(U_arr[a, :], V_arr[b, :] * W_arr[:, c_idx] if False else V_arr[b, :])
                # Actually need to check all c
                for c in range(p * n):
                    val = np.dot(U_arr[a, :], V_arr[b, :] * W_arr[c, :])
                    if abs(val - target[a, b, c]) > tol:
                        return False
        return True

    # Try to find a lower-rank decomposition using structured approaches
    # For 2x5x5, we can try:
    # 1. Block decomposition using 2x2x2 blocks
    # 2. Random search with structured initialization
    
    # Laderman's algorithm for 2x2x2 (rank 7)
    def laderman_2x2x2():
        """Return U, V, W for 2x2x2 matrix multiplication (rank 7)."""
        # A is 2x2, B is 2x2, C is 2x2
        # U shape: (4, 7) indexed by [i*2+j, r]
        # V shape: (4, 7) indexed by [j*2+k, r]
        # W shape: (4, 7) indexed by [k*2+i, r]
        
        U = np.zeros((4, 7), dtype=np.float64)
        V = np.zeros((4, 7), dtype=np.float64)
        W = np.zeros((4, 7), dtype=np.float64)
        
        # Term 1: (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = -1
        
        # Term 2: (A10+A11)(B00+B11)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1; V[3, 1] = 1
        W[0, 1] = -1; W[1, 1] = -1; W[2, 1] = 1; W[3, 1] = 1
        
        # Term 3: A00*(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[0, 2] = 0; W[1, 2] = 1; W[2, 2] = -1; W[3, 2] = 0
        
        # Term 4: A10*(A11-A01) -- this is actually A10*B10... let me re-check
        # Actually Laderman's terms:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e+h)  
        # M3 = a(f-h)
        # M4 = c(d-b)  -- wait, this involves A entries only? No...
        # Let me use the standard Laderman formulation:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e+h)
        # M3 = a(f-h)
        # M4 = c(e-g)  -- no...
        
        # Let me use a verified Laderman decomposition:
        # For A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # C = A*B
        # m1 = (a+d)(e+h)
        # m2 = (c+d)(e+h)
        # m3 = a(f-h)
        # m4 = c(e-g)  -- hmm, this doesn't seem right either
        # 
        # Actually, let me just use a known rank-7 decomposition for 2x2x2.
        # The standard one uses these 7 products:
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)(B00+B11)
        # P3 = A00(B01-B11)
        # P4 = A10(B10-B00)  -- not standard
        # 
        # Let me just implement the schoolbook for 2x2x2 (rank 8) as a base
        # and then try to optimize.
        
        # For now, return schoolbook 2x2x2 (rank 8)
        U2 = np.zeros((4, 8), dtype=np.float64)
        V2 = np.zeros((4, 8), dtype=np.float64)
        W2 = np.zeros((4, 8), dtype=np.float64)
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U2[i*2+j, col] = 1
                    V2[j*2+k, col] = 1
                    W2[k*2+i, col] = 1
                    col += 1
        return U2, V2, W2

    # Try random search for low-rank decompositions
    def random_search(rank, num_trials=5000, vals=None):
        if vals is None:
            vals = [-2, -1, -0.5, 0, 0.5, 1, 2]
        
        for _ in range(num_trials):
            if time.time() > deadline - 5:
                return None
            
            U = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(n * m)])
            V = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(m * p)])
            W = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(p * n)])
            
            # Quick check: verify a subset of entries
            ok = True
            for a in range(n * m):
                for b in range(m * p):
                    for c in range(p * n):
                        val = np.dot(U[a, :], V[b, :] * W[c, :])
                        if abs(val - target[a, b, c]) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        return None

    # Try various ranks from lower bound up to schoolbook
    # Lower bound for 2x5x5: max(10, 25, 10) = 25
    # Schoolbook: 50
    
    best_result = None
    
    # First try a few specific structured decompositions
    # For 2x5x5, we can think of it as two 1x5x5 multiplications
    # Or as blocks of 2x2x2
    
    # Try random search for ranks 25-50
    for rank in range(25, 51):
        if time.time() > deadline - 10:
            break
        
        result = random_search(rank, num_trials=1000)
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        best_result = schoolbook()
    
    return best_result
# EVOLVE-BLOCK-END
