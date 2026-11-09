"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    # Build tensor from decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                # For matrix mult tensor, T[i,j,k] should be 1
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition (rank = n*m*p)."""
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

def _strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2x2."""
    # Strassen's 7 multiplications
    # U[i,j,r], V[j,k,r], W[i,k,r] for i,j,k in {0,1}
    # r in {0,...,6}
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    U = [[0]*7 for _ in range(4)]  # n*m = 2*2 = 4
    V = [[0]*7 for _ in range(4)]  # m*p = 2*2 = 4
    W = [[0]*7 for _ in range(4)]  # p*n = 2*2 = 4
    
    # Index: U[i*2+j][r], V[j*2+k][r], W[k*2+i][r]
    
    # M1: (A00+A11)*(B00+B11)
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    W[0][0] = 1  # C00 (k=0,i=0)
    W[3][0] = 1  # C11 (k=1,i=1)
    
    # M2: (A10+A11)*B00
    U[2][1] = 1  # A10 (i=1,j=0)
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    W[2][1] = 1  # C10 (k=0,i=1)
    
    # M3: A00*(B01-B11)
    U[0][2] = 1  # A00
    V[1][2] = 1  # B01
    V[3][2] = -1 # B11
    W[1][2] = 1  # C01 (k=1,i=0)
    
    # M4: A11*(B10-B00)
    U[3][3] = 1  # A11
    V[2][3] = 1  # B10
    V[0][3] = -1 # B00
    W[0][3] = 1  # C00
    W[3][3] = 1  # C11
    
    # M5: (A00+A01)*B11
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01 (i=0,j=1)
    V[3][4] = 1  # B11
    W[0][4] = -1 # C00
    W[1][4] = 1  # C01
    
    # M6: (A10-A00)*(B00+B01)
    U[2][5] = 1  # A10
    U[0][5] = -1 # A00
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    W[2][5] = 1  # C10
    
    # M7: (A01-A11)*(B10+B11)
    U[1][6] = 1  # A01
    U[3][6] = -1 # A11
    V[2][6] = 1  # B10
    V[3][6] = 1  # B11
    W[0][6] = -1 # C00
    W[3][6] = 1  # C11
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try structured Strassen-based decomposition
    # For 4x4x9, we can split into blocks
    # Let's try: treat as 2x2 blocks where possible
    
    # Approach: Use Kronecker product structure
    # 4x4x9 = (2x2x4) x (2x2x9) approximately
    # Actually, let's try a direct search with structured initialization
    
    # First, try a simple block decomposition
    # Split the 4x4x9 into sub-problems
    
    # Method 1: Try rank around 100 with random search
    # Method 2: Use structured ansatz
    
    # Let's try a structured approach based on Strassen recursion
    # 4x4x4 can use rank 49 (Strassen), then add 4x4x5 part
    
    # For 4x4x5: schoolbook is 80, but we can do better
    # Actually, let's just try random search for lower rank
    
    # Try structured: combine Strassen for 2x2x2 blocks
    # The 4x4x9 tensor can be seen as having structure
    
    # Let's try: use the fact that we can decompose into 2x2x(9/2) type blocks
    # But 9 is odd, so let's split 9 = 4 + 5
    
    # Part 1: 4x4x4 (first 4 columns) - use Strassen rank 49
    # Part 2: 4x4x5 (last 5 columns) - schoolbook rank 80
    # Total: 129, which is better than 144
    
    # But we can do better. Let's try to optimize the 4x4x5 part.
    
    # Actually, let's implement a proper search
    # Try ranks from 100 to 143 and see if we can find a valid decomposition
    
    # For efficiency, use numpy for verification
    U_np = np.array(best["U"], dtype=float)
    V_np = np.array(best["V"], dtype=float)
    W_np = np.array(best["W"], dtype=float)
    
    # Try to find lower rank using random initialization + verification
    # This is a hard problem, so let's try a few structured approaches
    
    # Structured approach: Use the tensor's symmetry
    # T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor for the structure)
    # Wait, no. The matrix multiplication tensor has T[i,j,k] = delta_{i+k, ...}
    # Actually for C[i,k] = sum_j A[i,j] B[j,k], the tensor entry is T[i,j,k] = 1
    # for all i,j,k. So it's the all-ones tensor of shape (n*m, m*p, p*n)
    # No wait, that's not right either.
    
    # Let me re-examine. The tensor T has indices (i,j,k) where:
    # i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 (the structure tensor for matrix multiplication)
    # A rank-r decomposition means:
    # T[i,j,k] = sum_{r=0}^{r-1} U[i,j,r] V[j,k,r] W[k,i,r]
    # (using the trace convention W[k*n+i, r])
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is the "all-ones" constraint.
    
    # For the all-ones tensor, the rank is related to the matrix multiplication rank.
    
    # Let's try a simple construction:
    # If we set U[i,j,r] = a_{i,j,r}, V[j,k,r] = b_{j,k,r}, W[k,i,r] = c_{k,i,r}
    # such that sum_r a_{i,j,r} b_{j,k,r} c_{k,i,r} = 1
    
    # One trivial solution: rank = n*m*p with a,b,c being basis vectors.
    
    # For lower rank, we need clever combinations.
    
    # Let's try: for 4x4x9, use a 2x2 block structure
    # Split i into (i1, i2) where i1 in {0,1}, i2 in {0,1} (so i = 2*i1+i2)
    # Similarly j = 2*j1+j2, and k = 2*k1+k2 + k_extra where k_extra in {0,1,2}
    # This is messy.
    
    # Alternative: Just try random search for ranks 100-143
    # But this is computationally expensive.
    
    # Let's try a smarter approach: use the fact that the tensor is "simple"
    # and try low-rank approximations with structured entries.
    
    # For now, let's try a few specific ranks with random search
    for target_rank in [120, 110, 100, 90, 80]:
        if time.time() > deadline - 30:
            break
        
        # Try random initialization
        for trial in range(5):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(trial * 1000 + target_rank)
            U_r = np.random.randn(n*m, target_rank)
            V_r = np.random.randn(m*p, target_rank)
            W_r = np.random.randn(p*n, target_rank)
            
            # Verify
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This is a 3D tensor contraction
            # Efficient check: for each (i,k), the matrix U[:,i*m+j] ... 
            # Actually, let's just check all (i,j,k) tuples
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                        if abs(s - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best = {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
                break
    
    if time.time() > deadline - 10:
        return best
    
    # If we found a better rank, try to polish with integer/half-integer entries
    # For now, return what we have
    
    return best
# EVOLVE-BLOCK-END
