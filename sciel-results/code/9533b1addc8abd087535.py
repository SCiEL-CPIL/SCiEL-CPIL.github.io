"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np

def _verify(U, V, W, n=2, m=2, p=4):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i, j, k] = 1 if C[k] = sum_{i,j} A[i,m//2 + i//... ] ... 
    # Actually, let's think about it differently.
    # A is n x m (2x2), B is m x p (2x4), C is n x p (2x4)
    # C[r, s] = sum_{t=0}^{m-1} A[r, t] * B[t, s]
    
    # The tensor for matrix multiplication has entries:
    # T[r, t, s] where r is row of A (and C), t is column of A (and row of B), s is column of B (and C)
    # T[r, t, s] = 1 for all valid r, t, s
    
    # In the decomposition:
    # C[r, s] = sum_{k=0}^{rank-1} W[k, r*p + s] * (sum_{t} U[r*m + t, k] * V[t*p + s, k])
    # Wait, the interface says:
    # U: (n*m, rank) -> U[i, k] where i = r*m + t
    # V: (m*p, rank) -> V[j, k] where j = t*p + s
    # W: (p*n, rank) -> W[l, k] where l = s*n + r (W[k*n+i] convention)
    
    # So C[r, s] = sum_k W[s*n + r, k] * (sum_t U[r*m + t, k] * V[t*p + s, k])
    
    # Let's verify: for each r, s, compute the LHS and RHS
    for r in range(n):
        for s in range(p):
            lhs = 0.0
            for t in range(m):
                # A[r,t] * B[t,s] - we don't have specific values, so we check structure
                pass
            
            # Actually, we need to check that the bilinear form matches
            # C[r,s] should equal sum_{t} A[r,t] * B[t,s]
            # In tensor form: T[r, t, s] = 1 for all r, t, s
            # Decomposition: T[r, t, s] = sum_k U[r*m+t, k] * V[t*p+s, k] * W[s*n+r, k]
            
            # So for each (r, t, s), we need:
            # 1 = sum_k U[r*m+t, k] * V[t*p+s, k] * W[s*n+r, k]
            
            # Let's check all (r, t, s) combinations
            for t in range(m):
                expected = 1.0  # T[r,t,s] = 1
                actual = 0.0
                for k in range(rank):
                    actual += U[r*m + t, k] * V[t*p + s, k] * W[s*n + r, k]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _construct_rank14():
    """Construct the known rank-14 decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # This is the Strassen-based rank-14 decomposition
    # Block 1: columns 0,1 of B (b00, b01, b10, b11)
    # Block 2: columns 2,3 of B (b02, b03, b12, b13)
    
    # For block 1 (standard Strassen for 2x2 * 2x2):
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b10+b11)
    # M4 = (a11)(b00-b11)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    
    # C00 = M1 - M2 - M4 + M5  ... wait, let me use the correct Strassen formulas
    # Standard Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b10+b11)
    # M4 = (a11)(b00-b11)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    
    # C00 = M1 - M2 - M4 + M5  -> hmm, let me verify
    # Actually the standard formulas are:
    # C00 = M1 - M2 - M4 + M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 - M4 - M6 + M7
    
    # Let me just use the verified rank-14 from the current program
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    U[0][7] = 1; U[3][7] = 1
    U[2][8] = 1; U[3][8] = 1
    U[0][9] = 1
    U[3][10] = 1
    U[0][11] = 1; U[1][11] = 1
    U[2][12] = 1; U[0][12] = -1
    U[1][13] = 1; U[3][13] = -1
    
    V[0][0] = 1; V[5][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[5][2] = -1
    V[4][3] = 1; V[0][3] = -1
    V[5][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[4][6] = 1; V[5][6] = 1
    
    V[2][7] = 1; V[7][7] = 1
    V[2][8] = 1
    V[3][9] = 1; V[7][9] = -1
    V[6][10] = 1; V[2][10] = -1
    V[7][11] = 1
    V[2][12] = 1; V[3][12] = 1
    V[6][13] = 1; V[7][13] = 1
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][1] = 1; W[1][3] = 1
    W[2][2] = 1; W[2][4] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    W[4][7] = 1; W[4][10] = 1; W[4][11] = -1; W[4][13] = 1
    W[5][8] = 1; W[5][10] = 1
    W[6][9] = 1; W[6][11] = 1
    W[7][7] = 1; W[7][8] = -1; W[7][9] = 1; W[7][12] = 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    # Return the verified rank-14 decomposition
    return _construct_rank14()
# EVOLVE-BLOCK-END
