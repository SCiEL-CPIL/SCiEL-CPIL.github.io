"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank: Try to beat schoolbook (36)
    # Strategy: Use Strassen's algorithm for the top-left 2x2 block of the 3x3 part.
    # This reduces 8 schoolbook terms to 7 Strassen terms.
    # Remaining terms: 36 - 8 + 7 = 35.
    
    # We will construct the factors U, V, W explicitly.
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (12, rank)
    # W: (p*n, rank) = (12, rank)
    
    # We use a list of terms. Each term is a tuple (u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w)
    # Actually, U, V, W are matrices.
    # Let's build the matrices directly.
    
    # We will use rank 35 initially.
    # If we can find a better rank, we update.
    
    # For now, implement the Strassen reduction.
    
    # The 8 schoolbook terms for i,j,k in {0,1} are removed.
    # The 7 Strassen terms are added.
    
    # Strassen terms for 2x2 block (i,j,k in 0,1):
    # M1: (A00+A11)(B00+B11) -> u=(1,1,0,0), v=(1,0,1,0), w=(1,1,0,0) ?
    # Indices in U: i*m+j. For 2x2 block, i,j in 0,1.
    # U indices: 0,1,3,4 (0*3+0, 0*3+1, 1*3+0, 1*3+1)
    # V indices: j*p+k. For 2x2 block, j,k in 0,1.
    # V indices: 0,1,4,5 (0*4+0, 0*4+1, 1*4+0, 1*4+1)
    # W indices: k*n+i. For 2x2 block, k,i in 0,1.
    # W indices: 0,1,3,4 (0*3+0, 0*3+1, 1*3+0, 1*3+1)
    
    # Let's define the Strassen factors.
    # M1 = (A00 + A11)(B00 + B11)
    # u1: A00=1, A11=1 -> U[0]=1, U[4]=1
    # v1: B00=1, B11=1 -> V[0]=1, V[5]=1
    # w1: C00=1, C11=1 -> W[0]=1, W[4]=1
    
    # M2 = (A10 + A11)B00
    # u2: A10=1, A11=1 -> U[3]=1, U[4]=1
    # v2: B00=1 -> V[0]=1
    # w2: C10=1 -> W[3]=1
    
    # M3 = A00(B10 - B11)
    # u3: A00=1 -> U[0]=1
    # v3: B10=1, B11=-1 -> V[4]=1, V[5]=-1
    # w3: C01=1 -> W[1]=1
    
    # M4 = A11(B01 - B00)
    # u4: A11=1 -> U[4]=1
    # v4: B01=1, B00=-1 -> V[1]=1, V[0]=-1
    # w4: C00=1, C11=-1 -> W[0]=1, W[4]=-1
    
    # M5 = (A00 + A01)B11
    # u5: A00=1, A01=1 -> U[0]=1, U[1]=1
    # v5: B11=1 -> V[5]=1
    # w5: C00=1, C11=1 -> W[0]=1, W[4]=1
    
    # M6 = (A10 - A00)(B00 + B01)
    # u6: A10=1, A00=-1 -> U[3]=1, U[0]=-1
    # v6: B00=1, B01=1 -> V[0]=1, V[1]=1
    # w6: C11=1 -> W[4]=1
    
    # M7 = (A01 - A11)(B10 + B11)
    # u7: A01=1, A11=-1 -> U[1]=1, U[4]=-1
    # v7: B10=1, B11=1 -> V[4]=1, V[5]=1
    # w7: C10=1 -> W[3]=1
    
    # These 7 terms replace the 8 schoolbook terms for (i,j,k) in {0,1}^3.
    
    # The other 28 terms are schoolbook.
    # Total rank = 7 + 28 = 35.
    
    rank = 35
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Add Strassen terms (columns 0-6)
    # M1
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[4, 0] = 1
    
    # M2
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[3, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[4, 2] = 1; V[5, 2] = -1
    W[1, 2] = 1
    
    # M4
    U[4, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[4, 3] = -1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[5, 4] = 1
    W[0, 4] = 1; W[4, 4] = 1
    
    # M6
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[4, 5] = 1
    
    # M7
    U[1, 6] = 1; U[4, 6] = -1
    V[4, 6] = 1; V[5, 6] = 1
    W[3, 6] = 1
    
    # Add Schoolbook terms for the rest (columns 7-34)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this term is in the 2x2 block handled by Strassen
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Convert to lists of lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
