"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = 2, 3, 7
    # We will try to find a low-rank decomposition.
    # Known bounds for 2x3x7:
    # Schoolbook is 42.
    # We can use block decompositions.
    # Split B into 2x7 and 1x7?
    # A is 2x3.
    
    # Let's try a specific construction based on Strassen for 2x2x2 and additions.
    # Or use a random search / algebraic method?
    # Given the time constraint and the need for exact integer/half-integer entries,
    # random search is risky for exactness.
    
    # Let's stick to a structured block approach.
    # A is 2x3. B is 3x7.
    # Let's partition A into A1 (2x2) and A2 (2x1).
    # Let's partition B into B1 (2x7) and B2 (1x7).
    # C = A1 B1 + A2 B2.
    
    # A1 B1 is 2x2x7.
    # A2 B2 is 2x1x7.
    
    # Rank of 2x2x7:
    # We can use Strassen for the first 2 columns of B1 (rank 7).
    # Remaining 5 columns of B1: 2x2x5.
    # Rank of 2x2x5 is 2*5 = 10? No, maybe less.
    # Actually, rank of 2x2xp is max(2p, p+2)? No.
    # For 2x2x2, rank 7.
    # For 2x2x1, rank 2.
    # For 2x2xp, it is known that rank is 2p for p>=3? No.
    # Actually, for 2x2xp, the rank is 2p if p is odd?
    # Let's just use schoolbook for the remainder to be safe, unless we find a better way.
    
    # Strategy:
    # 1. A1 B1: Split B1 into B1a (2x2) and B1b (2x5).
    #    Rank(A1 B1a) = 7 (Strassen).
    #    Rank(A1 B1b) = 10 (Schoolbook 2x2x5).
    #    Total for A1 B1 = 17.
    # 2. A2 B2: 2x1x7. Rank = 7 (Schoolbook).
    # Total rank = 17 + 7 = 24.
    
    # Can we do better?
    # Maybe split B into 3 blocks of size 2?
    # B is 3x7.
    # A is 2x3.
    # Let's try to use a known low rank decomposition for 2x3x7.
    # According to literature, the rank of 2x3x7 is between 24 and 30?
    # Actually, I recall that for 2xn x nxm, there are bounds.
    
    # Let's try to implement the 24-rank decomposition described above.
    
    rank = 24
    U = np.zeros((n*m, rank)) # 6x24
    V = np.zeros((m*p, rank)) # 21x24
    W = np.zeros((p*n, rank)) # 14x24
    
    # Indices:
    # A[i,j], i in 0..1, j in 0..2
    # B[j,k], j in 0..2, k in 0..6
    # C[i,k], i in 0..1, k in 0..6
    
    # Partition A:
    # A1: rows 0,1; cols 0,1 (indices 0,1 in j)
    # A2: rows 0,1; col 2 (index 2 in j)
    
    # Partition B:
    # B1: rows 0,1 (indices 0,1 in j); cols 0..6
    # B2: row 2 (index 2 in j); cols 0..6
    
    # Term 1: A1 B1a (2x2x2) using Strassen (Rank 7)
    # A1 = [[A00, A01], [A10, A11]]
    # B1a = [[B00, B01], [B10, B11]]
    # Indices in U, V, W:
    # U index: i*m + j. For A1, i in 0..1, j in 0..1.
    # V index: j*p + k. For B1a, j in 0..1, k in 0..1.
    # W index: k*n + i. For C, k in 0..1, i in 0..1.
    
    # Strassen factors:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C01 += M1, C10 += M1, C11 += M1 ... wait.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    col = 0
    # M1
    # U: A00+A11 -> U[0,0]=1, U[3,0]=1 (i=0,j=0 and i=1,j=1)
    # V: B00+B11 -> V[0,0]=1, V[1+p,0]=1 (j=0,k=0 and j=1,k=1)
    # W: C00, C01, C10, C11 -> W[0*n+0]=1, W[1*n+0]=1, W[0*n+1]=1, W[1*n+1]=1
    U[0, col] += 1; U[3, col] += 1
    V[0, col] += 1; V[1*p, col] += 1
    W[0, col] += 1; W[1, col] += 1; W[n, col] += 1; W[n+1, col] += 1
    col += 1
    
    # M2 = (A10+A11)B00
    # U: A10+A11 -> U[1,0]=1 (i=1,j=0), U[3,0]=1 (i=1,j=1)
    # V: B00 -> V[0,0]=1
    # W: C10, C11 -> W[n,0]=1, W[n+1,0]=1
    U[1, col] += 1; U[3, col] += 1
    V[0, col] += 1
    W[n, col] -= 1; W[n+1, col] += 1 # C11 = ... - M2
    col += 1
    
    # M3 = A00(B10-B11)
    # U: A00 -> U[0,0]=1
    # V: B10-B11 -> V[1*p,0]=1, V[1*p+1,0]=-1
    # W: C01, C11 -> W[1,0]=1, W[n+1,0]=1
    U[0, col] += 1
    V[p, col] += 1; V[p+1, col] -= 1
    W[1, col] += 1; W[n+1, col] += 1
    col += 1
    
    # M4 = A11(B01-B00)
    # U: A11 -> U[3,0]=1
    # V: B01-B00 -> V[1,0]=1, V[0,0]=-1
    # W: C00, C10 -> W[0,0]=1, W[n,0]=1
    U[3, col] += 1
    V[1, col] += 1; V[0, col] -= 1
    W[0, col] += 1; W[n, col] += 1
    col += 1
    
    # M5 = (A00+A01)B11
    # U: A00+A01 -> U[0,0]=1, U[1,0]=1
    # V: B11 -> V[1*p+1,0]=1
    # W: C00, C01 -> W[0,0]=-1, W[1,0]=1
    U[0, col] += 1; U[1, col] += 1
    V[p+1, col] += 1
    W[0, col] -= 1; W[1, col] += 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    # U: A10-A00 -> U[1,0]=1, U[0,0]=-1
    # V: B00+B01 -> V[0,0]=1, V[1,0]=1
    # W: C11 -> W[n+1,0]=1
    U[1, col] += 1; U[0, col] -= 1
    V[0, col] += 1; V[1, col] += 1
    W[n+1, col] += 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    # U: A01-A11 -> U[1,0]=1, U[3,0]=-1
    # V: B10+B11 -> V[p,0]=1, V[p+1,0]=1
    # W: C00 -> W[0,0]=1
    U[1, col] += 1; U[3, col] -= 1
    V[p, col] += 1; V[p+1, col] += 1
    W[0, col] += 1
    col += 1
    
    # Term 2: A1 B1b (2x2x5) Schoolbook
    # A1: i in 0..1, j in 0..1
    # B1b: j in 0..1, k in 2..6
    for i in range(2):
        for j in range(2):
            for k in range(2, 7):
                U[i*m + j, col] += 1
                V[j*p + k, col] += 1
                W[k*n + i, col] += 1
                col += 1
    
    # Term 3: A2 B2 (2x1x7) Schoolbook
    # A2: i in 0..1, j=2
    # B2: j=2, k in 0..6
    for i in range(2):
        for k in range(7):
            U[i*m + 2, col] += 1
            V[2*p + k, col] += 1
            W[k*n + i, col] += 1
            col += 1
            
    # Check rank used
    assert col == rank, f"Rank mismatch: {col} vs {rank}"
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
