"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) tensor decomposition for 4x5 @ 5x6.
    Uses recursive block decomposition with Strassen for 2x2x2 blocks.
    """
    start_time = time.time()
    deadline = 110.0
    
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    def strassen_2x2x2():
        # U, V, W are lists of rows.
        # U: (2*2=4) x 7
        # V: (2*2=4) x 7
        # W: (2*2=4) x 7
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Indices:
        # U[i*2 + j]
        # V[j*2 + k]
        # W[k*2 + i]
        
        # Strassen terms:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map these to U, V, W.
        # Term r corresponds to:
        # U_r(A) = sum U_ij,r A_ij
        # V_r(B) = sum V_jk,r B_jk
        # C_ik += W_ki,r U_r V_r
        
        # M1:
        # U: A11, A22 -> 1
        # V: B11, B22 -> 1
        # W: C11(1), C22(1) -> 1
        U[0*2+0][0] = 1; U[1*2+1][0] = 1
        V[0*2+0][0] = 1; V[1*2+1][0] = 1
        W[0*2+0][0] = 1; W[1*2+1][0] = 1
        
        # M2:
        # U: A21, A22 -> 1
        # V: B11 -> 1
        # W: C21(1) -> 1
        U[1*2+0][1] = 1; U[1*2+1][1] = 1
        V[0*2+0][1] = 1
        W[0*2+1][1] = 1
        
        # M3:
        # U: A11 -> 1
        # V: B12(1), B22(-1)
        # W: C12(1), C22(1)
        U[0*2+0][2] = 1
        V[0*2+1][2] = 1; V[1*2+1][2] = -1
        W[1*2+0][2] = 1; W[1*2+1][2] = 1
        
        # M4:
        # U: A22 -> 1
        # V: B21(1), B11(-1)
        # W: C11(-1), C21(1)
        U[1*2+1][3] = 1
        V[1*2+0][3] = 1; V[0*2+0][3] = -1
        W[0*2+0][3] = -1; W[0*2+1][3] = 1
        
        # M5:
        # U: A11(1), A12(1)
        # V: B22 -> 1
        # W: C11(1), C12(1)
        U[0*2+0][4] = 1; U[0*2+1][4] = 1
        V[1*2+1][4] = 1
        W[0*2+0][4] = 1; W[1*2+0][4] = 1
        
        # M6:
        # U: A21(1), A11(-1)
        # V: B11(1), B12(1)
        # W: C22(1)
        U[1*2+0][5] = 1; U[0*2+0][5] = -1
        V[0*2+0][5] = 1; V[0*2+1][5] = 1
        W[1*2+1][5] = 1
        
        # M7:
        # U: A12(1), A22(-1)
        # V: B21(1), B22(1)
        # W: C11(-1)
        U[0*2+1][6] = 1; U[1*2+1][6] = -1
        V[1*2+0][6] = 1; V[1*2+1][6] = 1
        W[0*2+0][6] = -1
        
        return U, V, W

    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][c] = 1
                    V[j*p + k][c] = 1
                    W[k*n + i][c] = 1
                    c += 1
        return U, V, W

    def get_decomp(n, m, p):
        if n == 2 and m == 2 and p == 2:
            return strassen_2x2x2()
        
        if n == 1 or m == 1 or p == 1:
            return schoolbook(n, m, p)
        
        # Split the dimension that allows for better reduction
        # Heuristic: Split n, then m, then p
        if n > 1:
            n1 = n // 2
            n2 = n - n1
            U1, V1, W1 = get_decomp(n1, m, p)
            U2, V2, W2 = get_decomp(n2, m, p)
            r1 = len(U1[0])
            r2 = len(U2[0])
            r_tot = r1 + r2
            
            U = [[0]*r_tot for _ in range(n*m)]
            V = [[0]*r_tot for _ in range(m*p)]
            W = [[0]*r_tot for _ in range(p*n)]
            
            for i in range(n1):
                for j in range(m):
                    for c in range(r1):
                        U[i*m + j][c] = U1[i*m1 + j][c]
                # V and W are full for this block
                for j in range(m):
                    for k in range(p):
                        for c in range(r1):
                            V[j*p + k][c] = V1[j*p + k][c]
                for k in range(p):
                    for ii in range(n1):
                        for c in range(r1):
                            W[k*n1 + ii][c] = W1[k*n1 + ii][c] # Note: W1 is sized p x n1? No, W1 is p x n1? 
                            # W1 shape: (p*n1, r1). Index k*n1 + ii.
                            # But global W index is k*n + ii.
                            # We need to map local W1 indices to global W indices.
                            # Local W1 row: k*n1 + ii.
                            # Global W row: k*n + ii.
                            
            # This manual mapping is error prone. Let's use a helper.
            pass

    # Given the complexity and time limit, and that 108 is a solid improvement, 
    # we will implement the recursive combination carefully.
    
    def combine_blocks(U1, V1, W1, U2, V2, W2, n1, n2, m, p):
        r1 = len(U1[0])
        r2 = len(U2[0])
        r_tot = r1 + r2
        n = n1 + n2
        
        U = [[0]*r_tot for _ in range(n*m)]
        V = [[0]*r_tot for _ in range(m*p)]
        W = [[0]*r_tot for _ in range(p*n)]
        
        # Block 1 (top n1 rows)
        for i in range(n1):
            for j in range(m):
                for c in range(r1):
                    U[i*m + j][c] = U1[i*m + j][c]
                    V[j*p + k if False else j*p + k][c] # V is same for both blocks in terms of rows, but columns are disjoint
                    # V1 and V2 have same row indices (j*p+k).
                    # We just copy them to the first r1 columns.
        
        # Correct V and W filling
        for j in range(m):
            for k in range(p):
                for c in range(r1):
                    V[j*p + k][c] = V1[j*p + k][c]
                for c in range(r2):
                    V[j*p + k][r1 + c] = V2[j*p + k][c]
                    
        for k in range(p):
            for i in range(n1):
                for c in range(r1):
                    W[k*n + i][c] = W1[k*n1 + i][c]
            for i in range(n2):
                for c in range(r2):
                    W[k*n + (n1 + i)][r1 + c] = W2[k*n2 + i][c]
                    
        for i in range(n2):
            for j in range(m):
                for c in range(r2):
                    U[(n1 + i)*m + j][r1 + c] = U2[i*m + j][c]
                    
        return U, V, W

    # Recursively build
    U, V, W = get_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
