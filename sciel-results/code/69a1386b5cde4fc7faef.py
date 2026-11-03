"""Program implementing Strassen-based rank-54 decomposition for 2x5x6 matrix multiplication."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # We decompose M_{2,5,6} into blocks:
    # Split m=5 into [2, 2, 1]
    # Split p=6 into [2, 2, 2]
    # 
    # A is 2x5, split into A1 (2x2), A2 (2x2), A3 (2x1)
    # B is 5x6, split into B1 (2x6), B2 (2x6), B3 (1x6)
    # C = A1*B1 + A2*B2 + A3*B3
    #
    # Each A1*B1 is M_{2,2,6}, further split B1 into 3 blocks of 2x2:
    #   A1*B1 = A1*B11 + A1*B12 + A1*B13, each M_{2,2,2} -> Strassen rank 7
    # So A1*B1 costs 3*7 = 21
    # Similarly A2*B2 costs 21
    # A3*B3 is M_{2,1,6}, rank = 2*1*6 = 12 (schoolbook, no improvement possible)
    # Total: 21 + 21 + 12 = 54
    
    rank = 54
    U = [[0.0] * rank for _ in range(n * m)]   # 10 x 54
    V = [[0.0] * rank for _ in range(m * p)]   # 30 x 54
    W = [[0.0] * rank for _ in range(p * n)]   # 12 x 54
    
    r = 0  # current rank index
    
    # Helper: add a Strassen 2x2x2 block
    # A_block is 2x2 with rows (i0, i1), cols (j0, j1) in the original A
    # B_block is 2x2 with rows (j0, j1), cols (k0, k1) in the original B
    # C indices: k in {k0, k1}, i in {i0, i1}
    # W index convention: W[k*n + i, r]
    #
    # Strassen products:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B11-B21)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    #
    # C11 = P1 - P4 + P6
    # C12 = P3 + P5
    # C21 = P2 + P7
    # C22 = P1 - P3 + P7
    
    def strassen_2x2x2(i0, j0, k0):
        """Add 7 rank-1 terms for the 2x2x2 block starting at (i0, j0, k0)."""
        nonlocal r
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        
        # U indices: i*m + j  -> for i in {i0,i1}, j in {j0,j1}
        # V indices: j*p + k  -> for j in {j0,j1}, k in {k0,k1}
        # W indices: k*n + i  -> for k in {k0,k1}, i in {i0,i1}
        
        u_ii0_jj0 = i0 * m + j0
        u_ii0_jj1 = i0 * m + j1
        u_ii1_jj0 = i1 * m + j0
        u_ii1_jj1 = i1 * m + j1
        
        v_ij0_jk0 = j0 * p + k0
        v_ij0_jk1 = j0 * p + k1
        v_ij1_jk0 = j1 * p + k0
        v_ij1_jk1 = j1 * p + k1
        
        w_ik0_ii0 = k0 * n + i0
        w_ik0_ii1 = k0 * n + i1
        w_ik1_ii0 = k1 * n + i0
        w_ik1_ii1 = k1 * n + i1
        
        # P1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        U[u_ii0_jj0][r] = 1
        U[u_ii1_jj1][r] = 1
        V[v_ij0_jk0][r] = 1
        V[v_ij1_jk1][r] = 1
        # C11 += P1, C22 += P1
        W[w_ik0_ii0][r] = 1
        W[w_ik1_ii1][r] = 1
        r += 1
        
        # P2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        U[u_ii1_jj0][r] = 1
        U[u_ii1_jj1][r] = 1
        V[v_ij0_jk0][r] = 1
        # C21 += P2
        W[w_ik0_ii1][r] = 1
        r += 1
        
        # P3 = A[i0,j0] * (B[j1,k0]-B[j1,k1])
        U[u_ii0_jj0][r] = 1
        V[v_ij1_jk0][r] = 1
        V[v_ij1_jk1][r] = -1
        # C12 += P3, C22 -= P3
        W[w_ik0_ii0][r] = 1
        W[w_ik1_ii1][r] = -1
        r += 1
        
        # P4 = A[i1,j1] * (B[j0,k0]-B[j0,k1])
        U[u_ii1_jj1][r] = 1
        V[v_ij0_jk0][r] = 1
        V[v_ij0_jk1][r] = -1
        # C11 -= P4
        W[w_ik0_ii0][r] = -1
        r += 1
        
        # P5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        U[u_ii0_jj0][r] = 1
        U[u_ii0_jj1][r] = 1
        V[v_ij1_jk1][r] = 1
        # C12 += P5
        W[w_ik1_ii0][r] = 1
        r += 1
        
        # P6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        U[u_ii1_jj0][r] = 1
        U[u_ii0_jj0][r] = -1
        V[v_ij0_jk0][r] = 1
        V[v_ij0_jk1][r] = 1
        # C11 += P6
        W[w_ik0_ii0][r] = 1
        r += 1
        
        # P7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        U[u_ii0_jj1][r] = 1
        U[u_ii1_jj1][r] = -1
        V[v_ij1_jk0][r] = 1
        V[v_ij1_jk1][r] = 1
        # C21 += P7, C22 += P7
        W[w_ik0_ii1][r] = 1
        W[w_ik1_ii1][r] = 1
        r += 1
    
    # Block A1 (rows 0-1 of A, cols 0-1 of A) times B1 (rows 0-1 of B, all 6 cols)
    # B1 split into B11 (cols 0-1), B12 (cols 2-3), B13 (cols 4-5)
    for k0 in [0, 2, 4]:
        strassen_2x2x2(i0=0, j0=0, k0=k0)
    
    # Block A2 (rows 0-1 of A, cols 2-3 of A) times B2 (rows 2-3 of B, all 6 cols)
    # B2 split into B21 (cols 0-1), B22 (cols 2-3), B23 (cols 4-5)
    for k0 in [0, 2, 4]:
        strassen_2x2x2(i0=0, j0=2, k0=k0)
    
    # Block A3 (rows 0-1 of A, col 4 of A) times B3 (row 4 of B, all 6 cols)
    # This is M_{2,1,6}, schoolbook rank 12
    for i in range(n):
        for k in range(p):
            U[i * m + 4][r] = 1
            V[4 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    
    # Convert to proper types - ensure they are clean rationals
    # Snap to half-integers (should already be integers)
    def snap(val):
        if abs(val - round(val)) < 1e-9:
            return int(round(val))
        elif abs(val - round(val) * 0.5) < 1e-9:
            return round(val)
        return val
    
    U = [[snap(val) for val in row] for row in U]
    V = [[snap(val) for val in row] for row in V]
    W = [[snap(val) for val in row] for row in W]
    
    assert r == rank, f"Expected rank {rank}, got {r}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
