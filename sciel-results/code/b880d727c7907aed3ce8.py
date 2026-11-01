import numpy as np

def construct():
    """
    56-rank Strassen block decomposition for 2x4 * 4x8 matrix multiplication.
    
    A (2x4) is split into A_left (2x2) and A_right (2x2).
    B (4x8) is split into B_top (2x8) and B_bottom (2x8).
    C = A_left * B_top + A_right * B_bottom
    
    Each 2x2 * 2x2 product uses Strassen (7 multiplications).
    4 column blocks * 2 (top/bottom) * 7 = 56 total.
    """
    n, m, p = 2, 4, 8
    rank = 56
    
    U = np.zeros((n * m, rank), dtype=float)  # (8, 56)
    V = np.zeros((m * p, rank), dtype=float)  # (32, 56)
    W = np.zeros((p * n, rank), dtype=float)  # (16, 56)
    
    # U row indices for A (2x4):
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    # A_left = [[U[0],U[1]],[U[4],U[5]]]
    # A_right = [[U[2],U[3]],[U[6],U[7]]]
    
    # V row indices for B (4x8): V[j*p + k] = B[j,k]
    # B[0,k] = V[k], B[1,k] = V[8+k], B[2,k] = V[16+k], B[3,k] = V[24+k]
    # B_top row j, col k: V[j*8+k] for j in {0,1}
    # B_bottom row j, col k: V[(2+j)*8+k] for j in {0,1}
    
    # W row indices for C (2x8): W[k*n + i] = C[i,k]
    # C[0,k] = W[2k], C[1,k] = W[2k+1]
    
    r = 0
    for j in range(4):
        # A_left * BTj where BTj = B_top[:, 2j:2j+2]
        # BTj = [[V[2j], V[2j+1]], [V[8+2j], V[8+2j+1]]]
        # A_left = [[U[0],U[1]],[U[4],U[5]]]
        
        # Strassen M1 = (x00+x11)*(y00+y11)
        U[0, r] = 1; U[5, r] = 1
        V[2*j, r] = 1; V[8+2*j, r] = 1
        W[4*j, r] = 1      # Z00 += M1
        W[4*j+2, r] = 0    # Z01
        W[4*j+1, r] = 0    # Z10
        W[4*j+3, r] = 1    # Z11 += M1
        r += 1
        
        # M2 = (x10+x11)*y00
        U[4, r] = 1; U[5, r] = 1
        V[2*j, r] = 1
        W[4*j, r] = 0
        W[4*j+2, r] = 0
        W[4*j+1, r] = 1    # Z10 += M2
        W[4*j+3, r] = -1   # Z11 -= M2
        r += 1
        
        # M3 = x00*(y01-y11)
        U[0, r] = 1
        V[2*j+1, r] = 1; V[8+2*j+1, r] = -1
        W[4*j, r] = 0
        W[4*j+2, r] = 1    # Z01 += M3
        W[4*j+1, r] = 0
        W[4*j+3, r] = 1    # Z11 += M3
        r += 1
        
        # M4 = x11*(y10-y00)
        U[5, r] = 1
        V[8+2*j, r] = 1; V[2*j, r] = -1
        W[4*j, r] = 1      # Z00 += M4
        W[4*j+2, r] = 0
        W[4*j+1, r] = 1    # Z10 += M4
        W[4*j+3, r] = 0
        r += 1
        
        # M5 = (x00+x01)*y11
        U[0, r] = 1; U[1, r] = 1
        V[8+2*j+1, r] = 1
        W[4*j, r] = -1     # Z00 -= M5
        W[4*j+2, r] = 1    # Z01 += M5
        W[4*j+1, r] = 0
        W[4*j+3, r] = 0
        r += 1
        
        # M6 = (x10-x00)*(y00+y01)
        U[4, r] = 1; U[0, r] = -1
        V[2*j, r] = 1; V[2*j+1, r] = 1
        W[4*j, r] = 0
        W[4*j+2, r] = 0
        W[4*j+1, r] = 0
        W[4*j+3, r] = 1    # Z11 += M6
        r += 1
        
        # M7 = (x11-x01)*(y10+y11)
        U[5, r] = 1; U[1, r] = -1
        V[8+2*j, r] = 1; V[8+2*j+1, r] = 1
        W[4*j, r] = 1      # Z00 += M7
        W[4*j+2, r] = 0
        W[4*j+1, r] = 0
        W[4*j+3, r] = 1    # Z11 += M7
        r += 1
        
        # A_right * BBj where BBj = B_bottom[:, 2j:2j+2]
        # BBj = [[V[16+2j], V[16+2j+1]], [V[24+2j], V[24+2j+1]]]
        # A_right = [[U[2],U[3]],[U[6],U[7]]]
        
        # M1 = (x00+x11)*(y00+y11)
        U[2, r] = 1; U[7, r] = 1
        V[16+2*j, r] = 1; V[24+2*j, r] = 1
        W[4*j, r] = 1
        W[4*j+2, r] = 0
        W[4*j+1, r] = 0
        W[4*j+3, r] = 1
        r += 1
        
        # M2 = (x10+x11)*y00
        U[6, r] = 1; U[7, r] = 1
        V[16+2*j, r] = 1
        W[4*j, r] = 0
        W[4*j+2, r] = 0
        W[4*j+1, r] = 1
        W[4*j+3, r] = -1
        r += 1
        
        # M3 = x00*(y01-y11)
        U[2, r] = 1
        V[16+2*j+1, r] = 1; V[24+2*j+1, r] = -1
        W[4*j, r] = 0
        W[4*j+2, r] = 1
        W[4*j+1, r] = 0
        W[4*j+3, r] = 1
        r += 1
        
        # M4 = x11*(y10-y00)
        U[7, r] = 1
        V[24+2*j, r] = 1; V[16+2*j, r] = -1
        W[4*j, r] = 1
        W[4*j+2, r] = 0
        W[4*j+1, r] = 1
        W[4*j+3, r] = 0
        r += 1
        
        # M5 = (x00+x01)*y11
        U[2, r] = 1; U[3, r] = 1
        V[24+2*j+1, r] = 1
        W[4*j, r] = -1
        W[4*j+2, r] = 1
        W[4*j+1, r] = 0
        W[4*j+3, r] = 0
        r += 1
        
        # M6 = (x10-x00)*(y00+y01)
        U[6, r] = 1; U[2, r] = -1
        V[16+2*j, r] = 1; V[16+2*j+1, r] = 1
        W[4*j, r] = 0
        W[4*j+2, r] = 0
        W[4*j+1, r] = 0
        W[4*j+3, r] = 1
        r += 1
        
        # M7 = (x11-x01)*(y10+y11)
        U[7, r] = 1; U[3, r] = -1
        V[24+2*j, r] = 1; V[24+2*j+1, r] = 1
        W[4*j, r] = 1
        W[4*j+2, r] = 0
        W[4*j+1, r] = 0
        W[4*j+3, r] = 1
        r += 1
    
    assert r == 56, f"Expected rank 56, got {r}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
