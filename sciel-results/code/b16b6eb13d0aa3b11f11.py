"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (30, rank)
    # W: (p*n, rank) = (20, rank)
    
    # Rank 55 solution using 5 blocks of 2x3x2 multiplication
    # Each block uses Strassen (7) + Outer Product (4) = 11
    rank = 55
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Strassen for A' (2x2) and B' (2x2)
        # A' rows 0,1 cols 0,1
        # B' rows 0,1 cols k0,k1
        
        # M1 = (A00 + A11) (B0k0 + B1k1)
        U[0, col] = 1; U[3, col] = 1  # A00, A11 (indices 0*3+0, 1*3+1)
        V[0, col] = 1; V[10+k1, col] = 1 # B0k0, B1k1 (indices 0*10+k0, 1*10+k1)
        # Result: P00 += M1, P11 += M1
        W[k0*2+0, col] = 1; W[k1*2+1, col] = 1
        col += 1
        
        # M2 = (A10 + A11) B0k0
        U[3, col] = 1; U[4, col] = 1 # A10, A11
        V[0, col] = 1 # B0k0
        # Result: P10 += M2
        W[k0*2+1, col] = 1
        col += 1
        
        # M3 = A00 (B0k1 - B1k1)
        U[0, col] = 1 # A00
        V[10+k1, col] = 1; V[10+k1+10, col] = -1 # B0k1 - B1k1? No B0k1 is 0*10+k1, B1k1 is 1*10+k1
        # Wait, V index is j*p + k.
        # B0k1: j=0, k=k1 -> 0*10+k1 = k1
        # B1k1: j=1, k=k1 -> 1*10+k1 = 10+k1
        V[k1, col] = 1; V[10+k1, col] = -1
        # Result: P01 += M3
        W[k1*2+0, col] = 1
        col += 1
        
        # M4 = A11 (B1k0 - B0k0)
        U[4, col] = 1 # A11
        V[10+k0, col] = 1; V[k0, col] = -1
        # Result: P00 += M4, P10 += M4
        W[k0*2+0, col] = 1; W[k0*2+1, col] = 1
        col += 1
        
        # M5 = (A00 + A01) B1k1
        U[0, col] = 1; U[1, col] = 1 # A00, A01
        V[10+k1, col] = 1 # B1k1
        # Result: P01 += M5, P00 -= M5
        W[k1*2+0, col] = 1; W[k0*2+0, col] = -1
        col += 1
        
        # M6 = (A10 - A00) (B0k0 + B0k1)
        U[3, col] = 1; U[0, col] = -1 # A10 - A00
        V[k0, col] = 1; V[k1, col] = 1 # B0k0 + B0k1
        # Result: P11 += M6
        W[k1*2+1, col] = 1
        col += 1
        
        # M7 = (A01 - A11) (B1k0 + B1k1)
        U[1, col] = 1; U[4, col] = -1 # A01 - A11
        V[10+k0, col] = 1; V[10+k1, col] = 1 # B1k0 + B1k1
        # Result: P00 += M7, P11 -= M7
        W[k0*2+0, col] = 1; W[k1*2+1, col] = -1
        col += 1
        
        # Outer product A'' (2x1) and B'' (1x2)
        # A'' rows 0,1 col 2
        # B'' row 2 cols k0,k1
        
        # M8 = A02 B2k0
        U[2, col] = 1 # A02 (0*3+2)
        V[20+k0, col] = 1 # B2k0 (2*10+k0)
        W[k0*2+0, col] = 1
        col += 1
        
        # M9 = A02 B2k1
        U[2, col] = 1
        V[20+k1, col] = 1
        W[k1*2+0, col] = 1
        col += 1
        
        # M10 = A12 B2k0
        U[5, col] = 1 # A12 (1*3+2)
        V[20+k0, col] = 1
        W[k0*2+1, col] = 1
        col += 1
        
        # M11 = A12 B2k1
        U[5, col] = 1
        V[20+k1, col] = 1
        W[k1*2+1, col] = 1
        col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
