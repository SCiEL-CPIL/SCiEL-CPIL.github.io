"""
Matrix multiplication tensor decomposition for M_{2,3,10}.
Uses Strassen's algorithm on 2x2 sub-blocks for the first 2x2 part of the 
multiplication and schoolbook for the remaining 2x1x10 part.
Total rank: 5 * (7 + 4) = 55.
"""
import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Returns a rank-55 exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x10 matrix.
    """
    n, m, p = DIMENSIONS
    # Rank breakdown:
    # Split m=3 into m1=2 and m2=1.
    # A1 is 2x2, B1 is 2x10. Split B1 into 5 blocks of 2x2.
    # Each 2x2x2 block uses Strassen (rank 7). Total 35.
    # A2 is 2x1, B2 is 1x10. Split into 5 blocks of 2x1x2.
    # Each 2x1x2 block uses schoolbook (rank 4). Total 20.
    # Total rank = 55.
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # --- Strassen 7 products for A[:, 0:2] * B[0:2, k0:k1+1] ---
        # A indices:
        # A00 = U[0] (i=0, j=0)
        # A01 = U[1] (i=0, j=1)
        # A10 = U[3] (i=1, j=0)
        # A11 = U[4] (i=1, j=1)
        
        # B indices:
        # B0k0 = V[0*10 + k0] = V[k0]
        # B0k1 = V[0*10 + k1] = V[k1]
        # B1k0 = V[1*10 + k0] = V[10+k0]
        # B1k1 = V[1*10 + k1] = V[10+k1]
        
        # C indices:
        # C0k0 = W[k0*2 + 0] = W[2*k0]
        # C1k0 = W[k0*2 + 1] = W[2*k0+1]
        # C0k1 = W[k1*2 + 0] = W[2*k1]
        # C1k1 = W[k1*2 + 1] = W[2*k1+1]
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0][r] += 1
        U[4][r] += 1
        V[k0][r] += 1
        V[10+k1][r] += 1
        W[2*k0][r] += 1  # C00
        W[2*k1+1][r] += 1 # C11
        r += 1
        
        # M2 = (A10 + A11) * B0k0
        U[3][r] += 1
        U[4][r] += 1
        V[k0][r] += 1
        W[2*k0+1][r] += 1 # C10
        W[2*k1+1][r] -= 1 # -C11
        r += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0][r] += 1
        V[k1][r] += 1
        V[10+k1][r] -= 1
        W[2*k1][r] += 1 # C01
        W[2*k1+1][r] += 1 # C11
        r += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        U[4][r] += 1
        V[10+k0][r] += 1
        V[k0][r] -= 1
        W[2*k0][r] += 1 # C00
        W[2*k0+1][r] += 1 # C10
        r += 1
        
        # M5 = (A00 + A01) * B1k1
        U[0][r] += 1
        U[1][r] += 1
        V[10+k1][r] += 1
        W[2*k0][r] -= 1 # -C00
        W[2*k1][r] += 1 # C01
        r += 1
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[3][r] += 1
        U[0][r] -= 1
        V[k0][r] += 1
        V[k1][r] += 1
        W[2*k1+1][r] += 1 # C11
        r += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[1][r] += 1
        U[4][r] -= 1
        V[10+k0][r] += 1
        V[10+k1][r] += 1
        W[2*k0][r] += 1 # C00
        r += 1
        
        # --- Schoolbook 4 products for A[:, 2] * B[2, k0:k1+1] ---
        # A indices:
        # A02 = U[2] (i=0, j=2)
        # A12 = U[5] (i=1, j=2)
        
        # B indices:
        # B2k0 = V[2*10 + k0] = V[20+k0]
        # B2k1 = V[2*10 + k1] = V[20+k1]
        
        # C indices (same as above):
        # C0k0 = W[2*k0]
        # C1k0 = W[2*k0+1]
        # C0k1 = W[2*k1]
        # C1k1 = W[2*k1+1]
        
        # Term 1: A02 * B2k0 -> C0k0
        U[2][r] += 1
        V[20+k0][r] += 1
        W[2*k0][r] += 1
        r += 1
        
        # Term 2: A12 * B2k0 -> C1k0
        U[5][r] += 1
        V[20+k0][r] += 1
        W[2*k0+1][r] += 1
        r += 1
        
        # Term 3: A02 * B2k1 -> C0k1
        U[2][r] += 1
        V[20+k1][r] += 1
        W[2*k1][r] += 1
        r += 1
        
        # Term 4: A12 * B2k1 -> C1k1
        U[5][r] += 1
        V[20+k1][r] += 1
        W[2*k1+1][r] += 1
        r += 1

    assert r == rank, f"Rank mismatch: {r} != {rank}"
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    n, m, p = 2, 3, 10
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
                
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(T_recon - T))
    print("Error:", err)
