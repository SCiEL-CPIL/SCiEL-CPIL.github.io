import numpy as np
import time

# Matrix multiplication 2x3 x 3x9
# A: 2x3, B: 3x9, C: 2x9
# U: (n*m, rank) = (6, rank)
# V: (m*p, rank) = (27, rank)
# W: (p*n, rank) = (18, rank)

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    for the tensor decomposition of 2x3 * 3x9 matrix multiplication.
    """
    n, m, p = 2, 3, 9
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin from 120s limit
    
    # Base rank-1 tensors for the full multiplication
    # We will construct a rank 50 decomposition using Strassen on 2x2 blocks.
    
    R = 50
    U = np.zeros((6, R), dtype=float)
    V = np.zeros((27, R), dtype=float)
    W = np.zeros((18, R), dtype=float)
    
    col = 0
    
    def strassen_block(k0, k1):
        nonlocal col
        # U indices
        a = 0 # A00
        b = 1 # A01
        c = 3 # A10
        d = 4 # A11
        
        # V indices
        e = k0 # B0, k0
        f = k1 # B0, k1
        g = 9 + k0 # B1, k0
        h = 9 + k1 # B1, k1
        
        # W indices
        w00 = k0 * 2 # C0, k0
        w01 = k1 * 2 # C0, k1
        w10 = k0 * 2 + 1 # C1, k0
        w11 = k1 * 2 + 1 # C1, k1
        
        # M1 = (A00 + A11) * (B00 + B11) -> C00 += M1, C11 += M1
        U[a, col] += 1
        U[d, col] += 1
        V[e, col] += 1
        V[h, col] += 1
        W[w00, col] += 1
        W[w11, col] += 1
        col += 1
        
        # M2 = (A10 + A11) * B00 -> C10 += M2, C11 -= M2
        U[c, col] += 1
        U[d, col] += 1
        V[e, col] += 1
        W[w10, col] += 1
        W[w11, col] -= 1
        col += 1
        
        # M3 = A00 * (B01 - B11) -> C01 += M3, C11 += M3
        U[a, col] += 1
        V[f, col] += 1
        V[h, col] -= 1
        W[w01, col] += 1
        W[w11, col] += 1
        col += 1
        
        # M4 = A11 * (B10 - B00) -> C00 += M4, C10 += M4
        U[d, col] += 1
        V[g, col] += 1
        V[e, col] -= 1
        W[w00, col] += 1
        W[w10, col] += 1
        col += 1
        
        # M5 = (A00 + A01) * B11 -> C00 -= M5, C01 += M5
        U[a, col] += 1
        U[b, col] += 1
        V[h, col] += 1
        W[w00, col] -= 1
        W[w01, col] += 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01) -> C11 += M6
        U[c, col] += 1
        U[a, col] -= 1
        V[e, col] += 1
        V[f, col] += 1
        W[w11, col] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11) -> C00 += M7
        U[b, col] += 1
        U[d, col] -= 1
        V[g, col] += 1
        V[h, col] += 1
        W[w00, col] += 1
        col += 1

    # Apply Strassen to 4 blocks of 2 columns
    for i in range(4):
        if time.time() > deadline:
            break
        strassen_block(2*i, 2*i+1)
        
    # Last column (k=8) of B_top
    # 2x2 * 2x1. Naive rank 4.
    if time.time() <= deadline:
        k = 8
        a, b, c, d = 0, 1, 3, 4
        e = k # B0, k
        g = 9 + k # B1, k
        w0 = k * 2 # C0, k
        w1 = k * 2 + 1 # C1, k
        
        # C0 = A00*B0 + A01*B1
        U[a, col] += 1
        V[e, col] += 1
        W[w0, col] += 1
        col += 1
        
        U[b, col] += 1
        V[g, col] += 1
        W[w0, col] += 1
        col += 1
        
        # C1 = A10*B0 + A11*B1
        U[c, col] += 1
        V[e, col] += 1
        W[w1, col] += 1
        col += 1
        
        U[d, col] += 1
        V[g, col] += 1
        W[w1, col] += 1
        col += 1
        
        # Now A_right (2x1) * B_bottom (1x9)
        # A_right indices: A02=2, A12=5.
        # B_bottom indices: B2,k = 18+k.
        # W indices: C_{i,k} = k*2+i.
        
        ar0 = 2 # A02
        ar1 = 5 # A12
        
        for k in range(9):
            if time.time() > deadline:
                break
            v = 18 + k
            
            # C0,k += A02 * B2,k
            U[ar0, col] += 1
            V[v, col] += 1
            W[k*2, col] += 1
            col += 1
            
            # C1,k += A12 * B2,k
            U[ar1, col] += 1
            V[v, col] += 1
            W[k*2+1, col] += 1
            col += 1
            
    # If we broke out early, we might have a partial decomposition.
    # To ensure validity, we must complete the decomposition or return a valid one.
    # Since the loop is very fast, it's unlikely to break.
    # But to be safe, if col < R, we can leave the remaining columns as zeros?
    # No, that would make the decomposition incorrect (missing terms).
    # However, the deadline is 110s, and this takes microseconds.
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U, V, W = res['U'], res['V'], res['W']
    print(U.shape, V.shape, W.shape)
