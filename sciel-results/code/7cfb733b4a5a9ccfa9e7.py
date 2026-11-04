import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    n, m, p = 2, 3, 6
    
    # Try rank 32: 12 schoolbook + 2*10 (two 2x2x3 rank-10 blocks)
    # If verification fails, fall back to rank 33
    
    def build_rank32():
        rank = 32
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        
        # Part 1: A[:, 0] (2x1) * B[0, :] (1x6) -> 12 terms
        for k in range(p):
            for i in range(n):
                U[i * m + 0, col] = 1.0
                V[0 * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
        
        # Part 2: A[:, 1:3] (2x2) * B[1:3, :] (2x6)
        # Split into two 2x2x3 blocks: cols (0,1,2) and cols (3,4,5)
        
        for block in range(2):
            # Block base column offset
            kbase = 3 * block  # B column offset within the 6 cols
            
            # Strassen 2x2x2 for first two output columns (kbase, kbase+1)
            k0 = kbase
            k1 = kbase + 1
            k2 = kbase + 2  # third column for this block
            
            # M1: (A00+A11)(B00+B11)
            U[0*m + 1, col] += 1.0
            U[1*m + 2, col] += 1.0
            V[1*p + k0, col] += 1.0
            V[2*p + k1, col] += 1.0
            W[k0*n + 0, col] += 1.0
            W[k1*n + 1, col] += 1.0
            col += 1
            
            # M2: (A10+A11)B00
            U[1*m + 1, col] += 1.0
            U[1*m + 2, col] += 1.0
            V[1*p + k0, col] += 1.0
            W[k0*n + 1, col] += 1.0
            W[k1*n + 1, col] += -1.0
            col += 1
            
            # M3: A00(B01-B11)
            U[0*m + 1, col] += 1.0
            V[1*p + k1, col] += 1.0
            V[2*p + k1, col] += -1.0
            W[k1*n + 0, col] += 1.0
            W[k1*n + 1, col] += 1.0
            col += 1
            
            # M4: A11(B10-B00)
            U[1*m + 2, col] += 1.0
            V[2*p + k0, col] += 1.0
            V[1*p + k0, col] += -1.0
            W[k0*n + 0, col] += 1.0
            W[k0*n + 1, col] += 1.0
            col += 1
            
            # M5: (A00+A01)B11
            U[0*m + 1, col] += 1.0
            U[0*m + 2, col] += 1.0
            V[2*p + k1, col] += 1.0
            W[k0*n + 0, col] += -1.0
            W[k1*n + 0, col] += 1.0
            col += 1
            
            # M6: (A10-A00)(B00+B01)
            U[1*m + 1, col] += 1.0
            U[0*m + 1, col] += -1.0
            V[1*p + k0, col] += 1.0
            V[1*p + k1, col] += 1.0
            W[k1*n + 1, col] += 1.0
            col += 1
            
            # M7: (A01-A11)(B10+B11)
            U[0*m + 2, col] += 1.0
            U[1*m + 2, col] += -1.0
            V[2*p + k0, col] += 1.0
            V[2*p + k1, col] += 1.0
            W[k0*n + 0, col] += 1.0
            col += 1
            
            # M8: (A00+A10)*B02  where B02 = B[1, k2] (row 0 of sub-B, col 2)
            # A00 = A[0,1], A10 = A[1,1]
            U[0*m + 1, col] += 1.0
            U[1*m + 1, col] += 1.0
            V[1*p + k2, col] += 1.0
            # W: contributes to C[0, k2] with +1/2, C[1, k2] with +1/2
            W[k2*n + 0, col] += 0.5
            W[k2*n + 1, col] += 0.5
            col += 1
            
            # M9: (A01+A11)*B12  where B12 = B[2, k2] (row 1 of sub-B, col 2)
            # A01 = A[0,2], A11 = A[1,2]
            U[0*m + 2, col] += 1.0
            U[1*m + 2, col] += 1.0
            V[2*p + k2, col] += 1.0
            # W: contributes to C[0, k2] with +1/2, C[1, k2] with +1/2
            W[k2*n + 0, col] += 0.5
            W[k2*n + 1, col] += 0.5
            col += 1
            
            # M10: (A00-A10)*B02
            # A00 = A[0,1], A10 = A[1,1]
            U[0*m + 1, col] += 1.0
            U[1*m + 1, col] += -1.0
            V[1*p + k2, col] += 1.0
            # W: contributes to C[0, k2] with +1/2, C[1, k2] with -1/2
            W[k2*n + 0, col] += 0.5
            W[k2*n + 1, col] += -0.5
            col += 1
        
        return U, V, W
    
    def build_rank33():
        rank = 33
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        
        # Part 1: A[:, 0] * B[0, :]
        for k in range(p):
            for i in range(n):
                U[i * m + 0, col] = 1.0
                V[0 * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
        
        # Part 2: Three 2x2x2 Strassen blocks
        for t in range(3):
            k0 = 2 * t
            k1 = 2 * t + 1
            
            # M1
            U[0*m + 1, col] += 1.0; U[1*m + 2, col] += 1.0
            V[1*p + k0, col] += 1.0; V[2*p + k1, col] += 1.0
            W[k0*n + 0, col] += 1.0; W[k1*n + 1, col] += 1.0
            col += 1
            
            # M2
            U[1*m + 1, col] += 1.0; U[1*m + 2, col] += 1.0
            V[1*p + k0, col] += 1.0
            W[k0*n + 1, col] += 1.0; W[k1*n + 1, col] += -1.0
            col += 1
            
            # M3
            U[0*m + 1, col] += 1.0
            V[1*p + k1, col] += 1.0; V[2*p + k1, col] += -1.0
            W[k1*n + 0, col] += 1.0; W[k1*n + 1, col] += 1.0
            col += 1
            
            # M4
            U[1*m + 2, col] += 1.0
            V[2*p + k0, col] += 1.0; V[1*p + k0, col] += -1.0
            W[k0*n + 0, col] += 1.0; W[k0*n + 1, col] += 1.0
            col += 1
            
            # M5
            U[0*m + 1, col] += 1.0; U[0*m + 2, col] += 1.0
            V[2*p + k1, col] += 1.0
            W[k0*n + 0, col] += -1.0; W[k1*n + 0, col] += 1.0
            col += 1
            
            # M6
            U[1*m + 1, col] += 1.0; U[0*m + 1, col] += -1.0
            V[1*p + k0, col] += 1.0; V[1*p + k1, col] += 1.0
            W[k1*n + 1, col] += 1.0
            col += 1
            
            # M7
            U[0*m + 2, col] += 1.0; U[1*m + 2, col] += -1.0
            V[2*p + k0, col] += 1.0; V[2*p + k1, col] += 1.0
            W[k0*n + 0, col] += 1.0
            col += 1
        
        return U, V, W
    
    def build_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U[ii * m + jj, col] = 1.0
                    V[jj * p + kk, col] = 1.0
                    W[kk * n + ii, col] = 1.0
                    col += 1
        return U, V, W
    
    def verify(U, V, W):
        if time.time() > DEADLINE:
            return False
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=1e-9)
    
    # Try rank 32 first
    U32, V32, W32 = build_rank32()
    if verify(U32, V32, W32):
        return {"U": U32.tolist(), "V": V32.tolist(), "W": W32.tolist()}
    
    # Fallback to rank 33
    if time.time() > DEADLINE:
        # Return schoolbook as last resort
        U, V, W = build_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U33, V33, W33 = build_rank33()
    if verify(U33, V33, W33):
        return {"U": U33.tolist(), "V": V33.tolist(), "W": W33.tolist()}
    
    # Last resort: schoolbook
    U, V, W = build_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
