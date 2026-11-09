import numpy as np
import time

# Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
# Convention: C[i,k] = sum_j A[i,j]*B[j,k]
# Tensor T[i,j,k] = 1 for i,j,k in {0,1}
# We need: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]

# The 7 Strassen products for 2x2:
# M0 = (A00+A11)*(B00+B11)
# M1 = (A10+A11)*B00
# M2 = A00*(B01-B11)
# M3 = A11*(B10-B00)
# M4 = (A00+A01)*B11
# M5 = (A10-A00)*(B00+B10)
# M6 = (A01-A11)*(B01+B11)
#
# C00 = M0 + M3 - M4 - M6
# C01 = M2 + M4
# C10 = M1 + M5
# C11 = M0 - M1 + M2 + M5

# U entries: U[i,j,r] for i,j in {0,1}, r in 0..6
# V entries: V[j,k,r] for j,k in {0,1}, r in 0..6
# W entries: W[k,i,r] for k,i in {0,1}, r in 0..6

def strassen_2x2():
    """Return (U, V, W) each of shape (4, 7) for the 2x2x2 Strassen decomposition."""
    # U[i*2+j, r]
    U = np.zeros((4, 7))
    # V[j*2+k, r]
    V = np.zeros((4, 7))
    # W[k*2+i, r]
    W = np.zeros((4, 7))
    
    # r=0: M0 = (A00+A11)*(B00+B11)
    U[0, 0] = 1  # A[0,0]
    U[3, 0] = 1  # A[1,1]
    V[0, 0] = 1  # B[0,0]
    V[3, 0] = 1  # B[1,1]
    # C00 += M0, C11 += M0
    W[0, 0] = 1  # W[0,0] = C[0,0]
    W[3, 0] = 1  # W[1,1] = C[1,1]
    
    # r=1: M1 = (A10+A11)*B00
    U[2, 1] = 1  # A[1,0]
    U[3, 1] = 1  # A[1,1]
    V[0, 1] = 1  # B[0,0]
    # C10 += M1, C11 -= M1
    W[2, 1] = 1   # W[1,0] = C[1,0] (k=1, i=0)
    W[3, 1] = -1  # W[1,1] = C[1,1] (k=1, i=1)
    
    # r=2: M2 = A00*(B01-B11)
    U[0, 2] = 1  # A[0,0]
    V[1, 2] = 1  # B[0,1]
    V[3, 2] = -1 # B[1,1]
    # C01 += M2, C11 += M2
    W[1, 2] = 1  # W[0,1] = C[0,1] (k=0, i=1)
    W[3, 2] = 1  # W[1,1] = C[1,1] (k=1, i=1)
    
    # r=3: M3 = A11*(B10-B00)
    U[3, 3] = 1  # A[1,1]
    V[2, 3] = 1  # B[1,0]
    V[0, 3] = -1 # B[0,0]
    # C00 += M3
    W[0, 3] = 1  # W[0,0] = C[0,0] (k=0, i=0)
    
    # r=4: M4 = (A00+A01)*B11
    U[0, 4] = 1  # A[0,0]
    U[1, 4] = 1  # A[0,1]
    V[3, 4] = 1  # B[1,1]
    # C00 -= M4, C01 += M4
    W[0, 4] = -1 # W[0,0] = C[0,0]
    W[1, 4] = 1  # W[0,1] = C[0,1]
    
    # r=5: M5 = (A10-A00)*(B00+B10)
    U[2, 5] = 1  # A[1,0]
    U[0, 5] = -1 # A[0,0]
    V[0, 5] = 1  # B[0,0]
    V[2, 5] = 1  # B[1,0]
    # C10 += M5, C11 += M5
    W[2, 5] = 1  # W[1,0] = C[1,0]
    W[3, 5] = 1  # W[1,1] = C[1,1]
    
    # r=6: M6 = (A01-A11)*(B01+B11)
    U[1, 6] = 1  # A[0,1]
    U[3, 6] = -1 # A[1,1]
    V[1, 6] = 1  # B[0,1]
    V[3, 6] = 1  # B[1,1]
    # C00 -= M6
    W[0, 6] = -1 # W[0,0] = C[0,0]
    
    return U, V, W

def verify_2x2(U, V, W):
    """Verify the 2x2x2 decomposition."""
    # U: (4,7) -> reshape to (2,2,7)
    # V: (4,7) -> reshape to (2,2,7)
    # W: (4,7) -> reshape to (2,2,7) where W[k,i,r]
    U_r = U.reshape(2, 2, 7)
    V_r = V.reshape(2, 2, 7)
    W_r = W.reshape(2, 2, 7)
    
    T_rec = np.zeros((2, 2, 2))
    for r in range(7):
        # U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    T_rec[i,j,k] += U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    
    return np.allclose(T_rec, 1.0, atol=1e-10)

def construct():
    n, m, p = 2, 4, 4
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Verify Strassen 2x2 works
    U2, V2, W2 = strassen_2x2()
    if not verify_2x2(U2, V2, W2):
        # Fallback: schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build rank-28 decomposition by applying Strassen to each 2x2 block pair.
    # The 2x4 A matrix has columns j=0,1,2,3. We can group j into pairs: (0,1) and (2,3).
    # Similarly, the 4x4 B matrix has columns k=0,1,2,3, grouped as (0,1) and (2,3).
    # And the 2x4 C matrix has columns k=0,1,2,3, grouped as (0,1) and (2,3).
    # 
    # For each combination of (j-group, k-group), we do a 2x2x2 Strassen.
    # There are 2 j-groups and 2 k-groups = 4 Strassen decompositions.
    # Total rank = 4 * 7 = 28.
    
    # j-groups: g=0 -> j in {0,1}, g=1 -> j in {2,3}
    # k-groups: h=0 -> k in {0,1}, h=1 -> k in {2,3}
    
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for g in range(2):  # j-group
        for h in range(2):  # k-group
            base = (g * 2 + h) * 7
            
            for r_local in range(7):
                r_global = base + r_local
                
                # U: A[i, j] where j = 2*g + local_j
                for i in range(2):
                    for local_j in range(2):
                        val = U2[i*2 + local_j, r_local]
                        j_global = 2*g + local_j
                        U[i*m + j_global, r_global] = val
                
                # V: B[j, k] where j = 2*g + local_j, k = 2*h + local_k
                for local_j in range(2):
                    for local_k in range(2):
                        val = V2[local_j*2 + local_k, r_local]
                        j_global = 2*g + local_j
                        k_global = 2*h + local_k
                        V[j_global*p + k_global, r_global] = val
                
                # W: C[k, i] where k = 2*h + local_k
                for local_k in range(2):
                    for i in range(2):
                        val = W2[local_k*2 + i, r_local]
                        k_global = 2*h + local_k
                        W[k_global*n + i, r_global] = val
    
    # Verify the full 2x4x4 decomposition
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    T_rec = np.zeros((n, m, p))
    for rr in range(rank):
        u_r = U_r[:, :, rr]  # (2, 4)
        v_r = V_r[:, :, rr]  # (4, 4)
        w_r = W_r[:, :, rr]  # (4, 2)
        # T[i,j,k] += u_r[i,j] * v_r[j,k] * w_r[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_rec[i,j,k] += u_r[i,j] * v_r[j,k] * w_r[k,i]
    
    if not np.allclose(T_rec, 1.0, atol=1e-10):
        # Fallback: schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
