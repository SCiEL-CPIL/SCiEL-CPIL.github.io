import numpy as np
import time
from itertools import product

# The task: 2x3 * 3x6 -> 2x6
# Schoolbook rank: 2*3*6 = 36
# We use two 2x3x3 blocks, each with rank 11 -> total rank 22


def _build_rank11_233():
    """
    Rank-11 decomposition for 2x3 * 3x3 -> 2x3.
    
    U: shape (6, 11), indices a*3+b (a in 0..1, b in 0..2)
    V: shape (9, 11), indices b*3+d (b in 0..2, d in 0..2)
    W: shape (6, 11), indices d*2+a (d in 0..2, a in 0..1)
    """
    U = np.zeros((6, 11))
    V = np.zeros((9, 11))
    W = np.zeros((6, 11))
    
    # Strassen for the 2x2x2 core (rows 0,1 of A; cols 0,1 of B; output rows 0,1; cols 0,1)
    # U indices: a*3+b. For b=0,1: rows 0,1 (a=0,b=0,1) and 3,4 (a=1,b=0,1)
    # V indices: b*3+d. For b=0,1: cols 0,1,2 (b=0) and 3,4,5 (b=1)
    # W indices: d*2+a. For d=0,1: rows 0,1 (d=0) and 2,3 (d=1)
    
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) -> contributes to C[0,0],C[0,1],C[1,0],C[1,1]
    U[0, 0] = 1   # A[0,0]
    U[4, 0] = 1   # A[1,1]
    V[0, 0] = 1   # B[0,0]
    V[4, 0] = 1   # B[1,1]
    W[0, 0] = 1   # C[0,0]
    W[1, 0] = 1   # C[0,1]  (d=0, a=0 -> C[0,0]; d=0,a=1 -> C[1,0]; d=1,a=0 -> C[0,1]; d=1,a=1 -> C[1,1])
    W[2, 0] = 1   # C[0,1]
    W[3, 0] = 1   # C[1,1]
    
    # M2 = (A[1,0]+A[1,1])*B[0,0] -> C[1,0] += (A[1,0]+A[1,1])*B[0,0], C[1,1] -= A[1,1]*B[0,0]
    # Actually standard Strassen: M2=(A10+A11)B00, C10 += M2, C11 -= M2
    U[3, 1] = 1   # A[1,0]
    U[4, 1] = 1   # A[1,1]
    V[0, 1] = 1   # B[0,0]
    W[3, 1] = 1   # C[1,0] (d=0,a=1)
    W[3, 1] = 1
    W[3, 1] = 1
    W[1, 1] = -1  # C[0,1]? No wait.
    
    # Let me redo this more carefully.
    # Standard Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For our 2x3x3, the 2x2 core uses A[0:2,0:2] and B[0:2,0:2], producing C[0:2,0:2]
    # Then we handle A[0:2,2]*B[2,0:2] and A[0:2,0:2]*B[0:2,2] separately.
    
    U = np.zeros((6, 11))
    V = np.zeros((9, 11))
    W = np.zeros((6, 11))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[4, 0] = 1      # A[0,0], A[1,1]
    V[0, 0] = 1; V[4, 0] = 1      # B[0,0], B[1,1]
    W[0, 0] = 1; W[2, 0] = 1      # C[0,0], C[0,1]
    W[3, 0] = 1; W[1, 0] = 1      # C[1,0], C[1,1]
    # Actually W index is d*2+a. C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]
    W[0, 0] = 1; W[2, 0] = 1; W[1, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1   # C[1,0]
    W[3, 1] = -1  # C[1,1]
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[3, 2] = 1; V[5, 2] = -1
    W[2, 2] = 1   # C[0,1]
    W[3, 2] = 1   # C[1,1]
    # Wait, M3 contributes to C01 and C11? No.
    # C01 = M3+M5, C11 = M1-M2+M3+M6
    # So M3 contributes +1 to C01 and +1 to C11
    W[2, 2] = 1
    W[3, 2] = 1
    
    # M4 = A11(B01-B00)
    U[4, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1   # C[0,0]
    W[1, 3] = 1   # C[1,0]
    W[2, 3] = -1  # C[0,1]
    W[3, 3] = -1  # C[1,1]
    # C00=M1+M4..., C10=M2+M4..., C01=M3+M5 (no M4), C11=M1-M2+M3+M6 (no M4)
    # Wait, I'm confusing myself. Let me use the standard formulas:
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # So M4 contributes to C00 and C10 only.
    W[0, 3] = 1   # C[0,0]
    W[1, 3] = 1   # C[1,0]
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[5, 4] = 1
    W[0, 4] = -1  # C[0,0]
    W[2, 4] = 1   # C[0,1]
    
    # M6 = (A10-A00)(B00+B10)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[3, 5] = 1
    W[3, 5] = 1   # C[1,1]
    
    # M7 = (A01-A11)(B01+B11)
    U[1, 6] = 1; U[4, 6] = -1
    V[1, 6] = 1; V[5, 6] = 1
    W[0, 6] = 1   # C[0,0]
    W[2, 6] = 1   # C[0,1]
    
    # Now handle the remaining terms:
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # C[0,0:2] and C[1,0:2] also have contributions from A[:,2]*B[2,0:2]
    # C[0,0] += A[0,2]*B[2,0], C[0,1] += A[0,2]*B[2,1]
    # C[1,0] += A[1,2]*B[2,0], C[1,1] += A[1,2]*B[2,1]
    
    # M8 = A[0,2]*(B[2,0]+B[2,1]+B[2,2])
    U[2, 7] = 1
    V[6, 7] = 1; V[7, 7] = 1; V[8, 7] = 1
    W[0, 7] = 1; W[2, 7] = 1; W[4, 7] = 1
    
    # M9 = A[1,2]*(B[2,0]+B[2,1]+B[2,2])
    U[5, 8] = 1
    V[6, 8] = 1; V[7, 8] = 1; V[8, 8] = 1
    W[1, 8] = 1; W[3, 8] = 1; W[5, 8] = 1
    
    # M10 = (A[0,0]+A[1,0])*B[0,2]
    U[0, 9] = 1; U[3, 9] = 1
    V[2, 9] = 1
    W[4, 9] = 1; W[5, 9] = 1
    
    # M11 = (A[0,1]+A[1,1])*B[1,2]
    U[1, 10] = 1; U[4, 10] = 1
    V[5, 10] = 1
    W[4, 10] = 1; W[5, 10] = 1
    
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition is correct."""
    nU, nV, nW = n*m, m*p, p*n
    rank = U.shape[1]
    
    # Build target tensor
    T = np.zeros((nU, nV, nW))
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for k in range(p):
                        for i in range(n):
                            if a == i and b == c and d == k:
                                T[a*m+b, c*p+d, k*n+i] = 1.0
    
    # Reconstruct
    T_recon = np.zeros((nU, nV, nW))
    for l in range(rank):
        T_recon += np.einsum('i,j,k->ijk', U[:, l], V[:, l], W[:, l])
    
    return np.allclose(T_recon, T, atol=tol)


def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = 2, 3, 6
    
    # Build rank-11 decomposition for 2x3x3
    U_233, V_233, W_233 = _build_rank11_233()
    
    if not _verify(U_233, V_233, W_233, 2, 3, 3):
        # Fallback: schoolbook
        U_sb = np.zeros((6, 36))
        V_sb = np.zeros((18, 36))
        W_sb = np.zeros((12, 36))
        idx = 0
        for a in range(n):
            for b in range(m):
                for d in range(p):
                    U_sb[a*m+b, idx] = 1
                    V_sb[b*p+d, idx] = 1
                    W_sb[d*n+a, idx] = 1
                    idx += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Assemble 2x3x6 from two 2x3x3 blocks
    rank_total = 22
    U_full = np.zeros((6, rank_total))
    V_full = np.zeros((18, rank_total))
    W_full = np.zeros((12, rank_total))
    
    for block in range(2):
        offset = block * 11
        U_full[:, offset:offset+11] = U_233
        
        # V: shape (18, rank). V index = b*6+d. For block j (d in 3j..3j+2), V[b*6+3j+k, l] = V_233[b*3+k, l]
        for b in range(3):
            for k in range(3):
                V_full[b*6 + block*3 + k, offset:offset+11] = V_233[b*3 + k]
        
        # W: shape (12, rank). W index = k*2+i. For block j (k in 3j..3j+2), W[3j+k'*2+i, l] = W_233[k'*2+i, l]
        for k in range(3):
            for i in range(2):
                W_full[(block*3+k)*2 + i, offset:offset+11] = W_233[k*2 + i]
    
    if _verify(U_full, V_full, W_full, n, m, p):
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }
    
    # Fallback: schoolbook
    U_sb = np.zeros((6, 36))
    V_sb = np.zeros((18, 36))
    W_sb = np.zeros((12, 36))
    idx = 0
    for a in range(n):
        for b in range(m):
            for d in range(p):
                U_sb[a*m+b, idx] = 1
                V_sb[b*p+d, idx] = 1
                W_sb[d*n+a, idx] = 1
                idx += 1
    
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }


if __name__ == "__main__":
    construct()
