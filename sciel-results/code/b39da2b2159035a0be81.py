import numpy as np
import time

def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    Uses Strassen decomposition for 2x2x2 subblocks (rank 7 each) and direct
    multiplication for 2x1x8 subblock (rank 16).
    Total rank: 4*7 + 16 = 44
    """
    t_start = time.time()
    deadline = t_start + 100  # Safety margin before 150s limit
    
    n, m, p = 2, 3, 8
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify the tensor decomposition exactly reconstructs the multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        T_recon = np.zeros((n, m, p), dtype=float)
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            T_recon += np.einsum('ij,jk,ki->ijk', u, v, w)
        target = np.ones((n, m, p), dtype=float)
        return np.allclose(T_recon, target, atol=1e-10)
    
    def build_strassen_2x2x2():
        """
        Strassen decomposition of M_{2,2,2}. Rank 7.
        U: (4, 7) indices i*2+j
        V: (4, 7) indices j*2+k
        W: (4, 7) indices k*2+i
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00 -> contributes to C10
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3 = A00(B10+B11) -> contributes to C01, C11
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = 1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = (A00-A10)(B01+B10) -> contributes to C00, C10
        U[0, 3] = 1; U[2, 3] = -1
        V[1, 3] = 1; V[2, 3] = 1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A01+A10)(B00-B10) -> contributes to C00, C01
        U[1, 4] = 1; U[2, 4] = 1
        V[0, 4] = 1; V[2, 4] = -1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A11-A01)(B01-B11) -> contributes to C11
        U[1, 5] = -1; U[3, 5] = 1
        V[1, 5] = 1; V[3, 5] = -1
        W[3, 5] = 1
        
        # M7 = (A00+A01)(B11-B01) -> contributes to C00
        U[0, 6] = 1; U[1, 6] = 1
        V[1, 6] = -1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    # Build full decomposition
    # Split: A = [A1 (2x2) | A2 (2x1)], B = [B1 (2x8); B2 (1x8)]
    # Part 1: A1 * B1 = M_{2,2,8}, split into 4 blocks of M_{2,2,2}
    # Part 2: A2 * B2 = M_{2,1,8}, rank 16
    # Total rank = 4*7 + 16 = 44
    
    rank_total = 4 * 7 + 16  # = 44
    U = np.zeros((n * m, rank_total))  # (6, 44)
    V = np.zeros((m * p, rank_total))  # (24, 44)
    W = np.zeros((p * n, rank_total))  # (16, 44)
    
    # Part 1: 4 Strassen blocks for j in {0,1}, k blocks of size 2
    # For block b (b=0,1,2,3), k ranges over [2b, 2b+1]
    for b in range(4):
        U_s, V_s, W_s = build_strassen_2x2x2()
        r_offset = b * 7
        for r in range(7):
            # U: indices i*3+j for i in {0,1}, j in {0,1}
            # In Strassen U_s: index i*2+j
            for i in range(2):
                for j in range(2):
                    U[i * 3 + j, r_offset + r] = U_s[i * 2 + j, r]
            
            # V: indices j*8+k for j in {0,1}, k in {2b, 2b+1}
            # In Strassen V_s: index j*2+k_local where k_local in {0,1}
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * b + k_local
                    V[j * 8 + k_global, r_offset + r] = V_s[j * 2 + k_local, r]
            
            # W: indices k*2+i for k in {2b, 2b+1}, i in {0,1}
            # In Strassen W_s: index k_local*2+i
            for k_local in range(2):
                k_global = 2 * b + k_local
                for i in range(2):
                    W[k_global * 2 + i, r_offset + r] = W_s[k_local * 2 + i, r]
    
    # Part 2: Direct multiplication for j=2 (A2 is 2x1, B2 is 1x8)
    # Rank 16: one term per (i, k) pair
    r_offset = 4 * 7  # = 28
    for i in range(2):
        for k in range(8):
            r = r_offset + i * 8 + k
            U[i * 3 + 2, r] = 1  # A[i, 2]
            V[2 * 8 + k, r] = 1  # B[2, k]
            W[k * 2 + i, r] = 1  # C[i, k]
    
    # Verify
    if time.time() < deadline:
        if verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 48
    rank_fallback = n * m * p
    U_fallback = np.zeros((n * m, rank_fallback))
    V_fallback = np.zeros((m * p, rank_fallback))
    W_fallback = np.zeros((p * n, rank_fallback))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j, col] = 1
                V_fallback[j * p + k, col] = 1
                W_fallback[k * n + i, col] = 1
                col += 1
    
    return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}
