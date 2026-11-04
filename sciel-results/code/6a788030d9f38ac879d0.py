"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    nm, mp, pn = n * m, m * p, p * n  # 8, 20, 10

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        # Check all valid (i,j,k) positions: should equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check zero positions
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    if j2 == j:
                        continue
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j2 * p + k
                        idx_w = k * n + i
                        val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                        if abs(val) > tol:
                            return False
                for k in range(p):
                    for k2 in range(p):
                        if k2 == k:
                            continue
                        idx_u = i * m + j
                        idx_v = j * p + k2
                        idx_w = k * n + i
                        val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                        if abs(val) > tol:
                            return False
                    for i2 in range(n):
                        if i2 == i:
                            continue
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i2
                        val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                        if abs(val) > tol:
                            return False
        return True

    def schoolbook():
        rank0 = n * m * p
        U = np.zeros((nm, rank0))
        V = np.zeros((mp, rank0))
        W = np.zeros((pn, rank0))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    def strassen_2x2x2():
        """Return (U_sub, V_sub, W_sub) for 2x2 x 2x2 multiplication.
        U_sub: shape (4, 7) - rows indexed by (i,j) in {0,1}^2
        V_sub: shape (4, 7) - rows indexed by (j,k) in {0,1}^2  
        W_sub: shape (4, 7) - rows indexed by (k,i) in {0,1}^2
        """
        U_sub = np.zeros((4, 7))
        V_sub = np.zeros((4, 7))
        W_sub = np.zeros((4, 7))
        
        # Strassen 7 products for C = A*B where A,B are 2x2
        # C[0,0] = M1 - M4 + M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M6 + M7
        #
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # M1 = (A00+A11)(B00+B11)
        U_sub[0, 0] = 1  # A00
        U_sub[3, 0] = 1  # A11
        V_sub[0, 0] = 1  # B00
        V_sub[3, 0] = 1  # B11
        # C00 += M1, C11 += M1
        W_sub[0, 0] = 1  # (k=0,i=0) -> C00
        W_sub[3, 0] = 1  # (k=1,i=1) -> C11

        # M2 = (A10+A11)B00
        U_sub[2, 1] = 1  # A10
        U_sub[3, 1] = 1  # A11
        V_sub[0, 1] = 1  # B00
        # C10 += M2, C11 -= M2
        W_sub[2, 1] = 1  # (k=0,i=1) -> C10
        W_sub[3, 1] = -1  # (k=1,i=1) -> C11

        # M3 = A00(B01-B11)
        U_sub[0, 2] = 1  # A00
        V_sub[1, 2] = 1  # B01
        V_sub[3, 2] = -1  # B11
        # C01 += M3
        W_sub[1, 2] = 1  # (k=1,i=0) -> C01

        # M4 = A11(B10-B00)
        U_sub[3, 3] = 1  # A11
        V_sub[2, 3] = 1  # B10
        V_sub[0, 3] = -1  # B00
        # C00 -= M4, C10 += M4
        W_sub[0, 3] = -1  # (k=0,i=0) -> C00
        W_sub[2, 3] = 1  # (k=0,i=1) -> C10

        # M5 = (A00+A01)B11
        U_sub[0, 4] = 1  # A00
        U_sub[1, 4] = 1  # A01
        V_sub[3, 4] = 1  # B11
        # C00 += M5, C01 += M5
        W_sub[0, 4] = 1  # (k=0,i=0) -> C00
        W_sub[1, 4] = 1  # (k=1,i=0) -> C01

        # M6 = (A10-A00)(B00+B01)
        U_sub[2, 5] = 1  # A10
        U_sub[0, 5] = -1  # A00
        V_sub[0, 5] = 1  # B00
        V_sub[1, 5] = 1  # B01
        # C11 += M6
        W_sub[3, 5] = 1  # (k=1,i=1) -> C11

        # M7 = (A01-A11)(B10+B11)
        U_sub[1, 6] = 1  # A01
        U_sub[3, 6] = -1  # A11
        V_sub[2, 6] = 1  # B10
        V_sub[3, 6] = 1  # B11
        # C00 += M7, C11 += M7
        W_sub[0, 6] = 1  # (k=0,i=0) -> C00
        W_sub[3, 6] = 1  # (k=1,i=1) -> C11

        return U_sub, V_sub, W_sub

    def trivial_2x2x1():
        """Return (U_sub, V_sub, W_sub) for 2x2 x 2x1 multiplication.
        U_sub: shape (4, 4) - rows indexed by (i,j) in {0,1}^2
        V_sub: shape (2, 4) - rows indexed by (j,k=0) in {0,1}
        W_sub: shape (2, 4) - rows indexed by (k=0,i) in {0,1}
        """
        U_sub = np.zeros((4, 4))
        V_sub = np.zeros((2, 4))
        W_sub = np.zeros((2, 4))
        
        # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
        col = 0
        for i in range(2):
            for j in range(2):
                U_sub[i*2+j, col] = 1  # A[i,j]
                V_sub[j, col] = 1  # B[j,0]
                W_sub[0*2+i, col] = 1  # C[i,0] -> (k=0, i)
                col += 1
        return U_sub, V_sub, W_sub

    def trivial_2x1x1():
        """Return (U_sub, V_sub, W_sub) for 2x1 x 1x1 multiplication.
        U_sub: shape (2, 2) - rows indexed by (i,j=0)
        V_sub: shape (1, 2) - rows indexed by (j=0,k=0)
        W_sub: shape (1, 2) - rows indexed by (k=0,i)
        """
        U_sub = np.zeros((2, 2))
        V_sub = np.zeros((1, 2))
        W_sub = np.zeros((1, 2))
        
        col = 0
        for i in range(2):
            U_sub[i, col] = 1  # A[i,0]
            V_sub[0, col] = 1  # B[0,0]
            W_sub[0, col] = 1  # C[i,0] -> (k=0, i)
            col += 1
        return U_sub, V_sub, W_sub

    # Build the full decomposition
    # Split m=4 into j_blocks: {0,1} and {2,3}
    # Split p=5 into k_blocks: {0,1}, {2,3}, {4}
    
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3), (4, 4)]
    
    all_U = []
    all_V = []
    all_W = []
    
    for j_start, j_end in j_blocks:
        j_count = j_end - j_start + 1  # 2 for both blocks
        
        for k_start, k_end in k_blocks:
            k_count = k_end - k_start + 1
            
            if j_count == 2 and k_count == 2:
                # Use Strassen 2x2x2
                U_sub, V_sub, W_sub = strassen_2x2x2()
                
                # Map to global indices
                # U rows: (i, j) where i in {0,1}, j in {j_start, j_start+1}
                # Global U row index: i*m + j = i*4 + j
                U_full = np.zeros((nm, U_sub.shape[1]))
                for i in range(2):
                    for j_local in range(2):
                        j_global = j_start + j_local
                        U_full[i * m + j_global, :] = U_sub[i * 2 + j_local, :]
                
                # V rows: (j, k) where j in {j_start, j_start+1}, k in {k_start, k_start+1}
                # Global V row index: j*p + k
                V_full = np.zeros((mp, V_sub.shape[1]))
                for j_local in range(2):
                    j_global = j_start + j_local
                    for k_local in range(2):
                        k_global = k_start + k_local
                        V_full[j_global * p + k_global, :] = V_sub[j_local * 2 + k_local, :]
                
                # W rows: (k, i) where k in {k_start, k_start+1}, i in {0,1}
                # Global W row index: k*n + i
                W_full = np.zeros((pn, W_sub.shape[1]))
                for k_local in range(2):
                    k_global = k_start + k_local
                    for i in range(2):
                        W_full[k_global * n + i, :] = W_sub[k_local * 2 + i, :]
                
                all_U.append(U_full)
                all_V.append(V_full)
                all_W.append(W_full)
                
            elif j_count == 2 and k_count == 1:
                # Use trivial 2x2x1
                U_sub, V_sub, W_sub = trivial_2x2x1()
                
                U_full = np.zeros((nm, U_sub.shape[1]))
                for i in range(2):
                    for j_local in range(2):
                        j_global = j_start + j_local
                        U_full[i * m + j_global, :] = U_sub[i * 2 + j_local, :]
                
                V_full = np.zeros((mp, V_sub.shape[1]))
                for j_local in range(2):
                    j_global = j_start + j_local
                    V_full[j_global * p + k_start, :] = V_sub[j_local, :]
                
                W_full = np.zeros((pn, W_sub.shape[1]))
                for i in range(2):
                    W_full[k_start * n + i, :] = W_sub[i, :]
                
                all_U.append(U_full)
                all_V.append(V_full)
                all_W.append(W_full)
    
    U_full = np.hstack(all_U)
    V_full = np.hstack(all_V)
    W_full = np.hstack(all_W)
    
    rank = U_full.shape[1]
    
    if verify(U_full, V_full, W_full):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback to schoolbook
    U_s, V_s, W_s = schoolbook()
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
# EVOLVE-BLOCK-END
