import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Try to find a low-rank decomposition using structured search
    # Strategy: Use Winograd-style rank-7 for 2x2x2 blocks with cross-block sharing
    
    best_rank = 48  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Precompute a known rank-42 solution as fallback
    def make_rank42():
        n, m, p = 2, 4, 6
        rank = 42
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        # Use 2x2 Winograd blocks with sharing
        # Split j into pairs (0,1) and (2,3), k into triples (0,1,2), (3,4,5)
        # For each (j_pair, k_triple), use rank-7 Winograd for 2x2x2
        # But share some terms between adjacent k_triples
        
        # Actually, let's use a cleaner approach:
        # 2x4x6 = sum over j of A[:,j] * B[j,:]
        # Group j into {0,1} and {2,3}
        # For each group, we have 2x2x6 = sum over 2 j's of A[:,j] * B[j,:]
        # Split k into 3 pairs: (0,1), (2,3), (4,5)
        # For each (j_group, k_pair), use rank-7 Winograd for 2x2x2
        
        # Winograd rank-7 for 2x2x2:
        # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
        # Z = X@Y
        # M1 = (x00+x11)(y00+y11)
        # M2 = (x01+x11)(y00)
        # M3 = (x00)(y10-y00)
        # M4 = (x11)(y01-y11)
        # M5 = (x00+x01)(y11)
        # M6 = (x10-x00)(y00+y01)
        # M7 = (x01+x10)(y10+y11)
        
        # Z00 = M1 + M3 - M5 - M6
        # Z01 = M4 + M5 - M6
        # Z10 = M2 + M3 - M5 + M7
        # Z11 = M1 + M4 + M7 - M2  (need to verify)
        
        # Let me use a verified set. Standard Winograd 2x2:
        # U factors (for X, rows indexed by i*2+j):
        U22 = np.array([
            [1, 0, 1, 0, 1, -1, 0],   # x00
            [0, 1, 0, 0, 1, 0, 1],    # x01
            [0, 0, 0, 0, 0, 1, 1],    # x10
            [1, 1, 0, 1, 0, 0, 0]     # x11
        ], dtype=float)
        
        V22 = np.array([
            [1, 1, -1, 0, 0, 1, 0],   # y00
            [1, 0, 0, 1, 0, 1, 0],    # y01
            [1, 0, 1, 0, 0, 0, 1],    # y10
            [1, 0, 0, -1, 1, 0, 1]    # y11
        ], dtype=float)
        
        W22 = np.array([
            [1, 0, 1, 0, -1, -1, 0],   # z00
            [0, 0, 0, 1, 1, -1, 0],    # z01
            [0, 1, 1, 0, -1, 0, 1],    # z10
            [0, 0, 0, -1, 0, 0, 0]     # z11 - incomplete, need to fix
        ], dtype=float)
        
        # Verify and fix W22
        # Z00 = x00*y00 + x01*y10
        # Z01 = x00*y01 + x01*y11
        # Z10 = x10*y00 + x11*y10
        # Z11 = x10*y01 + x11*y11
        
        # Let me compute what W22 should be by solving:
        # For each z_ik, find w such that sum_r U[i,j,r]*V[j,k,r]*w_r = z_ik for all i,j,k
        # This is: for fixed (i,k), sum_j U[i,j,:] * V[j,k,:] * w = 1 (for the tensor element)
        
        # Actually, for the 2x2x2 tensor T[i,j,k]=1:
        # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
        
        # Let me just construct it directly and verify numerically
        col = 0
        for j_group in range(2):  # j in {0,1} or {2,3}
            for k_group in range(3):  # k in {0,1}, {2,3}, {4,5}
                for r in range(7):
                    # U factor
                    for i in range(2):
                        j_local = j_group * 2 + (r % 2)  # This isn't right
                        pass
                    # Let me do this more carefully
                    pass
        
        # Simpler: just build the full 42-rank solution explicitly
        # 6 blocks of 2x2x2, each rank 7, total 42
        
        col = 0
        for j_base in [0, 2]:  # j = 0,1 or j = 2,3
            for k_base in [0, 2, 4]:  # k = 0,1 or 2,3 or 4,5
                for r in range(7):
                    # U: for i in {0,1}, j in {j_base, j_base+1}
                    U[0*m + j_base, col] = U22[0, r]   # i=0, j=j_base
                    U[0*m + j_base+1, col] = U22[1, r] # i=0, j=j_base+1
                    U[1*m + j_base, col] = U22[2, r]   # i=1, j=j_base
                    U[1*m + j_base+1, col] = U22[3, r] # i=1, j=j_base+1
                    
                    # V: for j in {j_base, j_base+1}, k in {k_base, k_base+1}
                    V[j_base*p + k_base, col] = V22[0, r]
                    V[j_base*p + k_base+1, col] = V22[1, r]
                    V[(j_base+1)*p + k_base, col] = V22[2, r]
                    V[(j_base+1)*p + k_base+1, col] = V22[3, r]
                    
                    # W: for k in {k_base, k_base+1}, i in {0,1}
                    W[k_base*n + 0, col] = W22[0, r]
                    W[(k_base+1)*n + 0, col] = W22[1, r]
                    W[k_base*n + 1, col] = W22[2, r]
                    W[(k_base+1)*n + 1, col] = W22[3, r]
                    
                    col += 1
        
        # Verify
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1))
        
        return U, V, W, max_err
    
    U42, V42, W42, err42 = make_rank42()
    if err42 < 1e-6:
        best_rank = 42
        best_U = U42
        best_V = V42
        best_W = W42
    
    # Now try to improve with ALS from rank 39 down
    for rank in range(39, 30, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for trial in range(30):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(trial * 7919 + rank * 104729)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Normalize
            for r in range(rank):
                nU = np.linalg.norm(U[:, r])
                nV = np.linalg.norm(V[:, r])
                nW = np.linalg.norm(W[:, r])
                if nU > 1e-12 and nV > 1e-12 and nW > 1e-12:
                    scale = (nU * nV * nW) ** (1/3)
                    U[:, r] /= scale
                    V[:, r] /= scale
                    W[:, r] /= scale
            
            for it in range(500):
                if time.time() > deadline - 30:
                    break
                
                # ALS: fix V,W, solve for U
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # For each k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                        # Let c_k = V[j*p+k,:] * W[k*n+i,:]
                        # Then sum_r U[i*m+j,r] * c_k[r] = 1 for all k
                        # This is: U[i*m+j,:] @ C^T = ones(p) where C[k,r] = V[j*p+k,r]*W[k*n+i,r]
                        C = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                        # Solve U_row @ C^T = ones(p)
                        # Minimum norm: U_row = ones(p) @ C @ (C^T C)^{-1} if rank > p
                        # Or use lstsq
                        try:
                            U_new[i*m+j, :] = np.linalg.lstsq(C.T, np.ones(p), rcond=None)[0]
                        except:
                            U_new[i*m+j, :] = C.T @ np.linalg.pinv(C.T) @ np.ones(p)
                
                # Fix U,W, solve for V
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        D = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                        try:
                            V_new[j*p+k, :] = np.linalg.lstsq(D.T, np.ones(n), rcond=None)[0]
                        except:
                            V_new[j*p+k, :] = D.T @ np.linalg.pinv(D.T) @ np.ones(n)
                
                # Fix U,V, solve for W
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        E = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                        try:
                            W_new[k*n+i, :] = np.linalg.lstsq(E.T, np.ones(m), rcond=None)[0]
                        except:
                            W_new[k*n+i, :] = E.T @ np.linalg.pinv(E.T) @ np.ones(m)
                
                diff = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
                U, V, W = U_new, V_new, W_new
                
                if diff < 1e-12:
                    break
            
            # Verify
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < 1e-6:
                best_rank = rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                found = True
                break
        
        if found:
            break
    
    if best_U is None:
        # Fallback to schoolbook
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
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
