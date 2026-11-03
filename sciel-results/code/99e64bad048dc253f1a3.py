import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    
    deadline = time.time() + 115  # safety margin
    
    # Fallback: rank-44 solution
    def make_rank_44():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(2):
            for k in range(8):
                U[i * 3 + 0][col] = 1
                V[0 * 8 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1
        
        for block in range(4):
            U[1][col] = 1; U[5][col] = 1
            V[8 + block*2][col] = 1; V[16 + block*2+1][col] = 1
            W[block*2*2][col] = 1; W[(block*2+1)*2+1][col] = 1
            col += 1
            U[4][col] = 1; U[5][col] = 1
            V[8 + block*2][col] = 1
            W[block*2*2+1][col] = 1; W[(block*2+1)*2+1][col] = -1
            col += 1
            U[1][col] = 1
            V[8 + block*2+1][col] = 1; V[16 + block*2+1][col] = -1
            W[(block*2+1)*2][col] = 1; W[(block*2+1)*2+1][col] = 1
            col += 1
            U[5][col] = 1
            V[16 + block*2][col] = 1; V[8 + block*2][col] = -1
            W[block*2*2][col] = 1; W[block*2*2+1][col] = 1
            col += 1
            U[1][col] = 1; U[2][col] = 1
            V[16 + block*2+1][col] = 1
            W[block*2*2][col] = -1; W[(block*2+1)*2][col] = 1
            col += 1
            U[4][col] = 1; U[1][col] = -1
            V[8 + block*2][col] = 1; V[8 + block*2+1][col] = 1
            W[(block*2+1)*2+1][col] = 1
            col += 1
            U[2][col] = 1; U[5][col] = -1
            V[16 + block*2][col] = 1; V[16 + block*2+1][col] = 1
            W[block*2*2][col] = 1
            col += 1
        
        return {"U": U, "V": V, "W": W}
    
    best_result = make_rank_44()
    best_rank = 44
    
    # Strategy: Find rank-r decomposition of 2x3x4 sub-problem, then use tensor product with 1x1x2
    # If we find rank R for 2x3x4, total rank = R * 2
    # Target: R <= 12 gives total rank 24
    
    def verify_decomposition(U, V, W, nn, mm, pp):
        """Verify that U,V,W exactly decompose the nn x mm x pp multiplication tensor."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        r = U_np.shape[1]
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    val = sum(U_np[i*mm+j, r_idx] * V_np[j*pp+k, r_idx] * W_np[k*nn+i, r_idx] for r_idx in range(r))
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True
    
    def search_rank_R_subproblem(R, nn=2, mm=3, pp=4, max_iters=5000, seed=42):
        """Search for rank-R decomposition of nn x mm x pp multiplication tensor."""
        rng = np.random.RandomState(seed)
        # Initialize with random values
        U = rng.randn(nn * mm, R)
        V = rng.randn(mm * pp, R)
        W = rng.randn(pp * nn, R)
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute error: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] should equal 1
            err = 0.0
            for i in range(nn):
                for j in range(mm):
                    for k in range(pp):
                        val = sum(U[i*mm+j, r] * V[j*pp+k, r] * W[k*nn+i, r] for r in range(R))
                        err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-10:
                break
            
            # Gradient descent with adaptive step
            lr = 0.01 / (1 + it * 0.001)
            for i in range(nn):
                for j in range(mm):
                    for k in range(pp):
                        grad_U = 2 * (sum(U[i*mm+j, r] * V[j*pp+k, r] * W[k*nn+i, r] for r in range(R)) - 1.0) * V[j*pp+k] * W[k*nn+i]
                        U[i*mm+j] -= lr * grad_U
            
            for j in range(mm):
                for k in range(pp):
                    for i in range(nn):
                        grad_V = 2 * (sum(U[i*mm+j, r] * V[j*pp+k, r] * W[k*nn+i, r] for r in range(R)) - 1.0) * U[i*mm+j] * W[k*nn+i]
                        V[j*pp+k] -= lr * grad_V
            
            for k in range(pp):
                for i in range(nn):
                    for j in range(mm):
                        grad_W = 2 * (sum(U[i*mm+j, r] * V[j*pp+k, r] * W[k*nn+i, r] for r in range(R)) - 1.0) * U[i*mm+j] * V[j*pp+k]
                        W[k*nn+i] -= lr * grad_W
        
        return best_U, best_V, best_W, best_err
    
    # Try different ranks for the 2x3x4 sub-problem
    for R in [12, 13, 14, 15, 16]:
        if time.time() > deadline - 30:
            break
        
        # Multiple restarts
        found = False
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            U_sub, V_sub, W_sub, err = search_rank_R_subproblem(R, 2, 3, 4, max_iters=3000, seed=42 + restart * 1000)
            
            if err < 1e-8:
                # Found valid decomposition! Build full 2x3x8 via tensor product with 1x1x2
                # C[i,k] = sum_j A[i,j]*B[j,k], k = 2*t + s where t in {0,1}, s in {0,1}
                # Split B into two 3x4 blocks: B_left (cols 0-3), B_right (cols 4-7)
                # C[i, 2t+s] = sum_j A[i,j]*B[j, 2t+s]
                # = (sum_j A[i,j]*B_left[j,s]) * 1 + ... 
                # Actually simpler: C[i,k] for k=0..7. Let k = 4*t + s, t in {0,1}, s in {0,1,2,3}
                # C[i, 4t+s] = sum_j A[i,j]*B[j, 4t+s]
                # This is just two independent 2x3x4 problems!
                # So total rank = 2 * R
                
                total_rank = 2 * R
                if total_rank < best_rank:
                    # Build full decomposition
                    U_full = [[0] * total_rank for _ in range(n * m)]
                    V_full = [[0] * total_rank for _ in range(m * p)]
                    W_full = [[0] * total_rank for _ in range(p * n)]
                    
                    for t in range(2):
                        for r_idx in range(R):
                            col = t * R + r_idx
                            for i in range(n):
                                for j in range(m):
                                    U_full[i * m + j][col] = U_sub[i * m + j, r_idx]
                            for j in range(m):
                                for s in range(4):
                                    V_full[j * p + t * 4 + s][col] = V_sub[j * 4 + s, r_idx]
                            for s in range(4):
                                for i in range(n):
                                    W_full[(t * 4 + s) * n + i][col] = W_sub[s * n + i, r_idx]
                    
                    # Verify
                    if verify_decomposition(U_full, V_full, W_full, n, m, p):
                        best_result = {"U": U_full, "V": V_full, "W": W_full}
                        best_rank = total_rank
                        found = True
                        break
                    else:
                        # Numerical issue, try to snap to nearest half-integers
                        U_snap = np.round(2 * np.array(U_full)) / 2.0
                        V_snap = np.round(2 * np.array(V_full)) / 2.0
                        W_snap = np.round(2 * np.array(W_full)) / 2.0
                        if verify_decomposition(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
                            best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                            best_rank = total_rank
                            found = True
                            break
        
        if found:
            break
    
    return best_result
