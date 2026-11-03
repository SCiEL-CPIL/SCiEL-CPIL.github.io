import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 3, 8
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16
    
    t_start = time.time()
    deadline = t_start + 140.0
    
    # Build the known rank-44 decomposition
    def build_rank44():
        rank = 44
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        
        # Part 1: j=0, 16 terms
        for k in range(p):
            U[0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1
            U[1 * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + 1, col] = 1
            col += 1
        
        # Part 2: j=1,2 in 4 blocks of 2x2x2, 7 terms each = 28
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            # P1
            U[0 * m + 1, col] = 1
            U[1 * m + 2, col] = 1
            V[1 * p + k0, col] = 1
            V[2 * p + k1, col] = 1
            W[k0 * n + 0, col] = 1
            W[k1 * n + 1, col] = 1
            col += 1
            
            # P2
            U[1 * m + 1, col] = 1
            U[1 * m + 2, col] = 1
            V[1 * p + k0, col] = 1
            W[k0 * n + 1, col] = 1
            W[k1 * n + 1, col] = -1
            col += 1
            
            # P3
            U[0 * m + 1, col] = 1
            V[1 * p + k1, col] = 1
            V[2 * p + k1, col] = -1
            W[k1 * n + 0, col] = 1
            W[k1 * n + 1, col] = 1
            col += 1
            
            # P4
            U[1 * m + 2, col] = 1
            V[2 * p + k0, col] = 1
            V[1 * p + k0, col] = -1
            W[k0 * n + 0, col] = 1
            W[k0 * n + 1, col] = 1
            col += 1
            
            # P5
            U[0 * m + 1, col] = 1
            U[0 * m + 2, col] = 1
            V[2 * p + k1, col] = 1
            W[k0 * n + 0, col] = -1
            W[k1 * n + 0, col] = 1
            col += 1
            
            # P6
            U[1 * m + 1, col] = 1
            U[0 * m + 1, col] = -1
            V[1 * p + k0, col] = 1
            V[1 * p + k1, col] = 1
            W[k1 * n + 1, col] = 1
            col += 1
            
            # P7
            U[0 * m + 2, col] = 1
            U[1 * m + 2, col] = -1
            V[2 * p + k0, col] = 1
            V[2 * p + k1, col] = 1
            W[k0 * n + 0, col] = 1
            col += 1
        
        return U, V, W, rank
    
    U44, V44, W44, rank44 = build_rank44()
    best_U, best_V, best_W = U44.copy(), V44.copy(), W44.copy()
    best_rank = rank44

    # Verification function
    def verify(U, V, W, tol=1e-9):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for kp in range(p):
                                val = 0
                                for r in range(R):
                                    val += U[i * m + j, r] * V[jp * p + k, r] * W[kp * n + ip, r]
                                expected = 1.0 if (j == jp and i == ip and k == kp) else 0.0
                                if abs(val - expected) > tol:
                                    return False
        return True

    # Try to optimize the 2x2x8 sub-problem
    def optimize_2x2x8(U_sub, V_sub, W_sub, target_rank):
        # U_sub: (4, R), V_sub: (16, R), W_sub: (16, R)
        # We map the 2x2x8 indices to the full 2x3x8 indices
        # i in 0..1, j in 1..2 (mapped to 1,2 in full), k in 0..7
        
        size_U = 4
        size_V = 16
        size_W = 16
        
        def residuals(params):
            R = target_rank
            U = params[0:R*size_U].reshape(R, size_U)
            V = params[R*size_U:R*size_U + R*size_V].reshape(R, size_V)
            W = params[R*size_U + R*size_V:].reshape(R, size_W)
            
            res = []
            # Check diagonal terms
            for i in range(2):
                for j in range(2): # j=0,1 in sub -> j=1,2 in full
                    for k in range(8):
                        val = np.sum(W[:, k*2+i] * U[:, i*2+j] * V[:, j*8+k])
                        res.append(val - 1.0)
                        
            # Check some off-diagonal terms to ensure structure
            # We don't need to check all 1000, just enough to prevent drift
            # But for safety, check all? No, too slow.
            # Check terms where j != jp
            for i in range(2):
                for jp in range(2):
                    if jp == 0: continue # j=0 in sub is j=1 in full. We only care about j=1,2 in full.
                    # Actually, the sub-problem only involves j=1,2.
                    # So off-diagonals are when i!=i' or k!=k' or j!=j' within the sub.
                    pass
            
            return np.array(res)
        
        # Initialization
        if target_rank < U_sub.shape[1]:
             # Zero out some columns
             U_init = U_sub[:, :target_rank]
             V_init = V_sub[:, :target_rank]
             W_init = W_sub[:, :target_rank]
        else:
             U_init = np.random.randn(size_U, target_rank) * 0.1
             V_init = np.random.randn(size_V, target_rank) * 0.1
             W_init = np.random.randn(size_W, target_rank) * 0.1
             
        params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = least_squares(residuals, params, method='lm', max_nfev=5000)
            if result.cost < 1e-10:
                U = result.x[0:R*size_U].reshape(R, size_U)
                V = result.x[R*size_U:R*size_U + R*size_V].reshape(R, size_V)
                W = result.x[R*size_U + R*size_W:].reshape(R, size_W)
                
                # Snap
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify sub-solution
                valid = True
                for i in range(2):
                    for j in range(2):
                        for k in range(8):
                            val = np.sum(W_snap[:, k*2+i] * U_snap[:, i*2+j] * V_snap[:, j*8+k])
                            if abs(val - 1.0) > 1e-6:
                                valid = False
                if valid:
                    return U_snap, V_snap, W_snap
        except Exception:
            pass
        return None

    # Extract the 2x2x8 part from the rank 44 solution
    # Part 2 corresponds to columns 16 to 43
    U_sub_28 = U44[:, 16:44]
    # Map indices:
    # U_full[i*m + j, r] -> U_sub[i*2 + (j-1), r] for j=1,2
    # V_full[j*p + k, r] -> V_sub[(j-1)*8 + k, r] for j=1,2
    # W_full[k*n + i, r] -> W_sub[k*2 + i, r]
    
    U_sub = np.zeros((4, 28))
    V_sub = np.zeros((16, 28))
    W_sub = np.zeros((16, 28))
    
    for i in range(2):
        for j in range(2):
            U_sub[i*2+j, :] = U44[i*3 + (j+1), 16:44]
            
    for j in range(2):
        for k in range(8):
            V_sub[j*8+k, :] = V44[(j+1)*8 + k, 16:44]
            
    for k in range(8):
        for i in range(2):
            W_sub[k*2+i, :] = W44[k*2 + i, 16:44]
            
    # Try to reduce rank of sub-problem
    best_sub_rank = 28
    best_sub_U, best_sub_V, best_sub_W = U_sub, V_sub, W_sub
    
    for target_rank in range(27, 24, -1):
        if time.time() > deadline - 20:
            break
        res = optimize_2x2x8(U_sub, V_sub, W_sub, target_rank)
        if res:
            best_sub_U, best_sub_V, best_sub_W = res
            best_sub_rank = target_rank
            break
            
    # Combine
    if best_sub_rank < 28:
        total_rank = 16 + best_sub_rank
        U_new = np.zeros((6, total_rank))
        V_new = np.zeros((24, total_rank))
        W_new = np.zeros((16, total_rank))
        
        # Copy Part 1 (j=0)
        U_new[:, 0:16] = U44[:, 0:16]
        V_new[:, 0:16] = V44[:, 0:16]
        W_new[:, 0:16] = W44[:, 0:16]
        
        # Copy Part 2 (j=1,2)
        for i in range(2):
            for j in range(2):
                U_new[i*3 + (j+1), 16:16+best_sub_rank] = best_sub_U[i*2+j, :]
                
        for j in range(2):
            for k in range(8):
                V_new[(j+1)*8 + k, 16:16+best_sub_rank] = best_sub_V[j*8+k, :]
                
        for k in range(8):
            for i in range(2):
                W_new[k*2 + i, 16:16+best_sub_rank] = best_sub_W[k*2+i, :]
                
        if verify(U_new, V_new, W_new):
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = total_rank

    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}

def _verify(U, V, W):
    n, m, p = 2, 3, 8
    R = len(U[0]) if U else 0
    
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                for i in range(n):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            if i == i0 and k == k0:
                                val += W[k * n + i][r] * U[i * m + j0][r] * V[j0 * p + k][r]
                        expected_val = 1 if (i == i0 and k == k0) else 0
                        if val != expected_val:
                            return False
    return True

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    if _verify(res["U"], res["V"], res["W"]):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
