"""Matrix multiplication 2x3x6 - structural + search hybrid."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = schoolbook_rank
    
    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (val - 1.0) ** 2
        return err
    
    # Try structured rank-27 decomposition: split k into two groups of 3
    # Group 1: k=0,1,2; Group 2: k=3,4,5
    # Each group is a 2x3 x 3x3 problem with schoolbook rank 18
    # But we can try to find lower rank for each group
    
    # First, try to find a rank-9 decomposition for 2x3 x 3x3
    # T[i,j,k] = 1 for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # Need U(6x9), V(9x9), W(6x9) such that sum_c U[i*3+j,c]*V[j*3+k,c]*W[k*2+i,c] = 1
    
    def try_2x3x3(rank, iters=50000):
        """Try to find rank-r decomposition for 2x3 x 3x3."""
        nn, mm, pp = 2, 3, 3
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for it in range(iters):
            if time.time() > deadline - 5:
                break
            
            U = np.random.randn(nn*mm, rank)
            V = np.random.randn(mm*pp, rank)
            W = np.random.randn(pp*nn, rank)
            
            err = 0.0
            for i in range(nn):
                for j in range(mm):
                    for k in range(pp):
                        val = 0.0
                        for c in range(rank):
                            val += U[i*mm+j, c] * V[j*pp+k, c] * W[k*nn+i, c]
                        err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-12:
                return best_U, best_V, best_W, best_err
        
        return best_U, best_V, best_W, best_err
    
    # Try rank 12 for 2x3x3 (schoolbook is 18)
    for target_rank in [12, 13, 14, 15]:
        if time.time() > deadline - 15:
            break
        
        U_sub, V_sub, W_sub, err = try_2x3x3(target_rank, iters=30000)
        
        if err < 1e-9:
            # Verify
            if verify(U_sub, V_sub, W_sub):
                # Build full 2x3x6 using two copies
                full_rank = 2 * target_rank
                U_full = np.zeros((n*m, full_rank))
                V_full = np.zeros((m*p, full_rank))
                W_full = np.zeros((p*n, full_rank))
                
                # Group 1: k=0,1,2
                U_full[:, :target_rank] = U_sub
                V_full[:3*3, :target_rank] = V_sub  # j*3+k for k=0,1,2
                W_full[:3*2, :target_rank] = W_sub  # k*2+i for k=0,1,2
                
                # Group 2: k=3,4,5
                U_full[:, target_rank:] = U_sub
                V_full[3*3:, target_rank:] = V_sub  # j*3+k for k=3,4,5 -> index 9+j*3+k... 
                # Wait, V index is j*p+k = j*6+k. For k=3,4,5: j*6+3, j*6+4, j*6+5
                # V_sub has shape (9, target_rank) for indices j*3+k where j in 0..2, k in 0..2
                # We need to map: V_full[j*6+k, target_rank:] = V_sub[j*3+(k-3), target_rank:] for k=3,4,5
                for j in range(3):
                    for k in range(3):
                        V_full[j*6 + (k+3), target_rank:] = V_sub[j*3 + k, :]
                
                # W index is k*n+i = k*2+i. For k=3,4,5: 6+i, 7+i, 8+i
                # W_sub has shape (6, target_rank) for indices k*2+i where k in 0..2, i in 0..1
                # W_full[6+i, target_rank:] = W_sub[0*2+i, :] for k=3 -> index 6
                # W_full[7+i, target_rank:] = W_sub[1*2+i, :] for k=4 -> index 7
                # W_full[8+i, target_rank:] = W_sub[2*2+i, :] for k=5 -> index 8
                for k in range(3):
                    for i in range(2):
                        W_full[(k+3)*2 + i, target_rank:] = W_sub[k*2 + i, :]
                
                if verify(U_full, V_full, W_full):
                    best_result = {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
                    best_rank = full_rank
                    return best_result
    
    # Try direct random search for full problem at various ranks
    for rank in range(20, min(36, best_rank)):
        if time.time() > deadline - 10:
            break
        
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for it in range(20000):
            if time.time() > deadline - 5:
                break
            
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            err = compute_error(U, V, W)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-12:
                break
        
        if best_err < 1e-9 and best_U is not None:
            if verify(best_U, best_V, best_W):
                best_result = {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
                best_rank = rank
                return best_result
    
    return best_result
# EVOLVE-BLOCK-END
