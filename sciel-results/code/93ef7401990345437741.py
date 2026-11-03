"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p), dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    def schoolbook():
        """Standard schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_decomp = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    def verify(U, V, W, rank):
        """Verify that the decomposition exactly reconstructs T."""
        recon = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        recon[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
        return np.allclose(recon, T, atol=1e-8)
    
    def strassen_2x2x2():
        """
        Strassen's algorithm for 2x2x2 matrix multiplication.
        Returns U, V, W matrices of shape (4,7) each for the 2x2*2x2 case.
        Using the convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        # Strassen's 7 products for 2x2 matrices:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B21)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For the tensor decomposition, we need to express each C element
        # as a sum of products of A and B elements.
        
        # Let's build the decomposition directly.
        # A is 2x2: A[i,j] for i,j in {0,1}
        # B is 2x2: B[j,k] for j,k in {0,1}
        # C is 2x2: C[i,k] for i,k in {0,1}
        
        # The 7 intermediate products and how they contribute to C:
        # We'll create 7 rank-1 terms.
        
        U = np.zeros((4, 7))  # n*m = 2*2 = 4
        V = np.zeros((4, 7))  # m*p = 2*2 = 4
        W = np.zeros((4, 7))  # p*n = 2*2 = 4
        
        # M1 = (A00+A11)(B00+B11)
        # Contributes to: C00 (+1), C11 (+1)
        # A part: A00 + A11 -> U[0*2+0,0]=1, U[1*2+1,0]=1
        # B part: B00 + B11 -> V[0*2+0,0]=1, V[1*2+1,0]=1
        # C part: C00 + C11 -> W[0*2+0,0]=1, W[1*2+1,0]=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        # Contributes to: C10 (+1)
        # A part: A10 + A11 -> U[1*2+0,1]=1, U[1*2+1,1]=1
        # B part: B00 -> V[0*2+0,1]=1
        # C part: C10 -> W[0*2+1,1]=1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # M3 = A00(B10-B11)
        # Contributes to: C01 (+1)
        # A part: A00 -> U[0*2+0,2]=1
        # B part: B10 - B11 -> V[1*2+0,2]=1, V[1*2+1,2]=-1
        # C part: C01 -> W[1*2+0,2]=1
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1
        
        # M4 = A11(B00-B10)
        # Contributes to: C00 (+1), C10 (+1)
        # A part: A11 -> U[1*2+1,3]=1
        # B part: B00 - B10 -> V[0*2+0,3]=1, V[1*2+0,3]=-1
        # C part: C00 + C10 -> W[0*2+0,3]=1, W[0*2+1,3]=1
        U[3, 3] = 1
        V[0, 3] = 1; V[2, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5 = (A00+A01)B11
        # Contributes to: C00 (-1), C01 (+1)
        # A part: A00 + A01 -> U[0*2+0,4]=1, U[0*2+1,4]=1
        # B part: B11 -> V[1*2+1,4]=1
        # C part: -C00 + C01 -> W[0*2+0,4]=-1, W[1*2+0,4]=1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        # Contributes to: C11 (+1)
        # A part: A10 - A00 -> U[1*2+0,5]=1, U[0*2+0,5]=-1
        # B part: B00 + B01 -> V[0*2+0,5]=1, V[0*2+1,5]=1
        # C part: C11 -> W[1*2+1,5]=1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        # Contributes to: C00 (+1)
        # A part: A01 - A11 -> U[0*2+1,6]=1, U[1*2+1,6]=-1
        # B part: B10 + B11 -> V[1*2+0,6]=1, V[1*2+1,6]=1
        # C part: C00 -> W[0*2+0,6]=1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W, 7
    
    def peel_greedy(T_res, n, m, p, max_iters=30, seed=0):
        """
        Greedy rank-1 peeling for a tensor of shape (n, m, p).
        Returns U, V, W, rank or None if no progress.
        """
        rng = np.random.RandomState(seed)
        residual = T_res.copy()
        U_cols = []
        V_cols = []
        W_cols = []
        
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            if np.allclose(residual, 0, atol=1e-10):
                break
            
            best_score = -np.inf
            best_u, best_v, best_w = None, None, None
            
            n_trials = 100
            for t in range(n_trials):
                u = rng.randn(n * m)
                v = rng.randn(m * p)
                
                u_mat = u.reshape(n, m)
                v_mat = v.reshape(m, p)
                
                num = np.zeros((n, p))
                den = np.zeros((n, p))
                for i in range(n):
                    for k in range(p):
                        uj = u_mat[i, :]
                        vk = v_mat[:, k]
                        ujvk = uj * vk
                        num[i, k] = np.dot(residual[i, :, k], ujvk)
                        den[i, k] = np.dot(ujvk, ujvk)
                
                mask = den > 1e-12
                w_flat = np.zeros(p * n)
                for i in range(n):
                    for k in range(p):
                        if mask[i, k]:
                            w_flat[k * n + i] = num[i, k] / den[i, k]
                
                recon = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            recon[i, j, k] = u_mat[i, j] * v_mat[j, k] * w_flat[k * n + i]
                
                diff = residual - recon
                score = -np.sum(diff * diff)
                
                if score > best_score:
                    best_score = score
                    best_u, best_v, best_w = u.copy(), v.copy(), w_flat.copy()
            
            if best_u is None:
                break
            
            recon = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        recon[i, j, k] = best_u[i * m + j] * best_v[j * p + k] * best_w[k * n + i]
            
            new_residual = residual - recon
            if np.sum(new_residual * new_residual) >= np.sum(residual * residual) - 1e-12:
                break
            
            U_cols.append(best_u)
            V_cols.append(best_v)
            W_cols.append(best_w)
            residual = new_residual
        
        if len(U_cols) == 0:
            return None
        
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        rank = len(U_cols)
        
        # Verify
        recon = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        recon[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
        
        if np.allclose(recon, T_res, atol=1e-8):
            return U, V, W, rank
        else:
            return None
    
    # Try block decomposition approach
    # Partition m=5 into 4+1, p=6 into 4+2
    # Block 1: 4x4x4 -> use Strassen (rank 7 per 2x2x2, so 7*7=49 for 4x4x4? No, that's not right)
    # Actually for 4x4x4, we can use Strassen recursively: 4x4 = 2*(2x2), so rank 7^2 = 49
    # But let's just use schoolbook for now and see if peeling helps
    
    # Let's try a different approach: use Strassen for the 2x2x2 blocks within the larger problem
    # For 4x5x6, we can partition:
    # i: 0-3 (4 rows)
    # j: 0-4 (5 cols)  
    # k: 0-5 (6 cols)
    
    # Let's try to find a decomposition using block structure
    # Partition into 2x2x2 blocks where possible
    
    # For now, let's try peeling on the full tensor with multiple seeds
    for seed in range(50):
        if time.time() > deadline - 15.0:
            break
        
        result = peel_greedy(T, n, m, p, max_iters=20, seed=seed)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_rank = rank
                best_decomp = (U, V, W)
                print(f"Found rank {rank} with seed {seed}")
    
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
