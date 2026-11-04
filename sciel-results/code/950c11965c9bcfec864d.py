import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, leave margin
    
    n, m, p = 6, 6, 6
    # Tensor T[i*m+j, j*p+k, k*n+i] = 1 if the indices are consistent
    # Actually, the tensor for matrix multiplication is:
    # T_{(i,j),(j,k),(k,i)} = 1
    # In flattened form: T[i*m+j, j*p+k, k*n+i] = 1
    
    # Build the tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    best_rank = n*m*p  # schoolbook: 216
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, idx] = 1.0
                V_school[j*p + k, idx] = 1.0
                W_school[k*n + i, idx] = 1.0
                idx += 1
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Randomized rank-1 peeling
    # For each trial, try to find a low-rank decomposition
    num_trials = 50
    
    for trial in range(num_trials):
        if time.time() > deadline - 10.0:
            break
        
        np.random.seed(trial * 7919 + 13)
        
        # Start with a random initial rank estimate
        # Try different starting ranks
        start_rank = random.randint(100, 200)
        
        U = np.zeros((n*m, 0))
        V = np.zeros((m*p, 0))
        W = np.zeros((p*n, 0))
        
        residual = T.copy()
        
        max_iter = start_rank + 50
        min_improvement = 1e-10
        
        for it in range(max_iter):
            if time.time() > deadline - 10.0:
                break
            
            if it >= start_rank:
                # Check if we can stop early
                if np.linalg.norm(residual, 'fro') < 1e-6:
                    break
            
            # Find best rank-1 component of residual
            # Use randomized SVD-like approach
            # Project residual to random vectors to estimate dominant component
            
            # Try multiple random projections and pick best
            best_U_vec = None
            best_V_vec = None
            best_W_vec = None
            best_score = 0
            
            num_projections = 20
            
            for proj in range(num_projections):
                # Random vectors
                r1 = np.random.randn(n*m)
                r2 = np.random.randn(m*p)
                r3 = np.random.randn(p*n)
                
                # Compute u = sum_jk residual[i,j,k] * r2[j] * r3[k]
                u = np.tensordot(residual, r2, axes=([1],[0]))
                u = np.tensordot(u, r3, axes=([1],[0]))
                
                # Compute v = sum_ik residual[i,j,k] * r1[i] * r3[k]
                v = np.tensordot(residual, r1, axes=([0],[0]))
                v = np.tensordot(v, r3, axes=([1],[0]))
                
                # Compute w = sum_ij residual[i,j,k] * r1[i] * r2[j]
                w = np.tensordot(residual, r1, axes=([0],[0]))
                w = np.tensordot(w, r2, axes=([1],[0]))
                
                # Score: how well this rank-1 approximates residual
                score = np.abs(np.tensordot(u, v, axes=([0],[0])))
                score = np.tensordot(score, w, axes=([0],[0]))
                
                if score > best_score:
                    best_score = score
                    best_U_vec = u
                    best_V_vec = v
                    best_W_vec = w
            
            if best_score < min_improvement:
                break
            
            # Normalize
            norm_u = np.linalg.norm(best_U_vec)
            norm_v = np.linalg.norm(best_V_vec)
            norm_w = np.linalg.norm(best_W_vec)
            
            if norm_u < 1e-12 or norm_v < 1e-12 or norm_w < 1e-12:
                break
            
            # Scale so that product of norms = 1 (or similar)
            scale = 1.0 / (norm_u * norm_v * norm_w) ** (1.0/3.0)
            best_U_vec = best_U_vec / norm_u * scale
            best_V_vec = best_V_vec / norm_v * scale
            best_W_vec = best_W_vec / norm_w * scale
            
            # Update residual
            residual -= np.outer(np.outer(best_U_vec, best_V_vec), best_W_vec).reshape(n*m, m*p, p*n)
            
            # Append to factors
            U = np.hstack([U, best_U_vec.reshape(-1, 1)])
            V = np.hstack([V, best_V_vec.reshape(-1, 1)])
            W = np.hstack([W, best_W_vec.reshape(-1, 1)])
        
        current_rank = U.shape[1]
        
        # Verify the decomposition
        if current_rank > 0:
            # Check: sum_r U[:,r] V[:,r] W[:,r] should equal T
            reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
            error = np.linalg.norm(reconstructed - T, 'fro')
            
            if error < 1e-6 and current_rank < best_rank:
                best_rank = current_rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
        
        # Early termination if we've found something very good
        if best_rank <= 150:
            break
    
    # If no improvement found, return schoolbook
    if best_U is None:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
