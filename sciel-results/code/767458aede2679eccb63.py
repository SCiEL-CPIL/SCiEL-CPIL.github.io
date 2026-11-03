"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    nm = n * m  # 16
    mp = m * p  # 28
    pn = p * n  # 28
    
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p  # 112
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Strategy: for each candidate rank, try random initializations and optimize
    
    def compute_error(U, V, W):
        """Compute the error between T and the decomposition."""
        # Efficient computation: for each r, outer product U[:,r] @ V[:,r] @ W[:,r]
        # Use einsum or loop
        # T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        # For speed, do it in chunks
        err = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(U.shape[1]):
            err += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.max(np.abs(T - err))
    
    def try_rank(target_rank, n_restarts=5, max_iter=200):
        """Try to find a rank-target_rank decomposition."""
        nonlocal best_U, best_V, best_W, best_rank
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            scale = 0.5
            U = np.random.randn(nm, target_rank) * scale
            V = np.random.randn(mp, target_rank) * scale
            W = np.random.randn(pn, target_rank) * scale
            
            # Alternating minimization (ALS)
            # For each mode, fix the others and solve least squares
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each a, U[a,:] should satisfy
                # T[a,:,:] = sum_r U[a,r] * (V[:,r] outer W[:,r])
                # This is a linear system in U[a,:]
                # T[a,:,:] = U[a,:] @ (V * W) where V*W is target_rank x (mp*pn)
                # Actually: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # Reshape: for fixed a, T[a] (mp x pn) = sum_r U[a,r] (V[:,r] outer W[:,r])
                # So T[a] = sum_r U[a,r] * M_r where M_r = V[:,r] outer W[:,r]
                # This is: T[a] = U[a,:] @ M where M is target_rank x (mp*pn)
                # But we need to solve in a way that's efficient
                
                # Better: use the fact that T[a] = sum_r U[a,r] * np.outer(V[:,r], W[:,r])
                # Let M[r] = np.outer(V[:,r], W[:,r]).reshape(-1)  # mp*pn
                # Then T[a].reshape(-1) = U[a,:] @ M.T
                # So U[a,:] = pinv(M.T) @ T[a].reshape(-1)
                
                # For efficiency, compute M = np.stack([np.outer(V[:,r], W[:,r]).reshape(-1) for r in range(target_rank)])
                # M shape: (target_rank, mp*pn)
                # Then for each a: T[a].reshape(-1) = M.T @ U[a,:]  (mp*pn = target_rank x target_rank)
                # U[a,:] = pinv(M.T) @ T[a].reshape(-1)
                
                M_U = np.stack([np.outer(V[:, r], W[:, r]).reshape(-1) for r in range(target_rank)])
                # M_U shape: (target_rank, mp*pn)
                # We want: T[a].reshape(-1) = M_U.T @ U[a,:]
                # U[a,:] = pinv(M_U.T) @ T[a].reshape(-1)
                # But pinv(M_U.T) is (target_rank, target_rank) if M_U is full rank
                # Actually M_U.T is (mp*pn, target_rank), pinv is (target_rank, mp*pn)
                
                # More efficient: solve least squares for all a at once
                # T_flat: (nm, mp*pn)
                # U = (nm, target_rank)
                # T_flat = U @ M_U  (nm x target_rank @ target_rank x mp*pn = nm x mp*pn)
                # So U = pinv(M_U) @ T_flat.T ... no
                # T_flat[a] = M_U.T @ U[a,:]  => T_flat = M_U.T @ U.T? No.
                # T_flat[a, bc] = sum_r M_U[r, bc] * U[a, r]
                # So T_flat = U @ M_U  (nm x target_rank) @ (target_rank x mp*pn) = (nm x mp*pn)
                # To solve for U: T_flat = U @ M_U
                # U = T_flat @ pinv(M_U)  (nm x mp*pn @ mp*pn x target_rank = nm x target_rank)
                
                T_flat = T.reshape(nm, -1)  # (nm, mp*pn)
                # Solve U @ M_U = T_flat
                # U = T_flat @ pinv(M_U)
                try:
                    U_new = T_flat @ np.linalg.pinv(M_U)
                except:
                    break
                
                # Update V: T[:,b,c] = sum_r V[b,r] * (U[:,r] outer W[:,r])
                # For fixed b: T[:,b,:] (nm x pn) = sum_r V[b,r] * np.outer(U[:,r], W[:,r])
                # M_V[r] = np.outer(U[:,r], W[:,r]).reshape(-1)  # nm*pn
                # T[:,b,:].reshape(-1) = M_V.T @ V[b,:]
                # T_flat2: (mp, nm*pn) where T_flat2[b] = T[:,b,:].reshape(-1)
                M_V = np.stack([np.outer(U[:, r], W[:, r]).reshape(-1) for r in range(target_rank)])
                T_flat_V = np.zeros((mp, nm * pn))
                for b in range(mp):
                    T_flat_V[b] = T[:, b, :].reshape(-1)
                try:
                    V_new = T_flat_V @ np.linalg.pinv(M_V)
                except:
                    break
                
                # Update W: T[a,b,c] = sum_r W[c,r] * (U[:,r] outer V[:,r])
                # For fixed c: T[:,:,c] (nm x mp) = sum_r W[c,r] * np.outer(U[:,r], V[:,r])
                # M_W[r] = np.outer(U[:,r], V[:,r]).reshape(-1)  # nm*mp
                # T[:,:,c].reshape(-1) = M_W.T @ W[c,:]
                M_W = np.stack([np.outer(U[:, r], V[:, r]).reshape(-1) for r in range(target_rank)])
                T_flat_W = np.zeros((pn, nm * mp))
                for c in range(pn):
                    T_flat_W[c] = T[:, :, c].reshape(-1)
                try:
                    W_new = T_flat_W @ np.linalg.pinv(M_W)
                except:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                if it % 10 == 0 or it == max_iter - 1:
                    err = compute_error(U, V, W)
                    if err < 1e-8:
                        # Found exact solution!
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = target_rank
                        return True
                    if time.time() > deadline - 10:
                        break
            
            # Check after each restart
            err = compute_error(U, V, W)
            if err < 1e-8:
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = target_rank
                return True
        
        return False
    
    # Try ranks from 111 down to some minimum
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    # But realistically, for matrix multiplication, lower bounds are higher
    # Let's try a range of ranks
    
    # Try some promising ranks
    candidate_ranks = list(range(111, 80, -1))  # Try 111 down to 81
    
    for rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
        success = try_rank(rank, n_restarts=3, max_iter=100)
        if success:
            break
    
    # If we found a better rank, convert to lists
    if best_rank < rank_school:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    else:
        U_out = U_school.tolist()
        V_out = V_school.tolist()
        W_out = W_school.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
