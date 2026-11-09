import numpy as np
import time

def construct():
    n, m, p = 3, 4, 7
    t_start = time.time()
    deadline = t_start + 120.0
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_sb = np.zeros((n * m, rank_school))
    V_sb = np.zeros((m * p, rank_school))
    W_sb = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = rank_school

    def verify(U, V, W, tol=1e-6):
        """
        U: (n*m, R)
        V: (m*p, R)
        W: (p*n, R)
        Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        """
        R = U.shape[1]
        # Precompute for speed
        # We need to check for all i, j, k
        # Sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's iterate
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Compute sum
                    # U[u_idx, :] * V[v_idx, :] * W[w_idx, :]
                    # This is element-wise product then sum
                    prod = U[u_idx, :] * V[v_idx, :] * W[w_idx, :]
                    s = np.sum(prod)
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def solve_for_rank(R, max_iters=2000, lr=0.01, seed=0):
        np.random.seed(seed)
        # Initialize
        # Scale initialization to avoid huge gradients
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        # Gradient descent
        # Loss: L = 0.5 * sum_{i,j,k} (sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1)^2
        # dL/dU[i,j,r] = (sum_r' U[i,j,r'] V[j,k,r'] W[k,i,r'] - 1) * V[j,k,r] * W[k,i,r] * (number of k for this i,j? No, sum over k)
        # Actually, for a fixed U[i,j,r], it appears in terms for all k.
        # Let S[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # dL/dU[i*m+j, r] = sum_k (S[i,j,k] - 1) * V[j*p+k, r] * W[k*n+i, r]
        
        # We can compute gradients efficiently using matrix multiplications?
        # S is a tensor of shape (n, m, p).
        # Let's just compute it directly, n*m*p = 84 terms, R is small.
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Compute S[i, j, k]
            S = np.zeros((n, m, p))
            for r in range(R):
                # U[:, r] is vector of length n*m. Reshape to (n, m)
                Ur = U[:, r].reshape(n, m)
                # V[:, r] is vector of length m*p. Reshape to (m, p)
                Vr = V[:, r].reshape(m, p)
                # W[:, r] is vector of length p*n. Reshape to (p, n)
                Wr = W[:, r].reshape(p, n)
                
                # S[i, j, k] += Ur[i, j] * Vr[j, k] * Wr[k, i]
                # This is an outer product kind of thing.
                # For fixed r, contribution to S is Ur[i,j] * Vr[j,k] * Wr[k,i]
                # We can compute this by iterating i, j, k or using einsum
                # S += np.einsum('ij,jk,ki->ijk', Ur, Vr, Wr)
                S += np.einsum('ij,jk,ki->ijk', Ur, Vr, Wr)
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            # dU[i*m+j, r] = sum_k (S[i,j,k] - 1) * V[j*p+k, r] * W[k*n+i, r]
            # dV[j*p+k, r] = sum_i (S[i,j,k] - 1) * U[i*m+j, r] * W[k*n+i, r]
            # dW[k*n+i, r] = sum_j (S[i,j,k] - 1) * U[i*m+j, r] * V[j*p+k, r]
            
            # We can compute these using loops or einsum.
            # Let's use loops for simplicity and clarity, as n,m,p are small.
            # But vectorizing over r is better.
            
            # Let Err = S - 1. Shape (n, m, p).
            Err = S - 1.0
            
            for r in range(R):
                Ur = U[:, r].reshape(n, m)
                Vr = V[:, r].reshape(m, p)
                Wr = W[:, r].reshape(p, n)
                
                # dU[:, r] is (n, m)
                # dU[i, j] = sum_k Err[i, j, k] * Vr[j, k] * Wr[k, i]
                # dU_r = np.einsum('ijk,jk,ki->ij', Err, Vr, Wr)
                dU[:, r] = np.einsum('ijk,jk,ki->ij', Err, Vr, Wr).flatten()
                
                # dV[:, r] is (m, p)
                # dV[j, k] = sum_i Err[i, j, k] * Ur[i, j] * Wr[k, i]
                # dV_r = np.einsum('ijk,ij,ki->jk', Err, Ur, Wr)
                dV[:, r] = np.einsum('ijk,ij,ki->jk', Err, Ur, Wr).flatten()
                
                # dW[:, r] is (p, n)
                # dW[k, i] = sum_j Err[i, j, k] * Ur[i, j] * Vr[j, k]
                # dW_r = np.einsum('ijk,ij,jk->ki', Err, Ur, Vr)
                dW[:, r] = np.einsum('ijk,ij,jk->ki', Err, Ur, Vr).flatten()
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check convergence
            if np.max(np.abs(Err)) < 1e-8:
                if verify(U, V, W, tol=1e-7):
                    return U, V, W
                    
            # Adaptive learning rate?
            if it % 100 == 0:
                lr *= 0.99
                if lr < 1e-6:
                    break

        # Final check
        if verify(U, V, W, tol=1e-6):
            return U, V, W
        return None

    # Try ranks from 84 down to 70
    # Start from lower to find minimum, but we know 84 works.
    # Let's try 83, 82, ...
    # Actually, we can try a few random seeds for each rank.
    
    for R in range(84, 60, -1):
        if time.time() > deadline - 10:
            break
        found = False
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            res = solve_for_rank(R, max_iters=1000, lr=0.05, seed=seed)
            if res is not None:
                U, V, W = res
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                best_rank = R
                found = True
                # We can stop searching for this rank if we want, but maybe a lower rank exists?
                # The loop continues to R-1.
                break
        
        # If we found a solution for R, we don't need to try R+1 again (we are going down).
        # But we should keep going down to find the minimum.
        
    return {"U": best_U, "V": best_V, "W": best_W}
