"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    total_size = nm * mp * pn
    
    # Build the matrix multiplication tensor in vector form
    # Tensor T[a, b, c] where a = i*m+j (input A index), b = j'*p+k (input B index), c = k'*n+i' (output index)
    # T[a, b, c] = 1 if the (i,j) element of A and (j',k) element of B contribute to (i',k') element of C
    # i.e., i = i', j = j', k = k'
    
    # The tensor is: T[i*m+j, j'*p+k, k'*n+i'] = 1 iff i==i' and j==j' and k==k'
    
    # For rank-r decomposition:
    # U has shape (nm, r), V has shape (mp, r), W has shape (pn, r)
    # The reconstructed tensor is: T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # We need T_recon == T
    
    # Start with schoolbook (rank 64)
    rank_schoolbook = n * m * p
    U_school = np.zeros((nm, rank_schoolbook))
    V_school = np.zeros((mp, rank_schoolbook))
    W_school = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Verify schoolbook
    def verify_decomposition(U, V, W, tol=1e-9):
        r = U.shape[1]
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Check against true tensor
        # True tensor: T[i*m+j, j2*p+k, k2*n+i2] = 1 iff i==i2 and j==j2 and k==k2
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for k2 in range(p):
                            for i2 in range(n):
                                a = i * m + j
                                b = j2 * p + k
                                c = k2 * n + i2
                                expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                                recon = 0.0
                                for r_idx in range(r):
                                    recon += U[a, r_idx] * V[b, r_idx] * W[c, r_idx]
                                if abs(recon - expected) > tol:
                                    return False
        return True
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist(),
        "rank": rank_schoolbook
    }
    
    # Try Strassen-like or other structured decompositions
    # For 4x4x4, known best upper bound is 49 (from Lader, 1986)
    # But finding an explicit rank-49 decomposition is non-trivial.
    
    # Let's try a systematic search for lower rank using random initializations
    # and gradient-free optimization (e.g., random restarts with local search)
    
    # First, let's try some known structured approaches
    
    # Approach: Try to find a rank-r decomposition using randomized methods
    # We'll use a simple approach: random initialization + iterative refinement
    
    def try_random_decomposition(rank, max_iters=500, seed=0):
        rng = np.random.RandomState(seed)
        
        U = rng.randn(nm, rank)
        V = rng.randn(mp, rank)
        W = rng.randn(pn, rank)
        
        # Normalize to avoid overflow/underflow
        for r in range(rank):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            if norm_u > 0 and norm_v > 0 and norm_w > 0:
                scale = (norm_u * norm_v * norm_w) ** (1/3)
                U[:, r] /= norm_u
                V[:, r] /= norm_v
                W[:, r] /= norm_w
                # Redistribute
                U[:, r] *= scale ** (1/3)
                V[:, r] *= scale ** (1/3)
                W[:, r] *= scale ** (1/3)
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute gradient of loss = 0.5 * ||T - sum_r U[:,r] V[:,r] W[:,r]||^2
            # This is expensive for full tensor, so we'll sample
            
            # Sample random (a,b,c) triples
            n_samples = 1000
            a_samples = rng.randint(0, nm, n_samples)
            b_samples = rng.randint(0, mp, n_samples)
            c_samples = rng.randint(0, pn, n_samples)
            
            # Compute loss and gradients
            loss = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for s in range(n_samples):
                a, b, c = a_samples[s], b_samples[s], c_samples[s]
                # Determine expected value
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                
                recon = np.sum(U[a, :] * V[b, :] * W[c, :])
                diff = recon - expected
                
                loss += diff * diff
                for r in range(rank):
                    uvw = U[a, r] * V[b, r] * W[c, r]
                    grad_U[a, r] += diff * V[b, r] * W[c, r]
                    grad_V[b, r] += diff * U[a, r] * W[c, r]
                    grad_W[c, r] += diff * U[a, r] * V[b, r]
            
            loss /= n_samples
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Decay learning rate
            lr *= 0.995
            
            if loss < 1e-10:
                break
        
        return best_U, best_V, best_W, best_loss
    
    # Try various ranks starting from schoolbook and going down
    for rank in range(rank_schoolbook - 1, 30, -1):
        if time.time() > deadline:
            break
        
        # Try a few random seeds
        found = False
        for seed in range(5):
            if time.time() > deadline:
                break
            
            U_try, V_try, W_try, loss = try_random_decomposition(rank, max_iters=200, seed=seed)
            
            if loss < 1e-6:
                # Verify exactly
                if verify_decomposition(U_try, V_try, W_try, tol=1e-8):
                    best_result = {
                        "U": U_try.tolist(),
                        "V": V_try.tolist(),
                        "W": W_try.tolist(),
                        "rank": rank
                    }
                    found = True
                    break
        
        if found:
            # Try to go lower
            continue
        else:
            break
    
    # Return the best result
    return {
        "U": best_result["U"],
        "V": best_result["V"],
        "W": best_result["W"]
    }
# EVOLVE-BLOCK-END
