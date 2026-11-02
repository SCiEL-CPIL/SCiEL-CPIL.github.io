"""Find a rank-minimal exact tensor decomposition for 2x3 x 3x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]
n, m, p = DIMENSIONS
NMP = n * m * p  # 54

def schoolbook():
    """Standard rank n*m*p = 54 decomposition."""
    rank = NMP
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def verify(U, V, W):
    """Verify that U, V, W form a valid decomposition of the 2x3 x 3x9 multiplication tensor."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    # Build the target tensor T[i,j,k] = e_{i,k} (Kronecker-like)
    # The multiplication tensor for (n x m) * (m x p) -> (n x p):
    # T[a, b, c] where a indexes A (n*m), b indexes B (m*p), c indexes C (n*p)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and i==i' and k==k'
    # Actually the tensor is: T[A_idx, B_idx, C_idx] where
    # A_idx = i*m+j, B_idx = j'*p+k, C_idx = k'*n+i'
    # T = 1 iff j == j' and i == i' and k == k'
    
    # Compute via U, V, W: C_tensor = sum_r U[:,r] outer V[:,r] outer W[:,r]
    # i.e., C[A,B,C] = sum_r U[A,r] * V[B,r] * W[C,r]
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute the reconstructed tensor
    # For each output entry (c_idx), sum over r of W[c_idx, r] * (sum over b of U[a_idx, r] * V[b_idx, r])
    # Actually: T_recon[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Target: T_target[a, b, c] = 1 iff a=i*m+j, b=j'*p+k, c=k'*n+i' with j==j', i==i', k==k'
    
    # Check: for all a, b, c
    # T_recon[a,b,c] should equal T_target[a,b,c]
    
    # More efficient check: for each c_idx (output entry k'*n+i'),
    # the output C[k',i'] = sum_j A[i',j] * B[j,k']
    # So the coefficient of A[i',j] * B[j,k'] in C[k',i'] should be 1
    # And coefficient of A[i',j] * B[j',k'] for j' != j should be 0
    
    # T_recon[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # We need T_recon[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and i==i' and k==k', else 0
    
    # Let's check all (a, b, c) triples where the target is nonzero
    # and also spot-check some where target is zero
    
    # Efficient: compute the full tensor and compare
    # T_recon has shape (n*m, m*p, p*n) = (6, 27, 18) = 2916 entries
    # We can compute it as: for each r, outer product U[:,r] @ V[:,r] @ W[:,r]
    # T_recon = sum_r np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    # Simpler: T_recon[a,b,c] = (U_arr.T @ V_arr) doesn't work directly for 3-tensor
    # T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # But this might be slow for large r. Let's just do it.
    try:
        T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    except:
        return False
    
    # Build target
    T_target = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                T_target[a_idx, b_idx, c_idx] = 1
    
    return np.allclose(T_recon, T_target, atol=1e-9)

def construct():
    deadline = time.time() + 120
    
    # Start with schoolbook (guaranteed valid, rank 54)
    best = schoolbook()
    
    # Try to find a smaller decomposition
    # Strategy: Try to find rank R < 54 by random search / iterative refinement
    
    # First, try a structured approach:
    # The 2x3 x 3x9 multiplication can be seen as 3 independent 2x3 x 3x3 multiplications
    # if we group the 9 columns into 3 groups of 3.
    # Rank(2x3 x 3x3) = 18 (schoolbook) or possibly less.
    # If we find rank 10 for 2x3x3, total rank = 30.
    
    # Let's try to find a good decomposition for 2x3 x 3x3 first, then replicate.
    
    # Actually, let's try a direct random search for the full 2x3 x 3x9 case.
    # We'll use a random initialization and then do gradient-free refinement.
    
    # Alternative: use the fact that the tensor has a specific structure.
    # The multiplication tensor T_{2,3,9} has rank at most 54 (schoolbook).
    # It's known that R(2,3,3) = 18 (schoolbook is optimal for 2x3x3? No, actually it might be less)
    
    # Let me try a random search approach with deadline guard.
    
    def random_search(rank_target, max_iters):
        """Try to find a decomposition of given rank."""
        np.random.seed(42)
        n, m, p = DIMENSIONS
        nm = n*m
        mp = m*p
        pn = p*n
        
        for trial in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            # Initialize with random values
            U = np.random.randn(nm, rank_target)
            V = np.random.randn(mp, rank_target)
            W = np.random.randn(pn, rank_target)
            
            # Scale to help
            U /= np.sqrt(rank_target)
            V /= np.sqrt(rank_target)
            W /= np.sqrt(rank_target)
            
            # Try a few gradient steps to improve fit
            best_err = np.inf
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for step in range(20):
                if time.time() > deadline - 10:
                    break
                
                # Compute gradient-like update
                # err = T_recon - T_target
                # dU = sum_bc err[a,b,c] * V[b,r] * W[c,r] for each a, r
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                
                # Build target
                T_target = np.zeros((nm, mp, pn), dtype=float)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_target[i*m+j, j*p+k, k*n+i] = 1.0
                
                err = T_recon - T_target
                current_err = np.sum(err**2)
                
                if current_err < best_err:
                    best_err = current_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if current_err < 1e-10:
                    break
                
                # Gradient descent
                lr = 0.1
                dU = np.einsum('abc,br,cr->ar', err, V, W)
                dV = np.einsum('abc,ar,cr->br', err, U, W)
                dW = np.einsum('abc,ar,br->cr', err, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Normalize to prevent divergence
                normU = np.linalg.norm(U)
                normV = np.linalg.norm(V)
                normW = np.linalg.norm(W)
                if normU > 0 and normV > 0 and normW > 0:
                    scale = (normU * normV * normW) ** (1/3)
                    U /= scale * np.sqrt(rank_target)
                    V /= scale * np.sqrt(rank_target)
                    W /= scale * np.sqrt(rank_target)
            
            if best_err < 1e-8:
                # Convert to lists and verify
                U_list = best_U.tolist()
                V_list = best_V.tolist()
                W_list = best_W.tolist()
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try various ranks from low to high
    # Lower bounds: The rank must be at least max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # And at least n*p = 18 (number of outputs)
    # Known: R(2,3,3) >= 10 (Landsberg bound), actually R(2,3,3) = 18? 
    # Let's try ranks from 27 upward.
    
    for rank in range(27, 55):
        if time.time() > deadline - 15:
            break
        
        # Try a few random searches for this rank
        result = random_search(rank, max_iters=5)
        if result is not None:
            best = result
            break
    
    # Final check: make sure best is valid
    if not verify(best["U"], best["V"], best["W"]):
        # Fall back to schoolbook
        best = schoolbook()
    
    return best
