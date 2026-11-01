"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy import linalg
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Precompute target tensor
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is exact and has integer/half-integer entries."""
        r = U.shape[1]
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        
        if np.max(np.abs(R - target)) > tol:
            return False
        
        # Check integer/half-integer
        for mat in [U, V, W]:
            for val in mat:
                if not (abs(val * 2 - round(val * 2)) < tol):
                    return False
        return True
    
    def round_to_half_integer(mat):
        """Round each entry to the nearest half-integer."""
        return np.round(mat * 2) / 2.0
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    def compute_R(U, V, W):
        """Compute the reconstructed tensor."""
        r = U.shape[1]
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        return R
    
    def tensor_error(U, V, W):
        """Compute the squared error of the tensor reconstruction."""
        R = compute_R(U, V, W)
        return np.sum((R - target) ** 2)
    
    def compute_gradients(U, V, W):
        """Compute gradients of tensor_error w.r.t. U, V, W efficiently."""
        r = U.shape[1]
        R = compute_R(U, V, W)
        E = R - target  # error tensor
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for c in range(r):
            M_U = np.einsum('abc,c->ab', E, W[:, c])
            grad_U[:, c] = 2.0 * (M_U @ V[:, c])
            
            N_V = np.einsum('abc,a->bc', E, U[:, c])
            grad_V[:, c] = 2.0 * (N_V @ W[:, c])
            
            UV_outer = np.outer(U[:, c], V[:, c])  # shape (nm, mp)
            grad_W[:, c] = 2.0 * np.einsum('abc,ab->c', E, UV_outer)
        
        return grad_U, grad_V, grad_W
    
    def optimize(U, V, W, max_iters=500, lr_init=0.1):
        """Optimize U, V, W to minimize tensor error using gradient descent with momentum."""
        lr = lr_init
        prev_error = tensor_error(U, V, W)
        
        # Momentum
        vel_U = np.zeros_like(U)
        vel_V = np.zeros_like(V)
        vel_W = np.zeros_like(W)
        momentum = 0.9
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            grad_U, grad_V, grad_W = compute_gradients(U, V, W)
            
            # Update velocities
            vel_U = momentum * vel_U - lr * grad_U
            vel_V = momentum * vel_V - lr * grad_V
            vel_W = momentum * vel_W - lr * grad_W
            
            # Update
            U_new = U + vel_U
            V_new = V + vel_V
            W_new = W + vel_W
            
            new_error = tensor_error(U_new, V_new, W_new)
            
            if new_error < prev_error:
                U, V, W = U_new, V_new, W_new
                prev_error = new_error
                # Decay learning rate
                lr *= 0.995
            else:
                # Roll back and reduce learning rate
                lr *= 0.5
                if lr < 1e-10:
                    break
        
        return U, V, W, prev_error
    
    def polish(U, V, W, max_iters=1000, lr_init=0.01):
        """High-precision polish pass with tighter tolerance and more iterations."""
        lr = lr_init
        prev_error = tensor_error(U, V, W)
        
        vel_U = np.zeros_like(U)
        vel_V = np.zeros_like(V)
        vel_W = np.zeros_like(W)
        momentum = 0.95
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            grad_U, grad_V, grad_W = compute_gradients(U, V, W)
            
            vel_U = momentum * vel_U - lr * grad_U
            vel_V = momentum * vel_V - lr * grad_V
            vel_W = momentum * vel_W - lr * grad_W
            
            U_new = U + vel_U
            V_new = V + vel_V
            W_new = W + vel_W
            
            new_error = tensor_error(U_new, V_new, W_new)
            
            if new_error < prev_error:
                U, V, W = U_new, V_new, W_new
                prev_error = new_error
                lr *= 0.99
            else:
                lr *= 0.5
                if lr < 1e-12:
                    break
        
        return U, V, W, prev_error
    
    def algebraic_solve(rank, num_attempts=5):
        """
        Use algebraic structure: fix W to have simple structure, solve linear system for U, V.
        The tensor equation is: sum_c U[:,c] * V[:,c] * W[:,c] = target
        For each (a,b): sum_c U[a,c] * V[b,c] * W[:,c] = target[a,b,:]
        
        Strategy: Try specific structured W matrices and solve the resulting system.
        """
        for attempt in range(num_attempts):
            if time.time() > deadline - 20:
                break
            
            # Create a structured W matrix
            # Use a pattern that creates a linear system we can solve
            W = np.zeros((pn, rank))
            
            # Fill W with a structured pattern
            # Use different patterns for different attempts
            if attempt == 0:
                # Diagonal-like pattern
                for c in range(rank):
                    W[c % pn, c] = 1.0
            elif attempt == 1:
                # Block pattern
                for c in range(rank):
                    W[(c * 3) % pn, c] = 1.0
                    W[(c * 3 + 1) % pn, c] = 0.5
            elif attempt == 2:
                # Random sparse
                rng = np.random.RandomState(42 + attempt)
                for c in range(rank):
                    idx = rng.choice(pn, size=min(3, pn), replace=False)
                    for i in idx:
                        W[i, c] = rng.choice([0.5, 1.0, -0.5, -1.0])
            elif attempt == 3:
                # Cyclic pattern
                for c in range(rank):
                    for i in range(3):
                        W[(c + i * (rank // 3)) % pn, c] = 1.0 if i == 0 else 0.5
            else:
                # Another pattern
                rng = np.random.RandomState(100 + attempt)
                for c in range(rank):
                    W[c % pn, c] = 1.0
                    W[(c + 1) % pn, c] = 0.5
            
            # Now solve for U and V given W
            # The equation is: sum_c U[:,c] * V[:,c] * W[:,c] = target
            # For each c, define scaled vectors: Uc = U[:,c] * W[:,c] (element-wise)
            # Then: sum_c Uc * V[:,c]^T = target (as a set of matrices)
            
            # Actually, let's think of it differently.
            # target[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed r, this is a rank-1 tensor: U[:,r] * V[:,r] * W[:,r]
            
            # Let's set up a linear system. For each (a,b), we have:
            # sum_r U[a,r] * V[b,r] * W[:,r] = target[a,b,:]
            
            # This is bilinear in U and V. Let's fix V and solve for U, then alternate.
            
            # Initialize V
            rng = np.random.RandomState(200 + attempt)
            V = rng.randn(mp, rank) * 0.5
            
            # Alternate between solving for U and V
            for it in range(50):
                if time.time() > deadline - 15:
                    break
                
                # Solve for U given V and W
                # For each a: sum_r U[a,r] * (V @ diag(W[:,r]) ... no
                # target[a,b,:] = sum_r U[a,r] * V[b,r] * W[:,r]
                # Let M_b = sum_r V[b,r] * W[:,r] (this is a vector of length pn)
                # No, that's not right either.
                
                # target[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For fixed a,b: target[a,b,:] = sum_r (U[a,r] * V[b,r]) * W[:,r]
                # Let S[a,b] = sum_r U[a,r] * V[b,r] (this is U @ V^T)
                # Then target[a,b,c] = sum_r S[a,b] * W[c,r]... no that's wrong.
                
                # Actually: target[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # This is a 3-way tensor. Let's think of it as:
                # For each r, the contribution is U[:,r] * V[:,r] * W[:,r] (outer product)
                # So target = sum_r outer(outer(U[:,r], V[:,r]), W[:,r])
                
                # Let's use a different approach: vectorize and solve linear system
                # Fix V and W, solve for U.
                # target[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For fixed a: sum_r U[a,r] * (V[:,r] * W[:,r]) = target[a,:,:] (reshaped)
                # Let Z_b = V[b,:] * W[b,:] (element-wise, but dimensions don't match)
                
                # Wait, V has shape (mp, rank) and W has shape (pn, rank)
                # So V[b,r] and W[c,r] are scalars.
                # For fixed a: target[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # Let's define for each r: A_r = V[:,r] * W[:,r] (this doesn't make sense dimensionally)
                
                # Let me re-index. target[a,b,c] where a in [0,nm), b in [0,mp), c in [0,pn)
                # U[a,r], V[b,r], W[c,r]
                # For fixed a: sum_r U[a,r] * (V[b,r] * W[c,r]) = target[a,b,c]
                # Let P[b,c] = sum_r V[b,r] * W[c,r] (this is V @ W^T)
                # Then target[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # This is NOT equal to U[a,:] @ P because P[b,c] = sum_r V[b,r]*W[c,r] but we need
                # sum_r U[a,r]*V[b,r]*W[c,r] which is a different contraction.
                
                # Let's just do alternating least squares style.
                # Fix V and W, solve for U:
                # For each a: sum_r U[a,r] * V[b,r] * W[c,r] = target[a,b,c] for all b,c
                # This is: U[a,:] * (V * W) = target[a,:,:] where (V*W)[b,c,r] = V[b,r]*W[c,r]
                # Vectorize: for each a, we have a linear system in U[a,:]
                # Let M be a matrix of size (mp*pn) x rank
                # M[(b*pn+c), r] = V[b,r] * W[c,r]
                # Then M @ U[a,:] = target[a,:,:].flatten()
                
                M = np.zeros((mp * pn, rank))
                for b in range(mp):
                    for c in range(pn):
                        M[b * pn + c, :] = V[b, :] * W[c, :]
                
                # Solve least squares for each a
                U_new = np.zeros((nm, rank))
                for a in range(nm):
                    rhs = target[a, :, :].flatten()
                    try:
                        U_new[a, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                    except:
                        U_new[a, :] = np.zeros(rank)
                
                # Now fix U and W, solve for V
                # target[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For fixed b: sum_r V[b,r] * U[a,r] * W[c,r] = target[a,b,c] for all a,c
                # Let N be a matrix of size (nm*pn) x rank
                # N[(a*pn+c), r] = U[a,r] * W[c,r]
                # Then N @ V[b,:] = target[:,b,:].flatten()
                
                N_mat = np.zeros((nm * pn, rank))
                for a in range(nm):
                    for c in range(pn):
                        N_mat[a * pn + c, :] = U_new[a, :] * W[c, :]
                
                V_new = np.zeros((mp, rank))
                for b in range(mp):
                    rhs = target[:, b, :].flatten()
                    try:
                        V_new[b, :] = np.linalg.lstsq(N_mat, rhs, rcond=None)[0]
                    except:
                        V_new[b, :] = np.zeros(rank)
                
                U = U_new
                V = V_new
                
                # Check convergence
                err = tensor_error(U, V, W)
                if err < 1e-10:
                    break
            
            # Round to half-integers and verify
            U_rounded = round_to_half_integer(U)
            V_rounded = round_to_half_integer(V)
            W_rounded = round_to_half_integer(W)
            
            if verify(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded
            
            # Try polishing
            U_pol, V_pol, W_pol, err = polish(U_rounded, V_rounded, W_rounded, max_iters=200, lr_init=0.005)
            U_pol = round_to_half_integer(U_pol)
            V_pol = round_to_half_integer(V_pol)
            W_pol = round_to_half_integer(W_pol)
            
            if verify(U_pol, V_pol, W_pol):
                return U_pol, V_pol, W_pol
        
        return None, None, None
    
    def try_rank(rank, num_attempts=5):
        """Try to find a valid decomposition for a given rank using multiple strategies."""
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # Strategy 1: Random init + gradient descent
        for attempt in range(min(3, num_attempts)):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(hash((rank, attempt, 1)) % 2**32)
            U = np.random.randn(nm, rank) * 0.3
            V = np.random.randn(mp, rank) * 0.3
            W = np.random.randn(pn, rank) * 0.3
            
            U_opt, V_opt, W_opt, final_error = optimize(U, V, W, max_iters=200, lr_init=0.05)
            
            if final_error < best_error:
                best_error = final_error
                best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
            
            U_rounded = round_to_half_integer(U_opt)
            V_rounded = round_to_half_integer(V_opt)
            W_rounded = round_to_half_integer(W_opt)
            
            if verify(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded
        
        # Strategy 2: Algebraic approach
        if time.time() < deadline - 20:
            U_alg, V_alg, W_alg = algebraic_solve(rank, num_attempts=3)
            if U_alg is not None:
                return U_alg, V_alg, W_alg
        
        # Strategy 3: Start from schoolbook and try to reduce rank
        if rank < n * m * p and time.time() < deadline - 20:
            U_sb, V_sb, W_sb = schoolbook()
            # Zero out some columns to try to reduce rank
            rng = np.random.RandomState(300 + rank)
            U_red = U_sb[:, :rank]
            V_red = V_sb[:, :rank]
            W_red = W_sb[:, :rank]
            
            U_opt, V_opt, W_opt, final_error = optimize(U_red, V_red, W_red, max_iters=200, lr_init=0.05)
            
            U_rounded = round_to_half_integer(U_opt)
            V_rounded = round_to_half_integer(V_opt)
            W_rounded = round_to_half_integer(W_opt)
            
            if verify(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded
        
        return None, None, None
    
    best_result = None
    best_rank = 42
    
    # Try ranks from low to high
    for rank in range(25, 42):
        if time.time() > deadline - 20:
            break
        
        U, V, W = try_rank(rank, num_attempts=5)
        
        if U is not None:
            best_result = (U, V, W)
            best_rank = rank
            break
    
    if best_result is None:
        # Fall back to schoolbook
        U, V, W = schoolbook()
        best_result = (U, V, W)
        best_rank = 42
    
    # Convert to lists as required
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
