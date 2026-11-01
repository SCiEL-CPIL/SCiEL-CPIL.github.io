"""Combined approach: algebraic structured search for tensor decomposition."""
import time
import random
import numpy as np
from scipy import linalg
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s limit
    
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
    
    def try_rank(rank, num_attempts=10):
        """Try to find a valid decomposition for a given rank."""
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(hash((rank, attempt)) % 2**32)
            U = np.random.randn(nm, rank) * 0.3
            V = np.random.randn(mp, rank) * 0.3
            W = np.random.randn(pn, rank) * 0.3
            
            # Optimize
            U_opt, V_opt, W_opt, final_error = optimize(U, V, W, max_iters=300, lr_init=0.05)
            
            if final_error < best_error:
                best_error = final_error
                best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
            
            # Round to half-integers and verify
            U_rounded = round_to_half_integer(U_opt)
            V_rounded = round_to_half_integer(V_opt)
            W_rounded = round_to_half_integer(W_opt)
            
            if verify(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded
        
        return None, None, None
    
    def algebraic_search(rank, max_time=30.0):
        """
        Search for a low-rank decomposition using structured algebraic patterns.
        This leverages the fact that matrix multiplication has a specific structure
        that can sometimes be exploited to find non-trivial cancellations.
        """
        start = time.time()
        
        # Try different structured initializations
        # Pattern 1: Diagonal-like patterns based on index arithmetic
        # Pattern 2: Block-diagonal patterns
        # Pattern 3: Randomized algebraic patterns
        
        for pattern_idx in range(5):
            if time.time() - start > max_time:
                break
            
            # Generate a structured pattern
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            
            for c in range(rank):
                # Use index-based patterns
                seed_val = (c * 31 + pattern_idx * 17) % 100
                
                # For each position, decide whether to include it based on pattern
                # This creates a sparse, structured pattern
                for i in range(n):
                    for j in range(m):
                        a = i * m + j
                        # Pattern based on (i, j, c)
                        if (i + j + c) % 3 == pattern_idx % 3:
                            U[a, c] = 1.0 if (i * j + c) % 2 == 0 else -1.0
                
                for j in range(m):
                    for k in range(p):
                        b = j * p + k
                        if (j + k + c) % 3 == (pattern_idx + 1) % 3:
                            V[b, c] = 1.0 if (j * k + c) % 2 == 0 else -1.0
                
                for k in range(p):
                    for i in range(n):
                        cc = k * n + i
                        if (k + i + c) % 3 == (pattern_idx + 2) % 3:
                            W[cc, c] = 1.0 if (k * i + c) % 2 == 0 else -1.0
            
            # Optimize this structured initialization
            U_opt, V_opt, W_opt, final_error = optimize(U, V, W, max_iters=200, lr_init=0.05)
            
            # Round to half-integers and verify
            U_rounded = round_to_half_integer(U_opt)
            V_rounded = round_to_half_integer(V_opt)
            W_rounded = round_to_half_integer(W_opt)
            
            if verify(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded
        
        return None, None, None
    
    def combined_search(rank, max_time=60.0):
        """
        Combine random and algebraic search strategies.
        """
        start = time.time()
        
        # First try algebraic search (more structured, might find patterns)
        U, V, W = algebraic_search(rank, max_time=max_time * 0.4)
        if U is not None:
            return U, V, W
        
        # Then try random search
        remaining_time = max_time - (time.time() - start)
        if remaining_time > 10:
            U, V, W = try_rank(rank, num_attempts=min(5, int(remaining_time // 10) + 1))
            if U is not None:
                return U, V, W
        
        return None, None, None
    
    best_result = None
    best_rank = 42
    
    # Try ranks from low to high, with more effort on lower ranks
    # Use combined search for each rank
    for rank in range(25, 42):
        if time.time() > deadline - 20:
            break
        
        # Allocate time based on rank (lower ranks get more time)
        time_budget = min(30.0, (42 - rank) * 3.0 + 10.0)
        
        U, V, W = combined_search(rank, max_time=time_budget)
        
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
