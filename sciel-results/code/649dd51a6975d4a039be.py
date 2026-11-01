"""Matrix multiplication 2,3,7: rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

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
    
    def verify(U, V, W, tol=1e-9):
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
        """Compute the reconstructed tensor using vectorized operations."""
        r = U.shape[1]
        R = np.einsum('ac,bc,dc->abd', U, V, W)
        return R
    
    def tensor_error(U, V, W):
        """Compute the squared error of the tensor reconstruction."""
        R = compute_R(U, V, W)
        return np.sum((R - target) ** 2)
    
    def compute_gradients(U, V, W):
        """Compute gradients for U, V, W using vectorized operations."""
        r = U.shape[1]
        R = compute_R(U, V, W)
        E = R - target  # error tensor
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # grad_U[:, c] = 2 * (E @ W[:, c]) @ V[:, c]
        # For each c: grad_U[:, c] = 2 * einsum('abc,c->ab', E, W[:, c]) @ V[:, c]
        for c in range(r):
            M_U = np.einsum('abc,c->ab', E, W[:, c])
            grad_U[:, c] = 2.0 * (M_U @ V[:, c])
            
            N_V = np.einsum('abc,a->bc', E, U[:, c])
            grad_V[:, c] = 2.0 * (N_V @ W[:, c])
            
            UV_outer = np.outer(U[:, c], V[:, c])
            grad_W[:, c] = 2.0 * np.einsum('abc,ab->c', E, UV_outer)
        
        return grad_U, grad_V, grad_W
    
    def optimize(U, V, W, max_iters=500, lr_init=0.02, use_adaptive=True):
        """Optimize U, V, W to minimize tensor error using adaptive gradient descent."""
        lr = lr_init
        prev_error = tensor_error(U, V, W)
        best_error = prev_error
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Momentum
        vel_U = np.zeros_like(U)
        vel_V = np.zeros_like(V)
        vel_W = np.zeros_like(W)
        momentum = 0.9
        patience = 0
        plateau_count = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
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
                patience = 0
                plateau_count = 0
                
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if use_adaptive:
                    lr *= 0.998
            else:
                patience += 1
                plateau_count += 1
                lr *= 0.5
                
                if patience > 30 or lr < 1e-12:
                    break
        
        return best_U, best_V, best_W, best_error
    
    def polish(U, V, W, max_iters=200, lr_init=0.005):
        """High-precision polish pass with smaller steps."""
        lr = lr_init
        prev_error = tensor_error(U, V, W)
        best_error = prev_error
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        vel_U = np.zeros_like(U)
        vel_V = np.zeros_like(V)
        vel_W = np.zeros_like(W)
        momentum = 0.85
        patience = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
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
                patience = 0
                
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                lr *= 0.995
            else:
                patience += 1
                lr *= 0.5
                
                if patience > 20 or lr < 1e-14:
                    break
        
        return best_U, best_V, best_W, best_error
    
    def init_random(rank, scale=0.3):
        """Random initialization."""
        U = np.random.randn(nm, rank) * scale
        V = np.random.randn(mp, rank) * scale
        W = np.random.randn(pn, rank) * scale
        return U, V, W
    
    def init_structured(rank):
        """Structured initialization based on tensor support."""
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Initialize with partial structure
        for c in range(rank):
            # Random subset of indices
            idx_a = np.random.choice(nm, size=max(1, nm // 3), replace=False)
            idx_b = np.random.choice(mp, size=max(1, mp // 4), replace=False)
            idx_c = np.random.choice(pn, size=max(1, pn // 5), replace=False)
            
            for a in idx_a:
                U[a, c] = np.random.choice([-1, 1])
            for b in idx_b:
                V[b, c] = np.random.choice([-1, 1])
            for c_idx in idx_c:
                W[c_idx, c] = np.random.choice([-1, 1])
        
        return U, V, W
    
    def init_sparse(rank, sparsity=0.3):
        """Sparse initialization."""
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        for c in range(rank):
            # Random sparse pattern
            for a in range(nm):
                if np.random.random() < sparsity:
                    U[a, c] = np.random.choice([-1, 1])
            for b in range(mp):
                if np.random.random() < sparsity * 0.7:
                    V[b, c] = np.random.choice([-1, 1])
            for c_idx in range(pn):
                if np.random.random() < sparsity * 0.5:
                    W[c_idx, c] = np.random.choice([-1, 1])
        
        return U, V, W
    
    def try_rank(rank, num_attempts=5):
        """Try to find a valid decomposition for a given rank with multiple strategies."""
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        strategies = [
            ('random', lambda: init_random(rank, 0.3)),
            ('structured', lambda: init_structured(rank)),
            ('sparse', lambda: init_sparse(rank, 0.3)),
            ('random_small', lambda: init_random(rank, 0.1)),
            ('random_large', lambda: init_random(rank, 0.5)),
        ]
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(hash((rank, attempt)) % 2**32)
            
            # Choose strategy
            strat_name, init_func = strategies[attempt % len(strategies)]
            U, V, W = init_func()
            
            # Optimize
            U_opt, V_opt, W_opt, final_error = optimize(U, V, W, max_iters=300, lr_init=0.02)
            
            if final_error < best_error:
                best_error = final_error
                best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
            
            # If error is small, try polish
            if final_error < 1.0:
                U_pol, V_pol, W_pol, polish_error = polish(U_opt, V_opt, W_pol, max_iters=100, lr_init=0.005)
                if polish_error < best_error:
                    best_error = polish_error
                    best_U, best_V, best_W = U_pol.copy(), V_pol.copy(), W_pol.copy()
            
            # Round to half-integers and verify
            U_rounded = round_to_half_integer(best_U)
            V_rounded = round_to_half_integer(best_V)
            W_rounded = round_to_half_integer(best_W)
            
            if verify(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded
        
        return None, None, None
    
    def exact_search_small_rank(rank):
        """Exact combinatorial search for very small ranks."""
        if rank > 15:
            return None, None, None
        
        # For small ranks, try all sign patterns for a subset
        # This is only feasible for very small rank
        if rank > 10:
            return None, None, None
        
        # Try random sign patterns with optimization
        for attempt in range(20):
            if time.time() > deadline - 25:
                break
            
            np.random.seed(hash(('exact', rank, attempt)) % 2**32)
            
            U = np.random.choice([-1, 0, 1], size=(nm, rank))
            V = np.random.choice([-1, 0, 1], size=(mp, rank))
            W = np.random.choice([-1, 0, 1], size=(pn, rank))
            
            # Optimize
            U_opt, V_opt, W_opt, final_error = optimize(U, V, W, max_iters=200, lr_init=0.01)
            
            # Polish
            U_pol, V_pol, W_pol, polish_error = polish(U_opt, V_opt, W_opt, max_iters=50, lr_init=0.003)
            
            # Round and verify
            U_rounded = round_to_half_integer(U_pol)
            V_rounded = round_to_half_integer(V_pol)
            W_rounded = round_to_half_integer(W_pol)
            
            if verify(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded
        
        return None, None, None
    
    best_result = None
    best_rank = 42
    
    # Try ranks from low to high, with more attempts for lower ranks
    for rank in range(25, 42):
        if time.time() > deadline - 30:
            break
        
        # Use exact search for very small ranks
        if rank <= 10:
            U, V, W = exact_search_small_rank(rank)
        else:
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
