import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    Nm = n * m
    Mp = m * p
    Pn = p * n
    
    # Build sparse target: list of (a, b, c) where T[a,b,c] = 1
    # T[a,b,c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # i = a // m, j = a % m, k = b % p, and check b // p == j, c % n == i, c // n == k
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
    
    rank_sb = n * m * p  # 100
    
    # Schoolbook decomposition
    U_sb = np.zeros((Nm, rank_sb))
    V_sb = np.zeros((Mp, rank_sb))
    W_sb = np.zeros((Pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1.0
                V_sb[b, col] = 1.0
                W_sb[c, col] = 1.0
                col += 1
    
    def verify(U_mat, V_mat, W_mat, tol=1e-6):
        """Check if U, V, W decompose the target tensor exactly"""
        R = U_mat.shape[1]
        max_err = 0.0
        # Check target positions (should be 1)
        for (a, b, c) in target_indices:
            val = np.dot(U_mat[a, :], V_mat[b, :] * W_mat[c, :])
            err = abs(val - 1.0)
            if err > max_err:
                max_err = err
        # Check non-target positions (should be 0) - sample some
        # For efficiency, check a subset
        np.random.seed(12345)
        for _ in range(200):
            a = np.random.randint(Nm)
            b = np.random.randint(Mp)
            c = np.random.randint(Pn)
            # Skip if this is a target position
            if (a, b, c) in target_indices:
                continue
            val = np.dot(U_mat[a, :], V_mat[b, :] * W_mat[c, :])
            err = abs(val)
            if err > max_err:
                max_err = err
        return max_err < tol, max_err
    
    # Verify schoolbook
    is_valid, err = verify(U_sb, V_sb, W_sb)
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    if not is_valid:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    def objective_and_grad(U, V, W):
        """Compute objective (sum of squared errors) and gradients efficiently"""
        R = U.shape[1]
        obj = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Target positions: should be 1
        for (a, b, c) in target_indices:
            u = U[a, :]
            v = V[b, :]
            w = W[c, :]
            val = np.dot(u, v * w)
            diff = val - 1.0
            obj += diff * diff
            # grad_U[a,r] += 2*diff * V[b,r]*W[c,r]
            grad_U[a, :] += 2.0 * diff * (v * w)
            grad_V[b, :] += 2.0 * diff * (u * w)
            grad_W[c, :] += 2.0 * diff * (u * v)
        
        # Non-target positions: should be 0 (sample)
        np.random.seed(999)
        for _ in range(100):
            a = np.random.randint(Nm)
            b = np.random.randint(Mp)
            c = np.random.randint(Pn)
            if (a, b, c) in target_indices:
                continue
            u = U[a, :]
            v = V[b, :]
            w = W[c, :]
            val = np.dot(u, v * w)
            obj += val * val
            grad_U[a, :] += 2.0 * val * (v * w)
            grad_V[b, :] += 2.0 * val * (u * w)
            grad_W[c, :] += 2.0 * val * (u * v)
        
        return obj, grad_U, grad_V, grad_W
    
    def try_rank(target_rank, max_iters=500, lr_init=0.005):
        """Try to find a valid decomposition of given rank"""
        nonlocal best_rank, best_U, best_V, best_W
        if time.time() > deadline:
            return False
        
        # Initialize near schoolbook: use first target_rank columns with small perturbation
        np.random.seed(42 + target_rank)
        
        # Pick target_rank indices from the schoolbook
        indices = np.random.choice(rank_sb, target_rank, replace=False)
        U_curr = U_sb[:, indices].copy()
        V_curr = V_sb[:, indices].copy()
        W_curr = W_sb[:, indices].copy()
        
        # Add small random perturbation
        U_curr += np.random.randn(Nm, target_rank) * 0.01
        V_curr += np.random.randn(Mp, target_rank) * 0.01
        W_curr += np.random.randn(Pn, target_rank) * 0.01
        
        lr = lr_init
        best_obj = np.inf
        no_improve = 0
        
        for it in range(max_iters):
            if time.time() > deadline:
                return False
            
            obj, gU, gV, gW = objective_and_grad(U_curr, V_curr, W_curr)
            
            if obj < best_obj:
                best_obj = obj
                no_improve = 0
            else:
                no_improve += 1
            
            # Adaptive learning rate
            if no_improve > 20:
                lr *= 0.5
                no_improve = 0
                if lr < 1e-6:
                    break
            
            # Gradient step
            U_curr -= lr * gU
            V_curr -= lr * gV
            W_curr -= lr * gW
            
            # Check if we're close to a solution
            if obj < 1e-6:
                # Round to half-integers
                U_rounded = np.round(U_curr * 2) / 2
                V_rounded = np.round(V_curr * 2) / 2
                W_rounded = np.round(W_curr * 2) / 2
                
                is_valid, err = verify(U_rounded, V_rounded, W_rounded)
                if is_valid:
                    best_rank = target_rank
                    best_U = U_rounded.copy()
                    best_V = V_rounded.copy()
                    best_W = W_rounded.copy()
                    return True
        
        # Final attempt: round and verify
        U_rounded = np.round(U_curr * 2) / 2
        V_rounded = np.round(V_curr * 2) / 2
        W_rounded = np.round(W_curr * 2) / 2
        is_valid, err = verify(U_rounded, V_rounded, W_rounded)
        if is_valid:
            best_rank = target_rank
            best_U = U_rounded.copy()
            best_V = V_rounded.copy()
            best_W = W_rounded.copy()
            return True
        
        return False
    
    # Try to reduce rank
    # Start from rank 99 down to 85
    for r in range(99, 84, -1):
        if time.time() > deadline:
            break
        
        success = try_rank(r)
        if success:
            # Found a valid decomposition at this rank
            # Try to go lower
            continue
    
    # If we found something better, try a few more iterations with polish
    if best_rank < rank_sb:
        # Polish: do more iterations at the found rank
        if time.time() < deadline - 10:
            try_rank(best_rank, max_iters=200, lr_init=0.001)
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
