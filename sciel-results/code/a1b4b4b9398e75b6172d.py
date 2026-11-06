import numpy as np
import time

def construct():
    # Seed generation
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    c = np.array(p[:24], dtype=np.int8)
    
    N = 24
    # Use a dense grid for sup-norm estimation
    M = 4096
    theta = np.linspace(0, 2*np.pi, M, endpoint=False)
    # Precompute exp(ik*theta) for k=0..N-1
    # This is a matrix of size N x M
    # exp(i * k * theta_j)
    # We can compute this as (theta[None, :] * np.arange(N)[:, None]).T
    # But M=4096, N=24, so 24x4096 complex matrix is fine.
    
    # Actually, to update quickly, we can just compute the current values.
    # vals[j] = sum(c[k] * exp(i*k*theta[j]))
    
    # Let's precompute the basis
    k = np.arange(N)
    # theta is shape (M,)
    # We want exp(i * k * theta) -> shape (N, M)
    # exp(i * (k[:, None] * theta[None, :]))
    basis = np.exp(1j * (k[:, None] * theta[None, :]))
    
    # Initial values
    vals = c @ basis # shape (M,)
    
    # Function to get max norm
    def get_max_norm(vals):
        return np.max(np.abs(vals))
    
    current_max = get_max_norm(vals)
    best_c = c.copy()
    best_max = current_max
    
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # Local search
    # We will try flips.
    # To speed up, we can precompute the change for each flip?
    # Flip k: vals_new = vals - 2*c[k]*basis[k]
    # We can compute the new max for each k.
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try all single flips
        best_gain = 0
        best_flip_idx = -1
        
        for k in range(N):
            # Flip coefficient k
            # Change is -2 * c[k] * basis[k]
            delta = -2 * c[k] * basis[k]
            new_vals = vals + delta
            new_max = get_max_norm(new_vals)
            
            # We want to minimize max norm
            gain = current_max - new_max
            if gain > best_gain:
                best_gain = gain
                best_flip_idx = k
        
        if best_flip_idx != -1:
            # Apply the flip
            k = best_flip_idx
            c[k] *= -1
            vals = vals - 2 * c[k] * basis[k] # Wait, c[k] is already flipped in the array?
            # Let's be careful.
            # Old c[k] was c_old. New c[k] is -c_old.
            # delta = -2 * c_old * basis[k]
            # We calculated new_vals using c_old.
            # Now we update c array.
            # c[k] is now -c_old.
            # We should update vals to match the new c.
            # vals = sum(c_new * basis)
            # Since we already computed new_vals, we can just assign it.
            vals = new_vals
            current_max = new_max
            
            if current_max < best_max:
                best_max = current_max
                best_c = c.copy()
            
            improved = True
        
        if not improved:
            # Stuck at local minimum. Perturb.
            # Flip 2 random distinct coefficients
            idxs = np.random.choice(N, size=2, replace=False)
            for k in idxs:
                c[k] *= -1
                vals = vals - 2 * c[k] * basis[k] # Note: c[k] is now flipped, so change is -2 * new_c[k] * basis[k]?
                # Let's trace:
                # old_c -> new_c = -old_c
                # new_val = old_val + (new_c - old_c)*basis[k] = old_val - 2*old_c*basis[k]
                # In code: c[k] is updated to new_c.
                # So delta = -2 * old_c * basis[k] = 2 * new_c * basis[k]
                # So vals += 2 * c[k] * basis[k]
                # Let's re-verify.
                # If c[k] becomes -1 (was 1), delta = -2*1*basis = -2*basis.
                # Code: c[k] is -1. 2*c[k]*basis = -2*basis. Correct.
                # If c[k] becomes 1 (was -1), delta = -2*(-1)*basis = 2*basis.
                # Code: c[k] is 1. 2*c[k]*basis = 2*basis. Correct.
                pass
            
            # Recompute current max
            current_max = get_max_norm(vals)
            
            # If this perturbation is better than best, update best
            if current_max < best_max:
                best_max = current_max
                best_c = c.copy()
                
    return best_c.tolist()
