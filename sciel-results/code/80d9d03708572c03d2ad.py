import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_score = -np.inf
    best_x = np.ones(N)
    
    def compute_value(x):
        # x is a numpy array of shape (N,)
        # Compute autoconvolution
        # np.convolve(x, x, mode='full')
        c = np.convolve(x, x, mode='full')
        max_abs_conv = np.max(np.abs(c))
        
        # Compute squared integral
        # Assuming integral is sum(x)
        s = np.sum(x)
        sq_integral = s * s
        
        if sq_integral == 0:
            return np.inf
        
        return max_abs_conv / sq_integral

    current_value = compute_value(best_x)
    best_score = 1.0 / current_value if current_value > 0 else -np.inf

    # Helper to update best
    def update_best(x, val):
        nonlocal best_score, best_x
        if val < current_value:
            current_value = val
            best_x = x.copy()
            best_score = 1.0 / val if val > 0 else -np.inf

    # Initialize with several diverse starts
    inits = []
    inits.append(np.ones(N))
    inits.append(np.random.choice([-1, 1], N))
    
    # Chirp
    chirp = np.sign(np.cos(np.linspace(0, 10*np.pi, N)))
    chirp[chirp == 0] = 1
    inits.append(chirp)
    
    # Blocks
    block_size = 10
    blocks = np.random.choice([-1, 1], N // block_size + 1)
    block_x = np.repeat(blocks, block_size)[:N]
    inits.append(block_x)
    
    for init in inits:
        if time.time() > deadline:
            break
            
        x = init.copy()
        val = compute_value(x)
        
        if val < current_value:
            update_best(x, val)
            x = best_x.copy()
            val = current_value
            
        # Local search: sign flips
        # Random order of indices
        indices = np.arange(N)
        np.random.shuffle(indices)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in indices:
                if time.time() > deadline:
                    break
                
                # Flip sign at i
                old_sign = x[i]
                x[i] = -old_sign
                
                new_val = compute_value(x)
                
                if new_val < val:
                    val = new_val
                    improved = True
                    # Update global best
                    if val < current_value:
                        update_best(x, val)
                else:
                    # Revert
                    x[i] = old_sign

    return best_x

# EVOLVE-BLOCK-START
# (The code above is the content of the evolve block)
# EVOLVE-BLOCK-END
