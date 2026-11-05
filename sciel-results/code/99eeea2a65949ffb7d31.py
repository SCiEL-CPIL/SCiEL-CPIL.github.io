"""Program exploring non-constant step heights to reduce the third autocorrelation constant."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def _objective(h):
    s = np.sum(h)
    if s == 0:
        return 1e12
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s * s))

def construct():
    best_h = np.ones(N)
    best_val = _objective(best_h)
    
    # Multi-start strategy
    # 1. Constant
    # 2. Random uniform positive
    # 3. Random uniform signed
    # 4. Low frequency cosine variants
    
    starts = []
    starts.append(np.ones(N))
    
    # Random positive
    for _ in range(5):
        starts.append(np.random.rand(N))
        
    # Random signed
    for _ in range(5):
        starts.append(np.random.randn(N))
        
    # Cosine waves with random phase/freq
    for f in [1, 2, 3, 5]:
        t = np.linspace(0, 2*np.pi*f, N)
        starts.append(1.0 + 0.5*np.cos(t))
        starts.append(1.0 + 0.5*np.sin(t))

    rng = np.random.default_rng(42)
    
    for h0 in starts:
        h = h0.copy()
        current_val = _objective(h)
        
        # Hill Climbing
        # Step size decay
        step_size = 0.5
        no_improve_count = 0
        max_no_improve = 50
        
        for i in range(2000): # Iterations per start
            if no_improve_count > max_no_improve:
                break
                
            # Pick random index
            idx = rng.integers(N)
            
            # Try a few perturbations
            found_improvement = False
            for _ in range(10):
                delta = rng.uniform(-step_size, step_size)
                h_new = h.copy()
                h_new[idx] += delta
                
                val_new = _objective(h_new)
                
                if val_new < current_val:
                    h = h_new
                    current_val = val_new
                    no_improve_count = 0
                    found_improvement = True
                    break
            
            if not found_improvement:
                no_improve_count += 1
                # Shrink step size if stuck
                step_size *= 0.95
            
            # Update global best
            if current_val < best_val:
                best_val = current_val
                best_h = h.copy()
                
    return best_h
# EVOLVE-BLOCK-END
