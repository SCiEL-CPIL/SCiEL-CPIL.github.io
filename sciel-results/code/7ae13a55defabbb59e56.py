import numpy as np
import time

N = 100
TARGET_SUM = N / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # --- Core evaluation machinery (from inspirations) ---
    def get_correlations_linear(h):
        """
        Compute linear cross-correlation of h with (1-h) for all non-zero shifts.
        Uses np.correlate for O(N^2) -> O(N log N) via FFT internally for large N,
        but for N=100 direct is fine. np.correlate is efficient.
        Returns array of C_k for k != 0.
        """
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # c[N-1] is zero-shift. We exclude it.
        # c[0:N-1] are negative shifts, c[N:2N-1] are positive shifts.
        return np.concatenate([c[:N-1], c[N:]])

    def max_overlap(h):
        """Return the maximum overlap value."""
        c = get_correlations_linear(h)
        if len(c) == 0:
            return 0.0
        return np.max(c)

    def project(h):
        """
        Project h onto feasible set: 0 <= h_i <= 1 and sum(h) = N/2.
        Uses iterative clipping and adjustment.
        """
        h = np.clip(h, 0.0, 1.0)
        for _ in range(50):
            diff = TARGET_SUM - np.sum(h)
            if abs(diff) < 1e-12:
                break
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
        return h

    # --- Structural Idea: Binary/Discrete Search (from Cell Findings/Inspiration 1) ---
    # The problem landscape is "binary-like". Continuous optimization often gets stuck
    # on plateaus. We will use a Discrete Simulated Annealing approach where moves
    # are swaps or flips that preserve the sum constraint naturally (or via projection).
    # Specifically, we will work with a discrete grid of values or simply swap values
    # between indices.
    
    # Initialization: Start with a random feasible configuration
    # A good initial guess is a "block" structure or random noise projected.
    # Let's start with a random permutation of a base sequence to break symmetry.
    
    rng = np.random.default_rng(12345)
    
    # Base sequence: 50 zeros and 50 ones? No, sum must be N/2 = 50.
    # If we use 0s and 1s, sum is count of 1s. So 50 ones, 50 zeros.
    # This satisfies 0<=h<=1 and sum=50.
    # This is a valid candidate! The "Erdos" problem often relates to binary sequences.
    # Let's initialize with a random binary sequence with exactly 50 ones.
    
    def random_binary_sequence():
        idx = rng.permutation(N)
        h = np.zeros(N)
        h[idx[:N//2]] = 1.0
        return h

    h_curr = random_binary_sequence()
    best_h = h_curr.copy()
    best_score = max_overlap(best_h)
    
    # We can also try the uniform solution as a baseline, though binary is likely better
    # for minimizing max overlap (variance reduction).
    h_uniform = np.full(N, 0.5)
    uniform_score = max_overlap(h_uniform)
    if uniform_score < best_score:
        best_score = uniform_score
        best_h = h_uniform.copy()
        h_curr = best_h.copy() # Start SA from best so far

    # Simulated Annealing Parameters
    T_init = 0.1
    T_final = 0.001
    n_iterations = 20000 # Adjust based on speed
    
    # Pre-compute indices for faster swap generation? 
    # For N=100, random selection is cheap.
    
    for it in range(n_iterations):
        if time.time() > deadline - 5.0:
            break
        
        # Temperature schedule (exponential decay)
        progress = it / n_iterations
        T = T_init * (T_final / T_init) ** progress
        
        # Generate Move
        # Move Type 1: Swap two random indices (preserves sum exactly)
        # Move Type 2: Flip a value (h_i -> 1-h_i) and adjust another (h_j -> h_j + 2*h_i - 1)? 
        # Actually, swapping is the most robust sum-preserving move.
        # Let's stick to swaps for simplicity and robustness.
        
        i, j = rng.integers(0, N, size=2)
        if i == j:
            continue
            
        h_prop = h_curr.copy()
        h_prop[i], h_prop[j] = h_curr[j], h_curr[i]
        
        # Evaluate
        score_curr = max_overlap(h_curr)
        score_prop = max_overlap(h_prop)
        
        delta = score_prop - score_curr
        
        # Acceptance
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            h_curr = h_prop
            curr_score = score_prop
        else:
            curr_score = score_curr
            
        # Track Best
        if curr_score < best_score:
            best_score = curr_score
            best_h = h_curr.copy()
            
        # Periodic check for time
        if it % 1000 == 0:
            if time.time() > deadline - 10.0:
                break

    # --- Refinement Machinery (from Inspiration 2) ---
    # After the discrete search, we might be stuck in a local optimum of the binary landscape.
    # We can try to "relax" the solution slightly into the continuous domain to see if
    # a small perturbation improves the max overlap, then project back.
    # However, the cell finding said "Binary swap SA ... targets combinatorial nature ... avoiding plateau issues".
    # This suggests the binary landscape is the key. 
    # But let's add a final local search step that tries small continuous perturbations
    # around the best discrete solution, projecting back to feasibility.
    
    if time.time() < deadline - 5.0:
        h_refine = best_h.copy()
        # Try a few gradient steps on the smooth max objective?
        # Or just random local perturbations?
        # Let's do a few random local perturbations (change one value slightly) and project.
        
        for _ in range(500):
            if time.time() > deadline - 2.0:
                break
            i = rng.integers(0, N)
            h_prop = h_refine.copy()
            h_prop[i] += rng.uniform(-0.1, 0.1)
            h_prop = project(h_prop)
            
            score_prop = max_overlap(h_prop)
            if score_prop < best_score:
                best_score = score_prop
                best_h = h_prop.copy()
                h_refine = best_h.copy()

    # Final Validation and Return
    best_h = project(best_h)
    
    # Ensure strict constraints
    assert np.all(best_h >= 0) and np.all(best_h <= 1), "h out of bounds"
    assert abs(np.sum(best_h) - TARGET_SUM) < 1e-9, f"Sum constraint violated: {np.sum(best_h)}"
    
    return best_h
