"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))

def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _run_eca(rule, init_state, length):
    """
    Run a 1D elementary cellular automaton for 'length' steps.
    Returns the trajectory as a 1D array of 0s and 1s.
    """
    # Map rule (0-255) to transition table
    # bit 7 corresponds to 111, bit 0 to 000
    trans = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.int8)
    
    # State is a 1D array of 0s and 1s
    state = np.zeros(length, dtype=np.int8)
    state[:] = init_state
    
    trajectory = np.zeros(length * length, dtype=np.int8)
    
    for t in range(length):
        trajectory[t * length : (t + 1) * length] = state
        # Update state
        # Pad with 0s (boundary condition)
        left = np.roll(state, 1)
        right = np.roll(state, -1)
        # Indices for transition table: 4*left + 2*state + right
        idx = 4 * left + 2 * state + right
        state = trans[idx]
        
    return trajectory[:length * length]

def _generate_eca_signals(rng, deadline):
    """
    Generate a set of candidate signals from ECA trajectories.
    """
    candidates = []
    # Precompute all 256 rules
    rules = list(range(256))
    rng.shuffle(rules)
    
    # We only need to run enough steps to get N samples.
    # Actually, ECA trajectory is a time series. 
    # We can take a window of length N from the time series.
    # Or we can run it for N steps and take the final state? 
    # The task says "sequence of signed step heights".
    # A "trajectory" usually implies the sequence of states over time.
    # Let's interpret the sequence x[0..N-1] as a slice of the space-time diagram or just the time evolution of a single cell?
    # "1D binary cellular automaton trajectory" -> likely the state of the system over time.
    # If we take the state of the whole system at time t, that's N bits.
    # But we need a sequence of N heights.
    # Let's assume the sequence x is the time evolution of a single cell, or a slice through the space-time diagram.
    # Given N=400, a single cell's history is 1D.
    # Let's try: Run ECA for N steps. Take the history of cell index k.
    # Or better: Take a random window of length N from the flattened space-time diagram.
    
    # For efficiency, we will generate full space-time diagrams for a few steps (e.g., N steps)
    # and extract random length-N segments.
    
    steps_needed = N
    width = N
    
    for rule in rules:
        if time.time() > deadline:
            break
            
        # Try a few random initial states for this rule
        for _ in range(5):
            if time.time() > deadline:
                break
            init_state = rng.integers(0, 2, size=width).astype(np.int8)
            
            # Generate space-time diagram
            # This is O(N^2) per rule/init, which is 160k ops. 256*5 = 1280 diagrams -> 200M ops. Might be slow in pure Python loops.
            # Let's optimize the ECA run using numpy vectorization for the whole grid at once?
            # No, ECA is sequential in time.
            # We can use a fast C-based ECA if available, but standard library only.
            # Let's limit the number of rules we test fully if it's too slow.
            
            # Actually, let's just run it. 400 steps * 400 width = 160k updates.
            # 1280 * 160k = 204.8 million simple integer ops. In Python loop, this is ~20-40s.
            # We have 145s. This is feasible if we are careful.
            
            # Optimization: Use np.roll and vectorized table lookup.
            state = init_state.copy()
            # We store the history of a specific cell or a slice?
            # Let's store the entire space-time diagram in a (N, N) array? No, memory is fine (160k bytes).
            # But we just need a 1D sequence of length N.
            # Let's pick a random cell index c and track its history.
            # Or pick a random row t and take the state at that time?
            # The "trajectory" of a CA is often the sequence of states.
            # Let's assume the sequence x[0..N-1] corresponds to the values of a single cell c over time t=0..N-1.
            
            c = rng.integers(0, width)
            hist = np.zeros(N, dtype=np.int8)
            hist[0] = state[c]
            
            trans = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.int8)
            
            for t in range(1, N):
                left = np.roll(state, 1)
                right = np.roll(state, -1)
                idx = 4 * left + 2 * state + right
                state = trans[idx]
                hist[t] = state[c]
            
            # Convert 0/1 to -1/1
            x = 2.0 * hist - 1.0
            
            # Check metric
            val = _metric(x)
            candidates.append((val, x))
            
    return candidates

def _local_search_sign_flip(x, deadline, rng, max_iters=50000):
    """
    Local search by flipping signs of individual elements.
    This preserves the 'chaotic' structure while allowing fine-tuning.
    """
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    
    # Precompute FFT of x for faster updates? 
    # Flipping sign of x[i] changes X[k] by -2*x[i]*exp(-2*pi*i*k/N).
    # This allows O(N) update of spectrum, then O(N log N) for conv?
    # Actually, full recalculation is O(N log N). 
    # N=400. N log N ~ 3200 ops. 
    # 50000 iters * 3200 = 160M ops. Feasible in ~10-20s with NumPy.
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        if no_improve > 200:
            break
            
        # Pick random index
        idx = rng.integers(0, N)
        
        # Flip sign
        x_test = best_x.copy()
        x_test[idx] *= -1
        
        val_test = _metric(x_test)
        
        if val_test < best_val:
            best_val = val_test
            best_x = x_test
            no_improve = 0
        else:
            no_improve += 1
            
    return best_x, best_val

def _construct():
    deadline = time.time() + 145.0
    
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Generate candidates from ECA
    # We don't know how many we can generate in time, so we run until deadline - 30s
    gen_deadline = time.time() + 110.0
    
    candidates = _generate_eca_signals(rng, gen_deadline)
    
    # Sort by metric
    candidates.sort(key=lambda item: item[0])
    
    # Phase 2: Refine top candidates
    # Take top 10 candidates (or as many as we have)
    top_k = min(10, len(candidates))
    
    refine_deadline = deadline - 5.0
    time_per_refine = 10.0
    
    for i in range(top_k):
        if time.time() > refine_deadline:
            break
            
        val, x = candidates[i]
        if val >= best_val:
            # If the initial ECA signal is already worse than our best (which might be from a previous run or better ECA), skip?
            # No, local search might improve it significantly.
            pass
            
        x_opt, val_opt = _local_search_sign_flip(x, refine_deadline - 5.0, rng, max_iters=10000)
        
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt
            
    # Phase 3: If we have time, try a few random restarts of local search on the best found
    if time.time() < deadline - 10:
        x_final, val_final = _local_search_sign_flip(best_x, deadline - 2.0, rng, max_iters=20000)
        if val_final < best_val:
            best_val = val_final
            best_x = x_final

    return best_x

def construct():
    return _construct()
# EVOLVE-BLOCK-END
