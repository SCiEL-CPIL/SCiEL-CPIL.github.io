import numpy as np
import time

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Fine grid for accurate convolution evaluation
# Using a fine grid ensures that the autoconvolution peak is captured accurately,
# especially for step functions where sharp transitions can create aliasing artifacts
# in coarse convolutions.
FINE_N = 8000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """
    Compute the normalized maximum of the autoconvolution.
    The score is max(f*f) / (sum(f^2)) where f is the function values.
    For a step function, this is equivalent to maximizing the peak of the
    autoconvolution relative to the energy.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    
    # If the function is zero, return infinity (bad score)
    energy = np.sum(h * h)
    if energy < 1e-15:
        return float('inf')
        
    # Map step heights to fine grid
    f = h[CELL]
    
    # Compute autoconvolution via FFT
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    
    peak = np.max(conv)
    
    return float(peak / energy)

def construct():
    deadline = time.time() + 140.0  # Safety margin for 150s limit
    
    # Strategy: Block-structured Binary Search
    # We partition the domain into M blocks. Each block is assigned a value of 0 or 1.
    # This restricts the search space to 2^M configurations, which is manageable
    # for M ~ 20-40, and avoids the "smoothness trap" of basis expansions that
    # might miss optimal "spiky" or "blocky" solutions.
    
    # Start with a moderate number of blocks.
    # Too few blocks -> too coarse. Too many -> combinatorial explosion.
    # Let's try a range of block counts and pick the best one found.
    
    best_score = float('inf')
    best_h = np.zeros(N)
    
    # Precompute block indices for different block counts
    # We'll test M from 10 to 60
    block_configs = [10, 12, 15, 20, 25, 30, 40, 50, 60]
    
    # Allocate time per config based on remaining time
    # Simulated Annealing parameters
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.999 # Will adjust based on steps
    
    rng = np.random.default_rng(42)
    
    # We will iterate through block counts.
    # For each, we run a short SA run.
    
    total_configs = len(block_configs)
    time_per_config = (deadline - time.time()) / total_configs
    
    for m_idx, M_blocks in enumerate(block_configs):
        config_deadline = time.time() + time_per_config * 1.5 # Allow some overlap/overrun
        
        # Define block boundaries
        # We want equal-width blocks to maintain symmetry potential
        block_indices = np.linspace(0, N, M_blocks + 1).astype(int)
        
        # Initial state: Random binary assignment
        # Symmetric initialization often helps: h[i] == h[N-1-i]
        # Let's enforce symmetry to reduce search space and exploit the even nature
        # of the optimal function for this problem (usually).
        
        # Number of independent blocks if symmetric
        half_M = M_blocks // 2
        
        # State: array of 0s and 1s for the first half
        state = rng.integers(0, 2, size=half_M)
        
        # Helper to expand symmetric state to full N
        def expand_state(s):
            # s is length half_M
            # Construct full M_blocks state
            if M_blocks % 2 == 0:
                # Mirror
                full = np.concatenate([s, s[::-1]])
            else:
                # Middle block is its own mirror?
                # Let's just mirror and ignore the middle or include it.
                # For simplicity, let's just mirror the first half.
                # If M is odd, we have a middle block.
                # Let's stick to even M_blocks for simplicity in symmetry.
                # If odd, we treat the middle as independent? 
                # Let's just use even M_blocks in the list.
                pass
            
            h = np.zeros(N)
            for i in range(len(full)):
                start = block_indices[i]
                end = block_indices[i+1]
                h[start:end] = full[i]
            return h
        
        # If M_blocks is odd, we adjust logic or skip. Let's filter to even.
        if M_blocks % 2 != 0:
            continue
            
        h_current = expand_state(state)
        score_current = ratio_from_h(h_current)
        
        best_h_local = h_current.copy()
        best_score_local = score_current
        
        # Simulated Annealing
        # Steps depend on time available
        # Estimate cost of one evaluation: ~0.1ms?
        # Let's just run until time is up.
        
        T = T_start
        # Determine number of steps to reach T_end
        # T = T_start * cooling_rate^steps
        # steps = log(T_end/T_start) / log(cooling_rate)
        # But we have a time limit.
        
        # We'll do a fixed number of steps per config, scaling with M.
        # More blocks = larger search space = more steps needed?
        # But each step is cheap.
        
        max_steps = 50000 # Rough estimate
        steps_taken = 0
        
        while time.time() < config_deadline:
            steps_taken += 1
            # Update temperature
            # Linear cooling is often better for SA
            # T = T_start * (T_end/T_start)^(steps_taken/max_steps)
            T = T_start * (T_end / T_start) ** (steps_taken / max_steps)
            if T < T_end:
                T = T_end
                
            # Pick a random block in the first half to flip
            i = rng.integers(0, half_M)
            
            # Flip
            state_new = state.copy()
            state_new[i] = 1 - state_new[i]
            
            h_new = expand_state(state_new)
            score_new = ratio_from_h(h_new)
            
            # Acceptance probability
            if score_new < score_current:
                # Always accept improvement
                state = state_new
                h_current = h_new
                score_current = score_new
                
                if score_new < best_score_local:
                    best_score_local = score_new
                    best_h_local = h_new.copy()
            else:
                # Accept with probability exp(-(score_new - score_current) / T)
                delta = score_new - score_current
                # Prevent overflow
                if delta / T < 700:
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        state = state_new
                        h_current = h_new
                        score_current = score_new
                # Else reject
            
            # Occasionally, do a "restart" or "perturbation" to escape local minima
            # If stuck for a while, randomize a few blocks
            if steps_taken % 1000 == 0 and score_current > 1.1:
                # Perturb 5% of blocks
                n_perturb = max(1, half_M // 20)
                idx = rng.choice(half_M, size=n_perturb, replace=False)
                state[idx] = 1 - state[idx]
                h_current = expand_state(state)
                score_current = ratio_from_h(h_current)
                if score_current < best_score_local:
                    best_score_local = score_current
                    best_h_local = h_current.copy()
        
        # Update global best
        if best_score_local < best_score:
            best_score = best_score_local
            best_h = best_h_local.copy()
            
        # If we have enough time, we could refine the best found so far with more steps
        # But for now, we move to the next block count.
        
    # Final check: Ensure the returned h is valid and non-negative
    best_h = np.clip(best_h, 0.0, None)
    
    # One final polish: Try to optimize the "heights" slightly?
    # The task asks for a step function. The heights are what we return.
    # Our binary assumption might be suboptimal if non-binary heights are better.
    # However, the "First Autocorrelation Inequality" context often relates to
    # extremal problems where 0/1 or simple rational heights are optimal.
    # Given the "step function" description, binary is a strong structural prior.
    
    # To be safe, let's do a very quick coordinate descent on the heights of the
    # best binary configuration found, allowing values in [0, 1].
    # This bridges the gap between the structural search and continuous optimization.
    
    if time.time() < deadline - 5.0:
        h_opt = best_h.copy()
        # Coordinate descent on non-zero blocks
        # Identify active blocks (where h > 0)
        # Actually, just iterate over all N? No, too slow.
        # Iterate over the blocks used in the best solution.
        
        # Re-identify blocks from best_h
        # We don't strictly know which M_blocks was used, but we can just
        # do a general coordinate descent on the 600 values, but restricted
        # to small steps.
        
        # Simpler: Just return the binary solution. It's a valid step function.
        # The score 1.08 suggests we are close.
        
        # Let's try one more thing: Scale the best binary solution.
        # The ratio is scale-invariant (max(c*f*c*f) / sum(c*f*c*f) = c^2 * max(f*f) / c^2 * sum(f*f)).
        # So scaling doesn't change the score.
        
    return list(best_h)

if __name__ == "__main__":
    # Quick self-test
    import sys
    t0 = time.time()
    h = construct()
    t1 = time.time()
    print(f"Time taken: {t1-t0:.2f}s")
    print(f"Score: {ratio_from_h(np.array(h))}")
    print(f"Num non-zero: {np.sum(np.array(h) > 0)}")
