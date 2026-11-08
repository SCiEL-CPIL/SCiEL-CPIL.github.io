"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_psl(h):
    """Compute Peak Sidelobe Level: max(|R[k]|) for k != 0."""
    # h is a 1D array of length N
    # Autocorrelation R[k] = sum(h[j] * h[j+k])
    # np.correlate(h, h, mode='full') returns length 2N-1
    # The zero lag is at index N-1
    ac = np.correlate(h, h, mode='full')
    # Exclude the zero lag (index N-1)
    # Sidelobes are indices 0 to N-2 and N to 2N-2
    # Due to symmetry, we can just check one side, but abs handles it
    # ac[N-1] is the peak (N)
    sidelobes = np.concatenate([ac[:N-1], ac[N:]])
    return np.max(np.abs(sidelobes))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initialize with random signs
    # Random initialization is often good for starting local search
    h = np.random.choice([-1, 1], size=N).astype(np.float64)
    
    current_psl = _compute_psl(h)
    best_psl = current_psl
    best_h = h.copy()
    
    # Hill Climbing / Simulated Annealing
    # We will use a simple hill climbing with random restarts if stuck
    # Or just pure hill climbing for a while.
    # Given the discrete nature, pure hill climbing might get stuck.
    # Let's try a few random restarts.
    
    max_restarts = 5
    steps_per_restart = 100000  # Adjust based on speed
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        if r > 0:
            # Restart with new random sequence
            h = np.random.choice([-1, 1], size=N).astype(np.float64)
            current_psl = _compute_psl(h)
        
        stuck_count = 0
        max_stuck = 1000  # If no improvement for 1000 steps, break restart loop or restart
        
        while time.time() < deadline and stuck_count < max_stuck:
            # Pick a random index to flip
            i = np.random.randint(N)
            
            # Flip
            old_val = h[i]
            h[i] = -old_val
            
            # Compute new PSL
            new_psl = _compute_psl(h)
            
            if new_psl < current_psl:
                current_psl = new_psl
                stuck_count = 0
                if new_psl < best_psl:
                    best_psl = new_psl
                    best_h = h.copy()
            else:
                # Revert
                h[i] = old_val
                stuck_count += 1
                
    return best_h
# EVOLVE-BLOCK-END
