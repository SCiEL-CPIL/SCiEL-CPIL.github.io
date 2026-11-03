"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _score_sequence(x):
    """
    Compute the score for a signed sequence x (values in {-1, 1}).
    Score is max(|autoconv|) / (sum(x))^2.
    Lower is better.
    """
    # Autoconvolution (full)
    ac = np.convolve(x, x, mode='full')
    # Integral (sum of x)
    integral = np.sum(x)
    if integral == 0:
        return float('inf')
    # Maximum absolute value of autoconvolution
    max_ac = np.max(np.abs(ac))
    return max_ac / (integral ** 2)

def _construct_from_theta(theta):
    """
    Construct a signed sequence from continuous theta values.
    x_i = sign(theta_i), with 0 mapped to 1.
    """
    x = np.sign(theta)
    x[x == 0] = 1
    return x

def _objective(theta):
    """
    Objective function for optimization.
    We want to minimize max(|autoconv|) / (sum(x))^2.
    """
    x = _construct_from_theta(theta)
    return _score_sequence(x)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin: 110s out of 120s limit
    
    # Initial guess: random signs
    rng = np.random.default_rng(42)
    
    # Try a few different strategies within the time budget
    
    best_x = None
    best_score = float('inf')
    
    # Strategy 1: Start with a simple pattern and use local search
    # Let's try a sequence that alternates or has some structure
    
    # First, try a few heuristic initializations
    # 1. Alternating
    x_alt = np.array([1 if i % 2 == 0 else -1 for i in range(N)])
    s = _score_sequence(x_alt)
    if s < best_score:
        best_score = s
        best_x = x_alt.copy()
    
    # 2. Blocks of +1 and -1
    x_blocks = np.ones(N)
    x_blocks[N//4:3*N//4] = -1
    s = _score_sequence(x_blocks)
    if s < best_score:
        best_score = s
        best_x = x_blocks.copy()
    
    # 3. Random
    x_rand = rng.choice([-1, 1], size=N)
    s = _score_sequence(x_rand)
    if s < best_score:
        best_score = s
        best_x = x_rand.copy()
    
    # 4. Zadoff-Chu like (for N=400, not prime, but let's try a chirp)
    # Zadoff-Chu: x_n = exp(-j * pi * q * n^2 / N)
    # Real part: cos(-pi * q * n^2 / N)
    # Let's try q=1
    n = np.arange(N)
    x_zc = np.sign(np.cos(-np.pi * 1 * n**2 / N))
    x_zc[x_zc == 0] = 1
    s = _score_sequence(x_zc)
    if s < best_score:
        best_score = s
        best_x = x_zc.copy()
    
    # Now do local search (flip individual bits)
    # This is a simple hill climbing
    x = best_x.copy()
    current_score = best_score
    
    # Precompute for faster updates
    # When we flip x[i], the autoconvolution changes in a predictable way
    # But for simplicity, let's just recompute if N is small enough
    # N=400, convolve is O(N^2) = 160000, which is fast
    
    # Local search: try flipping each bit, keep the best
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for i in range(N):
            if (time.time() - start_time) > deadline:
                break
            # Flip bit i
            old_val = x[i]
            x[i] = -old_val
            new_score = _score_sequence(x)
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
            else:
                x[i] = old_val
    
    if current_score < best_score:
        best_score = current_score
        best_x = x.copy()
    
    # Try a second round of local search from a different starting point
    if (time.time() - start_time) < deadline:
        # Start from a different random point
        x2 = rng.choice([-1, 1], size=N)
        # Maybe add some structure: first half +1, second half -1, then randomize
        x2 = np.ones(N)
        x2[N//2:] = -1
        # Randomly flip some
        flip_idx = rng.choice(N, size=N//4, replace=False)
        x2[flip_idx] *= -1
        
        current_score2 = _score_sequence(x2)
        if current_score2 < best_score:
            best_score = current_score2
            best_x = x2.copy()
        
        # Local search on this
        x = x2.copy()
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            for i in range(N):
                if (time.time() - start_time) > deadline:
                    break
                old_val = x[i]
                x[i] = -old_val
                new_score = _score_sequence(x)
                if new_score < current_score2 - 1e-12:
                    current_score2 = new_score
                    improved = True
                else:
                    x[i] = old_val
        
        if current_score2 < best_score:
            best_score = current_score2
            best_x = x.copy()
    
    # If we still have time, try a few more random restarts
    if (time.time() - start_time) < deadline:
        for _ in range(10):
            if (time.time() - start_time) > deadline:
                break
            
            # Random start with some bias
            x3 = rng.choice([-1, 1], size=N)
            # Maybe try to make the sum large (to reduce denominator)
            # But the score is max_ac / sum^2, so we want large sum and small max_ac
            # Large sum means mostly +1 or mostly -1
            # But then max_ac might be large
            
            # Let's try: mostly +1, with some -1 scattered
            x3 = np.ones(N)
            num_neg = rng.integers(0, N//2)
            neg_idx = rng.choice(N, size=num_neg, replace=False)
            x3[neg_idx] = -1
            
            current_score3 = _score_sequence(x3)
            if current_score3 < best_score:
                best_score = current_score3
                best_x = x3.copy()
            
            # Local search
            x = x3.copy()
            improved = True
            while improved and (time.time() - start_time) < deadline:
                improved = False
                for i in range(N):
                    if (time.time() - start_time) > deadline:
                        break
                    old_val = x[i]
                    x[i] = -old_val
                    new_score = _score_sequence(x)
                    if new_score < current_score3 - 1e-12:
                        current_score3 = new_score
                        improved = True
                    else:
                        x[i] = old_val
            
            if current_score3 < best_score:
                best_score = current_score3
                best_x = x.copy()
    
    return best_x

# EVOLVE-BLOCK-END
