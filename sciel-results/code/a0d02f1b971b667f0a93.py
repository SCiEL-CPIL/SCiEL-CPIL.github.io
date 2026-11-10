"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _compute_max_discrepancy(signs):
    """Compute maximum absolute signed sum over all arithmetic progressions in [1, n]."""
    n = len(signs)
    arr = np.array(signs, dtype=np.int64)
    max_disc = 1  # Singletons have abs sum 1
    
    for step in range(1, n + 1):
        for r in range(step):
            if r >= n:
                break
            subseq = arr[r::step]
            if len(subseq) == 0:
                continue
            prefix_sums = np.cumsum(subseq)
            current_max = int(np.max(np.abs(prefix_sums)))
            if current_max > max_disc:
                max_disc = current_max
                if max_disc >= n:  # Can't get worse than n
                    return max_disc
    
    return max_disc

def _get_tight_ap_positions(signs, max_disc):
    """Find positions that appear in APs achieving the maximum discrepancy."""
    n = len(signs)
    arr = np.array(signs, dtype=np.int64)
    tight_positions = set()
    
    for step in range(1, n + 1):
        for r in range(step):
            if r >= n:
                break
            subseq = arr[r::step]
            if len(subseq) == 0:
                continue
            prefix_sums = np.cumsum(subseq)
            abs_prefix = np.abs(prefix_sums)
            # Find if this AP achieves max_disc
            if np.max(abs_prefix) >= max_disc:
                # All positions in this AP are "tight"
                for i in range(len(subseq)):
                    pos = r + i * step
                    if pos < n:
                        tight_positions.add(pos)
    
    return tight_positions

def _local_search_guided(signs, max_iterations=3000, time_limit=55.0):
    """Perform guided local search to minimize maximum absolute signed sum."""
    n = len(signs)
    signs = list(signs)
    current_disc = _compute_max_discrepancy(signs)
    best_disc = current_disc
    best_signs = signs.copy()
    
    start_time = time.time()
    rng = random.Random(42)
    
    no_improve_count = 0
    max_no_improve = 50
    
    for i in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Get tight positions (APs achieving current max)
        if current_disc > 1:
            tight_positions = _get_tight_ap_positions(signs, current_disc)
        else:
            tight_positions = set()
        
        # Decide which position to flip
        if tight_positions and len(tight_positions) > 0:
            # 70% chance to pick from tight positions, 30% random
            if rng.random() < 0.7:
                idx = rng.choice(list(tight_positions))
            else:
                idx = rng.randint(0, n - 1)
        else:
            idx = rng.randint(0, n - 1)
        
        old_sign = signs[idx]
        signs[idx] = -old_sign
        
        new_disc = _compute_max_discrepancy(signs)
        
        if new_disc < best_disc:
            best_disc = new_disc
            best_signs = signs.copy()
            current_disc = new_disc
            no_improve_count = 0
        elif new_disc == best_disc:
            # Accept equal to explore
            current_disc = new_disc
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Revert and try a different strategy
                signs[idx] = old_sign
                no_improve_count = 0
                # Force a random restart of the local search state
                for _ in range(5):
                    random_idx = rng.randint(0, n - 1)
                    signs[random_idx] = -signs[random_idx]
                current_disc = _compute_max_discrepancy(signs)
        else:
            # Revert
            signs[idx] = old_sign
            no_improve_count += 1
    
    return best_signs, best_disc

def construct():
    n = 256
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    best_signs = None
    best_disc = float('inf')
    
    # Multiple restarts with different seeds
    rng = random.Random(12345)
    
    for restart in range(50):
        if time.time() - start_time > deadline:
            break
        
        # Create initial random signs
        signs = [rng.choice((-1, 1)) for _ in range(n)]
        
        # Run guided local search with time limit
        remaining = deadline - (time.time() - start_time)
        if remaining < 2.0:
            break
        
        # Give more time to early restarts, less to later ones
        if restart < 3:
            time_limit = min(remaining * 0.3, 15.0)
        elif restart < 10:
            time_limit = min(remaining * 0.15, 8.0)
        else:
            time_limit = min(remaining * 0.08, 4.0)
        
        new_signs, new_disc = _local_search_guided(signs, max_iterations=5000, time_limit=time_limit)
        
        if new_disc < best_disc:
            best_disc = new_disc
            best_signs = new_signs
            # If we've found a good solution, we can be more aggressive
            if best_disc <= 4:
                break
    
    if best_signs is None:
        # Fallback: return random signs
        return [1] * n
    
    return best_signs
# EVOLVE-BLOCK-END
