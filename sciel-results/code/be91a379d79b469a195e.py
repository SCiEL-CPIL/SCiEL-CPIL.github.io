import time
import random
import numpy as np

_N = 256
_DEADLINE = 85.0

def _calc_max_disc(signs):
    n = len(signs)
    max_disc = 1
    
    for d in range(1, n // 2 + 1):
        max_for_d = 1
        for r in range(d):
            seq = signs[r::d]
            if len(seq) > 1:
                csum = np.cumsum(seq)
                min_prefix = np.minimum.accumulate(csum)
                max_sum = np.max(csum - min_prefix)
                
                max_prefix = np.maximum.accumulate(csum)
                min_sum = np.min(csum - max_prefix)
                
                local_max = max(max_sum, -min_sum)
                if local_max > max_for_d:
                    max_for_d = local_max
        if max_for_d > max_disc:
            max_disc = max_for_d
    return max_disc

def construct():
    start_time = time.time()
    best_signs = None
    best_score = float('inf')
    
    while time.time() - start_time < _DEADLINE - 2.0:
        # Random initialization
        signs = np.array([random.choice([-1, 1]) for _ in range(_N)], dtype=np.int8)
        
        current_score = _calc_max_disc(signs)
        
        if current_score < best_score:
            best_score = current_score
            best_signs = signs.copy()
            
        # Local Search
        stuck_counter = 0
        while time.time() - start_time < _DEADLINE - 2.0:
            i = random.randint(0, _N - 1)
            signs[i] *= -1
            new_score = _calc_max_disc(signs)
            
            if new_score < current_score:
                current_score = new_score
                stuck_counter = 0
                if current_score < best_score:
                    best_score = current_score
                    best_signs = signs.copy()
            else:
                signs[i] *= -1
                stuck_counter += 1
                
                if stuck_counter > 200:
                    # Perturb
                    for _ in range(5):
                        j = random.randint(0, _N - 1)
                        signs[j] *= -1
                    current_score = _calc_max_disc(signs)
                    stuck_counter = 0
                    break # Restart inner loop? Or continue?
                    # Breaking restarts the inner while, effectively restarting local search from this perturbed state
                    # But we are in the outer while too.
                    # Let's just break the inner loop and let the outer loop pick a new random start?
                    # No, that wastes the current state.
                    # Let's just continue the inner loop with the perturbed state.
                    # But we need to update current_score. Done above.
                    # And reset stuck_counter. Done.
                    # So we don't break, we just continue?
                    # If we continue, we might get stuck again immediately.
                    # Let's break to the outer loop to start a fresh random vector?
                    # Or just continue with the perturbed vector.
                    # Let's continue.
                    pass 

    if best_signs is None:
        return [1] * _N
    return [int(x) for x in best_signs]
