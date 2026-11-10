import time
import random
import math

_N = 256

def _legendre_seq(p):
    """Generate Legendre symbol sequence for 1..p-1"""
    seq = []
    for i in range(1, p):
        # Euler's criterion
        if pow(i, (p-1)//2, p) == 1:
            seq.append(1)
        else:
            seq.append(-1)
    return seq

def _max_abs_ap_sum(signs):
    """
    Compute max absolute signed sum over any AP in [0, n-1].
    Pure Python implementation for speed on small n.
    """
    n = len(signs)
    max_disc = 1
    # d=1 is handled by the general loop, but we can check it separately?
    # General loop:
    for d in range(1, n):
        for r in range(d):
            s = 0
            # Iterate through the AP: r, r+d, r+2d...
            for idx in range(r, n, d):
                s += signs[idx]
                abs_s = s if s > 0 else -s
                if abs_s > max_disc:
                    max_disc = abs_s
    return max_disc

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s to be safe
    
    # 1. Initialization
    # Use Legendre symbol mod 257
    init_seq = _legendre_seq(257)
    
    rng = random.Random(42)
    
    current = init_seq[:]
    current_D = _max_abs_ap_sum(current)
    
    best = current[:]
    best_D = current_D
    
    # Simulated Annealing parameters
    T = 10.0
    cooling = 0.999
    min_T = 0.1
    
    # To improve search, we can track "hot" indices
    # But simple random flip with SA is robust.
    
    while time.time() < deadline:
        # Pick a random index to flip
        i = rng.randint(0, _N - 1)
        
        # Flip
        current[i] *= -1
        
        # Evaluate
        new_D = _max_abs_ap_sum(current)
        
        # Acceptance criterion
        delta = new_D - current_D
        if delta < 0:
            # Improvement
            current_D = new_D
            if new_D < best_D:
                best = current[:]
                best_D = new_D
        else:
            # Deterioration or equal
            # Accept with probability exp(-delta / T)
            if delta == 0:
                # Always accept equal to explore
                current_D = new_D
                if new_D < best_D:
                    best = current[:]
                    best_D = new_D
            else:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    current_D = new_D
                    if new_D < best_D:
                        best = current[:]
                        best_D = new_D
                else:
                    # Reject: revert
                    current[i] *= -1
        
        # Cool down
        if T > min_T:
            T *= cooling
            if T < min_T:
                T = min_T
                # Optional: restart or keep going
                # If stuck, we might want to restart with a new random init
                # But Legendre is a good base.
        
        # Check time frequently
        if time.time() > deadline:
            break
            
    return best
