"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from collections import defaultdict

N = 5000
DEADLINE = 290.0  # Safety margin
START_TIME = time.time()

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N + 1)

def construct():
    global START_TIME
    if time.time() - START_TIME > DEADLINE:
        return list(range(N + 1))
    
    best_size = 0
    best_set = []
    
    # Strategies for selecting which residue class to remove
    # 'min_idx': min index among min-count candidates
    # 'random': random among min-count candidates
    # 'min_var': min variance among min-count candidates
    # 'max_var': max variance among min-count candidates
    # 'min_range': min range among min-count candidates
    # 'max_range': max range among min-count candidates
    # 'min_spread': min max-min range
    # 'max_spread': max max-min range
    
    strategies = ['min_idx', 'random', 'min_var', 'max_var', 'min_range', 'max_range', 'min_spread', 'max_spread']
    
    runs = 0
    max_runs = 5000
    
    while time.time() - START_TIME < DEADLINE and runs < max_runs:
        for strat in strategies:
            if time.time() - START_TIME > DEADLINE:
                break
            
            mask = np.ones(N + 1, dtype=bool)
            current_size = N + 1
            indices = np.arange(N + 1)
            
            for p in PRIMES:
                if p > current_size:
                    break
                
                active_vals = indices[mask]
                if len(active_vals) == 0:
                    break
                
                residues = active_vals % p
                counts = np.bincount(residues, minlength=p)
                
                if counts.min() > 0:
                    min_count = counts.min()
                    candidates = np.where(counts == min_count)[0]
                    
                    if strat == 'min_idx':
                        chosen = candidates[0]
                    elif strat == 'random':
                        chosen = np.random.choice(candidates)
                    elif strat in ['min_var', 'max_var']:
                        # Vectorized variance calculation
                        sum_vals = np.bincount(residues, weights=active_vals, minlength=p)
                        sum_sq_vals = np.bincount(residues, weights=active_vals**2, minlength=p)
                        
                        with np.errstate(divide='ignore', invalid='ignore'):
                            means = np.where(counts > 0, sum_vals / counts, 0)
                            variances = np.where(counts > 0, (sum_sq_vals / counts) - means**2, 0)
                        
                        cand_variances = variances[candidates]
                        
                        if strat == 'min_var':
                            best_idx = np.argmin(cand_variances)
                        else:  # max_var
                            best_idx = np.argmax(cand_variances)
                        
                        chosen = candidates[best_idx]
                        
                    elif strat in ['min_range', 'max_range', 'min_spread', 'max_spread']:
                        # Compute min and max for each residue
                        best_c = candidates[0]
                        best_range = float('inf') if strat in ['min_range', 'min_spread'] else -1
                        
                        for c in candidates:
                            vals = active_vals[residues == c]
                            if len(vals) > 0:
                                r = vals.max() - vals.min()
                                if strat in ['min_range', 'min_spread']:
                                    if r < best_range:
                                        best_range = r
                                        best_c = c
                                else:  # max_range, max_spread
                                    if r > best_range:
                                        best_range = r
                                        best_c = c
                        chosen = best_c
                    else:
                        chosen = candidates[0]
                        
                    # Remove
                    mask[indices % p == chosen] = False
                    current_size = mask.sum()
            
            if current_size > best_size:
                best_size = current_size
                best_set = indices[mask].tolist()
        
        runs += 1
    
    return best_set

# EVOLVE-BLOCK-END
