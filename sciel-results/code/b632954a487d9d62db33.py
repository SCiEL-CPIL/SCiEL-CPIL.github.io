import math
import time
import numpy as np

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# Pre-compute indices for each prime and residue
# indices[p][r] -> numpy array of x in [0, N] such that x % p == r
def _precompute_indices(primes):
    indices = {}
    for p in primes:
        idx = [np.array([], dtype=int) for _ in range(p)]
        for r in range(p):
            idx[r] = np.arange(r, N + 1, p)
        indices[p] = idx
    return indices

_primes_list = _primes(N)
_indices = _precompute_indices(_primes_list)

def _evaluate_config(config):
    """
    config: dict {p: r_p}
    returns: number of valid x
    """
    valid = np.ones(N + 1, dtype=bool)
    for p in _primes_list:
        r_p = config.get(p, 0)
        valid[_indices[p][r_p]] = False
    return np.sum(valid)

def _greedy_init():
    """Greedy initialization: pick residue class with minimum coverage among active elements."""
    active = np.ones(N + 1, dtype=bool)
    config = {}
    for p in _primes_list:
        # Count active elements in each residue class
        residue_counts = np.zeros(p, dtype=np.int64)
        for r in range(p):
            idx = _indices[p][r]
            residue_counts[r] = np.sum(active[idx])
        # Pick residue with minimum count (ties broken by smallest r)
        min_count = np.min(residue_counts)
        candidates = np.where(residue_counts == min_count)[0]
        chosen_r = int(candidates[0])
        config[p] = chosen_r
        # Update active: remove elements in chosen residue
        active[_indices[p][chosen_r]] = False
    return config, int(np.sum(active))

def _local_search(config, start_score, deadline, max_no_improvement=200):
    """
    Hill climbing local search: try changing one prime's residue at a time.
    Uses incremental evaluation for speed.
    """
    current = config.copy()
    current_score = start_score
    best = config.copy()
    best_score = start_score
    no_improvement = 0
    
    # Precompute for incremental evaluation
    # valid_mask: boolean array of which x are still valid
    valid_mask = np.ones(N + 1, dtype=bool)
    for p in _primes_list:
        valid_mask[_indices[p][current[p]]] = False
    current_score = int(np.sum(valid_mask))
    
    primes_arr = np.array(_primes_list)
    num_primes = len(_primes_list)
    
    while time.time() < deadline:
        # Pick a prime to modify - bias toward smaller primes for larger impact
        # Use a weighted selection: smaller primes get higher probability
        weights = 1.0 / primes_arr.astype(np.float64)
        weights /= weights.sum()
        p_idx = int(np.random.choice(num_primes, p=weights))
        p = primes_arr[p_idx]
        
        old_r = current[p]
        best_new_r = old_r
        best_new_score = current_score
        
        for new_r in range(p):
            if new_r == old_r:
                continue
            
            # Incremental: remove old residue's effect, add new residue's effect
            # Old: valid_mask has _indices[p][old_r] set to False
            # We need to: set those back to True, then set _indices[p][new_r] to False
            old_idx = _indices[p][old_r]
            new_idx = _indices[p][new_r]
            
            # Temporarily restore old residue positions
            valid_mask[old_idx] = True
            # Then remove new residue positions
            valid_mask[new_idx] = False
            
            new_score = int(np.sum(valid_mask))
            
            if new_score > best_new_score:
                best_new_score = new_score
                best_new_r = new_r
            
            # Restore for next iteration
            valid_mask[new_idx] = True
            valid_mask[old_idx] = False
        
        # Apply best change
        if best_new_r != old_r:
            current[p] = best_new_r
            valid_mask[_indices[p][old_r]] = True
            valid_mask[_indices[p][best_new_r]] = False
            current_score = best_new_score
        
        if current_score > best_score:
            best_score = current_score
            best = current.copy()
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement > max_no_improvement:
                # Try a random jump to escape local optimum
                # Pick 2-3 random primes and random residues
                num_jumps = np.random.randint(2, 5)
                for _ in range(num_jumps):
                    jp_idx = np.random.randint(num_primes)
                    jp = primes_arr[jp_idx]
                    old_jr = current[jp]
                    new_jr = np.random.randint(jp)
                    if new_jr != old_jr:
                        current[jp] = new_jr
                        valid_mask[_indices[jp][old_jr]] = True
                        valid_mask[_indices[jp][new_jr]] = False
                        current_score = int(np.sum(valid_mask))
                # Rebuild valid_mask from current to be safe
                valid_mask = np.ones(N + 1, dtype=bool)
                for pp in _primes_list:
                    valid_mask[_indices[pp][current[pp]]] = False
                current_score = int(np.sum(valid_mask))
                no_improvement = 0
                # Allow some more iterations after jump
                max_no_improvement = 100
                break
    
    return best, best_score

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    limit_time = start_time + 280.0  # Safety margin
    
    # Multi-start local search
    np.random.seed(42)
    
    best_config = None
    best_score = 0
    
    # Phase 1: Greedy init + local search
    config1, score1 = _greedy_init()
    config1, score1 = _local_search(config1, score1, limit_time, max_no_improvement=300)
    best_config = config1
    best_score = score1
    
    remaining = limit_time - time.time()
    if remaining < 30:
        # Not enough time for more, return best
        valid = np.ones(N + 1, dtype=bool)
        for p in _primes_list:
            valid[_indices[p][best_config[p]]] = False
        return [int(x) for x in np.where(valid)[0]]
    
    # Phase 2: Different greedy init strategies
    # Strategy 2: Greedy with random tie-breaking
    active = np.ones(N + 1, dtype=bool)
    config2 = {}
    for p in _primes_list:
        residue_counts = np.zeros(p, dtype=np.int64)
        for r in range(p):
            idx = _indices[p][r]
            residue_counts[r] = np.sum(active[idx])
        min_count = np.min(residue_counts)
        candidates = np.where(residue_counts == min_count)[0]
        # Random tie-breaking
        chosen_r = int(candidates[np.random.randint(len(candidates))])
        config2[p] = chosen_r
        active[_indices[p][chosen_r]] = False
    score2 = int(np.sum(active))
    
    config2, score2 = _local_search(config2, score2, limit_time, max_no_improvement=200)
    if score2 > best_score:
        best_score = score2
        best_config = config2
    
    remaining = limit_time - time.time()
    if remaining < 20:
        valid = np.ones(N + 1, dtype=bool)
        for p in _primes_list:
            valid[_indices[p][best_config[p]]] = False
        return [int(x) for x in np.where(valid)[0]]
    
    # Phase 3: Random restarts with short local search
    num_restarts = min(10, int(remaining / 15))
    for restart in range(num_restarts):
        if time.time() > limit_time:
            break
        
        # Random initial config
        config_r = {}
        for p in _primes_list:
            config_r[p] = np.random.randint(p)
        
        # Quick evaluation
        valid_r = np.ones(N + 1, dtype=bool)
        for p in _primes_list:
            valid_r[_indices[p][config_r[p]]] = False
        score_r = int(np.sum(valid_r))
        
        # Short local search
        config_r, score_r = _local_search(config_r, score_r, limit_time, max_no_improvement=50)
        
        if score_r > best_score:
            best_score = score_r
            best_config = config_r
    
    # Final: extract result
    valid = np.ones(N + 1, dtype=bool)
    for p in _primes_list:
        valid[_indices[p][best_config[p]]] = False
    
    result = [int(x) for x in np.where(valid)[0]]
    return result
# EVOLVE-BLOCK-END
