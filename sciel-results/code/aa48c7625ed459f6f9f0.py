"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return np.where(sieve)[0]

def _precompute_indices(primes):
    """
    Precompute list of indices for each residue class of each prime.
    indices[p_idx][r] is a numpy array of global indices x in [0, N] 
    such that x % primes[p_idx] == r.
    """
    indices = {}
    for p in primes:
        p_indices = {}
        for r in range(p):
            # Indices x such that x % p == r
            # x = r, r+p, r+2p, ...
            # Using np.arange is efficient
            if r <= N:
                p_indices[r] = np.arange(r, N + 1, p)
            else:
                p_indices[r] = np.array([], dtype=int)
        indices[p] = p_indices
    return indices

def _greedy_init(primes, indices):
    """
    Initialize c and cover_count using the original greedy logic.
    c is a dict mapping prime -> residue.
    """
    c = {}
    cover_count = np.zeros(N + 1, dtype=int)
    
    # We need to simulate the greedy choice.
    # The original greedy chose min count among active.
    # Here we can just use the same logic to get a good starting point.
    mask = np.ones(N + 1, dtype=bool)
    
    for p in primes:
        if p > N: break # Primes are <= N anyway
        
        active_indices = np.nonzero(mask)[0]
        if len(active_indices) == 0:
            break
            
        residues = active_indices % p
        counts = np.bincount(residues, minlength=p)
        
        if np.all(counts > 0):
            min_count = np.min(counts)
            candidates = np.where(counts == min_count)[0]
            min_class = candidates[0] # Deterministic for init
            
            c[p] = min_class
            # Update cover_count
            S = indices[p][min_class]
            cover_count[S] += 1
            # Update mask
            mask[S] = False
            
    # Fill in c for primes not processed (if any, though loop breaks only if active is 0)
    # If active is not 0, we process all primes <= N?
    # The loop `for p in primes` iterates all.
    # If `p > current_size` was the break condition in original, we should mimic that?
    # Original: `if p > current_size: break`
    # Let's just process all primes for the init to be safe, or mimic original.
    # Mimicking original is safer for a good start.
    
    # Actually, let's just run the original _greedy_construct to get the set, 
    # then reverse-engineer c? No, that's hard.
    # Let's just use the greedy logic above but ensure we process all primes 
    # up to the point where it makes sense, or just all primes.
    # If we process all primes, we guarantee admissibility.
    # Let's re-run the init loop properly.
    
    c = {}
    cover_count = np.zeros(N + 1, dtype=int)
    mask = np.ones(N + 1, dtype=bool)
    current_size = N + 1
    
    for p in primes:
        if p > current_size:
            break
            
        active_indices = np.nonzero(mask)[0]
        if len(active_indices) == 0:
            break
            
        residues = active_indices % p
        counts = np.bincount(residues, minlength=p)
        
        if np.all(counts > 0):
            min_count = np.min(counts)
            candidates = np.where(counts == min_count)[0]
            min_class = candidates[0]
            
            c[p] = min_class
            S = indices[p][min_class]
            cover_count[S] += 1
            mask[S] = False
            current_size -= min_count
        else:
            # If some count is 0, we don't need to remove a class?
            # But the task says "selecting a removed residue class for each prime".
            # If a class is already missed, we can select it (remove 0 new elements) 
            # or select another.
            # To be consistent with "admissible", we just need SOME class missed.
            # If we select the one already missed, we add no new removals.
            # If we select a full one, we remove elements.
            # Greedy would select the one with min count (which is 0).
            # So we select the class with count 0.
            min_class = np.where(counts == 0)[0][0]
            c[p] = min_class
            # cover_count update: S is already 0 in mask? 
            # No, S are indices where x%p==min_class.
            # Some of these might be active? No, count was 0, so none are active.
            # So cover_count[S] increases, but they were already removed by other primes?
            # Wait, if count is 0 in `active_indices`, it means NO active element has this residue.
            # So all elements with this residue are ALREADY removed by previous primes.
            # So `cover_count[S]` should be >= 1 for all x in S.
            # So adding to cover_count is correct.
            S = indices[p][min_class]
            cover_count[S] += 1

    return c, cover_count

def _hill_climb(primes, indices, c, cover_count, max_iters=100):
    """
    Perform hill climbing on the assignment c.
    """
    current_size = np.sum(cover_count == 0)
    
    for _ in range(max_iters):
        improved = False
        # Shuffle primes to break symmetry
        p_list = list(primes)
        np.random.shuffle(p_list)
        
        for p in p_list:
            if p not in c:
                continue
                
            r_old = c[p]
            S_old = indices[p][r_old]
            
            # Try all other residues
            # Optimization: only try if there's potential gain?
            # Gain comes from S_old elements having cover_count == 1.
            # If no element in S_old has count 1, gain is 0.
            # Loss comes from S_new elements having cover_count == 0.
            
            # We can pre-compute if any element in S_old has count 1
            # If not, any move will have gain 0, so we need loss < 0 (impossible) or loss=0.
            # If gain=0 and loss=0, no change in size.
            # So we only care if gain > loss.
            
            has_gain_potential = np.any(cover_count[S_old] == 1)
            if not has_gain_potential:
                continue
                
            best_r_new = -1
            best_net = 0
            
            for r_new in range(p):
                if r_new == r_old:
                    continue
                    
                S_new = indices[p][r_new]
                
                # Calculate gain and loss
                gain = np.sum(cover_count[S_old] == 1)
                # Note: gain is constant for this p, r_old!
                # So we only need to calculate loss for each r_new.
                
                loss = np.sum(cover_count[S_new] == 0)
                
                net = gain - loss
                if net > best_net:
                    best_net = net
                    best_r_new = r_new
                    
            if best_r_new != -1 and best_net > 0:
                # Apply move
                S_new = indices[p][best_r_new]
                cover_count[S_old] -= 1
                cover_count[S_new] += 1
                c[p] = best_r_new
                current_size += best_net
                improved = True
                
    return current_size, c

def construct():
    primes = _primes(N + 1)
    indices = _precompute_indices(primes)
    
    # Initialize with greedy
    c, cover_count = _greedy_init(primes, indices)
    
    # Hill climb
    size, c = _hill_climb(primes, indices, c, cover_count, max_iters=200)
    
    # Construct final tuple
    # Elements with cover_count == 0
    final_tuple = np.nonzero(cover_count == 0)[0].tolist()
    
    return final_tuple
# EVOLVE-BLOCK-END
