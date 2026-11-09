"""Combined program: bit-parallel verification + aggressive pruning + random restarts."""
import time
import random

N = 18
MAX_COMPARATORS = 4096

# Precompute constants for fast verification
# We use Python big integers to represent the state of all 2^18 inputs.
# Bit 'm' in the integer corresponds to input mask 'm'.
# Wire i state is an integer M_i.

def _precompute_targets():
    """
    Precompute the target state for each wire for all 2^18 inputs.
    Target_i has bit 'm' set if and only if the sorted output for input 'm' 
    has a 1 on wire i.
    Wire 0 is the minimum, Wire N-1 is the maximum.
    For input m with k ones, sorted output has 0s in positions 0..N-1-k and 1s in N-k..N-1.
    So wire i is 1 iff k >= N - i.
    """
    targets = [0] * N
    for m in range(1 << N):
        k = bin(m).count('1')
        for i in range(N):
            if k >= N - i:
                targets[i] |= (1 << m)
    return targets

def _precompute_initials():
    initials = [0] * N
    for i in range(N):
        mask = 0
        for m in range(1 << N):
            if (m >> i) & 1:
                mask |= (1 << m)
        initials[i] = mask
    return initials

_TARGETS = None
_INITIALS = None

def _init_constants():
    global _TARGETS, _INITIALS
    if _TARGETS is None:
        _TARGETS = _precompute_targets()
        _INITIALS = _precompute_initials()

def _verify_network_bitparallel(comparators):
    """
    Verifies the network using bit-parallel simulation.
    """
    if _TARGETS is None:
        _init_constants()
    
    states = list(_INITIALS)
    
    for i, j in comparators:
        si = states[i]
        sj = states[j]
        states[i] = si & sj
        states[j] = si | sj
        
    for i in range(N):
        if states[i] != _TARGETS[i]:
            return False
    return True

def _seed_network():
    """Odd-Even Transposition Sort."""
    comps = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _prune(comps, deadline):
    """
    Aggressively remove redundant comparators.
    """
    comps = list(comps)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(comps) - 1, -1, -1):
            if time.time() > deadline:
                return comps
            candidate = comps[:idx] + comps[idx+1:]
            if _verify_network_bitparallel(candidate):
                comps = candidate
                improved = True
                break
    return comps

def _random_valid_network(rng, deadline):
    """
    Try to build a random network that sorts, using greedy insertion.
    This is a structural change: instead of only mutating the seed, we try
    random construction.
    """
    if time.time() > deadline:
        return None
    
    # Start empty
    comps = []
    
    # Greedy: add random comparators until we sort
    max_iter = 200
    for _ in range(max_iter):
        if time.time() > deadline:
            return None
        
        # Try adding a random comparator
        i = rng.randint(0, N - 2)
        j = rng.randint(i + 1, N - 1)
        
        # Try inserting at random position
        pos = rng.randint(0, len(comps))
        candidate = comps[:pos] + [(i, j)] + comps[pos:]
        
        if _verify_network_bitparallel(candidate):
            comps = candidate
            # Check if it's a valid sorting network
            if len(comps) > 0:
                # We need to verify it actually sorts
                if _verify_network_bitparallel(comps):
                    return comps
        else:
            # Try just appending
            candidate = comps + [(i, j)]
            if _verify_network_bitparallel(candidate):
                comps = candidate
                if _verify_network_bitparallel(comps):
                    return comps
    
    return None

def construct():
    _init_constants()
    
    deadline = time.time() + 55  # 5s safety margin
    rng = random.Random(42)
    
    best = _seed_network()
    best_size = len(best)
    
    # Initial Pruning
    best = _prune(best, deadline)
    best_size = len(best)
    
    # Local Search with random restarts
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to remove any comparator
        for idx in range(len(best) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = best[:idx] + best[idx+1:]
            if _verify_network_bitparallel(candidate):
                best = candidate
                best_size -= 1
                improved = True
                break
        
        if not improved:
            # 2. Try random mutations
            for _ in range(5):
                if time.time() > deadline:
                    break
                mutated = list(best)
                action = rng.random()
                
                if action < 0.3 and len(mutated) > 1:
                    # Swap
                    i, j = rng.sample(range(len(mutated)), 2)
                    mutated[i], mutated[j] = mutated[j], mutated[i]
                elif action < 0.6:
                    # Insert
                    i = rng.randint(0, N - 2)
                    j = rng.randint(i + 1, N - 1)
                    pos = rng.randint(0, len(mutated))
                    mutated.insert(pos, (i, j))
                else:
                    # Delete
                    if len(mutated) > 1:
                        idx = rng.randint(0, len(mutated) - 1)
                        mutated.pop(idx)
                
                if _verify_network_bitparallel(mutated):
                    pruned = _prune(mutated, deadline)
                    if len(pruned) < best_size:
                        best = pruned
                        best_size = len(best)
                        improved = True
                        break
            
            # 3. Try random restart: build a new random network
            if not improved and time.time() < deadline:
                random_net = _random_valid_network(rng, deadline)
                if random_net is not None:
                    pruned = _prune(random_net, deadline)
                    if len(pruned) < best_size:
                        best = pruned
                        best_size = len(best)
                        improved = True
    
    return best
