import time
import random
import math

N = 18

# EVOLVE-BLOCK-START

# Precompute constants for fast verification
# We use Python big integers to represent the state of all 2^18 inputs.
# Bit 'm' in the integer corresponds to input mask 'm'.
# Wire i state is an integer M_i.

def _precompute_targets():
    """
    Precompute the target state for each wire for all 2^18 inputs.
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
        # Iterate backwards to handle indices safely if we modify
        for idx in range(len(comps) - 1, -1, -1):
            if time.time() > deadline:
                return comps
            candidate = comps[:idx] + comps[idx+1:]
            if _verify_network_bitparallel(candidate):
                comps = candidate
                improved = True
                break
    return comps

def construct():
    _init_constants()
    
    deadline = time.time() + 85 # 5s safety margin
    
    best = _seed_network()
    best_size = len(best)
    
    # Phase 1: Initial Pruning
    best = _prune(best, deadline)
    best_size = len(best)
    
    # Phase 2: Simulated Annealing
    current = list(best)
    current_size = len(current)
    
    T = 10.0 # Initial temperature
    T_min = 0.1
    cooling_rate = 0.99
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        if current_size < best_size:
            best = list(current)
            best_size = current_size
            
        # Generate neighbor
        action = rng.random()
        neighbor = list(current)
        
        if action < 0.5 and len(neighbor) > 1:
            # Remove
            idx = rng.randint(0, len(neighbor) - 1)
            neighbor.pop(idx)
        elif action < 0.8 and len(neighbor) > 1:
            # Swap
            i, j = rng.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        else:
            # Insert
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            pos = rng.randint(0, len(neighbor))
            neighbor.insert(pos, (i, j))
            
        # Verify neighbor
        if _verify_network_bitparallel(neighbor):
            neighbor_size = len(neighbor)
            delta = neighbor_size - current_size
            
            if delta < 0:
                # Always accept improvement
                current = neighbor
                current_size = neighbor_size
            else:
                # Accept worse move with probability exp(-delta / T)
                if rng.random() < math.exp(-delta / T):
                    current = neighbor
                    current_size = neighbor_size
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best

# EVOLVE-BLOCK-END
