import time
import random
import math
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 55.0  # Safety margin from 60s limit

def _build_mask(w, n_bits):
    block = 1 << w
    pattern = (1 << block) - 1
    pattern <<= block
    num_periods = n_bits // (2 * block)
    m = 0
    shift = 0
    for _ in range(num_periods):
        m |= (pattern << shift)
        shift += 2 * block
    return m

# Pre-compute masks at module load time to avoid doing it in construct()
_INITIAL_MASKS = [0] * N
for _w in range(N):
    _INITIAL_MASKS[_w] = _build_mask(_w, 1 << N)

def _verify(comparators):
    """
    Verify if the comparator list sorts all 2**N boolean inputs.
    Uses bitwise operations on Python integers representing the state of each wire.
    """
    state = _INITIAL_MASKS[:]
    
    for i, j in comparators:
        a = state[i]
        b = state[j]
        state[i] = a & b
        state[j] = a | b
        
    for i in range(N - 1):
        if (state[i] & ~state[i+1]) != 0:
            return False
            
    return True

def _odd_even_transposition():
    """Return the odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    # Start with the seed network
    current = _odd_even_transposition()
    best = current[:]
    best_len = len(best)
    
    # Initial verification of seed (should be true)
    if not _verify(best):
        return best
        
    # Phase 1: Greedy removal of redundant comparators
    changed = True
    while changed:
        changed = False
        for i in range(len(current) - 1, -1, -1):
            if time.time() > deadline:
                return best
            # Try removing current[i]
            neighbor = current[:i] + current[i+1:]
            if _verify(neighbor):
                current = neighbor
                changed = True
                if len(current) < best_len:
                    best = current[:]
                    best_len = len(current)
                    
    # Phase 2: Simulated Annealing
    remaining_time = deadline - time.time()
    if remaining_time < 1.0:
        return best
        
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.99
    
    rng = random.Random(42)
    
    all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Weights for moves
    weights = [0.5, 0.2, 0.2, 0.1] # remove, swap, move, add
    
    steps_since_improvement = 0
    restart_threshold = 500
    
    while time.time() < deadline and T > T_min:
        # Generate move
        r = rng.random()
        if r < weights[0]:
            move_type = 'remove'
        elif r < weights[0] + weights[1]:
            move_type = 'swap'
        elif r < weights[0] + weights[1] + weights[2]:
            move_type = 'move'
        else:
            move_type = 'add'
            
        neighbor = current[:]
        
        if move_type == 'remove' and len(neighbor) > 0:
            idx = rng.randrange(len(neighbor))
            neighbor.pop(idx)
            
        elif move_type == 'add':
            comp = rng.choice(all_comparators)
            pos = rng.randrange(len(neighbor) + 1)
            neighbor.insert(pos, comp)
            
        elif move_type == 'swap' and len(neighbor) >= 2:
            i = rng.randrange(len(neighbor))
            j = rng.randrange(len(neighbor))
            if i != j:
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                
        elif move_type == 'move' and len(neighbor) >= 2:
            i = rng.randrange(len(neighbor))
            j = rng.randrange(len(neighbor))
            if i != j:
                comp = neighbor.pop(i)
                if i < j:
                    j -= 1
                neighbor.insert(j, comp)
        else:
            continue

        if _verify(neighbor):
            delta = len(neighbor) - len(current)
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current = neighbor
                if len(current) < best_len:
                    best = current[:]
                    best_len = len(current)
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                steps_since_improvement += 1
        else:
            steps_since_improvement += 1
            
        # Restart logic
        if steps_since_improvement > restart_threshold:
            current = best[:]
            T = max(T, 2.0)
            steps_since_improvement = 0

        T *= cooling_rate
        
    # Final check
    if not _verify(best):
        return _odd_even_transposition()
        
    return best
