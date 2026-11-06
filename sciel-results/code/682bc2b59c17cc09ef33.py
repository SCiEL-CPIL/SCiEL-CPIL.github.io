N = 18
import time
import random
import math

# EVOLVE-BLOCK-START

# Precompute initial wire states for bit-parallel verification
_M = 1 << N
_wires_init = [0] * N
for w in range(N):
    period = 1 << (w + 1)
    half_period = 1 << w
    pattern = (1 << half_period) - 1
    pattern <<= half_period
    
    current_val = pattern
    current_len = period
    while current_len < _M:
        current_val |= (current_val << current_len)
        current_len *= 2
        
    _wires_init[w] = current_val

def _verify_network_bit_parallel(comparators):
    """
    Verify if the network sorts all 2^N boolean inputs.
    Uses bit-parallelism: state is a list of N big integers.
    """
    if not comparators:
        return N <= 1
    
    # Copy initial state
    state = _wires_init[:]
    
    for i, j in comparators:
        # Find inputs where wire i is 1 and wire j is 0
        swap_mask = state[i] & ~state[j]
        
        if swap_mask == 0:
            continue
            
        # Update wires i and j
        old_i = state[i]
        old_j = state[j]
        
        state[i] = (old_i & ~swap_mask) | (old_j & swap_mask)
        state[j] = (old_j & ~swap_mask) | (old_i & swap_mask)
        
    # Check if sorted
    bad = 0
    for w in range(N - 1):
        bad |= state[w] & ~state[w + 1]
        if bad:
            return False
            
    return True

def _odd_even_transposition(n=N):
    """Generate odd-even transposition sorting network."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators

def _depth_first_remove(comparators, deadline):
    """
    Depth-first search for removals: try removing each comparator,
    and if valid, recursively try to remove more from the result.
    This is a structural change from greedy: it explores the space of
    removable comparators more deeply, potentially finding combinations
    that are only valid when removed together.
    """
    current = list(comparators)
    best = list(current)
    best_size = len(current)
    
    def dfs(comps, depth):
        nonlocal best, best_size
        if time.time() > deadline:
            return
        
        if len(comps) < best_size:
            best = list(comps)
            best_size = len(comps)
        
        if depth >= 3:  # Limit recursion depth to avoid explosion
            return
        
        for i in range(len(comps) - 1, -1, -1):
            if time.time() > deadline:
                return
            
            # Try removing comparator i
            trial = comps[:i] + comps[i+1:]
            
            if _verify_network_bit_parallel(trial):
                # Recursively try to remove more
                dfs(trial, depth + 1)
                if time.time() > deadline:
                    return
            else:
                # This comparator is necessary, but maybe we can still
                # find a different set of removals
                pass
                
    dfs(current, 0)
    return best

def _hill_climbing_remove(comparators, deadline):
    """
    Hill climbing with removal: repeatedly find the best single removal
    that maintains validity, apply it, and repeat.
    """
    current = list(comparators)
    
    while time.time() < deadline:
        best_removal = -1
        for i in range(len(current)):
            if time.time() > deadline:
                break
            trial = current[:i] + current[i+1:]
            if _verify_network_bit_parallel(trial):
                best_removal = i
                break
        
        if best_removal == -1:
            break
        
        current = current[:best_removal] + current[best_removal+1:]
    
    return current

def _simulated_annealing(start_network, deadline):
    """
    Simulated Annealing to find a smaller sorting network.
    Moves: Remove, Add, Swap, Move, Replace, Slide.
    """
    current = list(start_network)
    best = list(current)
    best_size = len(current)
    
    # SA Parameters
    T = 2.0
    cooling_rate = 0.995
    min_T = 0.05
    
    steps_since_improvement = 0
    max_stall_steps = 100
    
    while time.time() < deadline and T > min_T:
        # Generate a move
        r = random.random()
        neighbor = list(current)
        move_valid = False
        delta = 0
        
        # Determine move type
        if r < 0.35:
            # Try Remove
            if len(neighbor) > 0:
                idx = random.randint(0, len(neighbor) - 1)
                neighbor.pop(idx)
                delta = -1
                move_valid = True
        elif r < 0.55:
            # Try Slide: Change (i, j) to (i+1, j+1) or (i-1, j-1)
            if len(neighbor) > 0:
                idx = random.randint(0, len(neighbor) - 1)
                i, j = neighbor[idx]
                if random.random() < 0.5:
                    if i + 1 < N - 1 and j + 1 < N:
                        ni, nj = i + 1, j + 1
                        if ni < nj:
                            neighbor[idx] = (ni, nj)
                            delta = 0
                            move_valid = True
                else:
                    if i - 1 >= 0 and j - 1 > 0:
                        ni, nj = i - 1, j - 1
                        if ni < nj:
                            neighbor[idx] = (ni, nj)
                            delta = 0
                            move_valid = True
        elif r < 0.70:
            # Try Replace: change the pair of a random comparator
            if len(neighbor) > 0:
                idx = random.randint(0, len(neighbor) - 1)
                if random.random() < 0.6:
                    dist = random.randint(1, 3)
                else:
                    dist = random.randint(1, N - 1)
                a = random.randint(0, N - 1 - dist)
                b = a + dist
                neighbor[idx] = (a, b)
                delta = 0
                move_valid = True
        elif r < 0.80:
            # Try Move: move a comparator to a different position
            if len(neighbor) > 1:
                idx1 = random.randint(0, len(neighbor) - 1)
                idx2 = random.randint(0, len(neighbor) - 1)
                if idx1 != idx2:
                    comp = neighbor.pop(idx1)
                    neighbor.insert(idx2, comp)
                    delta = 0
                    move_valid = True
        elif r < 0.90:
            # Try Swap: swap two comparators
            if len(neighbor) > 1:
                idx1, idx2 = random.sample(range(len(neighbor)), 2)
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                delta = 0
                move_valid = True
        else:
            # Try Add: add a new comparator at a random position
            if random.random() < 0.6:
                dist = random.randint(1, 3)
            else:
                dist = random.randint(1, N - 1)
            a = random.randint(0, N - 1 - dist)
            b = a + dist
            idx = random.randint(0, len(neighbor))
            neighbor.insert(idx, (a, b))
            delta = 1
            move_valid = True
        
        if not move_valid:
            T *= cooling_rate
            continue
            
        if _verify_network_bit_parallel(neighbor):
            # Accept the move
            current = neighbor
            if len(current) < best_size:
                best = list(current)
                best_size = len(current)
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
        else:
            # Reject the move
            steps_since_improvement += 1
            
        # Cooling
        T *= cooling_rate
        
        # Stall detection
        if steps_since_improvement > max_stall_steps:
            current = list(best)
            steps_since_improvement = 0
            T = max(T, 0.5)
            
    return best

def _construct():
    start_time = time.time()
    deadline = start_time + 50
    
    # Start with odd-even transposition
    best = _odd_even_transposition(N)
    best_len = len(best)
    
    if not _verify_network_bit_parallel(best):
        return []
        
    random.seed(42)
    
    # Phase 1: Depth-first removal (structural change)
    df_deadline = min(start_time + 10, deadline)
    current = _depth_first_remove(best, df_deadline)
    if len(current) < best_len:
        best = list(current)
        best_len = len(current)
        
    # Phase 2: Hill climbing removal
    hc_deadline = min(start_time + 20, deadline)
    current = _hill_climbing_remove(best, hc_deadline)
    if len(current) < best_len:
        best = list(current)
        best_len = len(current)
        
    # Phase 3: Simulated Annealing
    sa_deadline = deadline
    current = _simulated_annealing(best, sa_deadline)
    if len(current) < best_len:
        best = list(current)
        best_len = len(current)
    
    # Phase 4: Final depth-first pass
    if time.time() < deadline:
        final_deadline = deadline
        current = _depth_first_remove(best, final_deadline)
        if len(current) < best_len:
            best = list(current)
            best_len = len(current)
    
    return best

def construct():
    return _construct()

# EVOLVE-BLOCK-END
