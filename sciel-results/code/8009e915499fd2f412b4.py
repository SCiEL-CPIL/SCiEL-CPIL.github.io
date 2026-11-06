"""Size-efficient sorting networks on 17 wires via bit-vector simulation and GA."""
import time
import random

N = 17
NUM_INPUTS = 1 << N
DEADLINE = 55.0  # Safety margin before 60s limit
START_TIME = time.time()

# Precompute bit-vector representation of initial states
# Each wire is represented as a Python integer where bit k is the value on that wire for input k.
# There are 2^N inputs, so we need 2^N bits per wire.
# For N=17, 131072 bits = 16KB per wire. 17 wires = 272KB.
# Python big integers handle this efficiently.

def _init_bit_vectors():
    """
    Create initial bit-vector states for each wire.
    state[wire] is an integer where bit k is set if wire has value 1 for input k.
    Input k is an integer from 0 to 2^N - 1.
    Wire w has value 1 if bit w of k is 1.
    """
    states = [0] * N
    for k in range(NUM_INPUTS):
        for w in range(N):
            if (k >> w) & 1:
                states[w] |= (1 << k)
    return states

INITIAL_STATES = _init_bit_vectors()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def verify_network(comps):
    """
    Verify network using bit-vector simulation.
    comps: list of (i, j) tuples.
    Returns True if valid.
    """
    if not comps:
        return False
    
    # Copy initial states
    state = list(INITIAL_STATES)
    
    for i, j in comps:
        wi = state[i]
        wj = state[j]
        
        # Comparator (i, j): min to i, max to j.
        # min(wi, wj) = wi & wj
        # max(wi, wj) = wi | wj
        state[i] = wi & wj
        state[j] = wi | wj
        
    # Check if sorted: for all rows, state[:, k] <= state[:, k+1]
    # This means for all k, if state[w] has bit k set, state[w+1] must also have bit k set.
    # i.e., state[w] & ~state[w+1] must be 0 for all w.
    for w in range(N - 1):
        if state[w] & ~state[w + 1]:
            return False
            
    return True

def make_seed():
    """Odd-even transposition network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def greedy_remove(comps, rng=None):
    """
    Try to remove each comparator if the network remains valid.
    Uses randomized order to escape local minima.
    """
    if rng is None:
        rng = random.Random()
    n = len(comps)
    improved = True
    while improved:
        improved = False
        indices = list(range(n))
        rng.shuffle(indices)
        for idx in indices:
            if _time_left() < 2.0:
                return comps
            candidate = comps[:idx] + comps[idx+1:]
            if verify_network(candidate):
                comps = candidate
                n = len(comps)
                improved = True
                break # Restart scan
    return comps

def add_remove_search(comps, rng):
    """
    Perform a local search step: add a random comparator, then greedily remove.
    This helps escape local minima.
    """
    if _time_left() < 2.0:
        return comps
    
    n = len(comps)
    # Try adding a random comparator
    # We want to add a comparator that might help sorting, but any is fine for the kick
    i = rng.randrange(N)
    j = rng.randrange(N)
    if i == j:
        return comps
    if i > j:
        i, j = j, i
    
    # Insert at a random position
    pos = rng.randrange(n + 1)
    candidate = comps[:pos] + [(i, j)] + comps[pos:]
    
    # Greedily remove from this new candidate
    # We allow it to grow if it helps, but we aim to shrink
    candidate = greedy_remove(candidate, rng)
    
    return candidate

def construct():
    """Construct a sorting network for 17 wires, minimizing comparator count."""
    global START_TIME
    START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Start with odd-even transposition
    comps = make_seed()
    
    # First pass: aggressive greedy removal
    comps = greedy_remove(comps, rng)
    
    best = list(comps)
    best_len = len(best)
    current = list(comps)
    
    # Simulated Annealing
    T = 10.0  # Initial "temperature" (max allowed size increase)
    cooling = 0.99
    
    while _time_left() > 5.0:
        # Local search on current
        current = greedy_remove(current, rng)
        
        if len(current) < best_len:
            best = list(current)
            best_len = len(current)
            T = 10.0 # Reset temperature on improvement
        
        # Generate neighbor
        # 1. Try add-remove (kick)
        if rng.random() < 0.3:
            neighbor = add_remove_search(current, rng)
        else:
            # 2. Try random mutation (remove, swap, move, insert)
            neighbor = list(current)
            n = len(neighbor)
            op = rng.choice(['remove', 'swap', 'move', 'insert'])
            
            if op == 'remove' and n > 0:
                idx = rng.randrange(n)
                neighbor.pop(idx)
            elif op == 'swap' and n >= 2:
                i1 = rng.randrange(n)
                i2 = rng.randrange(n)
                if i1 != i2:
                    neighbor[i1], neighbor[i2] = neighbor[i2], neighbor[i1]
            elif op == 'move' and n >= 1:
                i1 = rng.randrange(n)
                i2 = rng.randrange(n)
                if i1 != i2:
                    elem = neighbor.pop(i1)
                    neighbor.insert(i2, elem)
            elif op == 'insert':
                i = rng.randrange(N)
                j = rng.randrange(N)
                if i > j: i, j = j, i
                if i != j:
                    pos = rng.randrange(n + 1)
                    neighbor.insert(pos, (i, j))
            
            # Verify and shrink
            if verify_network(neighbor):
                neighbor = greedy_remove(neighbor, rng)
            else:
                # If invalid, revert to current (or just skip)
                neighbor = list(current)
        
        # Acceptance criterion
        if len(neighbor) <= len(current):
            current = neighbor
        else:
            # Accept worse solution with probability exp(-delta/T)
            delta = len(neighbor) - len(current)
            if delta <= T:
                prob = 0.5 ** (delta / max(T, 1e-6))
                if rng.random() < prob:
                    current = neighbor
        
        T *= cooling
    
    # Final greedy pass
    best = greedy_remove(best, rng)
    
    return best
