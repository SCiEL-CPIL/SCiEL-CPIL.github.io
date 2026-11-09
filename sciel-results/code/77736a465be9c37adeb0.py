"""HP lattice protein folding: Diverse Greedy Growth + Fast Simulated Annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 115.0  # Safety margin for 120s budget


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbours(point):
    x, y = point
    for dx, dy in _DELTAS:
        yield (x + dx, y + dy)


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    if len(chain) < 2:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts // 2


def _grow(sequence, dimension, rng, bias_hh, bias_free):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Score: weighted sum of contacts and free space, plus noise
            score = bias_hh * gain + bias_free * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        # Pick the best option with some randomness (top-k selection)
        options.sort(key=lambda x: x[0], reverse=True)
        top_k = min(3, len(options))
        # Exponential weighting for selection
        weights = [math.exp(options[i][0] / 2.0) for i in range(top_k)]
        total_w = sum(weights)
        r = rng.random() * total_w
        cumulative = 0
        selected_site = options[0][1]
        for i in range(top_k):
            cumulative += weights[i]
            if r <= cumulative:
                selected_site = options[i][1]
                break
        
        occupied[selected_site] = i
        chain.append(selected_site)
    return chain


def _is_valid_chain(chain):
    """Check self-avoidance."""
    return len(set(chain)) == len(chain)


def _apply_reversal(chain, i, j):
    """
    Attempt a reversal move. Returns new chain if valid, else None.
    i, j are indices. 0 <= i < j < len(chain).
    """
    n = len(chain)
    if i >= j:
        return None
    
    # Check boundary conditions
    # Left cut: chain[i-1] must be adjacent to chain[j]
    if i > 0:
        p_prev = chain[i - 1]
        p_end = chain[j]
        if abs(p_prev[0] - p_end[0]) + abs(p_prev[1] - p_end[1]) != 1:
            return None
            
    # Right cut: chain[i] must be adjacent to chain[j+1]
    if j < n - 1:
        p_start = chain[i]
        p_next = chain[j + 1]
        if abs(p_start[0] - p_next[0]) + abs(p_start[1] - p_next[1]) != 1:
            return None
            
    # Construct new chain
    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
    
    # Check self-avoidance
    if not _is_valid_chain(new_chain):
        return None
        
    return new_chain


def _apply_pivot(chain, i, angle):
    """
    Attempt a pivot move. Rotate tail chain[i+1:] around chain[i] by 'angle' (1, 2, or 3 * 90 deg).
    Returns new chain if valid, else None.
    """
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return None
        
    pivot = chain[i]
    tail = chain[i+1:]
    prefix = chain[:i+1]
    prefix_set = set(prefix)
    
    # Rotate tail
    rotated_tail = []
    valid = True
    for p in tail:
        x, y = p[0] - pivot[0], p[1] - pivot[1]
        for _ in range(angle):
            x, y = -y, x
        new_p = (pivot[0] + x, pivot[1] + y)
        rotated_tail.append(new_p)
        if new_p in prefix_set:
            valid = False
            break
    
    if not valid:
        return None
        
    # Check connection between i-1 and i+1 (which is now rotated)
    # chain[i-1] must be adjacent to rotated[0]
    p_prev = chain[i-1]
    p_new_start = rotated_tail[0]
    if abs(p_prev[0] - p_new_start[0]) + abs(p_prev[1] - p_new_start[1]) != 1:
        return None
        
    new_chain = prefix + rotated_tail
    return new_chain


def _anneal(sequence, initial_chain, time_limit, rng):
    """
    Simulated Annealing starting from initial_chain.
    Returns best_chain, best_energy.
    """
    n = len(initial_chain)
    current_chain = initial_chain[:]
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy
    
    T = 5.0
    T_end = 0.1
    cooling_rate = 0.999
    
    deadline = time.time() + time_limit
    steps = 0
    stagnation = 0
    
    while time.time() < deadline:
        if _remaining_time() < 1.0:
            break
        
        # Choose move type: 50% Reversal, 50% Pivot
        if rng.random() < 0.5:
            # Reversal
            if n < 4:
                continue
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            new_chain = _apply_reversal(current_chain, i, j)
            if new_chain is None:
                continue
        else:
            # Pivot
            if n < 4:
                continue
            i = rng.randint(1, n - 2)
            angle = rng.choice([1, 2, 3])
            new_chain = _apply_pivot(current_chain, i, angle)
            if new_chain is None:
                continue
        
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Metropolis criterion
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 0.01)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
                stagnation = 0
            else:
                stagnation += 1
        else:
            stagnation += 1
            
        # Adaptive cooling: reheat if stuck
        if stagnation > 100:
            T = min(T * 1.5, 5.0)
            stagnation = 0
        else:
            T *= cooling_rate
            if T < T_end:
                T = T_end
                
        steps += 1
        if steps % 1000 == 0:
            # Check time more frequently to avoid overshoot
            pass
            
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    if n < 2:
        return [[0, 0]] * n
        
    best_chain = None
    best_energy = 0 # Energy is negative, so 0 is worse than any contact
    
    # Phase 1: Generate diverse initial chains
    # Use different bias parameters to explore different regions of the solution space
    bias_configs = [
        (10.0, 0.1),  # Strong HH bias
        (5.0, 0.5),   # Balanced
        (1.0, 1.0),   # Strong Free space bias
        (20.0, 0.0),  # Very strong HH bias
        (0.5, 2.0),   # Very strong free space
    ]
    
    initial_chains = []
    
    # Allocate 20% of time to growth
    total_time = _remaining_time()
    growth_time_limit = total_time * 0.2
    growth_deadline = time.time() + growth_time_limit
    
    # Distribute growth time among bias configs
    num_configs = len(bias_configs)
    
    for bias_hh, bias_free in bias_configs:
        if time.time() >= growth_deadline:
            break
            
        # Try a few times with this bias
        attempts = 0
        max_attempts = 50
        while time.time() < growth_deadline and attempts < max_attempts:
            chain = _grow(sequence, dimension, rng, bias_hh, bias_free)
            if chain is not None:
                initial_chains.append(chain)
            attempts += 1
                
    # If no chains generated, fallback to simple straight line
    if not initial_chains:
        initial_chains.append([(i, 0) for i in range(n)])
        
    # Evaluate and sort initial chains
    scored_chains = []
    for chain in initial_chains:
        e = _energy(sequence, chain)
        scored_chains.append((e, chain))
        
    # Sort by energy (most negative first)
    scored_chains.sort(key=lambda x: x[0])
    
    # Keep top K chains for annealing
    top_k = min(5, len(scored_chains))
    candidates = [c[1] for c in scored_chains[:top_k]]
    
    # Phase 2: Anneal each candidate
    # Allocate remaining time to annealing
    sa_time_remaining = _remaining_time()
    
    # Define time fractions for each candidate
    if top_k == 1:
        fractions = [1.0]
    elif top_k == 2:
        fractions = [0.6, 0.4]
    elif top_k == 3:
        fractions = [0.5, 0.3, 0.2]
    elif top_k == 4:
        fractions = [0.4, 0.3, 0.2, 0.1]
    else:
        fractions = [0.3, 0.25, 0.2, 0.15, 0.1]
    
    for i in range(top_k):
        if _remaining_time() < 1.0:
            break
            
        chain_copy = candidates[i][:]
        time_for_this = sa_time_remaining * fractions[i]
        time_for_this = min(time_for_this, _remaining_time() - 0.5)
        
        if time_for_this <= 0:
            continue
            
        best_c, best_e = _anneal(sequence, chain_copy, time_for_this, rng)
        
        if best_e < best_energy:
            best_energy = best_e
            best_chain = best_c
    
    if best_chain is None:
        best_chain = candidates[0]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
