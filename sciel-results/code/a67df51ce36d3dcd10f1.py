"""HP lattice protein folding on cubic lattice - diverse growth + intensive local search."""
import random
import time
import math

_GLOBAL_START = time.time()
_HARD_DEADLINE = _GLOBAL_START + 310.0  # 310s hard limit, 20s safety margin


def _neighbours(point):
    """Generate 6 neighbours in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy: -number of H-H contacts (non-adjacent)."""
    n = len(chain)
    occupied = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _delta_energy_move(sequence, chain, i, old_pos, new_pos, occupied):
    """Compute energy change when moving residue i from old_pos to new_pos.
    Returns delta such that new_energy = old_energy - delta (delta > 0 means improvement)."""
    if sequence[i] != 'H':
        return 0
    delta = 0
    # Lost contacts at old_pos
    for q in _neighbours(old_pos):
        j = occupied.get(q)
        if j is not None and j != i and sequence[j] == 'H':
            delta -= 1
    # Gained contacts at new_pos
    for q in _neighbours(new_pos):
        j = occupied.get(q)
        if j is not None and j != i and sequence[j] == 'H':
            delta += 1
    return delta


def _grow(sequence, dimension, rng):
    """Random self-avoiding walk growth."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0, 0)]
    
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        neighbors = list(_neighbours(last))
        rng.shuffle(neighbors)
        
        placed = False
        for site in neighbors:
            if site not in occupied:
                chain.append(site)
                occupied[site] = i
                placed = True
                break
        
        if not placed:
            return None
    
    return chain


def _local_search(sequence, chain, occupied, deadline, rng):
    """Simulated Annealing with single-residue moves."""
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_occupied = dict(occupied)
    current_chain = list(chain)
    current_occupied = dict(occupied)
    current_e = _compute_energy(sequence, current_chain)
    best_e = current_e
    
    T = 5.0
    cooling_rate = 0.995
    min_T = 0.01
    
    while time.time() < deadline:
        i = rng.randint(1, n - 2)
        prev = current_chain[i - 1]
        next_pos = current_chain[i + 1]
        old_pos = current_chain[i]
        
        candidates = []
        for site in _neighbours(prev):
            if site == old_pos:
                continue
            if site in current_occupied and current_occupied[site] != i:
                continue
            if sum(abs(site[a] - next_pos[a]) for a in range(3)) == 1:
                candidates.append(site)
        
        if not candidates:
            continue
        
        new_pos = rng.choice(candidates)
        delta_contacts = _delta_energy_move(sequence, current_chain, i, old_pos, new_pos, current_occupied)
        delta_e = -delta_contacts
        
        accept = False
        if delta_e < 0:
            accept = True
        else:
            if T > min_T:
                p = math.exp(-delta_e / T)
                if rng.random() < p:
                    accept = True
        
        if accept:
            current_occupied[old_pos] = -1
            current_occupied[new_pos] = i
            current_chain[i] = new_pos
            current_e += delta_e
            
            if current_e < best_e:
                best_e = current_e
                best_chain = list(current_chain)
                best_occupied = dict(current_occupied)
        
        T *= cooling_rate
        if T < min_T:
            T = min_T
    
    return best_chain, best_e


def fold(sequence, dimension):
    """Best conformation via diverse growth + intensive local search."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    # Seed RNG for reproducibility
    seed = 42 + len(sequence) * 7 + sum(ord(c) for c in sequence)
    rng = random.Random(seed)
    
    # Time budget: use up to 11s per sequence, but track global deadline
    per_seq_deadline = min(time.time() + 11.0, _HARD_DEADLINE - 5.0)
    
    # Phase 1: Generate diverse initial chains via random walks
    initial_chains = []
    growth_deadline = per_seq_deadline - 4.0
    
    count = 0
    max_growths = 100
    while count < max_growths and time.time() < growth_deadline:
        chain = _grow(sequence, dimension, rng)
        if chain is not None and len(chain) == n:
            initial_chains.append(chain)
        count += 1
    
    if not initial_chains:
        initial_chains.append([(i, 0, 0) for i in range(n)])
    
    # Phase 2: Simulated Annealing on top-k chains
    remaining_for_search = per_seq_deadline - time.time()
    
    scored_chains = []
    for chain in initial_chains:
        e = _compute_energy(sequence, chain)
        scored_chains.append((e, chain))
    
    scored_chains.sort(key=lambda x: x[0])
    
    if remaining_for_search > 6.0:
        k = min(5, len(scored_chains))
    elif remaining_for_search > 3.0:
        k = min(3, len(scored_chains))
    else:
        k = min(2, len(scored_chains))
    
    best_chain = None
    best_energy = float('inf')
    
    search_deadline = per_seq_deadline - 1.0
    per_chain_time = max(0.5, (search_deadline - time.time()) / k)
    
    for rank in range(k):
        e, chain = scored_chains[rank]
        chain_occupied = {p: i for i, p in enumerate(chain)}
        
        chain_deadline = min(time.time() + per_chain_time, search_deadline)
        if time.time() >= chain_deadline:
            break
        
        polished_chain, polished_e = _local_search(sequence, chain, chain_occupied, chain_deadline, rng)
        
        if polished_e < best_energy:
            best_energy = polished_e
            best_chain = polished_chain
    
    # If no chain was polished (shouldn't happen), use the best initial
    if best_chain is None:
        best_chain = scored_chains[0][1]
        best_energy = scored_chains[0][0]
    
    # Verify self-avoidance
    if len(set(best_chain)) != n:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]
