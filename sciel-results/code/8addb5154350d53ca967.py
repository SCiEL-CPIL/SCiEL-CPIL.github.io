"""HP lattice protein folding: Iterated Local Search with Diverse Greedy Initialization."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 115.0  # Safety margin for 120s budget


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    if len(chain) < 2:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            for dx, dy in _DELTAS:
                q = (x + dx, y + dy)
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts // 2


def _grow(sequence, dimension, rng, bias_hh=1.0, bias_free=0.1):
    """One randomized greedy chain growth."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
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
            weight = gain * bias_hh * 10.0 + free * bias_free * 0.1 + rng.random() * 0.5
            options.append((weight, site))
        if not options:
            return None
        options.sort(key=lambda x: x[0], reverse=True)
        top_k = min(3, len(options))
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


def _neighbours(point):
    x, y = point
    for dx, dy in _DELTAS:
        yield (x + dx, y + dy)


def _apply_move_reversal(chain, a, b):
    """Apply a reversal move from index a to b. Returns True if valid and applied."""
    if a == 0 or b == len(chain) - 1 or b <= a:
        return False
    
    p_prev = chain[a-1]
    p_end = chain[b]
    p_start = chain[a]
    p_next = chain[b+1]
    
    if abs(p_prev[0] - p_end[0]) + abs(p_prev[1] - p_end[1]) != 1:
        return False
    if abs(p_start[0] - p_next[0]) + abs(p_start[1] - p_next[1]) != 1:
        return False
    
    # Reversal preserves self-avoidance of the segment internally.
    # We just need to check that the new connections don't collide with the rest.
    # The segment points are the same, so no new collisions with outside points 
    # except the ones connected to the boundaries which we checked for adjacency.
    # Wait, does the reversal cause the segment to intersect itself?
    # No, the set of points is identical. The order is reversed.
    # Self-avoidance is a property of the set of points occupied.
    # Since the set is unchanged, and it was self-avoiding before, it is self-avoiding after.
    # The only potential issue is if the new boundary connections cross the segment?
    # No, the connections are just lattice edges.
    # So, if the boundary adjacencies are valid, the move is valid.
    
    chain[a:b+1] = reversed(chain[a:b+1])
    return True


def _apply_move_pivot(chain, i, k):
    """Apply a pivot move at index i, rotating the tail by 90 degrees k times."""
    if i == 0 or i == len(chain) - 1:
        return False
    
    pivot = chain[i]
    tail = chain[i+1:]
    prefix = chain[:i+1]
    prefix_set = set(prefix)
    
    rotated = []
    valid = True
    for p in tail:
        x = p[0] - pivot[0]
        y = p[1] - pivot[1]
        for _ in range(k):
            x, y = -y, x
        new_p = (pivot[0] + x, pivot[1] + y)
        rotated.append(new_p)
        if new_p in prefix_set:
            valid = False
            break
    
    if not valid:
        return False
    
    p_prev = chain[i-1]
    p_new_start = rotated[0]
    if abs(p_prev[0] - p_new_start[0]) + abs(p_prev[1] - p_new_start[1]) != 1:
        return False
    
    chain[i+1:] = rotated
    return True


def _local_search(sequence, chain, rng, max_iters=1000):
    """Greedy local search using short reversals and pivots."""
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    n = len(chain)
    
    for _ in range(max_iters):
        improved = False
        
        # Try short reversals (length 2-4)
        for length in range(2, 5):
            if n < length + 1:
                continue
            a = rng.randint(0, n - length - 1)
            b = a + length
            old_seg = chain[a:b+1]
            if _apply_move_reversal(chain, a, b):
                new_energy = _energy(sequence, chain)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = chain[:]
                    improved = True
                else:
                    chain[a:b+1] = old_seg
            if improved:
                break
        
        if improved:
            continue
        
        # Try pivots
        if n > 3:
            i = rng.randint(1, n-2)
            k = rng.randint(1, 3)
            old_tail = chain[i+1:]
            if _apply_move_pivot(chain, i, k):
                new_energy = _energy(sequence, chain)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = chain[:]
                    improved = True
                else:
                    chain[i+1:] = old_tail
        
        if not improved:
            break
    
    chain[:] = best_chain
    return best_energy


def fold(sequence, dimension):
    """Best conformation from Iterated Local Search."""
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    n = len(sequence)
    
    # Phase 1: Diverse Greedy Initialization
    init_chains = []
    init_deadline = time.time() + _remaining_time() * 0.1
    
    while time.time() < init_deadline:
        bias_hh = rng.uniform(0.5, 2.0)
        bias_free = rng.uniform(0.0, 0.5)
        chain = _grow(sequence, dimension, rng, bias_hh, bias_free)
        if chain:
            e = _energy(sequence, chain)
            init_chains.append((e, chain))
            if e < best_energy:
                best_energy = e
                best_chain = chain[:]
        
        if len(init_chains) > 20:
            break

    if not init_chains:
        return [[i, 0] for i in range(n)]
    
    init_chains.sort(key=lambda x: x[0])
    
    # Phase 2: Iterated Local Search on top candidates
    num_runs = min(3, len(init_chains))
    time_per_run = _remaining_time() * 0.8 / num_runs
    
    for i in range(num_runs):
        if _remaining_time() < 1.0:
            break
        
        e, c = init_chains[i]
        c_copy = c[:]
        
        # Initial local search
        _local_search(sequence, c_copy, rng, max_iters=2000)
        e = _energy(sequence, c_copy)
        if e < best_energy:
            best_energy = e
            best_chain = c_copy[:]
        
        current_chain = c_copy[:]
        current_energy = e
        deadline = time.time() + time_per_run
        
        while time.time() < deadline:
            # Perturb
            candidate = current_chain[:]
            if len(candidate) > 5:
                if rng.random() < 0.5:
                    a = rng.randint(1, len(candidate)-6)
                    b = rng.randint(a+4, len(candidate)-2)
                    _apply_move_reversal(candidate, a, b)
                else:
                    i = rng.randint(1, len(candidate)-2)
                    k = rng.randint(1, 3)
                    _apply_move_pivot(candidate, i, k)
            
            # Local search
            _local_search(sequence, candidate, rng, max_iters=500)
            cand_energy = _energy(sequence, candidate)
            
            # Acceptance
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_chain = candidate[:]
                current_chain = candidate[:]
                current_energy = cand_energy
            elif cand_energy < current_energy:
                current_chain = candidate[:]
                current_energy = cand_energy
            else:
                # Restart-like acceptance
                if rng.random() < 0.1:
                    current_chain = candidate[:]
                    current_energy = cand_energy

    if best_chain is None:
        best_chain = init_chains[0][1]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
