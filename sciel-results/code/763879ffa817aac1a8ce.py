"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math

_GLOBAL_DEADLINE = None


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy as minus the number of non-adjacent H-H contacts."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng, temperature=1.0):
    """Randomized greedy growth with simulated annealing-like acceptance."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        candidates = []
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
            score = gain + 0.1 * free + rng.random() * temperature
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Pick best with some randomness
        candidates.sort(reverse=True)
        if rng.random() < 0.8:
            site = candidates[0][1]
        elif len(candidates) > 1:
            site = candidates[1][1]
        else:
            site = candidates[0][1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain


def _heuristic_grow(sequence, dimension, rng):
    """
    More sophisticated growth that considers future potential.
    """
    n = len(sequence)
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            potential = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q not in occupied:
                        potential += 0.5
            
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            trap_score = 0
            if free < 2:
                trap_score = 1.0
            
            score = gain * 2.0 + potential * 0.5 + 0.1 * free - trap_score + rng.random() * 0.3
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        candidates.sort(reverse=True)
        k = min(3, len(candidates))
        if rng.random() < 0.7:
            site = candidates[0][1]
        else:
            site = candidates[rng.randint(0, k - 1)][1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain


def _pivot_move(sequence, chain, i, rng):
    """
    Perform a pivot move at index i.
    Rotates the suffix chain[i:] by 90 degrees around a random axis through chain[i-1].
    Returns (new_chain, new_occupied, energy_delta) or None if invalid.
    """
    n = len(chain)
    if i <= 0 or i >= n:
        return None
    
    pivot = chain[i-1]
    suffix = chain[i:]
    
    axis = rng.randint(0, 2)
    sign = rng.choice([-1, 1])
    
    new_suffix = []
    for p in suffix:
        dx = p[0] - pivot[0]
        dy = p[1] - pivot[1]
        dz = p[2] - pivot[2]
        
        if axis == 0: # Rotate around X
            new_dx = dx
            new_dy = -sign * dz
            new_dz = sign * dy
        elif axis == 1: # Rotate around Y
            new_dx = sign * dz
            new_dy = dy
            new_dz = -sign * dx
        else: # Rotate around Z
            new_dx = -sign * dy
            new_dy = sign * dx
            new_dz = dz
            
        new_p = (pivot[0] + new_dx, pivot[1] + new_dy, pivot[2] + new_dz)
        new_suffix.append(new_p)
        
    # Check validity
    seen = set()
    for p in new_suffix:
        if p in seen:
            return None
        seen.add(p)
        
    prefix_set = set(chain[:i])
    for p in new_suffix:
        if p in prefix_set:
            return None
            
    new_chain = chain[:i] + new_suffix
    
    # Compute energy delta
    old_occupied = {p: idx for idx, p in enumerate(chain)}
    new_occupied = {p: idx for idx, p in enumerate(new_chain)}
    
    # Count contacts in old state involving moved residues
    old_contacts = 0
    for idx in range(i, n):
        if sequence[idx] == 'H':
            p = chain[idx]
            for q in _neighbours(p):
                j = old_occupied.get(q)
                if j is not None and j != idx and abs(j - idx) > 1 and sequence[j] == 'H':
                    old_contacts += 1
                    
    # Count contacts in new state involving moved residues
    new_contacts = 0
    for idx in range(i, n):
        if sequence[idx] == 'H':
            p = new_chain[idx]
            for q in _neighbours(p):
                j = new_occupied.get(q)
                if j is not None and j != idx and abs(j - idx) > 1 and sequence[j] == 'H':
                    new_contacts += 1
                    
    delta = new_contacts - old_contacts
    
    return new_chain, new_occupied, delta


def _segment_move(sequence, chain, rng, occupied):
    """
    Try to rotate a segment of the chain. Pick a random segment [i, j]
    and try to rotate it by 90 or 180 degrees around a pivot.
    Returns (new_chain, new_occupied, energy_delta) or None.
    """
    n = len(chain)
    if n < 4:
        return None
    
    seg_len = rng.choice([2, 3, 4, 5])
    if n <= seg_len + 2:
        seg_len = max(2, n - 2)
    
    i = rng.randint(1, n - seg_len - 1)
    j = i + seg_len - 1
    
    prev_pos = chain[i-1]
    next_pos = chain[j+1]
    
    seg_positions = [(chain[k][0] - chain[i][0], chain[k][1] - chain[i][1], chain[k][2] - chain[i][2]) for k in range(i, j+1)]
    
    old_first = chain[i]
    possible_new_firsts = []
    for site in _neighbours(prev_pos):
        if site != old_first:
            conflict = False
            for k in range(i+1, j+1):
                new_k = (site[0] + seg_positions[k-i][0], site[1] + seg_positions[k-i][1], site[2] + seg_positions[k-i][2])
                occ = occupied.get(new_k)
                if occ is not None and occ != k:
                    conflict = True
                    break
            if not conflict:
                new_last = (site[0] + seg_positions[seg_len-1][0], site[1] + seg_positions[seg_len-1][1], site[2] + seg_positions[seg_len-1][2])
                if new_last != next_pos:
                    diff = abs(new_last[0] - next_pos[0]) + abs(new_last[1] - next_pos[1]) + abs(new_last[2] - next_pos[2])
                    if diff == 1:
                        possible_new_firsts.append(site)
    
    if not possible_new_firsts:
        return None
    
    new_first = rng.choice(possible_new_firsts)
    
    new_chain = list(chain)
    for k in range(i, j+1):
        new_chain[k] = (new_first[0] + seg_positions[k-i][0], new_first[1] + seg_positions[k-i][1], new_first[2] + seg_positions[k-i][2])
    
    seen = set()
    for k, pos in enumerate(new_chain):
        if pos in seen:
            return None
        seen.add(pos)
    
    old_occupied = dict(occupied)
    new_occupied = dict(occupied)
    for k in range(i, j+1):
        del new_occupied[chain[k]]
        new_occupied[new_chain[k]] = k
    
    old_energy = _energy(sequence, chain)
    new_energy = _energy(sequence, new_chain)
    delta = old_energy - new_energy
    
    if delta <= 0:
        return None
    
    return new_chain, new_occupied, delta


def _local_search(sequence, chain, rng, deadline):
    """
    Monte Carlo local search using pivot moves and segment moves.
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    current_occupied = {p: i for i, p in enumerate(current_chain)}
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    temperature = 2.0
    cooling_rate = 0.999
    
    while time.time() < deadline and temperature > 0.1:
        move_type = rng.random()
        
        if move_type < 0.7:
            i = rng.randint(1, n - 1)
            result = _pivot_move(sequence, current_chain, i, rng)
        else:
            result = _segment_move(sequence, current_chain, rng, current_occupied)
            
        if result is not None:
            new_chain, new_occupied, delta = result
            # delta is change in contacts (positive = better)
            # Energy change is -delta
            if delta > 0:
                current_chain = new_chain
                current_energy -= delta
                current_occupied = new_occupied
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
            else:
                if delta > -50:
                    if rng.random() < math.exp(delta / temperature):
                        current_chain = new_chain
                        current_energy -= delta
                        current_occupied = new_occupied
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(current_chain)
        
        temperature *= cooling_rate
        
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Find a good conformation for the given HP sequence.
    Returns a list of lattice points (lists of integers).
    """
    global _GLOBAL_DEADLINE
    if _GLOBAL_DEADLINE is None:
        _GLOBAL_DEADLINE = time.time() + 330.0
    
    start_time = time.time()
    remaining = _GLOBAL_DEADLINE - start_time
    if remaining < 1.0:
        n = len(sequence)
        return [[i, 0, 0] for i in range(n)]
    
    budget = min(remaining * 0.95, 20.0)
    deadline = start_time + budget
    
    n = len(sequence)
    rng = random.Random(12345 + len(sequence) + int(time.time() * 1000) % 10000)
    
    best_chain = None
    best_energy = float('inf')
    
    n_grows = max(2, int((deadline - time.time()) * 3))
    n_grows = min(n_grows, 40)
    
    grow_deadline = time.time() + (deadline - time.time()) * 0.25
    
    for _ in range(n_grows):
        if time.time() > grow_deadline:
            break
        
        strategy = rng.random()
        if strategy < 0.4:
            chain = _greedy_grow(sequence, dimension, rng, temperature=1.0)
        elif strategy < 0.7:
            chain = _heuristic_grow(sequence, dimension, rng)
        else:
            chain = _greedy_grow(sequence, dimension, rng, temperature=2.0)
        
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        best_chain = [[i, 0, 0] for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    if time.time() < deadline - 1.0:
        best_chain, best_energy = _local_search(sequence, best_chain, rng, deadline - 1.0)
    
    return best_chain

# EVOLVE-BLOCK-END
