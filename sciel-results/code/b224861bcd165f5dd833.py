"""HP lattice protein folding on the cubic lattice.
Combines greedy growth with pivot moves, tail re-growth, and simulated annealing.
"""
import random
import time
import math

# Global deadline: evaluator kills at 330s wall-clock. Use 315s safety margin.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 315.0


def _neighbours(point):
    """Yield the 6 face-neighbours of point in 3D."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _build_contact_map(sequence, chain):
    """
    Build a set of (i, j) pairs where i < j, chain[i] and chain[j] are
    H residues on neighbouring lattice sites, and |i - j| > 1.
    Also returns a dict mapping position -> index for O(1) lookup.
    """
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts.add((i, j))
    return contacts, where


def _energy_from_contacts(contacts):
    """Energy is negative of the number of H-H contacts."""
    return -len(contacts)


def _pivot_move(sequence, chain, i, rng):
    """
    Try a pivot move at index i: rotate the tail from i to end by 90 degrees
    around the axis defined by chain[i-1] -> chain[i].
    Returns (new_chain, new_contacts, new_where, new_energy) or (None, None, None, None).
    """
    n = len(chain)
    if i < 1 or i >= n:
        return None, None, None, None
    
    ax = chain[i][0] - chain[i-1][0]
    ay = chain[i][1] - chain[i-1][1]
    az = chain[i][2] - chain[i-1][2]
    
    cx, cy, cz = chain[i-1]
    
    new_tail = []
    for k in range(i, n):
        px, py, pz = chain[k]
        dx = px - cx
        dy = py - cy
        dz = pz - cz
        
        if ax != 0:
            new_dx = dx
            new_dy = -dz
            new_dz = dy
        elif ay != 0:
            new_dx = dz
            new_dy = dy
            new_dz = -dx
        else:
            new_dx = -dy
            new_dy = dx
            new_dz = dz
        
        new_tail.append((cx + new_dx, cy + new_dy, cz + new_dz))
    
    new_chain = chain[:i] + new_tail
    
    prefix_occupied = {p: idx for idx, p in enumerate(chain[:i])}
    for p in new_tail:
        if p in prefix_occupied:
            return None, None, None, None
    
    p_prev = chain[i-1]
    p_new = new_tail[0]
    if abs(p_prev[0] - p_new[0]) + abs(p_prev[1] - p_new[1]) + abs(p_prev[2] - p_new[2]) != 1:
        return None, None, None, None
    
    for k in range(len(new_tail) - 1):
        p1 = new_tail[k]
        p2 = new_tail[k+1]
        if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2]) != 1:
            return None, None, None, None
    
    tail_set = set(new_tail)
    if len(tail_set) != len(new_tail):
        return None, None, None, None
    
    new_where = {p: idx for idx, p in enumerate(new_chain)}
    new_contacts = set()
    for i_idx, p in enumerate(new_chain):
        if sequence[i_idx] == "H":
            for q in _neighbours(p):
                j_idx = new_where.get(q)
                if j_idx is not None and j_idx > i_idx + 1 and sequence[j_idx] == "H":
                    new_contacts.add((i_idx, j_idx))
    
    new_energy = _energy_from_contacts(new_contacts)
    return new_chain, new_contacts, new_where, new_energy


def _grow(sequence, dimension, rng):
    """
    One randomized greedy chain growth.
    """
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain + 0.1 * free + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        best_score, best_site = max(options)
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain


def _regrow_tail(sequence, chain, i, rng):
    """
    Re-grow the tail from index i onwards.
    Returns (new_chain, new_contacts, new_where, new_energy) or (None, None, None, None).
    """
    prefix = chain[:i]
    occupied = {p: idx for idx, p in enumerate(prefix)}
    new_chain = list(prefix)
    
    for k in range(i, len(sequence)):
        last = new_chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[k] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < k - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain + 0.1 * free + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None, None, None, None
        
        best_score, best_site = max(options)
        new_chain.append(best_site)
        occupied[best_site] = k
    
    new_contacts, new_where = _build_contact_map(sequence, new_chain)
    new_energy = _energy_from_contacts(new_contacts)
    return new_chain, new_contacts, new_where, new_energy


def fold(sequence, dimension):
    """
    Fold the sequence on the cubic lattice.
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    time_remaining = _DEADLINE - time.time()
    if time_remaining < 1.0:
        return [[i, 0, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    # Phase 1: Multiple greedy growths
    phase1_time_budget = time_remaining * 0.3
    est_growth_time = n * 0.002
    num_growths = max(5, int(phase1_time_budget / est_growth_time))
    num_growths = min(num_growths, 2000)
    
    for _ in range(num_growths):
        if time.time() > _DEADLINE - 15:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        contacts, where = _build_contact_map(sequence, chain)
        energy = _energy_from_contacts(contacts)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = 0
    
    # Phase 2: Local optimization
    current_chain = list(best_chain)
    current_contacts, current_where = _build_contact_map(sequence, current_chain)
    current_energy = _energy_from_contacts(current_contacts)
    
    time_remaining = _DEADLINE - time.time()
    if time_remaining < 1.0:
        return [list(p) for p in best_chain]
    
    # Simulated annealing parameters
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < _DEADLINE - 5:
        if rng.random() < 0.7:
            i = rng.randint(1, n - 1)
            new_chain, new_contacts, new_where, new_energy = _pivot_move(
                sequence, current_chain, i, rng)
        else:
            if rng.random() < 0.5:
                i = rng.randint(1, max(1, n // 2))
            else:
                i = rng.randint(n // 2, n - 1)
            i = min(i, n - 1)
            new_chain, new_contacts, new_where, new_energy = _regrow_tail(
                sequence, current_chain, i, rng)
        
        if new_chain is None:
            continue
        
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or (T > T_min and rng.random() < min(1.0, math.exp(-delta_e / T))):
            current_chain = new_chain
            current_contacts = new_contacts
            current_where = new_where
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        T *= cooling
    
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
