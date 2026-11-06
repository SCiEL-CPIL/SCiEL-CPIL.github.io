"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import sys
from collections import defaultdict

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 195.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute the HP contact energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _build_contact_map(sequence, chain):
    """Build a set of all H-H contacts (as frozensets of index pairs) and total count."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts.add(frozenset((i, j)))
    return contacts, len(contacts)


def _energy_from_contacts(contacts):
    """Energy is minus the number of H-H contacts."""
    return -len(contacts)


def _grow(sequence, dimension, rng, bias=1.0):
    """One randomized greedy chain growth: prefer sites with new H-H contacts."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * bias + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move(chain, rng):
    """Perform a single pivot move: rotate a subchain by 90, 180, or 270 degrees around a pivot point."""
    if len(chain) < 4:
        return None
    
    # Choose a random pivot index (not the first or last)
    pivot_idx = rng.randint(1, len(chain) - 2)
    pivot = chain[pivot_idx]
    
    # Get the segment to rotate (from pivot_idx to end)
    segment = chain[pivot_idx:]
    
    # Precompute fixed part positions for fast collision check
    fixed_set = set(chain[:pivot_idx])
    
    # Try different rotations
    for angle in [90, -90, 180, 270]:
        # Rotate segment around pivot
        rotated = []
        valid = True
        for p in segment:
            dx, dy = p[0] - pivot[0], p[1] - pivot[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            elif angle == 180:
                ndx, ndy = -dx, -dy
            else:  # 270
                ndx, ndy = dy, -dx
            new_pos = (pivot[0] + ndx, pivot[1] + ndy)
            # Check collision with fixed part (exclude pivot itself)
            if new_pos in fixed_set and new_pos != pivot:
                valid = False
                break
            rotated.append(new_pos)
        
        if not valid:
            continue
        
        # Check for self-collisions within the rotated segment
        rotated_set = set(rotated)
        if len(rotated_set) != len(rotated):
            continue
        
        # Check unit steps between rotated segment and fixed part, and within rotated
        # Check connection at pivot
        if pivot_idx > 0:
            prev_pos = chain[pivot_idx - 1]
            dx = abs(rotated[0][0] - prev_pos[0])
            dy = abs(rotated[0][1] - prev_pos[1])
            if dx + dy != 1:
                continue
        
        # Check unit steps within rotated segment
        for i in range(1, len(rotated)):
            dx = abs(rotated[i][0] - rotated[i-1][0])
            dy = abs(rotated[i][1] - rotated[i-1][1])
            if dx + dy != 1:
                valid = False
                break
        if not valid:
            continue
        
        new_chain = chain[:pivot_idx] + rotated
        return new_chain
    
    return None


def _slither_move(chain, rng):
    """Perform a slither move: move the last residue to an adjacent free site."""
    n = len(chain)
    if n < 3:
        return None
    
    last = chain[-1]
    occupied = set(chain[:-1])  # All except last
    
    options = []
    for site in _neighbours(last):
        if site in occupied:
            continue
        # Check if moving last here maintains unit step from second-to-last
        prev = chain[-2]
        dx = abs(site[0] - prev[0])
        dy = abs(site[1] - prev[1])
        if dx + dy == 1:
            options.append(site)
    
    if not options:
        return None
    
    new_last = rng.choice(options)
    new_chain = chain[:-1] + [new_last]
    return new_chain


def _crankshaft_move(chain, rng):
    """Perform a crankshaft move: rotate a short segment (length 2-3) around two pivot points."""
    n = len(chain)
    if n < 5:
        return None
    
    # Choose segment length
    seg_len = rng.choice([2, 3])
    
    # Choose start index such that segment fits
    max_start = n - seg_len - 2
    if max_start < 1:
        return None
    start = rng.randint(1, max_start)
    
    # The segment is chain[start:start+seg_len]
    # Pivot points are chain[start-1] and chain[start+seg_len]
    pivot1 = chain[start - 1]
    pivot2 = chain[start + seg_len]
    
    segment = chain[start:start + seg_len]
    
    # Try rotating the entire segment around pivot1
    for angle in [90, -90, 180]:
        rotated = []
        for p in segment:
            dx, dy = p[0] - pivot1[0], p[1] - pivot1[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            else:  # 180
                ndx, ndy = -dx, -dy
            rotated.append((pivot1[0] + ndx, pivot1[1] + ndy))
        
        # Check if the last point connects to pivot2
        if rotated[-1] != pivot2:
            continue
        
        # Check self-avoidance with the rest of the chain
        new_chain = chain[:start] + rotated + chain[start + seg_len:]
        new_occupied = set(new_chain)
        if len(new_occupied) == len(new_chain):
            # Check unit steps
            valid = True
            for i in range(1, len(new_chain)):
                dx = abs(new_chain[i][0] - new_chain[i-1][0])
                dy = abs(new_chain[i][1] - new_chain[i-1][1])
                if dx + dy != 1:
                    valid = False
                    break
            if valid:
                return new_chain
    
    return None


def _local_search(sequence, chain, max_iterations=1000, time_limit=5.0, use_crankshaft=True):
    """Simulated annealing local search to improve a conformation with incremental energy."""
    if chain is None:
        return None
    
    current_chain = list(chain)
    current_contacts, n_contacts = _build_contact_map(sequence, current_chain)
    current_energy = -n_contacts
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    temperature = 3.0
    cooling = 0.998
    
    rng = random.Random(hash(tuple(current_chain)) % (2**32))
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break
        
        # Choose move type
        r = rng.random()
        if use_crankshaft and r < 0.2:
            new_chain = _crankshaft_move(current_chain, rng)
        elif r < 0.5:
            new_chain = _slither_move(current_chain, rng)
        else:
            new_chain = _pivot_move(current_chain, rng)
        
        if new_chain is None:
            continue
        
        # Recompute energy for the new chain
        new_contacts, new_n_contacts = _build_contact_map(sequence, new_chain)
        new_energy = -new_n_contacts
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < pow(2.718281828, -delta / max(temperature, 0.1)):
            current_chain = new_chain
            current_contacts = new_contacts
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        temperature *= cooling
    
    return best_chain


def _polish(sequence, chain, max_iterations=500, time_limit=3.0):
    """Deterministic polish: try all single-site moves and accept improvements."""
    if chain is None:
        return None
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    improved = True
    
    while improved and time.time() - start_time < time_limit and _time_left() > 15:
        improved = False
        
        for i in range(1, len(current_chain) - 1):
            if time.time() - start_time > time_limit or _time_left() < 15:
                break
            
            pos = current_chain[i]
            occupied = set(current_chain)
            occupied.discard(pos)
            
            for site in _neighbours(pos):
                if site in occupied:
                    continue
                
                # Check unit step constraints
                prev = current_chain[i - 1]
                nxt = current_chain[i + 1]
                if abs(site[0] - prev[0]) + abs(site[1] - prev[1]) != 1:
                    continue
                if abs(site[0] - nxt[0]) + abs(site[1] - nxt[1]) != 1:
                    continue
                
                new_chain = current_chain[:i] + [site] + current_chain[i + 1:]
                new_energy = _energy(sequence, new_chain)
                
                if new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    break
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using multiple strategies: greedy growth + local search + polish."""
    seq_len = len(sequence)
    
    # Adjust strategy based on sequence length
    if seq_len <= 30:
        num_grow = 400
        local_search_iters = 3000
        local_search_time = 10.0
        num_ls_chains = 30
    elif seq_len <= 50:
        num_grow = 300
        local_search_iters = 2000
        local_search_time = 8.0
        num_ls_chains = 25
    elif seq_len <= 70:
        num_grow = 200
        local_search_iters = 1500
        local_search_time = 6.0
        num_ls_chains = 20
    else:
        num_grow = 150
        local_search_iters = 1000
        local_search_time = 4.0
        num_ls_chains = 15
    
    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0
    
    # First pass: greedy growth with varying bias
    for i in range(num_grow):
        if _time_left() < 20:
            break
        
        # Vary the bias to explore different strategies
        if i < num_grow // 3:
            bias = 1.0
        elif i < 2 * num_grow // 3:
            bias = 1.5
        else:
            bias = 0.5
        
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain
    
    # Second pass: local search on best chains
    if best_chain is not None and _time_left() > 30:
        # Try local search on the best chain
        improved = _local_search(sequence, best_chain, max_iterations=local_search_iters, 
                                  time_limit=local_search_time, use_crankshaft=True)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved
        
        # Try local search on a few other good chains
        for i in range(num_ls_chains):
            if _time_left() < 25:
                break
            
            chain = _grow(sequence, dimension, rng, bias=1.0 + 0.5 * rng.random())
            if chain is None:
                continue
            
            value = _energy(sequence, chain)
            if value < best_energy + 4:  # Only improve chains close to best
                improved = _local_search(sequence, chain, max_iterations=local_search_iters // 2,
                                          time_limit=local_search_time / 2, use_crankshaft=True)
                if improved is not None:
                    improved_energy = _energy(sequence, improved)
                    if improved_energy < best_energy:
                        best_energy = improved_energy
                        best_chain = improved
    
    # Third pass: deterministic polish
    if best_chain is not None and _time_left() > 15:
        polished = _polish(sequence, best_chain, max_iterations=500, time_limit=3.0)
        if polished is not None:
            polished_energy = _energy(sequence, polished)
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_chain = polished
    
    # Final fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(seq_len)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
