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


def _identify_priority_pairs(sequence):
    """
    Identify high-priority H-H pairs based on:
    1. Pairs of H residues that are far apart in sequence (long-range contacts)
    2. Pairs where both H residues have few H neighbours in sequence (isolated H's)
    Returns a list of (i, j) pairs sorted by priority.
    """
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    if len(h_indices) < 2:
        return []
    
    # Count H-H adjacency in sequence for each H
    h_set = set(h_indices)
    h_neighbour_count = {}
    for i in h_indices:
        count = 0
        if i - 1 in h_set:
            count += 1
        if i + 1 in h_set:
            count += 1
        h_neighbour_count[i] = count
    
    # Score each pair: prioritize long-range, isolated H's
    pairs = []
    for idx_i, i in enumerate(h_indices):
        for j in h_indices[idx_i+1:]:
            seq_dist = abs(i - j)
            # Isolation bonus: fewer sequence neighbours means more important contact
            iso_score = (2 - h_neighbour_count[i]) + (2 - h_neighbour_count[j])
            # Long-range bonus
            long_range_score = min(seq_dist, 30) / 30.0
            # Combined priority
            priority = long_range_score * 2.0 + iso_score * 0.5
            pairs.append((priority, i, j))
    
    pairs.sort(key=lambda x: -x[0])
    return [(i, j) for _, i, j in pairs]


def _count_priority_contacts(sequence, chain, priority_pairs):
    """Count how many priority H-H pairs are satisfied as contacts in the chain."""
    where = {p: i for i, p in enumerate(chain)}
    count = 0
    for i, j in priority_pairs:
        if sequence[i] != 'H' or sequence[j] != 'H':
            continue
        if abs(i - j) <= 1:
            continue
        pi = chain[i]
        pj = chain[j]
        # Check if they are adjacent on lattice
        if abs(pi[0] - pj[0]) + abs(pi[1] - pj[1]) == 1:
            count += 1
    return count


def _grow_with_priority(sequence, dimension, rng, bias=1.0, priority_pairs=None):
    """
    Chain growth that prioritizes placing new H residues near existing H residues
    that are part of high-priority pairs.
    """
    if priority_pairs is None:
        priority_pairs = []
    
    # Build a map: for each H index, which other H indices it should be near
    h_to_targets = defaultdict(set)
    for i, j in priority_pairs:
        h_to_targets[i].add(j)
        h_to_targets[j].add(i)
    
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            gain = 0
            priority_gain = 0
            if sequence[i] == "H":
                # Standard H-H contact gain
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
                
                # Priority gain: bonus for being near H residues we want to contact
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            if j in h_to_targets.get(i, set()):
                                priority_gain += 2.0  # Strong bonus for priority pairs
                            elif abs(i - j) > 5:
                                priority_gain += 0.5  # Smaller bonus for other long-range H-H
                
                # Also check if placing here helps future priority pairs
                # (look ahead: are there upcoming H residues that should be near this one?)
                for future_i in range(i + 2, min(i + 10, len(sequence))):
                    if sequence[future_i] == "H" and i in h_to_targets.get(future_i, set()):
                        # This H should be near future_i, so give bonus for open space
                        free_near = sum(1 for q in _neighbours(site) if q not in occupied)
                        priority_gain += 0.3 * free_near
                        break
            
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * bias + priority_gain + 0.1 * free + rng.random()
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
    
    pivot_idx = rng.randint(1, len(chain) - 2)
    pivot = chain[pivot_idx]
    
    segment = chain[pivot_idx:]
    
    fixed_set = set(chain[:pivot_idx])
    
    for angle in [90, -90, 180, 270]:
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
            else:
                ndx, ndy = dy, -dx
            new_pos = (pivot[0] + ndx, pivot[1] + ndy)
            if new_pos in fixed_set and new_pos != pivot:
                valid = False
                break
            rotated.append(new_pos)
        
        if not valid:
            continue
        
        rotated_set = set(rotated)
        if len(rotated_set) != len(rotated):
            continue
        
        if pivot_idx > 0:
            prev_pos = chain[pivot_idx - 1]
            dx = abs(rotated[0][0] - prev_pos[0])
            dy = abs(rotated[0][1] - prev_pos[1])
            if dx + dy != 1:
                continue
        
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
    occupied = set(chain[:-1])
    
    options = []
    for site in _neighbours(last):
        if site in occupied:
            continue
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
    
    seg_len = rng.choice([2, 3])
    
    max_start = n - seg_len - 2
    if max_start < 1:
        return None
    start = rng.randint(1, max_start)
    
    pivot1 = chain[start - 1]
    pivot2 = chain[start + seg_len]
    
    segment = chain[start:start + seg_len]
    
    for angle in [90, -90, 180]:
        rotated = []
        for p in segment:
            dx, dy = p[0] - pivot1[0], p[1] - pivot1[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            else:
                ndx, ndy = -dx, -dy
            rotated.append((pivot1[0] + ndx, pivot1[1] + ndy))
        
        if rotated[-1] != pivot2:
            continue
        
        new_chain = chain[:start] + rotated + chain[start + seg_len:]
        new_occupied = set(new_chain)
        if len(new_occupied) == len(new_chain):
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


def _targeted_refinement(sequence, chain, priority_pairs, max_iterations=2000, time_limit=8.0):
    """
    Targeted refinement that specifically tries to satisfy priority H-H pairs.
    Uses a hybrid approach: random moves weighted toward satisfying priority contacts.
    """
    if chain is None:
        return None
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    current_priority = _count_priority_contacts(sequence, current_chain, priority_pairs)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    best_priority = current_priority
    
    start_time = time.time()
    temperature = 5.0
    cooling = 0.999
    
    rng = random.Random(hash(tuple(current_chain)) % (2**32))
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break
        
        r = rng.random()
        if r < 0.15:
            new_chain = _crankshaft_move(current_chain, rng)
        elif r < 0.45:
            new_chain = _slither_move(current_chain, rng)
        else:
            new_chain = _pivot_move(current_chain, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        new_priority = _count_priority_contacts(sequence, new_chain, priority_pairs)
        
        # Combined score: energy + priority bonus
        old_score = current_energy + 0.3 * current_priority
        new_score = new_energy + 0.3 * new_priority
        delta_score = new_score - old_score
        
        # Accept if improvement in score, or with probability based on temperature
        if delta_score < 0 or rng.random() < pow(2.718281828, -max(delta_score, 0) / max(temperature, 0.5)):
            current_chain = new_chain
            current_energy = new_energy
            current_priority = new_priority
            
            # Track best by combined score
            if new_score < best_energy + 0.3 * best_priority:
                best_energy = new_energy
                best_priority = new_priority
                best_chain = list(new_chain)
        
        temperature *= cooling
    
    # Return best chain (by energy primarily, then priority)
    return best_chain


def _local_search(sequence, chain, max_iterations=1000, time_limit=5.0, use_crankshaft=True):
    """Simulated annealing local search to improve a conformation."""
    if chain is None:
        return None
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
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
        
        r = rng.random()
        if use_crankshaft and r < 0.2:
            new_chain = _crankshaft_move(current_chain, rng)
        elif r < 0.5:
            new_chain = _slither_move(current_chain, rng)
        else:
            new_chain = _pivot_move(current_chain, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < pow(2.718281828, -delta / max(temperature, 0.1)):
            current_chain = new_chain
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
    """
    Best conformation using constraint-satisfaction approach:
    1. Identify priority H-H pairs
    2. Grow chains biased toward satisfying priority pairs
    3. Targeted refinement focusing on priority contacts
    4. General local search and polish
    """
    seq_len = len(sequence)
    
    # Identify priority pairs
    priority_pairs = _identify_priority_pairs(sequence)
    
    # Adjust strategy based on sequence length
    if seq_len <= 30:
        num_grow = 300
        targeted_iters = 3000
        targeted_time = 12.0
        local_search_iters = 2000
        local_search_time = 8.0
        num_ls_chains = 20
    elif seq_len <= 50:
        num_grow = 250
        targeted_iters = 2000
        targeted_time = 10.0
        local_search_iters = 1500
        local_search_time = 6.0
        num_ls_chains = 15
    elif seq_len <= 70:
        num_grow = 200
        targeted_iters = 1500
        targeted_time = 8.0
        local_search_iters = 1000
        local_search_time = 5.0
        num_ls_chains = 12
    else:
        num_grow = 150
        targeted_iters = 1000
        targeted_time = 6.0
        local_search_iters = 800
        local_search_time = 4.0
        num_ls_chains = 10
    
    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0
    
    # First pass: greedy growth with priority bias
    for i in range(num_grow):
        if _time_left() < 25:
            break
        
        # Vary the bias and priority emphasis
        if i < num_grow // 4:
            bias = 1.0
        elif i < num_grow // 2:
            bias = 1.5
        elif i < 3 * num_grow // 4:
            bias = 0.7
        else:
            bias = 1.2
        
        chain = _grow_with_priority(sequence, dimension, rng, bias=bias, priority_pairs=priority_pairs)
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain
    
    # Second pass: targeted refinement on best chain
    if best_chain is not None and _time_left() > 40:
        refined = _targeted_refinement(sequence, best_chain, priority_pairs,
                                        max_iterations=targeted_iters, time_limit=targeted_time)
        if refined is not None:
            refined_energy = _energy(sequence, refined)
            if refined_energy < best_energy:
                best_energy = refined_energy
                best_chain = refined
    
    # Third pass: targeted refinement on a few other good chains
    if best_chain is not None and _time_left() > 35:
        for i in range(num_ls_chains):
            if _time_left() < 25:
                break
            
            chain = _grow_with_priority(sequence, dimension, rng, bias=1.0 + 0.3 * rng.random(),
                                         priority_pairs=priority_pairs)
            if chain is None:
                continue
            
            value = _energy(sequence, chain)
            if value < best_energy + 5:
                refined = _targeted_refinement(sequence, chain, priority_pairs,
                                                max_iterations=targeted_iters // 2,
                                                time_limit=targeted_time / 2)
                if refined is not None:
                    refined_energy = _energy(sequence, refined)
                    if refined_energy < best_energy:
                        best_energy = refined_energy
                        best_chain = refined
    
    # Fourth pass: general local search
    if best_chain is not None and _time_left() > 25:
        improved = _local_search(sequence, best_chain, max_iterations=local_search_iters,
                                  time_limit=local_search_time, use_crankshaft=True)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved
    
    # Fifth pass: deterministic polish
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
