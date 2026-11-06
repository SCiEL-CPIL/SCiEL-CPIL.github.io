"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import sys
import math
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


def _neighbours_tuple(point):
    """Return tuple of 4 neighbours of a 2D lattice point."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


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
    """Build a set of all H-H contacts and total count."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours_tuple(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts.add(frozenset((i, j)))
    return contacts, len(contacts)


def _energy_delta(sequence, chain, old_pos, new_pos, idx):
    """Compute energy delta when moving residue idx from old_pos to new_pos."""
    where = {p: i for i, p in enumerate(chain)}
    delta = 0
    if sequence[idx] == "H":
        for q in _neighbours_tuple(old_pos):
            j = where.get(q)
            if j is not None and j != idx and sequence[j] == "H":
                if j > idx + 1 or j < idx - 1:
                    delta += 1
        for q in _neighbours_tuple(new_pos):
            j = where.get(q)
            if j is not None and j != idx and sequence[j] == "H":
                if j > idx + 1 or j < idx - 1:
                    delta -= 1
    return delta


def _spiral_fold(sequence, dimension):
    """Generate a spiral conformation as a strong initial guess."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_idx = 0
    step_count = 0
    steps_needed = 1
    
    for i in range(1, n):
        placed = False
        for attempt in range(100):
            dx, dy = dirs[dir_idx % 4]
            new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
            
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied.add(new_pos)
                placed = True
                step_count += 1
                if step_count >= steps_needed:
                    step_count = 0
                    dir_idx += 1
                    if dir_idx % 4 == 0:
                        steps_needed += 1
                break
            else:
                dir_idx += 1
                if dir_idx >= 16:
                    break
        
        if not placed:
            # Fallback: try to find any empty neighbor
            found = False
            for q in _neighbours_tuple(chain[-1]):
                if q not in occupied:
                    chain.append(q)
                    occupied.add(q)
                    found = True
                    break
            if not found:
                chain.append((chain[-1][0] + 1, chain[-1][1]))
                occupied.add((chain[-1][0] + 1, chain[-1][1]))
    
    return chain


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


def _grow_constrained(sequence, dimension, rng, bias=1.0, compactness=0.5):
    """Greedy growth with additional compactness preference to encourage dense packing."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        min_x = min(p[0] for p in chain)
        max_x = max(p[0] for p in chain)
        min_y = min(p[1] for p in chain)
        max_y = max(p[1] for p in chain)
        
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            
            # Compactness: prefer sites closer to the center of mass
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            dist_to_center = abs(site[0] - cx) + abs(site[1] - cy)
            
            # Penalize expanding the bounding box
            box_penalty = 0
            if site[0] < min_x or site[0] > max_x or site[1] < min_y or site[1] > max_y:
                box_penalty = 1.0
            
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * bias + compactness * (-dist_to_center) - box_penalty + 0.1 * free + rng.random()
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
    """Slither move: move the last residue to a random empty neighbor of its current position."""
    if len(chain) < 3:
        return None
    
    last_idx = len(chain) - 1
    last_pos = chain[last_idx]
    chain_set = set(chain)
    
    empty_neighbors = [q for q in _neighbours_tuple(last_pos) if q not in chain_set]
    
    if not empty_neighbors:
        return None
    
    new_pos = rng.choice(empty_neighbors)
    new_chain = chain[:-1] + [new_pos]
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


def _local_search(sequence, chain, max_iterations=1000, time_limit=5.0, use_crankshaft=True):
    """Simulated annealing local search to improve a conformation."""
    if chain is None:
        return None
    
    current_chain = list(chain)
    current_contacts, n_contacts = _build_contact_map(sequence, current_chain)
    current_energy = -n_contacts
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    temperature = 2.0
    cooling = 0.995
    
    rng = random.Random(hash(tuple(current_chain)) % (2**32))
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break
        
        move_type = rng.random()
        if use_crankshaft and move_type < 0.25:
            new_chain = _crankshaft_move(current_chain, rng)
        elif move_type < 0.5:
            new_chain = _slither_move(current_chain, rng)
        else:
            new_chain = _pivot_move(current_chain, rng)
        
        if new_chain is None:
            continue
        
        new_contacts, new_n_contacts = _build_contact_map(sequence, new_chain)
        new_energy = -new_n_contacts
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 0.1)):
            current_chain = new_chain
            current_contacts = new_contacts
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        temperature *= cooling
    
    # Deterministic polish pass: accept only strictly improving moves
    polish_chain = list(best_chain)
    polish_contacts, polish_n = _build_contact_map(sequence, polish_chain)
    polish_energy = -polish_n
    polish_time_limit = min(2.0, max(0.5, time_limit * 0.2))
    polish_start = time.time()
    
    for _ in range(200):
        if time.time() - polish_start > polish_time_limit:
            break
        if _time_left() < 8:
            break
        
        improved = False
        for idx in range(1, len(polish_chain) - 1):
            old_pos = polish_chain[idx]
            for new_pos in _neighbours_tuple(old_pos):
                if new_pos in set(polish_chain):
                    continue
                if idx > 0:
                    prev = polish_chain[idx - 1]
                    if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
                        continue
                if idx < len(polish_chain) - 1:
                    nxt = polish_chain[idx + 1]
                    if abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) != 1:
                        continue
                
                delta = _energy_delta(sequence, polish_chain, old_pos, new_pos, idx)
                if delta < 0:
                    polish_chain[idx] = new_pos
                    polish_energy += delta
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    if polish_energy < best_energy:
        best_chain = polish_chain
        best_energy = polish_energy
    
    return best_chain


def _reheat_and_polish(sequence, best_chain, best_energy, time_budget=3.0):
    """A final high-precision polish pass with multiple restarts from the best solution."""
    if _time_left() < 5:
        return best_chain, best_energy
    
    start_time = time.time()
    rng = random.Random(99999)
    
    # Multiple short SA runs from the best solution with different temperature schedules
    for restart in range(3):
        if _time_left() < 5:
            break
        if time.time() - start_time > time_budget:
            break
        
        # Start from the best solution
        current_chain = list(best_chain)
        current_energy = best_energy
        
        # Use a lower initial temperature for more focused search
        temperature = 0.5 + 0.3 * restart
        cooling = 0.998
        max_iters = 500
        
        for iteration in range(max_iters):
            if _time_left() < 5:
                break
            if time.time() - start_time > time_budget:
                break
            
            move_type = rng.random()
            if move_type < 0.3:
                new_chain = _crankshaft_move(current_chain, rng)
            elif move_type < 0.5:
                new_chain = _slither_move(current_chain, rng)
            else:
                new_chain = _pivot_move(current_chain, rng)
            
            if new_chain is None:
                continue
            
            new_contacts, new_n_contacts = _build_contact_map(sequence, new_chain)
            new_energy = -new_n_contacts
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 0.05)):
                current_chain = new_chain
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
            
            temperature *= cooling
        
        # Quick deterministic polish
        polish_chain = list(best_chain)
        polish_energy = best_energy
        polish_start = time.time()
        
        for _ in range(50):
            if time.time() - polish_start > 0.5:
                break
            if _time_left() < 4:
                break
            
            improved = False
            for idx in range(1, len(polish_chain) - 1):
                old_pos = polish_chain[idx]
                for new_pos in _neighbours_tuple(old_pos):
                    if new_pos in set(polish_chain):
                        continue
                    if idx > 0:
                        prev = polish_chain[idx - 1]
                        if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
                            continue
                    if idx < len(polish_chain) - 1:
                        nxt = polish_chain[idx + 1]
                        if abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) != 1:
                            continue
                    
                    delta = _energy_delta(sequence, polish_chain, old_pos, new_pos, idx)
                    if delta < 0:
                        polish_chain[idx] = new_pos
                        polish_energy += delta
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        if polish_energy < best_energy:
            best_chain = polish_chain
            best_energy = polish_energy
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using multiple strategies: spiral + greedy growth + local search."""
    seq_len = len(sequence)
    
    # Adjust strategy based on sequence length
    if seq_len <= 30:
        num_grow = 600
        local_search_iters = 3000
        local_search_time = 12.0
        num_ls_chains = 35
    elif seq_len <= 50:
        num_grow = 450
        local_search_iters = 2500
        local_search_time = 10.0
        num_ls_chains = 30
    elif seq_len <= 70:
        num_grow = 300
        local_search_iters = 1800
        local_search_time = 7.0
        num_ls_chains = 25
    else:
        num_grow = 200
        local_search_iters = 1200
        local_search_time = 5.0
        num_ls_chains = 18
    
    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0
    
    # First: try spiral fold as a strong initial guess
    spiral_chain = _spiral_fold(sequence, dimension)
    if spiral_chain is not None and len(spiral_chain) == seq_len:
        spiral_energy = _energy(sequence, spiral_chain)
        if spiral_energy < best_energy:
            best_energy = spiral_energy
            best_chain = spiral_chain
    
    # Second pass: greedy growth with varying bias and compactness
    for i in range(num_grow):
        if _time_left() < 25:
            break
        
        if i < num_grow // 5:
            bias = 1.0
            compactness = 0.3
        elif i < 2 * num_grow // 5:
            bias = 1.5
            compactness = 0.5
        elif i < 3 * num_grow // 5:
            bias = 0.5
            compactness = 0.7
        elif i < 4 * num_grow // 5:
            bias = 2.0
            compactness = 0.4
        else:
            bias = 1.2
            compactness = 0.8
        
        # Alternate between standard and constrained growth
        if i % 2 == 0:
            chain = _grow(sequence, dimension, rng, bias=bias)
        else:
            chain = _grow_constrained(sequence, dimension, rng, bias=bias, compactness=compactness)
        
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain
    
    # Third pass: local search on best chains
    if best_chain is not None and _time_left() > 35:
        improved = _local_search(sequence, best_chain, max_iterations=local_search_iters, 
                                  time_limit=local_search_time, use_crankshaft=True)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved
        
        # Try local search on a few other good chains
        for i in range(num_ls_chains):
            if _time_left() < 30:
                break
            
            if i % 2 == 0:
                chain = _grow(sequence, dimension, rng, bias=1.0 + 0.5 * rng.random())
            else:
                chain = _grow_constrained(sequence, dimension, rng, bias=1.0 + 0.5 * rng.random(),
                                           compactness=0.5 + 0.3 * rng.random())
            
            if chain is None:
                continue
            
            value = _energy(sequence, chain)
            if value < best_energy + 4:
                improved = _local_search(sequence, chain, max_iterations=local_search_iters // 2,
                                          time_limit=local_search_time / 2, use_crankshaft=True)
                if improved is not None:
                    improved_energy = _energy(sequence, improved)
                    if improved_energy < best_energy:
                        best_energy = improved_energy
                        best_chain = improved
    
    # Final reheat and polish pass
    if best_chain is not None and _time_left() > 10:
        best_chain, best_energy = _reheat_and_polish(sequence, best_chain, best_energy, 
                                                       time_budget=min(3.0, max(1.0, _time_left() * 0.1)))
    
    # Final fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(seq_len)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
