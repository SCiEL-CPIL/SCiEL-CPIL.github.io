"""Improved HP lattice protein folding using A*-guided search with admissible heuristic."""
# EVOLVE-BLOCK-START
import heapq
import time


def _neighbours(point):
    """Yield all 2*dimension neighbours of a lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the HP energy (negative number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _a_star_fold(sequence, dimension, time_budget=25.0):
    """
    A* search over partial chains.
    State: (chain_tuple, occupied_frozenset, contacts, h_list)
    h_list: list of (position, index) for H residues placed so far.
    Priority: -(contacts + admissible_heuristic)
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]
    
    start = (0,) * dimension
    counter = 0
    start_time = time.time()
    
    initial_chain = (start,)
    initial_occupied = frozenset([start])
    initial_contacts = 0
    initial_h = [(start, 0)] if sequence[0] == 'H' else []
    
    remaining_h = sum(1 for i in range(1, n) if sequence[i] == 'H')
    placed_h_count = len(initial_h)
    h0 = remaining_h * min(placed_h_count, 4)
    initial_priority = -(initial_contacts + h0)
    
    open_heap = [(initial_priority, counter, initial_chain, initial_occupied, initial_contacts, initial_h)]
    counter += 1
    
    best_chain = None
    best_contacts = -1
    nodes_expanded = 0
    
    while open_heap:
        if time.time() - start_time > time_budget:
            break
        
        priority, _, chain, occupied, contacts, h_list = heapq.heappop(open_heap)
        current_index = len(chain) - 1
        
        if current_index == n - 1:
            if contacts > best_contacts:
                best_contacts = contacts
                best_chain = chain
            continue
        
        nodes_expanded += 1
        
        last = chain[-1]
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            new_occupied = occupied | {site}
            new_contacts = contacts
            new_h = h_list
            
            if sequence[current_index + 1] == 'H':
                for (ph, pidx) in h_list:
                    dist = sum(abs(a - b) for a, b in zip(site, ph))
                    if dist == 1 and abs(pidx - (current_index + 1)) > 1:
                        new_contacts += 1
                new_h = h_list + [(site, current_index + 1)]
            
            remaining_h = sum(1 for i in range(current_index + 2, n) if sequence[i] == 'H')
            placed_h_count = len(new_h)
            h = remaining_h * min(placed_h_count, 4)
            priority = -(new_contacts + h)
            
            new_chain = chain + (site,)
            heapq.heappush(open_heap, (priority, counter, new_chain, new_occupied, new_contacts, new_h))
            counter += 1
    
    if best_chain is not None:
        return list(best_chain)
    
    return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]


def _beam_fold(sequence, dimension, beam_width=80, time_budget=25.0):
    """
    Beam search with incremental contact tracking.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]
    
    start = (0,) * dimension
    start_time = time.time()
    
    beam = [((start,), frozenset([start]), 0, [(start, 0)] if sequence[0] == 'H' else [])]
    
    for step in range(1, n):
        if time.time() - start_time > time_budget:
            break
        
        candidates = []
        
        for chain, occupied, contacts, h_list in beam:
            last = chain[-1]
            
            for site in _neighbours(last):
                if site in occupied:
                    continue
                
                new_occupied = occupied | {site}
                new_contacts = contacts
                new_h = h_list
                
                if sequence[step] == 'H':
                    for (ph, pidx) in h_list:
                        dist = sum(abs(a - b) for a, b in zip(site, ph))
                        if dist == 1 and abs(pidx - step) > 1:
                            new_contacts += 1
                    new_h = h_list + [(site, step)]
                
                new_chain = chain + (site,)
                candidates.append((new_chain, new_occupied, new_contacts, new_h))
        
        if not candidates:
            break
        
        candidates.sort(key=lambda x: -x[2])
        beam = candidates[:beam_width]
    
    if not beam:
        return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    best_chain = max(beam, key=lambda x: x[2])
    return list(best_chain[0])


def _greedy_fold(sequence, dimension):
    """Simple greedy fallback: always pick the neighbor with most H-H contacts."""
    n = len(sequence)
    start = (0,) * dimension
    chain = [start]
    pos_to_idx = {start: 0}
    
    for i in range(1, n):
        best_site = None
        best_score = -1
        
        for site in _neighbours(chain[-1]):
            if site in pos_to_idx:
                continue
            
            score = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in pos_to_idx and pos_to_idx[q] < i - 1 and sequence[pos_to_idx[q]] == 'H':
                        score += 1
            
            if score > best_score:
                best_score = score
                best_site = site
        
        if best_site is None:
            last = chain[-1]
            for j in range(i, n):
                found = False
                for axis in range(dimension):
                    for step_dir in (1, -1):
                        candidate = list(last)
                        candidate[axis] += step_dir
                        if tuple(candidate) not in pos_to_idx:
                            chain.append(tuple(candidate))
                            pos_to_idx[tuple(candidate)] = j
                            last = tuple(candidate)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    chain.append(list(last))
                    pos_to_idx[tuple(last)] = j
            break
        
        chain.append(best_site)
        pos_to_idx[best_site] = i
    
    return [list(p) for p in chain]


def fold(sequence, dimension):
    """
    Fold the sequence using A* search for short chains, beam search for longer ones.
    Returns a list of lattice points (lists of ints).
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension for _ in range(n)]
    
    try:
        if n <= 25:
            chain = _a_star_fold(sequence, dimension, time_budget=20.0)
        else:
            beam_width = max(40, min(120, 2500 // n))
            chain = _beam_fold(sequence, dimension, beam_width, time_budget=20.0)
        
        if chain is None or len(chain) != n:
            raise ValueError("Invalid chain from search")
        
        pos_set = set(tuple(p) for p in chain)
        if len(pos_set) != n:
            raise ValueError("Chain is not self-avoiding")
        
        for i in range(1, n):
            diff = sum(abs(a - b) for a, b in zip(chain[i-1], chain[i]))
            if diff != 1:
                raise ValueError("Non-unit step detected")
        
        return [list(p) for p in chain]
    
    except Exception:
        try:
            return _greedy_fold(sequence, dimension)
        except Exception:
            return [[i] + [0] * (dimension - 1) for i in range(n)]
# EVOLVE-BLOCK-END
