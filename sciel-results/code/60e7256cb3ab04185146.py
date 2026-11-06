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


def _precompute_h_positions(sequence):
    """Precompute indices of H residues for faster heuristic computation."""
    return [i for i, c in enumerate(sequence) if c == 'H']


def _max_remaining_contacts(sequence, current_index, h_positions):
    """
    Admissible heuristic: maximum possible additional H-H contacts from residues
    current_index+1 to end.
    """
    h_count_remaining = sum(1 for i in h_positions if i > current_index)
    h_count_placed = sum(1 for i in h_positions if i <= current_index)
    # Each remaining H can contact at most 2 placed H's (conservative upper bound)
    return min(h_count_placed * h_count_remaining, h_count_remaining * 2)


def _a_star_fold(sequence, dimension, max_nodes=500000, time_limit=220):
    """
    A* search over partial chains with efficient state management.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]
    
    h_positions = _precompute_h_positions(sequence)
    start = (0,) * dimension
    counter = 0
    initial_state = (0, counter, (start,), {start: 0}, 0)
    counter += 1
    open_heap = [initial_state]
    
    best_chain = None
    best_contacts = -1
    nodes_expanded = 0
    start_time = time.time()
    
    while open_heap:
        if time.time() - start_time > time_limit:
            break
        
        priority, _, chain, pos_to_idx, contacts = heapq.heappop(open_heap)
        current_index = len(chain) - 1
        
        if current_index == n - 1:
            if contacts > best_contacts:
                best_contacts = contacts
                best_chain = chain
            continue
        
        nodes_expanded += 1
        if nodes_expanded > max_nodes:
            break
        
        last = chain[-1]
        
        for site in _neighbours(last):
            if site in pos_to_idx:
                continue
            
            new_chain = chain + (site,)
            new_pos_to_idx = dict(pos_to_idx)
            new_pos_to_idx[site] = current_index + 1
            new_contacts = contacts
            
            if sequence[current_index + 1] == 'H':
                for q in _neighbours(site):
                    if q in new_pos_to_idx:
                        j = new_pos_to_idx[q]
                        if j < current_index and sequence[j] == 'H':
                            new_contacts += 1
            
            heuristic = _max_remaining_contacts(sequence, current_index + 1, h_positions)
            priority = -(new_contacts + heuristic)
            
            heapq.heappush(open_heap, (priority, counter, new_chain, new_pos_to_idx, new_contacts))
            counter += 1
    
    if best_chain is not None:
        return list(best_chain)
    
    return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]


def _beam_fold(sequence, dimension, beam_width=100):
    """
    Optimized beam search with incremental contact tracking.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]
    
    h_positions = _precompute_h_positions(sequence)
    start = (0,) * dimension
    beam = [((start,), {start: 0}, 0)]
    
    for step in range(1, n):
        candidates = []
        
        for chain, pos_to_idx, contacts in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in pos_to_idx:
                    continue
                
                new_chain = chain + (site,)
                new_pos_to_idx = dict(pos_to_idx)
                new_pos_to_idx[site] = step
                new_contacts = contacts
                
                if sequence[step] == 'H':
                    for q in _neighbours(site):
                        if q in new_pos_to_idx:
                            j = new_pos_to_idx[q]
                            if j < step - 1 and sequence[j] == 'H':
                                new_contacts += 1
                
                candidates.append((new_chain, new_pos_to_idx, new_contacts))
        
        if not candidates:
            break
        
        candidates.sort(key=lambda x: -x[2])
        beam = candidates[:beam_width]
    
    if not beam:
        return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    best_chain = max(beam, key=lambda x: x[2])
    return list(best_chain[0])


def _local_search_refine(sequence, chain, max_iterations=100):
    """
    Simple local search: try to improve by moving individual residues.
    Only moves that maintain self-avoidance and unit steps are allowed.
    """
    n = len(sequence)
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)
    
    for _ in range(max_iterations):
        improved = False
        for i in range(1, n - 1):
            if sequence[i] != 'H':
                continue
            
            for site in _neighbours(chain[i]):
                if site == chain[i-1] or site == chain[i+1]:
                    continue
                
                # Check if site is occupied
                occupied = any(p == site for j, p in enumerate(chain) if j != i)
                if occupied:
                    continue
                
                # Try the move
                new_chain = list(chain)
                new_chain[i] = site
                
                # Verify unit steps
                valid = True
                for j in range(1, n):
                    diff = sum(abs(a - b) for a, b in zip(new_chain[j-1], new_chain[j]))
                    if diff != 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Verify self-avoidance
                if len(set(tuple(p) for p in new_chain)) != n:
                    continue
                
                new_energy = _energy(sequence, new_chain)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = new_chain
                    improved = True
        
        if not improved:
            break
    
    return best_chain


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
        if n <= 30:
            chain = _a_star_fold(sequence, dimension, max_nodes=200000, time_limit=200)
        else:
            beam_width = max(50, min(200, 5000 // n))
            chain = _beam_fold(sequence, dimension, beam_width)
        
        if chain is None or len(chain) != n:
            raise ValueError("Invalid chain from search")
        
        pos_set = set(tuple(p) for p in chain)
        if len(pos_set) != n:
            raise ValueError("Chain is not self-avoiding")
        
        for i in range(1, n):
            diff = sum(abs(a - b) for a, b in zip(chain[i-1], chain[i]))
            if diff != 1:
                raise ValueError("Non-unit step detected")
        
        # Apply local search refinement
        refined = _local_search_refine(sequence, chain, max_iterations=50)
        
        # Verify refined chain
        pos_set = set(tuple(p) for p in refined)
        if len(pos_set) == n:
            valid = True
            for i in range(1, n):
                diff = sum(abs(a - b) for a, b in zip(refined[i-1], refined[i]))
                if diff != 1:
                    valid = False
                    break
            if valid:
                return [list(p) for p in refined]
        
        return [list(p) for p in chain]
    
    except Exception:
        try:
            return _greedy_fold(sequence, dimension)
        except Exception:
            return [[i] + [0] * (dimension - 1) for i in range(n)]
# EVOLVE-BLOCK-END
