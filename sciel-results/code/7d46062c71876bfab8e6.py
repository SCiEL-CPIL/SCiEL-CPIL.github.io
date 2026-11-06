"""Improved HP lattice protein folding using A*-guided search with admissible heuristic."""
# EVOLVE-BLOCK-START
import heapq
import time
from collections import defaultdict


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


def _max_remaining_contacts(sequence, current_index):
    """
    Admissible heuristic: maximum possible additional H-H contacts from residues
    current_index+1 to end. Each new H can form at most 2 contacts with previously
    placed H residues (since it's adjacent to at most 2 already-placed positions
    that are H and not its chain neighbor). This is an upper bound.
    """
    h_count_remaining = sum(1 for i in range(current_index + 1, len(sequence)) if sequence[i] == 'H')
    # Each remaining H can potentially contact up to 2 earlier H residues
    # But we need to be careful: a contact is counted once per pair
    # Upper bound: each pair of H residues (one placed, one remaining) can form at most 1 contact
    h_count_placed = sum(1 for i in range(current_index + 1) if sequence[i] == 'H')
    # Max contacts = min(h_count_placed * h_count_remaining, h_count_remaining * 2) 
    # Actually, each remaining H can contact at most 2 placed H's (since it has 6 neighbors, 
    # but only 2 can be from earlier in chain due to self-avoidance constraints being loose)
    # Conservative upper bound: h_count_remaining * 2
    return min(h_count_placed * h_count_remaining, h_count_remaining * 2)


def _a_star_fold(sequence, dimension, max_nodes=500000):
    """
    A* search over partial chains. State: (chain_tuple, pos_to_idx_dict, contacts_so_far)
    Priority: -contacts_so_far - heuristic (we want to maximize contacts, so minimize negative)
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]
    
    start = (0,) * dimension
    # State: (priority, counter, chain_tuple, pos_to_idx, contacts)
    # We use a counter to break ties and ensure heap stability
    counter = 0
    initial_state = (0, counter, (start,), {start: 0}, 0)
    counter += 1
    open_heap = [initial_state]
    
    # Track best solution found
    best_chain = None
    best_contacts = -1
    
    nodes_expanded = 0
    start_time = time.time()
    time_limit = 220  # Leave margin for all sequences
    
    while open_heap:
        # Check time limit
        if time.time() - start_time > time_limit:
            break
        
        priority, _, chain, pos_to_idx, contacts = heapq.heappop(open_heap)
        current_index = len(chain) - 1
        
        # If we've placed all residues, this is a complete solution
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
            
            # Compute new H-H contacts if current residue is H
            if sequence[current_index + 1] == 'H':
                for q in _neighbours(site):
                    if q in new_pos_to_idx:
                        j = new_pos_to_idx[q]
                        # Only count if j is not the previous residue (chain neighbor)
                        if j < current_index and sequence[j] == 'H':
                            new_contacts += 1
            
            # Compute heuristic: max additional contacts possible
            heuristic = _max_remaining_contacts(sequence, current_index + 1)
            
            # Priority: we want to maximize total contacts = new_contacts + future_contacts
            # So we minimize -(new_contacts + heuristic)
            # But we also want to prefer states with more actual contacts
            # Use: priority = -(new_contacts) - heuristic * 0.1 (heuristic is admissible upper bound)
            # Actually for A* to be admissible, h must not overestimate. 
            # _max_remaining_contacts is an upper bound on remaining contacts, so it's admissible.
            # We want to find the path with maximum total contacts.
            # g = new_contacts (contacts so far)
            # h = heuristic (max possible additional contacts)
            # We want to maximize g + h, so we minimize -(g + h)
            priority = -(new_contacts + heuristic)
            
            heapq.heappush(open_heap, (priority, counter, new_chain, new_pos_to_idx, new_contacts))
            counter += 1
    
    if best_chain is not None:
        return list(best_chain)
    
    # Fallback: return a valid straight-line chain
    return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]


def _beam_fold(sequence, dimension, beam_width=100):
    """
    Beam search with incremental contact tracking.
    State: (chain_tuple, pos_to_idx_dict, contacts)
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]
    
    start = (0,) * dimension
    # Each beam entry: (chain_tuple, pos_to_idx_dict, contacts)
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
                
                # Compute new H-H contacts if current residue is H
                if sequence[step] == 'H':
                    for q in _neighbours(site):
                        if q in new_pos_to_idx:
                            j = new_pos_to_idx[q]
                            if j < step - 1 and sequence[j] == 'H':
                                new_contacts += 1
                
                candidates.append((new_chain, new_pos_to_idx, new_contacts))
        
        if not candidates:
            break
        
        # Sort by contacts (descending) and keep top beam_width
        candidates.sort(key=lambda x: -x[2])
        beam = candidates[:beam_width]
    
    if not beam:
        return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Return the best chain
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
            # No valid move; extend in a fixed direction
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
                    # Shouldn't happen, but pad
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
            # Use A* for shorter chains where it's tractable
            chain = _a_star_fold(sequence, dimension, max_nodes=200000)
        else:
            # Use beam search for longer chains
            beam_width = max(50, min(200, 5000 // n))
            chain = _beam_fold(sequence, dimension, beam_width)
        
        if chain is None or len(chain) != n:
            raise ValueError("Invalid chain from search")
        
        # Verify self-avoidance
        pos_set = set(tuple(p) for p in chain)
        if len(pos_set) != n:
            raise ValueError("Chain is not self-avoiding")
        
        # Verify unit steps
        for i in range(1, n):
            diff = sum(abs(a - b) for a, b in zip(chain[i-1], chain[i]))
            if diff != 1:
                raise ValueError("Non-unit step detected")
        
        return [list(p) for p in chain]
    
    except Exception:
        # Fallback: greedy
        try:
            return _greedy_fold(sequence, dimension)
        except Exception:
            # Last resort: straight line
            return [[i] + [0] * (dimension - 1) for i in range(n)]
# EVOLVE-BLOCK-END
