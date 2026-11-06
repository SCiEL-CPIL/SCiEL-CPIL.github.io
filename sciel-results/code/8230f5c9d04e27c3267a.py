"""Improved HP lattice protein folding using A*-guided search with admissible heuristic."""
# EVOLVE-BLOCK-START
import heapq
import time
import random
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


def _count_contacts(sequence, chain):
    """Count H-H contacts in a complete chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return contacts


def _max_remaining_contacts(sequence, current_index):
    """
    Admissible heuristic: maximum possible additional H-H contacts from residues
    current_index+1 to end. Each new H can form at most 2 contacts with previously
    placed H residues. This is an upper bound.
    """
    h_count_remaining = sum(1 for i in range(current_index + 1, len(sequence)) if sequence[i] == 'H')
    h_count_placed = sum(1 for i in range(current_index + 1) if sequence[i] == 'H')
    return min(h_count_placed * h_count_remaining, h_count_remaining * 2)


def _a_star_fold(sequence, dimension, max_nodes=500000, time_limit=200):
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
            
            heuristic = _max_remaining_contacts(sequence, current_index + 1)
            priority = -(new_contacts + heuristic)
            
            heapq.heappush(open_heap, (priority, counter, new_chain, new_pos_to_idx, new_contacts))
            counter += 1
    
    if best_chain is not None:
        return list(best_chain)
    
    return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]


def _beam_fold(sequence, dimension, beam_width=100):
    """
    Beam search with incremental contact tracking.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]
    
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


def _local_search_refine(sequence, dimension, chain, iterations=2000):
    """
    Local search refinement using move-based optimization.
    Try to improve contacts by moving individual residues.
    """
    n = len(sequence)
    if n <= 2:
        return chain
    
    # Build position map
    pos_map = {tuple(p): i for i, p in enumerate(chain)}
    current_contacts = _count_contacts(sequence, [tuple(p) for p in chain])
    
    best_chain = [list(p) for p in chain]
    best_contacts = current_contacts
    
    for _ in range(iterations):
        # Pick a random residue to move (not endpoints)
        i = random.randint(1, n - 2)
        old_pos = tuple(best_chain[i])
        
        # Try all 6 neighbors
        for new_pos in _neighbours(old_pos):
            if new_pos in pos_map:
                continue
            
            # Check if moving residue i to new_pos is valid
            # It must not collide with adjacent residues
            if i > 0:
                adj_prev = tuple(best_chain[i-1])
                if new_pos == adj_prev:
                    continue
            if i < n - 1:
                adj_next = tuple(best_chain[i+1])
                if new_pos == adj_next:
                    continue
            
            # Tentatively move
            pos_map.pop(old_pos)
            pos_map[new_pos] = i
            best_chain[i] = list(new_pos)
            
            new_contacts = _count_contacts(sequence, [tuple(p) for p in best_chain])
            
            if new_contacts > best_contacts:
                best_contacts = new_contacts
                # Keep the move
                break
            else:
                # Revert
                pos_map.pop(new_pos)
                pos_map[old_pos] = i
                best_chain[i] = list(old_pos)
        
        # Occasionally try a 2-residue move
        if random.random() < 0.3 and n >= 4:
            i = random.randint(1, n - 3)
            j = i + 1
            old_i = tuple(best_chain[i])
            old_j = tuple(best_chain[j])
            
            for new_i in _neighbours(old_i):
                if new_i in pos_map or new_i == old_j:
                    continue
                for new_j in _neighbours(old_j):
                    if new_j in pos_map or new_j == old_i:
                        continue
                    # Check adjacency
                    if i > 0 and new_i == tuple(best_chain[i-1]):
                        continue
                    if j < n - 1 and new_j == tuple(best_chain[j+1]):
                        continue
                    # Check that new_i and new_j are adjacent
                    if sum(abs(a-b) for a, b in zip(new_i, new_j)) != 1:
                        continue
                    
                    pos_map.pop(old_i)
                    pos_map.pop(old_j)
                    pos_map[new_i] = i
                    pos_map[new_j] = j
                    best_chain[i] = list(new_i)
                    best_chain[j] = list(new_j)
                    
                    new_contacts = _count_contacts(sequence, [tuple(p) for p in best_chain])
                    
                    if new_contacts > best_contacts:
                        best_contacts = new_contacts
                        break
                    else:
                        pos_map.pop(new_i)
                        pos_map.pop(new_j)
                        pos_map[old_i] = i
                        pos_map[old_j] = j
                        best_chain[i] = list(old_i)
                        best_chain[j] = list(old_j)
                else:
                    continue
                break
    
    return best_chain


def fold(sequence, dimension):
    """
    Fold the sequence using A* search for short chains, beam search for longer ones,
    then refine with local search.
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension for _ in range(n)]
    
    global_start = time.time()
    
    try:
        if n <= 30:
            # Use A* for shorter chains where it's tractable
            chain = _a_star_fold(sequence, dimension, max_nodes=200000, time_limit=150)
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
        
        # Refine with local search if time permits
        elapsed = time.time() - global_start
        remaining = 280 - elapsed  # Leave 20s margin
        if remaining > 10:
            iterations = min(5000, int(remaining * 50))
            chain = _local_search_refine(sequence, dimension, chain, iterations=iterations)
        
        # Final verification
        pos_set = set(tuple(p) for p in chain)
        if len(pos_set) != n:
            raise ValueError("Chain is not self-avoiding after refinement")
        
        for i in range(1, n):
            diff = sum(abs(a - b) for a, b in zip(chain[i-1], chain[i]))
            if diff != 1:
                raise ValueError("Non-unit step detected after refinement")
        
        return [list(p) for p in chain]
    
    except Exception:
        # Fallback: greedy
        try:
            chain = _greedy_fold(sequence, dimension)
            # Try to refine
            elapsed = time.time() - global_start
            remaining = 280 - elapsed
            if remaining > 10:
                iterations = min(3000, int(remaining * 30))
                chain = _local_search_refine(sequence, dimension, chain, iterations=iterations)
            return chain
        except Exception:
            return [[i] + [0] * (dimension - 1) for i in range(n)]
# EVOLVE-BLOCK-END
