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


def _h_heuristic(sequence, current_index, h_placed):
    """
    Admissible heuristic: upper bound on additional H-H contacts from residues
    current_index+1 to end. Each remaining H can form at most 2 contacts with
    previously placed H residues.
    """
    h_remaining = sum(1 for i in range(current_index + 1, len(sequence)) if sequence[i] == 'H')
    return min(h_placed * h_remaining, h_remaining * 2)


def _a_star_fold(sequence, dimension, time_budget=180):
    """
    A* search over partial chains with admissible heuristic and best-so-far pruning.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]
    
    start = (0,) * dimension
    counter = 0
    initial_h = 1 if sequence[0] == 'H' else 0
    initial_state = (0, counter, (start,), {start: 0}, 0, initial_h)
    counter += 1
    open_heap = [initial_state]
    
    best_chain = None
    best_contacts = -1
    start_time = time.time()
    
    while open_heap:
        if time.time() - start_time > time_budget:
            break
        
        priority, _, chain, pos_to_idx, contacts, h_placed = heapq.heappop(open_heap)
        current_index = len(chain) - 1
        
        if current_index == n - 1:
            if contacts > best_contacts:
                best_contacts = contacts
                best_chain = chain
            continue
        
        last = chain[-1]
        
        for site in _neighbours(last):
            if site in pos_to_idx:
                continue
            
            new_h_placed = h_placed + (1 if sequence[current_index + 1] == 'H' else 0)
            new_contacts = contacts
            
            if sequence[current_index + 1] == 'H':
                for q in _neighbours(site):
                    if q in pos_to_idx:
                        j = pos_to_idx[q]
                        if j < current_index and sequence[j] == 'H':
                            new_contacts += 1
            
            heuristic = _h_heuristic(sequence, current_index + 1, new_h_placed)
            
            # Prune if cannot beat best
            if best_contacts >= 0 and new_contacts + heuristic <= best_contacts:
                continue
            
            priority = -(new_contacts + heuristic)
            new_chain = chain + (site,)
            new_pos_to_idx = dict(pos_to_idx)
            new_pos_to_idx[site] = current_index + 1
            
            heapq.heappush(open_heap, (priority, counter, new_chain, new_pos_to_idx, new_contacts, new_h_placed))
            counter += 1
    
    if best_chain is not None:
        return list(best_chain)
    
    return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]


def _beam_fold(sequence, dimension, beam_width=100, time_budget=180):
    """
    Beam search with incremental contact tracking and adaptive width.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]
    
    start = (0,) * dimension
    beam = [((start,), {start: 0}, 0, 1 if sequence[0] == 'H' else 0)]
    start_time = time.time()
    
    for step in range(1, n):
        elapsed = time.time() - start_time
        if time_budget - elapsed <= 0:
            break
        
        progress = step / max(1, n - 1)
        if progress < 0.3:
            bw = beam_width
        elif progress < 0.7:
            bw = max(20, beam_width // 2)
        else:
            bw = max(10, beam_width // 4)
        
        candidates = []
        
        for chain, pos_to_idx, contacts, h_placed in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in pos_to_idx:
                    continue
                
                new_h_placed = h_placed + (1 if sequence[step] == 'H' else 0)
                new_contacts = contacts
                
                if sequence[step] == 'H':
                    for q in _neighbours(site):
                        if q in pos_to_idx:
                            j = pos_to_idx[q]
                            if j < step - 1 and sequence[j] == 'H':
                                new_contacts += 1
                
                new_chain = chain + (site,)
                new_pos_to_idx = dict(pos_to_idx)
                new_pos_to_idx[site] = step
                
                candidates.append((new_chain, new_pos_to_idx, new_contacts, new_h_placed))
        
        if not candidates:
            break
        
        candidates.sort(key=lambda x: -x[2])
        beam = candidates[:bw]
    
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
            chain = _a_star_fold(sequence, dimension, time_budget=180)
        else:
            beam_width = max(30, min(150, 3000 // n))
            chain = _beam_fold(sequence, dimension, beam_width, time_budget=180)
        
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
