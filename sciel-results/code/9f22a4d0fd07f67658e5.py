"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute the energy: -1 * (number of H-H non-adjacent contacts)."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _grow_greedy(sequence, dimension, rng):
    """Grow a chain greedily, preferring positions that create H-H contacts."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
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
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10 + free * 0.1 + rng.random() * 0.5
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        candidates.sort(reverse=True)
        top_k = min(3, len(candidates))
        chosen = rng.choice(candidates[:top_k])
        site = chosen[1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def _local_search(sequence, chain, max_iters=50000, timeout=10.0):
    """Improve a conformation using single-residue moves."""
    n = len(chain)
    if n < 3:
        return list(chain)
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    start_time = time.time()
    
    # Build position lookup for fast self-avoidance checks
    pos_to_idx = {p: i for i, p in enumerate(best_chain)}
    
    for _ in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        # Pick a random interior residue to move
        i = random.Random().randint(1, n - 2)
        old_pos = best_chain[i]
        
        for new_pos in _neighbours(old_pos):
            # Cannot move to adjacent chain positions
            if new_pos == best_chain[i - 1] or new_pos == best_chain[i + 1]:
                continue
            
            # Check if new_pos is occupied
            if new_pos in pos_to_idx:
                continue
            
            # Perform the move
            del pos_to_idx[old_pos]
            best_chain[i] = new_pos
            pos_to_idx[new_pos] = i
            
            # Check unit steps
            valid = True
            diff1 = sum(abs(a - b) for a, b in zip(best_chain[i - 1], best_chain[i]))
            diff2 = sum(abs(a - b) for a, b in zip(best_chain[i], best_chain[i + 1]))
            if diff1 != 1 or diff2 != 1:
                valid = False
            
            if not valid:
                # Revert
                del pos_to_idx[new_pos]
                best_chain[i] = old_pos
                pos_to_idx[old_pos] = i
                continue
            
            new_energy = _compute_energy(sequence, best_chain)
            if new_energy < best_energy:
                best_energy = new_energy
            else:
                # Revert
                del pos_to_idx[new_pos]
                best_chain[i] = old_pos
                pos_to_idx[old_pos] = i
    
    return best_chain


def fold(sequence, dimension):
    """Find a good conformation for the given HP sequence."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    start_time = time.time()
    deadline = start_time + 280  # Leave margin for 330s limit
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0  # 0 means no contacts
    
    # Phase 1: Greedy growth with multiple restarts
    max_restarts = 300
    for attempt in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        chain = _grow_greedy(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _compute_energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    # Phase 2: Local search to improve the best conformation
    remaining = deadline - time.time()
    if remaining > 3:
        improved = _local_search(sequence, best_chain, max_iters=100000, timeout=remaining - 2)
        improved_energy = _compute_energy(sequence, improved)
        if improved_energy < best_energy:
            best_chain = improved
            best_energy = improved_energy
    
    # Convert to list of lists
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
