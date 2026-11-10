"""HP lattice protein folding - improved growth + fast local search."""
# EVOLVE-BLOCK-START
import random
import time

_TOTAL_START = time.time()
_HARD_DEADLINE = _TOTAL_START + 320.0  # 10s safety margin before 330s kill

def _remaining():
    return _HARD_DEADLINE - time.time()

def _manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])

def _neighbours6(p):
    return (
        (p[0]+1, p[1], p[2]), (p[0]-1, p[1], p[2]),
        (p[0], p[1]+1, p[2]), (p[0], p[1]-1, p[2]),
        (p[0], p[1], p[2]+1), (p[0], p[1], p[2]-1),
    )

def _energy(sequence, chain):
    """Compute energy: -count of H-H non-bonded contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours6(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2  # each contact counted twice

def _grow(sequence, dimension, rng, temp=1.0):
    """Randomized greedy growth with contact-aware scoring."""
    n = len(sequence)
    if n == 0:
        return []
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours6(last):
            if site in occupied:
                continue
            score = rng.random() * temp
            if sequence[i] == 'H':
                # Count potential H-H contacts this site would make
                for q in _neighbours6(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        score += 3.0
            # Prefer sites with more free neighbors (avoid dead ends)
            free = 0
            for q in _neighbours6(site):
                if q not in occupied:
                    free += 1
            score += 0.3 * free
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best with some randomness
        options.sort(key=lambda x: -x[0])
        if len(options) > 1 and rng.random() < 0.3:
            # Sometimes pick second best for diversity
            site = options[1][1]
        else:
            site = options[0][1]
        
        chain.append(site)
        occupied[site] = i
    
    return chain

def _local_search_fast(sequence, chain, time_limit, rng):
    """
    Fast local search using incremental energy deltas.
    Moves: single residue, 2-crank, 3-crank.
    """
    n = len(sequence)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    chain = list(chain)
    occupied = {p: i for i, p in enumerate(chain)}
    
    # Precompute H positions
    is_h = [c == 'H' for c in sequence]
    
    # Compute initial energy and contact map
    def compute_energy():
        contacts = 0
        for i, p in enumerate(chain):
            if is_h[i]:
                for q in _neighbours6(p):
                    j = occupied.get(q)
                    if j is not None and abs(i - j) > 1 and is_h[j]:
                        contacts += 1
        return -contacts // 2
    
    current_e = compute_energy()
    best_e = current_e
    best_chain = list(chain)
    
    # Contact count per residue for fast delta
    # contact_count[i] = number of H-H contacts involving residue i
    contact_count = [0] * n
    for i, p in enumerate(chain):
        if is_h[i]:
            for q in _neighbours6(p):
                j = occupied.get(q)
                if j is not None and abs(i - j) > 1 and is_h[j]:
                    contact_count[i] += 1
                    contact_count[j] += 1
    
    iters = 0
    stuck_count = 0
    
    while _remaining() > 1.0 and time.time() < time_limit:
        iters += 1
        
        r = rng.random()
        if r < 0.35:
            move_type = 0  # single
        elif r < 0.70:
            move_type = 1  # 2-crank
        else:
            move_type = 2  # 3-crank
        
        improved = False
        
        if move_type == 0 and n >= 3:
            i = rng.randrange(1, n - 1)
            old = chain[i]
            prev = chain[i-1]
            nxt = chain[i+1]
            
            for new in _neighbours6(old):
                if new == prev or new == nxt:
                    continue
                if new in occupied:
                    continue
                if _manhattan(new, prev) != 1 or _manhattan(new, nxt) != 1:
                    continue
                
                # Delta energy
                delta = 0
                if is_h[i]:
                    # Remove old contacts
                    for q in _neighbours6(old):
                        j = occupied.get(q)
                        if j is not None and j != i and is_h[j] and abs(i-j) > 1:
                            delta -= 1
                    # Add new contacts
                    for q in _neighbours6(new):
                        j = occupied.get(q)
                        if j is not None and j != i and is_h[j] and abs(i-j) > 1:
                            delta += 1
                
                if delta > 0:
                    occupied.pop(old)
                    occupied[new] = i
                    chain[i] = new
                    current_e += delta
                    if current_e < best_e:
                        best_e = current_e
                        best_chain = list(chain)
                    improved = True
                    break
        
        elif move_type == 1 and n >= 4:
            i = rng.randrange(1, n - 2)
            j = i + 1
            old_i = chain[i]
            old_j = chain[j]
            prev = chain[i-1]
            nxt = chain[j+1] if j+1 < n else None
            
            for new_i in _neighbours6(old_i):
                if new_i == prev:
                    continue
                if new_i in occupied and occupied[new_i] != i and occupied[new_i] != j:
                    continue
                if _manhattan(new_i, prev) != 1:
                    continue
                
                for new_j in _neighbours6(new_i):
                    if new_j == old_i:
                        continue
                    if new_j in occupied and occupied[new_j] != i and occupied[new_j] != j:
                        continue
                    if nxt is not None and _manhattan(new_j, nxt) != 1:
                        continue
                    if _manhattan(new_i, new_j) != 1:
                        continue
                    
                    delta = 0
                    if is_h[i]:
                        for q in _neighbours6(old_i):
                            k = occupied.get(q)
                            if k is not None and k != i and k != j and is_h[k] and abs(i-k) > 1:
                                delta -= 1
                        for q in _neighbours6(new_i):
                            k = occupied.get(q)
                            if k is not None and k != i and k != j and is_h[k] and abs(i-k) > 1:
                                delta += 1
                    if is_h[j]:
                        for q in _neighbours6(old_j):
                            k = occupied.get(q)
                            if k is not None and k != i and k != j and is_h[k] and abs(j-k) > 1:
                                delta -= 1
                        for q in _neighbours6(new_j):
                            k = occupied.get(q)
                            if k is not None and k != i and k != j and is_h[k] and abs(j-k) > 1:
                                delta += 1
                    
                    if delta > 0:
                        occupied.pop(old_i)
                        occupied.pop(old_j)
                        occupied[new_i] = i
                        occupied[new_j] = j
                        chain[i] = new_i
                        chain[j] = new_j
                        current_e += delta
                        if current_e < best_e:
                            best_e = current_e
                            best_chain = list(chain)
                        improved = True
                        break
                if improved:
                    break
        
        elif move_type == 2 and n >= 5:
            i = rng.randrange(1, n - 3)
            j = i + 1
            k = i + 2
            old_i = chain[i]
            old_j = chain[j]
            old_k = chain[k]
            prev = chain[i-1]
            nxt = chain[k+1] if k+1 < n else None
            
            for new_i in _neighbours6(old_i):
                if new_i == prev:
                    continue
                if new_i in occupied and occupied[new_i] not in (i, j, k):
                    continue
                if _manhattan(new_i, prev) != 1:
                    continue
                
                for new_j in _neighbours6(new_i):
                    if new_j == old_i:
                        continue
                    if new_j in occupied and occupied[new_j] not in (i, j, k):
                        continue
                    if _manhattan(new_i, new_j) != 1:
                        continue
                    
                    for new_k in _neighbours6(new_j):
                        if new_k == old_j:
                            continue
                        if new_k in occupied and occupied[new_k] not in (i, j, k):
                            continue
                        if nxt is not None and _manhattan(new_k, nxt) != 1:
                            continue
                        if _manhattan(new_j, new_k) != 1:
                            continue
                        
                        delta = 0
                        for idx, old_p, new_p in [(i, old_i, new_i), (j, old_j, new_j), (k, old_k, new_k)]:
                            if is_h[idx]:
                                for q in _neighbours6(old_p):
                                    m = occupied.get(q)
                                    if m is not None and m not in (i, j, k) and is_h[m] and abs(idx-m) > 1:
                                        delta -= 1
                                for q in _neighbours6(new_p):
                                    m = occupied.get(q)
                                    if m is not None and m not in (i, j, k) and is_h[m] and abs(idx-m) > 1:
                                        delta += 1
                        
                        if delta > 0:
                            for old_p, idx in [(old_i, i), (old_j, j), (old_k, k)]:
                                occupied.pop(old_p)
                            occupied[new_i] = i
                            occupied[new_j] = j
                            occupied[new_k] = k
                            chain[i] = new_i
                            chain[j] = new_j
                            chain[k] = new_k
                            current_e += delta
                            if current_e < best_e:
                                best_e = current_e
                                best_chain = list(chain)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            stuck_count += 1
            if stuck_count > 500:
                break
        else:
            stuck_count = 0
    
    return best_chain, best_e

def fold(sequence, dimension):
    """Fold HP chain on cubic lattice. Returns list of [x,y,z] points."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    if n == 2:
        return [[0, 0, 0], [1, 0, 0]]
    
    rng = random.Random(42 + len(sequence) * 7)
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Growth
    # Scale number of growths by remaining time and chain length
    if n <= 48:
        num_grow = 200
    elif n <= 80:
        num_grow = 120
    else:
        num_grow = 80
    
    candidates = []
    for attempt in range(num_grow):
        if _remaining() < 20.0:
            break
        # Vary temperature for diversity
        temp = 0.5 + rng.random() * 1.5
        chain = _grow(sequence, dimension, rng, temp)
        if chain is None:
            continue
        e = _energy(sequence, chain)
        candidates.append((e, chain))
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [[i, 0, 0] for i in range(n)]
        best_energy = _energy(sequence, best_chain)
        return best_chain
    
    candidates.sort(key=lambda x: x[0])
    
    # Phase 2: Local search on top candidates
    remaining = _remaining()
    if remaining > 15.0:
        # Allocate time to local search runs
        num_ls = min(len(candidates), max(2, int(remaining / 12.0)))
        per_run = (remaining - 10.0) / num_ls
        
        for idx in range(num_ls):
            if _remaining() < 8.0:
                break
            
            if idx == 0:
                start = candidates[0][1]
            elif idx < len(candidates):
                start = candidates[idx][1]
            else:
                start = candidates[-1][1]
            
            ls_limit = time.time() + per_run
            improved_chain, improved_e = _local_search_fast(
                sequence, list(start), ls_limit, rng
            )
            if improved_e < best_energy:
                best_energy = improved_e
                best_chain = improved_chain
    
    # Phase 3: Final polish on best chain if time remains
    if _remaining() > 10.0:
        polish_limit = _TOTAL_START + 315.0  # 15s before hard deadline
        polished_chain, polished_e = _local_search_fast(
            sequence, list(best_chain), polish_limit, rng
        )
        if polished_e < best_energy:
            best_energy = polished_e
            best_chain = polished_chain
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
