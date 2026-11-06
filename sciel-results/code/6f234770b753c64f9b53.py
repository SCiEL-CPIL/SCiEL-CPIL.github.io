# EVOLVE-BLOCK-START
import time
import random
from collections import deque

_START_TIME = time.time()
_DEADLINE = _START_TIME + 200  # 200s safety margin before 210s kill

# Precompute neighbour offsets for 2D
_DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def _neighbours2d(point):
    x, y = point
    yield (x, y + 1)
    yield (x + 1, y)
    yield (x, y - 1)
    yield (x - 1, y)


def _energy(sequence, chain):
    """Compute total contact energy (negative number of H-H contacts)."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            for dx, dy in _DIRS:
                q = (x + dx, y + dy)
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng):
    """Randomized greedy chain growth preferring sites with new H-H contacts."""
    n = len(sequence)
    if n == 0:
        return []
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    h_positions = {0} if sequence[0] == 'H' else set()
    
    for i in range(1, n):
        best_site = None
        best_score = -float('inf')
        last = chain[-1]
        x, y = last
        candidates = []
        for dx, dy in _DIRS:
            site = (x + dx, y + dy)
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                # Count H-H contacts this site would form
                sx, sy = site
                for edx, edy in _DIRS:
                    nq = (sx + edx, sy + edy)
                    if nq in occupied:
                        j = occupied[nq]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1
            # Prefer free space slightly to avoid trapping
            free = 0
            sx, sy = site
            for edx, edy in _DIRS:
                if (sx + edx, sy + edy) not in occupied:
                    free += 1
            score = gain * 10 + free + rng.random()
            if score > best_score:
                best_score = score
                best_site = site
        if best_site is None:
            return None
        chain.append(best_site)
        occupied[best_site] = i
        if sequence[i] == 'H':
            h_positions.add(i)
    return chain


def _local_search(sequence, chain, max_iter=5000):
    """Local search with O(1) delta-energy for moves and pivots."""
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    # Track H positions for fast contact counting
    h_set = set(i for i in range(n) if sequence[i] == 'H')
    
    rng = random.Random(42)
    improved = True
    iterations = 0
    
    while improved and iterations < max_iter:
        improved = False
        iterations += 1
        
        if time.time() > _DEADLINE - 5:
            break
        
        # Try moving each residue
        for i in range(1, n - 1):
            if i == 0 or i == n - 1:
                continue
            x, y = chain[i]
            for dx, dy in _DIRS:
                new_pos = (x + dx, y + dy)
                if new_pos in where:
                    continue
                # Check it's not adjacent in chain
                if where.get(new_pos) in (i - 1, i + 1):
                    continue
                
                # Calculate delta energy
                delta = 0
                if sequence[i] == 'H':
                    # Remove old contacts
                    old_x, old_y = chain[i]
                    for edx, edy in _DIRS:
                        nq = (old_x + edx, old_y + edy)
                        j = where.get(nq)
                        if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                            delta += 1
                    # Add new contacts
                    for edx, edy in _DIRS:
                        nq = (new_pos[0] + edx, new_pos[1] + edy)
                        j = where.get(nq)
                        if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                            delta -= 1
                
                # Also check if moving affects neighbors' contacts (only for H neighbors)
                # This is complex; approximate by only considering direct contacts of moved residue
                
                if delta < 0:
                    # Accept improving move
                    old_pos = chain[i]
                    del where[old_pos]
                    chain[i] = new_pos
                    where[new_pos] = i
                    current_energy += delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = chain[:]
                    improved = True
                    break
            if improved:
                break
        
        # Try pivot moves (rotate subchain)
        if not improved and n >= 5:
            pivot = rng.randint(2, n - 3)
            sub_len = min(rng.randint(2, n - pivot - 1), 5)
            if sub_len < 2:
                continue
            
            # Try rotating subchain [pivot..pivot+sub_len-1] around chain[pivot]
            anchor = chain[pivot]
            ax, ay = anchor
            
            # Get subchain positions
            subchain = []
            for k in range(pivot + 1, pivot + sub_len + 1):
                if k >= n:
                    break
                subchain.append(k)
            
            if len(subchain) < 2:
                continue
            
            # Try 90 degree rotation
            rotated = {}
            for k in subchain:
                ox, oy = chain[k]
                dx = ox - ax
                dy = oy - ay
                # Rotate 90 degrees CCW: (x,y) -> (-y,x)
                rotated[k] = (ax - dy, ax + dx)
            
            # Check validity
            valid = True
            for k, new_p in rotated.items():
                if new_p in where and new_p != chain[k]:
                    j = where[new_p]
                    if j not in subchain:
                        valid = False
                        break
                # Check chain adjacency within subchain
            if not valid:
                continue
            
            # Also check adjacency between subchain ends and outside
            first_k = subchain[0]
            last_k = subchain[-1]
            prev_k = pivot
            next_k = last_k + 1 if last_k + 1 < n else None
            
            # Check new position of first_k adjacent to prev_k
            prev_p = chain[prev_k]
            fp = rotated[first_k]
            if abs(fp[0] - prev_p[0]) + abs(fp[1] - prev_p[1]) != 1:
                valid = False
            if valid and next_k is not None:
                next_p = chain[next_k]
                lp = rotated[last_k]
                if abs(lp[0] - next_p[0]) + abs(lp[1] - next_p[1]) != 1:
                    valid = False
            
            if not valid:
                continue
            
            # Check internal adjacency
            for idx in range(len(subchain) - 1):
                a = rotated[subchain[idx]]
                b = rotated[subchain[idx + 1]]
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Calculate delta energy for pivot
            delta = 0
            for k in subchain:
                old_p = chain[k]
                new_p = rotated[k]
                if old_p == new_p:
                    continue
                if sequence[k] == 'H':
                    # Remove old contacts
                    ox, oy = old_p
                    for edx, edy in _DIRS:
                        nq = (ox + edx, oy + edy)
                        j = where.get(nq)
                        if j is not None and j not in subchain and abs(j - k) > 1 and sequence[j] == 'H':
                            delta += 1
                    # Add new contacts
                    nx, ny = new_p
                    for edx, edy in _DIRS:
                        nq = (nx + edx, ny + edy)
                        j = where.get(nq)
                        if j is not None and j not in subchain and abs(j - k) > 1 and sequence[j] == 'H':
                            delta -= 1
            
            if delta < 0:
                # Apply pivot
                for k in subchain:
                    old_p = chain[k]
                    if old_p != rotated[k]:
                        del where[old_p]
                    chain[k] = rotated[k]
                    where[rotated[k]] = k
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
                improved = True
    
    return best_chain, best_energy


def _simulate_annealing(sequence, chain, max_iter=10000, t_start=2.0, cooling=0.99):
    """Simulated annealing with O(1) delta-energy."""
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    h_set = set(i for i in range(n) if sequence[i] == 'H')
    rng = random.Random(123)
    
    T = t_start
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        i = rng.randint(1, n - 2)
        x, y = chain[i]
        candidates = []
        for dx, dy in _DIRS:
            new_pos = (x + dx, y + dy)
            if new_pos not in where:
                j = where.get(new_pos)
                if j not in (i - 1, i + 1):
                    candidates.append(new_pos)
        
        if not candidates:
            continue
        
        new_pos = candidates[rng.randint(0, len(candidates) - 1)]
        
        # Calculate delta energy
        delta = 0
        if sequence[i] == 'H':
            old_x, old_y = chain[i]
            for edx, edy in _DIRS:
                nq = (old_x + edx, old_y + edy)
                j = where.get(nq)
                if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                    delta += 1
            for edx, edy in _DIRS:
                nq = (new_pos[0] + edx, new_pos[1] + edy)
                j = where.get(nq)
                if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                    delta -= 1
        
        if delta < 0 or rng.random() < (2.718281828459045 ** (-delta / max(T, 0.01))):
            old_pos = chain[i]
            del where[old_pos]
            chain[i] = new_pos
            where[new_pos] = i
            current_energy += delta
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
        
        T *= cooling
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold HP sequence into low-energy conformation on square lattice."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    if n == 2:
        return [[0, 0], [1, 0]]
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0  # Start at 0 (no contacts)
    
    # Phase 1: Greedy growth with multiple restarts
    num_growth = 50
    for attempt in range(num_growth):
        if time.time() > _DEADLINE - 30:
            break
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            continue
        e = _energy(sequence, chain)
        if e < best_energy:
            best_energy = e
            best_chain = chain[:]
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local search on best chain
    if time.time() < _DEADLINE - 20:
        best_chain, best_energy = _local_search(sequence, best_chain, max_iter=20000)
    
    # Phase 3: Simulated annealing refinement
    if time.time() < _DEADLINE - 10:
        refined, e = _simulate_annealing(sequence, best_chain, max_iter=20000, t_start=1.0, cooling=0.999)
        if e < best_energy:
            best_chain = refined
            best_energy = e
    
    # Final verification
    final_energy = _energy(sequence, best_chain)
    if final_energy < best_energy:
        best_energy = final_energy
    
    # Ensure self-avoiding
    seen = set()
    valid = True
    for i, p in enumerate(best_chain):
        if p in seen:
            valid = False
            break
        seen.add(p)
    if not valid:
        # Fallback to straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
