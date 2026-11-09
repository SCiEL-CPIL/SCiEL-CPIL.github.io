"""HP lattice protein folding on the cubic lattice.
Combines fast greedy growth with aggressive iterative tail re-growth refinement.
"""
import random
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 318.0  # Safety margin before 330s kill


def _neighbours(point):
    """Yield the 6 face-neighbours of point in 3D (fast path)."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _energy(sequence, chain):
    """Compute total energy: negative count of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y, z = p
            for q in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng, temperature=1.0):
    """One randomized greedy chain growth with temperature control."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        x, y, z = chain[-1]
        for site in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                sx, sy, sz = site
                for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            # Count free neighbors for entropy bonus
            free = 0
            sx, sy, sz = site
            for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                if q not in occupied:
                    free += 1
            score = gain + 0.1 * free + rng.random() * temperature
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _regrow_tail(sequence, chain, start_idx, rng, max_iters=100, temperature=1.5):
    """Regrow the tail of the chain from start_idx to improve energy.
    Returns (best_chain, best_energy)."""
    if start_idx >= len(chain) - 1:
        return list(chain), _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    n = len(sequence)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 3:
            break
        
        prefix = chain[:start_idx + 1]
        occupied = {p: i for i, p in enumerate(prefix)}
        
        current_tail = [tuple(p) for p in prefix]
        success = True
        
        for i in range(start_idx + 1, n):
            x, y, z = current_tail[-1]
            options = []
            for site in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == "H":
                    sx, sy, sz = site
                    for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == "H":
                            gain += 1
                free = 0
                sx, sy, sz = site
                for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                    if q not in occupied:
                        free += 1
                score = gain + 0.05 * free + rng.random() * temperature
                options.append((score, site))
            if not options:
                success = False
                break
            site = max(options)[1]
            occupied[site] = i
            current_tail.append(site)
        
        if not success:
            continue
        
        new_energy = _energy(sequence, current_tail)
        
        if new_energy < best_energy:
            best_chain = current_tail
            best_energy = new_energy
            # Anneal: reduce temperature slightly on improvement
            temperature = max(0.5, temperature * 0.98)
    
    return best_chain, best_energy


def _local_search(sequence, chain, rng, time_budget=30.0):
    """Iterative local search: repeatedly try regrowing from various positions."""
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    n = len(sequence)
    start_time = time.time()
    
    # Generate candidate split points
    split_points = set()
    split_points.add(max(0, n - 15))
    split_points.add(max(0, n - 30))
    split_points.add(max(0, n - 50))
    split_points.add(n // 4)
    split_points.add(n // 2)
    split_points.add(3 * n // 4)
    split_points.add(max(0, n - 10))
    split_points.add(max(0, n - 20))
    
    # Sort by priority: shorter tails first (faster to evaluate)
    split_points = sorted(split_points, key=lambda sp: n - sp)
    
    for sp in split_points:
        if time.time() - start_time > time_budget:
            break
        if sp >= n - 1:
            continue
        
        improved_chain, improved_energy = _regrow_tail(
            sequence, best_chain, sp, rng, max_iters=50, temperature=1.0
        )
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    # Second pass with more iterations on the best split points
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 5:
        for sp in split_points[:4]:
            if time.time() - start_time > time_budget:
                break
            if sp >= n - 1:
                continue
            improved_chain, improved_energy = _regrow_tail(
                sequence, best_chain, sp, rng, max_iters=80, temperature=0.8
            )
            if improved_energy < best_energy:
                best_chain = improved_chain
                best_energy = improved_energy
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using fast growth + aggressive local search with deadline guard."""
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    n = len(sequence)
    remaining = _DEADLINE - time.time()
    
    # Phase 1: Multiple randomized greedy growths
    # Scale number of initial tries based on chain length and time budget
    if n <= 50:
        num_initial = min(80, int(remaining * 2))
    elif n <= 100:
        num_initial = min(40, int(remaining))
    else:
        num_initial = min(20, int(remaining * 0.5))
    
    num_initial = max(num_initial, 10)
    
    for trial in range(num_initial):
        if time.time() > _DEADLINE - 60:
            break
        
        # Vary temperature: early trials more exploratory, later more greedy
        temp = 1.5 if trial < num_initial // 3 else (0.8 if trial < 2 * num_initial // 3 else 0.4)
        chain = _grow(sequence, dimension, rng, temperature=temp)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best_chain = chain
            best_energy = value
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
        return [list(p) for p in best_chain]
    
    # Phase 2: Local search refinement
    remaining = _DEADLINE - time.time() - 5  # Leave 5s safety margin
    if remaining > 10:
        # Allocate time budget for local search
        ls_budget = min(remaining, 200.0)
        best_chain, best_energy = _local_search(sequence, best_chain, rng, time_budget=ls_budget)
    
    # Phase 3: Final polish if time remains
    remaining = _DEADLINE - time.time() - 3
    if remaining > 10:
        # Try a few more targeted regrowths
        n = len(sequence)
        for sp in [max(0, n - 10), max(0, n - 25), n // 2]:
            if time.time() > _DEADLINE - 3:
                break
            if sp >= n - 1:
                continue
            improved_chain, improved_energy = _regrow_tail(
                sequence, best_chain, sp, rng, max_iters=40, temperature=0.5
            )
            if improved_energy < best_energy:
                best_chain = improved_chain
                best_energy = improved_energy
    
    return [list(p) for p in best_chain]
