"""HP lattice protein folding with local search refinement."""
# EVOLVE-BLOCK-START
import random
import time
import math


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact (non-adjacent in chain)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _grow_greedy(sequence, dimension, rng, bias=1.0):
    """Greedy chain growth with contact bias."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * bias + 0.05 * free + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        options.sort(key=lambda x: -x[0])
        site = options[0][1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move(chain, rng):
    """Random pivot move: rotate a subchain around a pivot point."""
    n = len(chain)
    if n < 3:
        return None
    pivot_idx = rng.randint(1, n - 2)
    # Try all 4 directions for the new direction from pivot to next
    old_dir = (chain[pivot_idx + 1][0] - chain[pivot_idx][0],
               chain[pivot_idx + 1][1] - chain[pivot_idx][1])
    # Perpendicular directions
    if old_dir == (1, 0) or old_dir == (-1, 0):
        new_dirs = [(0, 1), (0, -1)]
    else:
        new_dirs = [(1, 0), (-1, 0)]
    
    for new_dir in new_dirs:
        new_chain = list(chain)
        # Rotate the subchain
        subchain = chain[pivot_idx + 1:]
        # Compute new positions
        new_sub = []
        for i, pt in enumerate(subchain):
            if i == 0:
                new_pt = (chain[pivot_idx][0] + new_dir[0],
                          chain[pivot_idx][1] + new_dir[1])
            else:
                prev_new = new_sub[-1]
                old_prev = subchain[i - 1]
                dx = old_prev[0] - subchain[i - 2][0] if i >= 2 else old_dir[0]
                dy = old_prev[1] - subchain[i - 2][1] if i >= 2 else old_dir[1]
                # Apply rotation
                if new_dir == (0, 1) and old_dir == (1, 0):
                    # 90 deg CCW
                    new_dx, new_dy = -dy, dx
                elif new_dir == (0, -1) and old_dir == (1, 0):
                    # 90 deg CW
                    new_dx, new_dy = dy, -dx
                elif new_dir == (0, 1) and old_dir == (-1, 0):
                    # 90 deg CW
                    new_dx, new_dy = dy, -dx
                elif new_dir == (0, -1) and old_dir == (-1, 0):
                    # 90 deg CCW
                    new_dx, new_dy = -dy, dx
                elif new_dir == (1, 0) and old_dir == (0, 1):
                    # 90 deg CW
                    new_dx, new_dy = dy, -dx
                elif new_dir == (-1, 0) and old_dir == (0, 1):
                    # 90 deg CCW
                    new_dx, new_dy = -dy, dx
                elif new_dir == (1, 0) and old_dir == (0, -1):
                    # 90 deg CCW
                    new_dx, new_dy = -dy, dx
                elif new_dir == (-1, 0) and old_dir == (0, -1):
                    # 90 deg CW
                    new_dx, new_dy = dy, -dx
                else:
                    continue
                new_pt = (prev_new[0] + new_dx, prev_new[1] + new_dy)
            new_sub.append(new_pt)
        new_chain[pivot_idx + 1:] = new_sub
        
        # Check self-avoidance
        occupied_set = set()
        for i, pt in enumerate(new_chain):
            if pt in occupied_set:
                break
            occupied_set.add(pt)
        else:
            return new_chain
    return None


def _simple_move(chain, rng):
    """Try to move a single residue to an adjacent empty site."""
    n = len(chain)
    if n < 2:
        return None
    idx = rng.randint(1, n - 2)
    pt = chain[idx]
    for new_pt in _neighbours(pt):
        # Check if new position is free
        occupied = set(chain)
        occupied.remove(pt)
        if new_pt in occupied:
            continue
        # Check that the move doesn't create adjacency issues
        # The residue at idx must still be adjacent to idx-1 and idx+1
        # Actually, simple move changes position, so we need to check
        # that new_pt is adjacent to both neighbors... but that's not how it works.
        # Simple move: move residue idx to new_pt, but it must remain connected.
        # This is tricky. Let's use a different approach.
        pass
    return None


def _local_search(sequence, chain, rng, max_iters=5000, time_limit=5.0):
    """Local search using pivot moves with simulated annealing."""
    n = len(chain)
    current = list(chain)
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    t0 = time.time()
    temp = 2.0
    cooling = 0.999
    
    for it in range(max_iters):
        if time.time() - t0 > time_limit:
            break
        
        new_chain = _pivot_move(current, rng)
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 0.01)):
            current = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best = list(current)
                best_energy = current_energy
        
        temp *= cooling
    
    return best, best_energy


def fold(sequence, dimension):
    """Fold the sequence using multiple strategies and local search."""
    n = len(sequence)
    start_time = time.time()
    deadline = start_time + 180.0  # Leave margin for 210s limit
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0  # 0 means no contacts (energy = 0)
    
    # Generate initial conformations using different strategies
    initial_chains = []
    
    # Strategy 1: Greedy with high contact bias
    for _ in range(10):
        chain = _grow_greedy(sequence, dimension, rng, bias=2.0)
        if chain is not None:
            initial_chains.append(chain)
    
    # Strategy 2: Greedy with low contact bias (more exploratory)
    for _ in range(10):
        chain = _grow_greedy(sequence, dimension, rng, bias=0.5)
        if chain is not None:
            initial_chains.append(chain)
    
    # Strategy 3: Random walk
    for _ in range(5):
        chain = [ (0, 0) ]
        occupied = {(0, 0): 0}
        for i in range(1, n):
            neighbors = list(_neighbours(chain[-1]))
            rng.shuffle(neighbors)
            placed = False
            for site in neighbors:
                if site not in occupied:
                    occupied[site] = i
                    chain.append(site)
                    placed = True
                    break
            if not placed:
                chain = None
                break
        if chain is not None and len(chain) == n:
            initial_chains.append(chain)
    
    if not initial_chains:
        # Fallback: straight line
        return [[i, 0] for i in range(n)]
    
    # Evaluate and pick best initial
    for chain in initial_chains:
        e = _energy(sequence, chain)
        if e < best_energy:
            best_energy = e
            best_chain = list(chain)
    
    # Local search on best few initial conformations
    initial_chains.sort(key=lambda c: _energy(sequence, c))
    to_refine = initial_chains[:5]
    
    for chain in to_refine:
        if time.time() > deadline - 5.0:
            break
        refined, e = _local_search(sequence, chain, rng, max_iters=3000, time_limit=3.0)
        if e < best_energy:
            best_energy = e
            best_chain = refined
    
    # If we have time, do more refinement on the best
    if time.time() < deadline - 10.0 and best_chain is not None:
        # More aggressive local search
        refined, e = _local_search(sequence, best_chain, rng, max_iters=10000, time_limit=10.0)
        if e < best_energy:
            best_energy = e
            best_chain = refined
    
    if best_chain is None:
        return [[i, 0] for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
