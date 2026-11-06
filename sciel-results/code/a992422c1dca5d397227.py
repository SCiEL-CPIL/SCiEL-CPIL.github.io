"""Rewritten program using CSP-style backtracking with heuristic ordering and local search."""
# EVOLVE-BLOCK-START
import random
import heapq
import math


def _neighbours(point):
    """Yield all 6 neighbours in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts (non-consecutive residues on adjacent sites)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _count_hh_contacts_in_prefix(sequence, chain, occupied):
    """Count H-H contacts formed so far in the chain."""
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return contacts


def _greedy_grow(sequence, dimension, rng):
    """Randomized greedy chain growth."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_search(sequence, dimension, chain, rng, iterations=5000):
    """
    Local search via 2-opt style moves: pick two non-adjacent residues and try to
    rotate/shift a segment to improve energy.
    """
    n = len(chain)
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    
    for _ in range(iterations):
        # Pick a random internal segment to try to move
        if n < 4:
            break
        i = rng.randint(1, n - 3)
        j = rng.randint(i + 1, n - 2)
        # Try to shift segment [i..j] by one step in a random direction
        direction = rng.choice([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
        
        # Check if all positions in segment shifted are valid
        new_positions = []
        valid = True
        occupied = set()
        for k in range(n):
            if i <= k <= j:
                new_pos = (chain[k][0] + direction[0], chain[k][1] + direction[1], chain[k][2] + direction[2])
            else:
                new_pos = chain[k]
            if new_pos in occupied:
                valid = False
                break
            occupied.add(new_pos)
            new_positions.append(new_pos)
        
        if not valid:
            continue
        
        # Check connectivity: consecutive residues must be adjacent
        connected = True
        for k in range(n - 1):
            diff = sum(abs(new_positions[k][a] - new_positions[k+1][a]) for a in range(dimension))
            if diff != 1:
                connected = False
                break
        
        if not connected:
            continue
        
        new_energy = _energy(sequence, new_positions)
        if new_energy < best_energy:
            best_chain = new_positions
            best_energy = new_energy
        elif rng.random() < 0.1:  # occasional acceptance of equal or slightly worse
            best_chain = new_positions
            best_energy = new_energy
    
    return best_chain, best_energy


def _backtrack_csp(sequence, dimension, time_budget=1.0):
    """
    CSP backtracking search with dynamic variable ordering and forward checking.
    Returns best chain found within time budget.
    """
    import time
    start_time = time.time()
    n = len(sequence)
    
    # For small chains, use full backtracking; for larger, use heuristic growth
    if n <= 15:
        # Full backtracking with MRV heuristic
        best_chain = None
        best_energy = 1  # positive infinity placeholder
        
        def backtrack(pos, chain, occupied):
            nonlocal best_chain, best_energy
            if time.time() - start_time > time_budget:
                return False
            
            if pos == n:
                e = _energy(sequence, chain)
                if e < best_energy:
                    best_energy = e
                    best_chain = list(chain)
                return True
            
            # MRV: try sites with fewest free neighbours first (more constrained)
            candidates = []
            for site in _neighbours(chain[-1]):
                if site in occupied:
                    continue
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                gain = 0
                if sequence[pos] == "H":
                    gain = sum(1 for q in _neighbours(site)
                               if q in occupied and occupied[q] < pos - 1 and sequence[occupied[q]] == "H")
                # Prefer higher gain, lower free space
                candidates.append((-gain, free, site))
            
            candidates.sort()
            for _, _, site in candidates:
                chain.append(site)
                occupied[site] = pos
                if backtrack(pos + 1, chain, occupied):
                    chain.pop()
                    del occupied[site]
                    return True
                chain.pop()
                del occupied[site]
            
            return False
        
        chain = [(0, 0, 0)]
        occupied = {(0, 0, 0): 0}
        backtrack(1, chain, occupied)
        
        if best_chain is not None:
            return best_chain
        else:
            return None
    
    else:
        # For longer chains, use multiple greedy grows + local search
        rng = random.Random(42 + len(sequence))
        best_chain = None
        best_energy = 1
        
        for _ in range(50):
            chain = _greedy_grow(sequence, dimension, rng)
            if chain is None:
                continue
            chain, e = _local_search(sequence, dimension, chain, rng, iterations=2000)
            if e < best_energy:
                best_energy = e
                best_chain = list(chain)
        
        return best_chain


def fold(sequence, dimension):
    """
    Find best conformation for the given HP sequence on the simple cubic lattice.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    
    # Try CSP backtracking for small chains
    if n <= 15:
        result = _backtrack_csp(sequence, dimension, time_budget=5.0)
        if result is not None:
            return [list(p) for p in result]
    
    # For all chains: multiple greedy grows + local search
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 1
    
    num_grows = 200 if n <= 20 else 100
    
    for _ in range(num_grows):
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            continue
        # Apply local search
        chain, e = _local_search(sequence, dimension, chain, rng, iterations=3000)
        if e < best_energy:
            best_energy = e
            best = list(chain)
    
    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
