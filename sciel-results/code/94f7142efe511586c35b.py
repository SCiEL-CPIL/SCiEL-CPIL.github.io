"""Rewritten HP lattice folding program."""
# EVOLVE-BLOCK-START
import random
import time

# Deadline guard: evaluator kills after 210s wall-clock.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 195.0  # 15s safety margin


def _remaining():
    return _DEADLINE - time.time()


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: minus the number of non-consecutive H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng, temperature=0.05):
    """Randomized chain growth with contact-aware scoring."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            # Count potential H-H contacts this placement would create
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            # Also count free neighbours to avoid trapping
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Score: prefer contacts, then free space, with randomness
            score = gain * 2.0 + 0.1 * free + rng.random() * temperature
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best option (with some randomness via the score)
        options.sort(key=lambda x: -x[0])
        # Sometimes pick second best to add diversity
        if len(options) > 1 and rng.random() < 0.15:
            site = options[1][1]
        else:
            site = options[0][1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain


def _local_search(sequence, chain, rng, max_moves=500):
    """Local search: try to improve energy by moving a single residue."""
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for _ in range(max_moves):
        if _remaining() < 5.0:
            break
        # Pick a random residue to move
        i = rng.randint(1, len(chain) - 2)  # not first or last
        old_pos = chain[i]
        
        # Try moving to a neighbouring empty site
        for new_pos in _neighbours(old_pos):
            if new_pos in where:
                continue
            
            # Check that the move doesn't violate self-avoidance
            # (it's a single site move, so only need to check the new site is free)
            
            # Tentatively move
            del where[old_pos]
            where[new_pos] = i
            chain[i] = new_pos
            
            new_energy = _energy(sequence, chain)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
            else:
                # Revert
                del where[new_pos]
                where[old_pos] = i
                chain[i] = old_pos
            
            if _remaining() < 5.0:
                break
    
    return best_chain, best_energy


def _simulated_annealing(sequence, chain, rng, max_iters=2000, t_start=1.0, t_end=0.01):
    """Simulated annealing on the chain."""
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    current_chain = list(chain)
    
    n_iters = max_iters
    for it in range(n_iters):
        if _remaining() < 5.0:
            break
        temp = t_start * (t_end / t_start) ** (it / max(1, n_iters - 1))
        
        # Pick a random internal residue
        i = rng.randint(1, len(current_chain) - 2)
        old_pos = current_chain[i]
        
        # Try moving to a random neighbouring empty site
        candidates = [q for q in _neighbours(old_pos) if q not in where]
        if not candidates:
            continue
        new_pos = rng.choice(candidates)
        
        # Perform move
        del where[old_pos]
        where[new_pos] = i
        current_chain[i] = new_pos
        
        new_energy = _energy(sequence, current_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < (2.718281828 ** (-delta / max(temp, 1e-10))):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(current_chain)
        else:
            # Revert
            del where[new_pos]
            where[old_pos] = i
            current_chain[i] = old_pos
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Find best conformation using multiple strategies within time budget."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    best_chain = None
    best_energy = 0  # energy is negative or zero
    
    # Strategy 1: Multiple randomized greedy growths
    growth_count = 0
    while _remaining() > 30.0:
        chain = _grow(sequence, dimension, rng, temperature=0.05)
        if chain is None:
            break
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
        growth_count += 1
        # Limit growths per pass
        if growth_count >= 500:
            break
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Strategy 2: Local search on best chain
    if _remaining() > 15.0:
        improved_chain, improved_energy = _local_search(sequence, best_chain, rng, max_moves=1000)
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain
    
    # Strategy 3: Simulated annealing
    if _remaining() > 10.0:
        sa_chain, sa_energy = _simulated_annealing(sequence, best_chain, rng, max_iters=3000)
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = sa_chain
    
    # Strategy 4: More growths with different temperatures if time remains
    if _remaining() > 10.0:
        for temp in [0.1, 0.2, 0.5]:
            if _remaining() < 10.0:
                break
            for _ in range(100):
                if _remaining() < 10.0:
                    break
                chain = _grow(sequence, dimension, rng, temperature=temp)
                if chain is None:
                    continue
                energy = _energy(sequence, chain)
                if energy < best_energy:
                    best_energy = energy
                    best_chain = list(chain)
                # Quick local search on good candidates
                if energy <= best_energy and _remaining() > 10.0:
                    imp_chain, imp_energy = _local_search(sequence, best_chain, rng, max_moves=200)
                    if imp_energy < best_energy:
                        best_energy = imp_energy
                        best_chain = imp_chain
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
