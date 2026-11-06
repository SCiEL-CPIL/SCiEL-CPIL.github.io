"""HP lattice protein folding on the cubic lattice."""
import random
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy = -number of H-H contacts (non-consecutive)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth with contact-aware scoring."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            # Also consider potential future contacts: how many H residues in chain
            # that are near this site but not yet adjacent
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Penalize trapping: if free space is very low, it's risky
            score = gain * 2.0 + 0.05 * free + rng.random()
            if free <= 1:
                score -= 10.0  # avoid trapping
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_search(sequence, chain, max_iter=2000, rng=None):
    """Monte Carlo local search to improve energy."""
    if rng is None:
        rng = random.Random()
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    energy = _energy(sequence, chain)
    
    # Precompute H positions
    h_positions = [i for i, c in enumerate(sequence) if c == 'H']
    
    for _ in range(max_iter):
        # Pick a random interior residue to move
        i = rng.randint(1, n - 2)
        if sequence[i] == 'P' and len(h_positions) > 0 and rng.random() > 0.3:
            # Bias toward moving H residues
            i = rng.choice(h_positions)
            if i <= 0 or i >= n - 1:
                i = rng.randint(1, n - 2)
        
        old_pos = chain[i]
        # Try all 6 neighbor positions
        best_new_pos = None
        best_new_energy = energy
        for new_pos in _neighbours(old_pos):
            if new_pos in where:
                # Check if it's the previous or next residue (allowed)
                if i > 0 and where[new_pos] == i - 1:
                    continue
                if i < n - 1 and where[new_pos] == i + 1:
                    continue
                continue
            
            # Remove old position from where
            del where[old_pos]
            where[new_pos] = i
            
            # Check self-avoidance: new_pos must not be occupied by other residues
            # (already checked above via where)
            
            # Compute energy change efficiently
            # Only need to check contacts involving residue i and its neighbors
            delta = 0
            if sequence[i] == 'H':
                # Count H-H contacts for residue i
                for q in _neighbours(new_pos):
                    j = where.get(q)
                    if j is not None and j != i and j != i - 1 and j != i + 1:
                        if sequence[j] == 'H':
                            delta += 1
                # Subtract old contacts
                for q in _neighbours(old_pos):
                    j = where.get(q)
                    if j is not None and j != i and j != i - 1 and j != i + 1:
                        if sequence[j] == 'H':
                            delta -= 1
            
            new_energy = energy - delta  # energy is negative, so -delta means add contacts
            
            # Restore
            del where[new_pos]
            where[old_pos] = i
            
            if new_energy < best_new_energy:
                best_new_energy = new_energy
                best_new_pos = new_pos
        
        if best_new_pos is not None:
            # Accept the move
            del where[old_pos]
            where[best_new_pos] = i
            chain[i] = best_new_pos
            energy = best_new_energy
        else:
            # With some probability, accept a worse move (simulated annealing lite)
            if rng.random() < 0.1:
                # Pick a random valid neighbor
                valid = [np_ for np_ in _neighbours(old_pos)
                         if np_ not in where or where.get(np_) in (i - 1, i + 1)]
                if valid:
                    new_pos = rng.choice(valid)
                    del where[old_pos]
                    where[new_pos] = i
                    chain[i] = new_pos
                    energy = _energy(sequence, chain)
    
    return chain, energy


def fold(sequence, dimension):
    """Best conformation using greedy growth + local search with time budget."""
    t_start = time.time()
    deadline = t_start + 230.0  # 10s safety margin before 240s limit
    
    # First, do a few quick greedy runs to get a baseline
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    # Quick greedy phase
    for _ in range(20):
        if time.time() > deadline - 60:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        energy = _energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        # Straight line fallback
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = _energy(sequence, best_chain)
    
    # Local search phase: refine best chain
    if time.time() < deadline - 30:
        refined, energy = _local_search(sequence, list(best_chain), max_iter=5000, rng=rng)
        if energy < best_energy:
            best_chain = refined
            best_energy = energy
    
    # More greedy + local search iterations
    while time.time() < deadline - 10:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        energy = _energy(sequence, chain)
        
        # Local search on this chain
        remaining = deadline - 10 - time.time()
        if remaining > 5:
            iter_count = min(2000, int(remaining * 100))
            if iter_count > 100:
                chain, energy = _local_search(sequence, chain, max_iter=iter_count, rng=rng)
        
        if energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    return [list(p) for p in best_chain]
