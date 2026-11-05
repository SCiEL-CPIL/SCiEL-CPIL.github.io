"""HP lattice protein folding using greedy growth with local search refinement."""
# EVOLVE-BLOCK-START
import random
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute HP contact energy: -1 for each H-H pair on neighbouring lattice sites not adjacent in chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
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
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_search(sequence, chain, max_iters=2000, rng=None):
    """
    Local search via pivot moves: pick a residue, try rotating it to an adjacent free site.
    Accept if energy improves or is equal (with probability to escape plateaus).
    """
    if rng is None:
        rng = random.Random()
    
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    # Precompute which residues are H
    is_h = [c == 'H' for c in sequence]
    
    for iteration in range(max_iters):
        # Pick a random interior residue to move
        i = rng.randint(1, n - 2)
        
        # Try each neighbour of current position
        for new_site in _neighbours(current_chain[i]):
            if new_site in occupied:
                continue
            
            # Check if moving residue i to new_site keeps self-avoidance
            # Only need to check that new_site is not occupied (already checked)
            # and that the chain remains connected (unit steps preserved automatically
            # since we only move one residue to an adjacent site)
            
            # Compute energy change
            # Remove residue i from occupied
            occupied.pop(current_chain[i], None)
            occupied[new_site] = i
            
            # Check if this creates a self-intersection: new_site must not be adjacent
            # to any other part of the chain in a way that violates self-avoidance.
            # Actually, self-avoidance just means no two residues share the same site.
            # Since we checked new_site is not in occupied, we're fine.
            # But we also need to ensure the chain is still a valid walk:
            # consecutive residues must be unit apart. Moving residue i to new_site
            # is fine as long as new_site is adjacent to both residue i-1 and i+1.
            
            if abs(new_site[0] - current_chain[i-1][0]) + abs(new_site[1] - current_chain[i-1][1]) != 1:
                occupied.pop(new_site, None)
                occupied[current_chain[i]] = i
                continue
            if abs(new_site[0] - current_chain[i+1][0]) + abs(new_site[1] - current_chain[i+1][1]) != 1:
                occupied.pop(new_site, None)
                occupied[current_chain[i]] = i
                continue
            
            # Compute new energy
            new_energy = current_energy
            # Remove old contacts involving residue i
            if is_h[i]:
                for q in _neighbours(current_chain[i]):
                    j = occupied.get(q)
                    if j is not None and j > i + 1 and is_h[j]:
                        new_energy += 1  # removing a contact increases energy
                    elif j is not None and j < i - 1 and is_h[j]:
                        new_energy += 1
            # Add new contacts involving residue i at new_site
            if is_h[i]:
                for q in _neighbours(new_site):
                    j = occupied.get(q)
                    if j is not None and j > i + 1 and is_h[j]:
                        new_energy -= 1
                    elif j is not None and j < i - 1 and is_h[j]:
                        new_energy -= 1
            
            if new_energy <= current_energy:
                # Accept the move
                current_chain[i] = new_site
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                break
            else:
                # Reject, restore state
                occupied.pop(new_site, None)
                occupied[current_chain[i]] = i
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation via multiple greedy growths followed by local search refinement."""
    n = len(sequence)
    deadline = time.time() + 110  # 110s safety margin within 120s budget
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    # Number of restarts based on sequence length
    if n <= 30:
        n_restarts = 30
    elif n <= 50:
        n_restarts = 20
    elif n <= 70:
        n_restarts = 12
    else:
        n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
        
        # Local search refinement
        refined_chain, refined_energy = _local_search(sequence, chain, max_iters=1500, rng=rng)
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = list(refined_chain)
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
