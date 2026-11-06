"""HP lattice protein folding: multi-strategy with simulated annealing and local search."""
# EVOLVE-BLOCK-START
import random
import time
import math

_T0 = time.time()
_DEADLINE = 200.0  # wall-clock limit with safety margin (210s kill, leave 10s buffer)


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy from scratch using position->index map."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _delta_energy(sequence, chain, where, i, new_pos):
    """
    Compute the change in energy when moving residue i from chain[i] to new_pos.
    Returns delta_energy (new_energy - old_energy).
    Assumes new_pos is not occupied by any other residue.
    """
    old_pos = chain[i]
    if old_pos == new_pos:
        return 0
    
    # Remove contacts involving residue i at old_pos
    old_contacts = 0
    if sequence[i] == "H":
        for q in _neighbours(old_pos):
            j = where.get(q)
            if j is not None and j != i:
                if sequence[j] == "H":
                    old_contacts += 1
    
    # Add contacts involving residue i at new_pos
    new_contacts = 0
    if sequence[i] == "H":
        for q in _neighbours(new_pos):
            j = where.get(q)
            if j is not None and j != i:
                if sequence[j] == "H":
                    new_contacts += 1
    
    # Also need to account for contacts between neighbors of i that might change
    # Actually, the contacts are between pairs of H residues. When we move residue i,
    # we only affect contacts that involve residue i.
    # But we also need to check if moving i breaks or creates contacts between other residues?
    # No - contacts are defined by lattice adjacency, not by chain adjacency.
    # Moving residue i only affects contacts that include residue i.
    
    return -(new_contacts - old_contacts)


def _grow(sequence, dimension, rng, greedy_weight=0.8):
    """One randomized greedy chain growth."""
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
            score = greedy_weight * gain + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _simulated_annealing(sequence, chain, rng, max_iters, t_start=1.0, t_end=0.01):
    """
    Simulated annealing with single-residue moves (pick a residue, try moving to adjacent free site).
    Returns the best chain found.
    """
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)
    
    cooling = (t_end / t_start) ** (1.0 / max(1, max_iters - 1))
    temp = t_start
    
    for it in range(max_iters):
        if _remaining() < 2.0:
            break
        
        # Pick a random residue to move (not endpoints to reduce risk of breaking chain)
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        
        # Try each neighbor of old_pos
        candidates = []
        for new_pos in _neighbours(old_pos):
            if new_pos in where:
                # Check if it's occupied by a non-adjacent residue
                j = where[new_pos]
                if abs(j - i) > 1:
                    continue  # Can't move here, occupied by non-adjacent
                # If adjacent, we need to be careful - but for single moves, 
                # we can't swap with adjacent easily. Skip for simplicity.
                continue
            # Check that moving here doesn't create self-intersection
            # The new position is free, so we just need to ensure the chain remains valid
            # Since we're only moving one residue and the new position is free, 
            # the only issue is if the old position becomes "hanging" but that's fine
            # as long as the chain is still connected (which it is, we just moved one node)
            
            # But wait - we need to check that the new position doesn't cause the chain
            # to intersect itself. Since new_pos is not in where, it's free.
            # However, we also need to ensure that removing old_pos and adding new_pos
            # doesn't break the chain connectivity. Since we're moving a single interior
            # residue, the chain remains a path as long as the new position is adjacent
            # to both neighbors of i in the chain.
            
            # Check adjacency to chain neighbors
            if i > 0 and chain[i-1] not in [q for q in _neighbours(new_pos)]:
                continue
            if i < n - 1 and chain[i+1] not in [q for q in _neighbours(new_pos)]:
                continue
            
            candidates.append(new_pos)
        
        if not candidates:
            temp *= cooling
            continue
        
        new_pos = rng.choice(candidates)
        
        # Compute delta energy
        delta_e = _delta_energy(sequence, chain, where, i, new_pos)
        
        if delta_e <= 0 or rng.random() < math.exp(-delta_e / max(temp, 1e-10)):
            # Accept the move
            # Update where
            del where[old_pos]
            where[new_pos] = i
            chain[i] = new_pos
            current_energy += delta_e
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        temp *= cooling
    
    return best_chain, best_energy


def _local_search(sequence, chain, rng, max_iters=500):
    """
    Local search: try single-residue moves that improve energy.
    Returns the best chain found.
    """
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)
    
    for it in range(max_iters):
        if _remaining() < 2.0:
            break
        
        improved = False
        # Try moving each residue
        for i in range(1, n - 1):
            old_pos = chain[i]
            for new_pos in _neighbours(old_pos):
                if new_pos in where:
                    j = where[new_pos]
                    if abs(j - i) > 1:
                        continue
                    continue
            
            # Check adjacency to chain neighbors
            if i > 0 and chain[i-1] not in [q for q in _neighbours(new_pos)]:
                continue
            if i < n - 1 and chain[i+1] not in [q for q in _neighbours(new_pos)]:
                continue
            
            delta_e = _delta_energy(sequence, chain, where, i, new_pos)
            if delta_e < 0:
                # Accept improving move
                del where[old_pos]
                where[new_pos] = i
                chain[i] = new_pos
                current_energy += delta_e
                improved = True
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                break  # Restart scan after a move
        
        if not improved:
            break
    
    return best_chain, best_energy


def _perturb(sequence, chain, rng, num_moves=5):
    """
    Perturb the chain by making random valid moves.
    Returns the perturbed chain.
    """
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    chain = list(chain)
    
    for _ in range(num_moves):
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        candidates = []
        for new_pos in _neighbours(old_pos):
            if new_pos in where:
                j = where[new_pos]
                if abs(j - i) > 1:
                    continue
                continue
            if i > 0 and chain[i-1] not in [q for q in _neighbours(new_pos)]:
                continue
            if i < n - 1 and chain[i+1] not in [q for q in _neighbours(new_pos)]:
                continue
            candidates.append(new_pos)
        
        if candidates:
            new_pos = rng.choice(candidates)
            del where[old_pos]
            where[new_pos] = i
            chain[i] = new_pos
    
    return chain


def fold(sequence, dimension):
    """
    Best conformation using multi-strategy approach:
    1. Generate initial chains via greedy growth
    2. Apply simulated annealing to each
    3. Apply local search for final polish
    4. Use perturbation + local search to escape local minima
    """
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    # Generate a set of initial chains
    initial_chains = []
    num_grow = max(10, min(50, 200 // max(1, n // 10)))
    for _ in range(num_grow):
        if _remaining() < 5.0:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is not None:
            initial_chains.append(chain)
    
    if not initial_chains:
        # Fallback: straight line
        return [list(p) for p in [(i, 0) for i in range(n)]]
    
    best_chain = None
    best_energy = 1
    
    # For each initial chain, apply SA + local search
    for chain in initial_chains:
        if _remaining() < 5.0:
            break
        
        # Simulated annealing
        sa_iters = max(100, min(2000, 5000 // max(1, n // 20)))
        sa_chain, sa_energy = _simulated_annealing(sequence, chain, rng, sa_iters)
        
        # Local search
        ls_chain, ls_energy = _local_search(sequence, sa_chain, rng, max_iters=300)
        
        if ls_energy < best_energy:
            best_energy = ls_energy
            best_chain = ls_chain
    
    # Try to escape local minima with perturbation + local search
    if best_chain is not None and _remaining() > 10.0:
        for _ in range(5):
            if _remaining() < 5.0:
                break
            perturbed = _perturb(sequence, best_chain, rng, num_moves=rng.randint(3, 10))
            ls_chain, ls_energy = _local_search(sequence, perturbed, rng, max_iters=200)
            if ls_energy < best_energy:
                best_energy = ls_energy
                best_chain = ls_chain
    
    if best_chain is None:
        best_chain = initial_chains[0]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
