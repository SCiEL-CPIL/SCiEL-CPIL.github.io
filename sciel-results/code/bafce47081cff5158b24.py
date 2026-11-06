"""Rewritten HP lattice protein folding program."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = time.time() + 325  # safety margin before 330s kill


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the energy (negative number of H-H contacts)."""
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


def _local_refine(sequence, chain, dimension, rng, max_iterations=5000):
    """Local refinement using 1-residue and 2-residue moves."""
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    # Precompute contact sets for energy updates
    def compute_contacts(seq, occ):
        """Return set of H-H contact pairs (i, j) with i < j."""
        contacts = set()
        for i, p in enumerate(occ):
            if seq[i] == "H":
                for q in _neighbours(p):
                    j = occ.get(q)
                    if j is not None and j > i + 1 and seq[j] == "H":
                        contacts.add((i, j))
        return contacts
    
    current_contacts = compute_contacts(sequence, occupied)
    current_energy = -len(current_contacts)
    
    for _ in range(max_iterations):
        if time.time() > _DEADLINE - 5:
            break
        
        # Choose a random residue index to try moving
        idx = rng.randint(1, n - 2)  # avoid endpoints
        
        # Try 1-residue move: move residue idx to a neighbouring site
        moved = False
        for site in _neighbours(best_chain[idx]):
            if site in occupied:
                # Check if it's the residue itself or adjacent in chain
                if occupied.get(site) == idx:
                    continue
                if abs(occupied.get(site, -1) - idx) <= 1:
                    continue
                continue
            
            # Check self-avoidance: site must not be occupied
            # and moving idx to site must not create adjacency issues
            
            # Temporarily remove idx from occupied
            old_pos = best_chain[idx]
            del occupied[old_pos]
            
            # Check if site conflicts with neighbours in chain
            valid = True
            for nb_idx in (idx - 1, idx + 1):
                if nb_idx >= 0 and nb_idx < n:
                    if best_chain[nb_idx] == site:
                        valid = False
                        break
            
            if not valid:
                occupied[old_pos] = idx
                continue
            
            # Compute new contacts
            new_contacts = set(current_contacts)
            # Remove contacts involving idx
            new_contacts = {(i, j) for (i, j) in new_contacts if i != idx and j != idx}
            # Add new contacts involving idx
            if sequence[idx] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j != idx:
                        if j > idx + 1:
                            new_contacts.add((idx, j))
                        elif j < idx - 1:
                            new_contacts.add((j, idx))
            
            new_energy = -len(new_contacts)
            
            if new_energy < current_energy:
                # Accept the move
                best_chain[idx] = site
                occupied[site] = idx
                current_contacts = new_contacts
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                moved = True
                break
            else:
                # Reject
                occupied[old_pos] = idx
        
        if moved:
            continue
        
        # Try 2-residue move (kink): move two consecutive residues
        if n >= 5:
            idx = rng.randint(1, n - 3)
            moved = False
            for site1 in _neighbours(best_chain[idx]):
                for site2 in _neighbours(best_chain[idx + 1]):
                    # Check validity
                    valid = True
                    old_pos1 = best_chain[idx]
                    old_pos2 = best_chain[idx + 1]
                    
                    # site1 must not be occupied (except by idx or idx+1)
                    if site1 in occupied:
                        if occupied[site1] not in (idx, idx + 1):
                            valid = False
                    if site2 in occupied:
                        if occupied[site2] not in (idx, idx + 1):
                            valid = False
                    if site1 == site2:
                        valid = False
                    
                    # Check adjacency constraints
                    if idx > 0 and best_chain[idx - 1] == site1:
                        valid = False
                    if idx + 2 < n and best_chain[idx + 2] == site2:
                        valid = False
                    # site1 and site2 must be adjacent
                    diff = sum(abs(a - b) for a, b in zip(site1, site2))
                    if diff != 1:
                        valid = False
                    # site1 must not equal old_pos2 and vice versa (unless swapping)
                    if site1 == old_pos2 and site2 == old_pos1:
                        # This is a swap, check it's valid
                        pass
                    else:
                        if site1 == old_pos2:
                            valid = False
                        if site2 == old_pos1:
                            valid = False
                    
                    if not valid:
                        continue
                    
                    # Temporarily remove idx and idx+1
                    del occupied[old_pos1]
                    del occupied[old_pos2]
                    occupied[site1] = idx
                    occupied[site2] = idx + 1
                    
                    # Check self-avoidance more carefully
                    chain_copy = list(best_chain)
                    chain_copy[idx] = site1
                    chain_copy[idx + 1] = site2
                    occ_set = set(chain_copy)
                    if len(occ_set) != n:
                        # Collision
                        del occupied[site1]
                        del occupied[site2]
                        occupied[old_pos1] = idx
                        occupied[old_pos2] = idx + 1
                        continue
                    
                    # Compute new contacts
                    new_contacts = compute_contacts(sequence, occupied)
                    new_energy = -len(new_contacts)
                    
                    if new_energy < current_energy:
                        best_chain[idx] = site1
                        best_chain[idx + 1] = site2
                        current_contacts = new_contacts
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                        moved = True
                        break
                    else:
                        # Reject
                        del occupied[site1]
                        del occupied[site2]
                        occupied[old_pos1] = idx
                        occupied[old_pos2] = idx + 1
                
                if moved:
                    break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using randomized greedy growth and local refinement."""
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 1
    
    n = len(sequence)
    
    # Determine number of restarts based on time budget
    # Each growth takes O(n^2) roughly, refinement takes O(n * max_iter)
    # For n=48, growth is fast; for n=136, it's slower
    
    # First, do several quick greedy growths
    num_growths = 50 if n <= 64 else 20
    
    for _ in range(num_growths):
        if time.time() > _DEADLINE - 10:
            break
        chain = _grow(sequence, dimension, rng)
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
    
    # Local refinement on the best chain
    if time.time() < _DEADLINE - 10:
        refined_chain, refined_energy = _local_refine(sequence, best_chain, dimension, rng, max_iterations=10000)
        if refined_energy < best_energy:
            best_chain = refined_chain
            best_energy = refined_energy
    
    # If we still have time, do more growths and refinements
    while time.time() < _DEADLINE - 5:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best_chain = chain
            best_energy = value
        
        # Refine this new candidate
        if time.time() < _DEADLINE - 5:
            refined_chain, refined_energy = _local_refine(sequence, best_chain, dimension, rng, max_iterations=5000)
            if refined_energy < best_energy:
                best_chain = refined_chain
                best_energy = refined_energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
