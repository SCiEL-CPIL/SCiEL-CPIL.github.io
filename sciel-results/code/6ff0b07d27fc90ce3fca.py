"""HP lattice protein folding on the cubic lattice with multi-start local search."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import defaultdict

# Global deadline guard
_DEADLINE = None
_BUDGET_FRACTION = 0.85  # Use 85% of the 330s budget to be safe


def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330 * _BUDGET_FRACTION
    return _DEADLINE


def _is_deadline_exceeded():
    return time.time() > _get_deadline()


def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D."""
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


def _contact_count(sequence, chain):
    """Count H-H contacts (positive)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return contacts


def _try_move(chain, i, new_pos, occupied):
    """Try to move residue i to new_pos. Returns (success, new_occupied)."""
    old_pos = chain[i]
    if new_pos in occupied and occupied[new_pos] != i:
        return False, occupied
    # Check self-avoidance: new_pos must not be occupied by any other residue
    # Check that the move doesn't create adjacency violations
    # The new position must not be adjacent to itself along the chain (i-1 or i+1)
    # Actually, we need to ensure the chain remains a valid walk
    # After moving, check that new_pos is not the same as chain[i-1] or chain[i+1]
    if i > 0 and new_pos == chain[i - 1]:
        return False, occupied
    if i < len(chain) - 1 and new_pos == chain[i + 1]:
        return False, occupied
    # Also check that new_pos is not occupied by any other residue
    for j, pos in enumerate(chain):
        if j != i and pos == new_pos:
            return False, occupied
    # Perform the move
    new_occupied = dict(occupied)
    del new_occupied[old_pos]
    new_occupied[new_pos] = i
    return True, new_occupied


def _local_search(sequence, chain, rng, max_iterations=5000):
    """Local search using single-residue moves (pulling moves)."""
    n = len(chain)
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)
    
    for iteration in range(max_iterations):
        if _is_deadline_exceeded():
            break
        
        # Pick a random residue to move (not the ends, to keep it simple)
        if n <= 2:
            break
        i = rng.randint(1, n - 2)
        
        # Try all 6 possible moves
        best_move = None
        best_move_energy = current_energy
        
        for new_pos in _neighbours(chain[i]):
            # Check if new_pos is valid
            if new_pos in occupied and occupied[new_pos] != i:
                continue
            if i > 0 and new_pos == chain[i - 1]:
                continue
            if i < n - 1 and new_pos == chain[i + 1]:
                continue
            # Check self-avoidance more thoroughly
            valid = True
            for j, pos in enumerate(chain):
                if j != i and pos == new_pos:
                    valid = False
                    break
            if not valid:
                continue
            
            # Compute the energy change
            # Remove contacts involving residue i
            old_contacts = 0
            for q in _neighbours(chain[i]):
                j = occupied.get(q)
                if j is not None and j != i and sequence[j] == "H" and sequence[i] == "H":
                    if abs(j - i) > 1:
                        old_contacts += 1
            
            # Add new contacts
            new_contacts = 0
            for q in _neighbours(new_pos):
                if q == chain[i - 1] if i > 0 else False:
                    continue
                if q == chain[i + 1] if i < n - 1 else False:
                    continue
                j = occupied.get(q)
                if j is not None and j != i and sequence[j] == "H" and sequence[i] == "H":
                    if abs(j - i) > 1:
                        new_contacts += 1
            
            # Also need to check if moving i breaks contacts between other residues
            # This is complex, so we just compute the full energy for the candidate
            candidate_chain = list(chain)
            candidate_chain[i] = new_pos
            candidate_energy = _energy(sequence, candidate_chain)
            
            if candidate_energy < best_move_energy:
                best_move_energy = candidate_energy
                best_move = new_pos
        
        if best_move is not None:
            chain[i] = best_move
            occupied = {p: idx for idx, p in enumerate(chain)}
            current_energy = best_move_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        else:
            # No improving move found, try a random move to escape local optima
            i = rng.randint(1, n - 2)
            options = []
            for new_pos in _neighbours(chain[i]):
                if new_pos in occupied and occupied[new_pos] != i:
                    continue
                if i > 0 and new_pos == chain[i - 1]:
                    continue
                if i < n - 1 and new_pos == chain[i + 1]:
                    continue
                valid = True
                for j, pos in enumerate(chain):
                    if j != i and pos == new_pos:
                        valid = False
                        break
                if valid:
                    options.append(new_pos)
            if options:
                new_pos = rng.choice(options)
                chain[i] = new_pos
                occupied = {p: idx for idx, p in enumerate(chain)}
                current_energy = _energy(sequence, chain)
    
    return best_chain, best_energy


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


def fold(sequence, dimension):
    """Best conformation using multi-start local search."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0  # 0 contacts = energy 0, we want to maximize contacts (minimize energy)
    
    # Number of restarts based on sequence length
    if n <= 50:
        n_restarts = 30
    elif n <= 100:
        n_restarts = 20
    else:
        n_restarts = 10
    
    for restart in range(n_restarts):
        if _is_deadline_exceeded():
            break
        
        # Generate an initial chain
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            # Try a different random seed
            chain = _grow(sequence, dimension, rng)
            if chain is None:
                continue
        
        # Local search
        chain, energy = _local_search(sequence, chain, rng, max_iterations=2000 if n <= 50 else 1000)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
        
        # Check deadline
        if _is_deadline_exceeded():
            break
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
