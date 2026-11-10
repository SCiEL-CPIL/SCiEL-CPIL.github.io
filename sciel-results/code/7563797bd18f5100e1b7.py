"""Rewritten program for HP lattice protein folding with local search."""
# EVOLVE-BLOCK-START
import time
import random
import math


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
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


def _local_search(sequence, chain, max_iters, deadline):
    """
    Local search using single-residue moves.
    Accepts any energy improvement to escape local minima.
    Returns (best_chain, best_energy).
    """
    n = len(sequence)
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    current_chain = list(best_chain)
    current_energy = best_energy
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    iters = 0
    while iters < max_iters:
        if time.time() > deadline:
            break
        iters += 1
        
        # Try moving a random residue
        idx = random.randint(1, n - 2)  # Don't move endpoints
        old_pos = current_chain[idx]
        
        # Try all 6 neighbor positions
        for new_pos in _neighbours(old_pos):
            if new_pos in occupied:
                continue
            if idx > 1 and new_pos == current_chain[idx - 1]:
                continue
            if idx < n - 2 and new_pos == current_chain[idx + 1]:
                continue
            
            # Compute energy change
            # Remove old position effects, add new position effects
            # This is complex, so just compute full energy for simplicity
            # But that's O(n^2) per move... let's try incremental
            
            # Actually, let's do incremental energy computation
            # Contacts involving idx:
            # 1. Remove contacts where idx is the "newer" residue (j > i+1)
            # 2. Remove contacts where idx is the "older" residue (i < j-1)
            # 3. Add contacts at new position
            
            # For now, just compute full energy - simpler and less error-prone
            test_chain = current_chain.copy()
            test_chain[idx] = new_pos
            test_energy = _energy(sequence, test_chain)
            
            if test_energy < current_energy:
                # Accept improvement
                current_chain = test_chain
                current_energy = test_energy
                occupied = {p: i for i, p in enumerate(current_chain)}
                
                if current_energy < best_energy:
                    best_chain = list(current_chain)
                    best_energy = current_energy
                
                break  # Restart neighbor search from new position
            elif test_energy == current_energy and random.random() < 0.1:
                # Plateau escape: accept equal energy with 10% probability
                current_chain = test_chain
                current_energy = test_energy
                occupied = {p: i for i, p in enumerate(current_chain)}
                break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation via randomized greedy growth + local search with time-aware budget."""
    start_time = time.time()
    total_budget = 320.0  # safety margin before 330s kill
    
    rng = random.Random(12345 + len(sequence))
    
    best, best_energy_val = None, 0
    best_is_valid = False
    
    while True:
        remaining = total_budget - (time.time() - start_time)
        if remaining < 0.5:
            break
        
        # Grow a new chain
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Local search with time-aware budget
        ls_time = remaining * 0.8
        ls_max_iters = max(50, int(ls_time * 1000))
        
        polished, polished_energy = _local_search(
            sequence, chain, ls_max_iters, start_time + total_budget
        )
        
        if not best_is_valid or polished_energy < best_energy_val:
            best = polished
            best_energy_val = polished_energy
            best_is_valid = True
    
    if not best_is_valid:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
