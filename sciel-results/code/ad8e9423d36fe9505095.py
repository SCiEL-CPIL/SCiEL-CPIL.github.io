"""HP lattice protein folding: greedy growth + deterministic local-search polish with time guard."""
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
    """Compute HP contact energy (negative count of H-H non-adjacent contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(sequence)
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
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


def _polish(sequence, chain, max_iter=5000):
    """Deterministic local-search polish: accept only strictly improving single-residue moves."""
    n = len(sequence)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for _ in range(max_iter):
        improved = False
        # Try moving each interior residue (not endpoints) to a neighbouring free site
        for i in range(1, n - 1):
            old_pos = tuple(chain[i])
            for new_pos in _neighbours(old_pos):
                if new_pos == tuple(chain[i - 1]) or new_pos == tuple(chain[i + 1]):
                    continue
                if new_pos in where:
                    continue
                # Tentatively move
                del where[old_pos]
                where[new_pos] = i
                chain[i] = new_pos
                
                new_energy = _energy(sequence, chain)
                
                if new_energy < current_energy:
                    current_energy = new_energy
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                else:
                    # Revert
                    del where[new_pos]
                    where[old_pos] = i
                    chain[i] = old_pos
        
        if not improved:
            break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation from multiple greedy growths, each followed by local-search polish."""
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin within 120s budget
    
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Number of growth attempts based on sequence length
    if n <= 30:
        num_growths = 300
    elif n <= 50:
        num_growths = 200
    else:
        num_growths = 100
    
    for attempt in range(num_growths):
        if time.time() > deadline:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Polish the grown chain
        polished_chain, polished_energy = _polish(sequence, chain, max_iter=3000)
        
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_chain = polished_chain
        
        # If we've hit a very good energy, reduce polish iterations for remaining attempts
        if best_energy <= -0.4 * n:
            # Good enough, use shorter polish for remaining
            pass
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Final polish with more iterations if time permits
    if time.time() < deadline - 5.0:
        final_chain, final_energy = _polish(sequence, best_chain, max_iter=10000)
        if final_energy < best_energy:
            best_chain = final_chain
            best_energy = final_energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
