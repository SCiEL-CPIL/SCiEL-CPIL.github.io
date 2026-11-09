"""HP lattice protein folding on cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import defaultdict


def _neighbours(point):
    """Yield all 6 neighbours of a point in 3D."""
    for axis in range(3):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy (negative number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _delta_energy_move(sequence, chain, occupied, i, new_pos):
    """
    Compute delta energy when moving residue i from chain[i] to new_pos.
    Returns (delta_energy, is_valid) where delta_energy is new_energy - old_energy.
    """
    old_pos = chain[i]
    if new_pos == old_pos:
        return 0, True
    if new_pos in occupied:
        return 0, False
    
    # Check connectivity: new_pos must be adjacent to chain[i-1] and chain[i+1]
    if i > 0:
        prev = chain[i-1]
        if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) + abs(new_pos[2] - prev[2]) != 1:
            return 0, False
    if i < len(chain) - 1:
        nxt = chain[i+1]
        if abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) + abs(new_pos[2] - nxt[2]) != 1:
            return 0, False
    
    # Compute energy change
    delta = 0
    is_h = sequence[i] == "H"
    
    # Remove contacts involving old_pos
    if is_h:
        for q in _neighbours(old_pos):
            j = occupied.get(q, -1)
            if j != -1 and j != i - 1 and j != i + 1 and sequence[j] == "H":
                # Check if this contact was counted (j > i)
                if j > i:
                    delta += 1  # We're losing a contact, energy increases by 1
    
    # Add contacts involving new_pos
    if is_h:
        for q in _neighbours(new_pos):
            j = occupied.get(q, -1)
            if j != -1 and j != i - 1 and j != i + 1 and sequence[j] == "H":
                # Check if this contact will be counted (j > i)
                if j > i:
                    delta -= 1  # We're gaining a contact, energy decreases by 1
    
    # Also need to check: when we remove old_pos, some other H-H contacts might be affected
    # Actually, the contacts are between non-adjacent residues, so moving i only affects
    # contacts involving i. The above should be correct.
    
    return delta, True


def _anneal(sequence, dimension, rng, time_limit):
    """Simulated annealing to find low-energy conformation."""
    n = len(sequence)
    
    # Initialize with a random self-avoiding walk
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                options.append(site)
        if not options:
            return None
        site = rng.choice(options)
        chain.append(site)
        occupied[site] = i
    
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random residue to move
        i = rng.randint(1, n - 2)  # Don't move ends
        
        # Try random nearby positions
        old_pos = chain[i]
        candidates = []
        for site in _neighbours(old_pos):
            if site not in occupied or occupied[site] == i:
                candidates.append(site)
        # Also try positions 2 steps away
        for site in _neighbours(old_pos):
            for site2 in _neighbours(site):
                if site2 != old_pos and (site2 not in occupied or occupied[site2] == i):
                    if site2 not in candidates:
                        candidates.append(site2)
        
        if not candidates:
            T *= cooling
            continue
        
        new_pos = rng.choice(candidates)
        if new_pos == old_pos:
            continue
        
        delta, valid = _delta_energy_move(sequence, chain, occupied, i, new_pos)
        if not valid:
            continue
        
        new_energy = current_energy + delta
        
        # Acceptance criterion
        if new_energy < current_energy or rng.random() < math.exp(-delta / T):
            # Apply move
            del occupied[old_pos]
            chain[i] = new_pos
            occupied[new_pos] = i
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        T *= cooling
    
    return best_chain


def _greedy_grow(sequence, dimension, rng):
    """Greedy growth with randomization."""
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def fold(sequence, dimension):
    """Find best conformation for the given sequence."""
    n = len(sequence)
    rng = random.Random(42 + len(sequence))
    
    # Time budget: ~8 seconds per sequence (240s / 26 sequences ≈ 9.2s, use 8s)
    time_budget = 8.0
    start_time = time.time()
    
    best_chain = None
    best_energy = 0  # Start with 0 (no contacts)
    
    # Phase 1: Multiple greedy growths
    greedy_time = min(1.0, time_budget * 0.2)
    greedy_end = start_time + greedy_time
    
    for _ in range(100):
        if time.time() > greedy_end:
            break
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is not None:
            energy = _compute_energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
    
    # Phase 2: Annealing from best greedy result
    if best_chain is not None:
        remaining = time_budget - (time.time() - start_time)
        if remaining > 0.5:
            anneal_result = _anneal(sequence, dimension, rng, remaining * 0.7)
            if anneal_result is not None:
                energy = _compute_energy(sequence, anneal_result)
                if energy < best_energy:
                    best_energy = energy
                    best_chain = anneal_result
    
    # Phase 3: More annealing from random starts if time remains
    remaining = time_budget - (time.time() - start_time)
    if remaining > 1.0:
        num_anneals = int(remaining / 1.5)
        for _ in range(num_anneals):
            if time.time() - start_time > time_budget:
                break
            # Random start
            chain = [(0, 0, 0)]
            occupied = {(0, 0, 0): 0}
            for i in range(1, n):
                options = [s for s in _neighbours(chain[-1]) if s not in occupied]
                if not options:
                    break
                site = rng.choice(options)
                chain.append(site)
                occupied[site] = i
            
            if len(chain) == n:
                anneal_result = _anneal(sequence, dimension, rng, 1.0)
                if anneal_result is not None:
                    energy = _compute_energy(sequence, anneal_result)
                    if energy < best_energy:
                        best_energy = energy
                        best_chain = anneal_result
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
