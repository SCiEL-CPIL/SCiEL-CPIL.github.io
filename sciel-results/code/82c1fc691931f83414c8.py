"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 175.0


def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _compute_energy(sequence, chain):
    """
    Compute the negative number of non-adjacent H-H contacts.
    Optimized for speed: O(N * 4) lookups.
    """
    n = len(chain)
    if n == 0:
        return 0
    
    # Map position to index
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            # Check 4 neighbors
            if (x + 1, y) in pos_to_idx:
                j = pos_to_idx[(x + 1, y)]
                if abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
            if (x - 1, y) in pos_to_idx:
                j = pos_to_idx[(x - 1, y)]
                if abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
            if (x, y + 1) in pos_to_idx:
                j = pos_to_idx[(x, y + 1)]
                if abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
            if (x, y - 1) in pos_to_idx:
                j = pos_to_idx[(x, y - 1)]
                if abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
                    
    return -contacts // 2


def _greedy_grow(sequence, start_pos, occupied, rng, temperature=0.0):
    """
    Randomized greedy chain growth.
    Returns a list of tuples (the chain) or None if stuck.
    """
    chain = [start_pos]
    local_occupied = dict(occupied)
    local_occupied[start_pos] = 0 # Placeholder index for the start
    
    n = len(sequence)
    
    # We need to grow n-1 more residues if start_pos corresponds to index 0 of the segment
    # But here we are regrowing a suffix. Let's assume we are growing from index 0 of the new segment.
    # The caller should pass the correct sequence slice.
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        
        for site in _neighbours(last):
            if site in local_occupied:
                continue
            
            # Calculate gain (potential new H-H contacts)
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = local_occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            # Calculate free space (heuristic to avoid dead ends)
            free = 0
            for q in _neighbours(site):
                if q not in local_occupied:
                    free += 1
            
            # Score
            # High gain is very good. Free space is good to avoid trapping.
            score = gain * 10.0 + free * 0.5 + rng.random() * temperature
            options.append((score, site))
            
        if not options:
            return None
            
        # Pick best or stochastic pick
        if temperature == 0.0:
            best_score, best_site = max(options, key=lambda x: x[0])
        else:
            # Softmax selection
            max_score = max(o[0] for o in options)
            exp_scores = []
            total = 0.0
            for s, site in options:
                e = math.exp((s - max_score) / temperature)
                exp_scores.append(e)
                total += e
            r = rng.random() * total
            cumulative = 0.0
            best_site = options[-1][1]
            for e, (s, site) in zip(exp_scores, options):
                cumulative += e
                if r <= cumulative:
                    best_site = site
                    break
        
        local_occupied[best_site] = i
        chain.append(best_site)
        
    return chain


def fold(sequence, dimension):
    """
    Fold the sequence using Simulated Annealing with Regrowth moves.
    """
    rng = random.Random(42)
    n = len(sequence)
    
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
        
    # Initial fold using greedy growth
    start = (0, 0)
    initial_occupied = {start: 0}
    current_chain = _greedy_grow(sequence, start, initial_occupied, rng, temperature=0.5)
    
    if current_chain is None:
        # Fallback: simple straight line (might not be optimal but valid)
        current_chain = [(i, 0) for i in range(n)]
        
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Simulated Annealing Parameters
    # We want to explore significantly.
    start_temp = 2.0
    cool_rate = 0.995
    temp = start_temp
    
    # Time management
    start_time = time.time()
    
    # Main SA Loop
    while time.time() < _DEADLINE - 5:
        # Choose a cut point. 
        # Biasing towards the end makes moves cheaper (less regrowth) but less global.
        # Biasing towards the beginning makes moves expensive but more global.
        # A mix is good. Let's pick uniformly from [1, n-2] to allow changing the tail.
        # For small n, we can pick anywhere.
        
        if n <= 5:
            cut = 1
        else:
            # Weighted choice: higher probability for larger cut (regrow shorter suffix)
            # but allow some large regrowths.
            # Let's just pick uniformly from 1 to n-2
            cut = rng.randint(1, n - 2)
            
        prefix = current_chain[:cut]
        suffix_seq = sequence[cut:]
        
        # Regrow the suffix
        # The occupied set for the prefix
        prefix_occupied = {p: i for i, p in enumerate(prefix)}
        
        # The start position for regrowth is the last element of the prefix
        # But we need to be careful: the regrowth function expects to start from index 0 of the new segment
        # and places it at start_pos.
        # The first element of the new segment (index 0) is already placed at prefix[-1].
        # So we should pass suffix_seq[0] as the first residue to be placed?
        # No, _greedy_grow places residue 0 at start_pos.
        # So we should pass suffix_seq and start_pos = prefix[-1].
        # The function will place suffix_seq[0] at prefix[-1] (which is already there).
        # Wait, _greedy_grow adds start_pos to the chain.
        # So if we call _greedy_grow(suffix_seq, prefix[-1], prefix_occupied, ...),
        # it will create a chain [prefix[-1], ...].
        # The first element of this new chain is prefix[-1].
        # We want to replace current_chain[cut:] with this new chain[1:].
        
        new_suffix_chain = _greedy_grow(suffix_seq, prefix[-1], prefix_occupied, rng, temperature=temp)
        
        if new_suffix_chain is None:
            # Failed to regrow, cool down and continue
            temp *= cool_rate
            continue
            
        # Construct new full chain
        # new_suffix_chain[0] is prefix[-1], which is current_chain[cut-1]
        # We want to append new_suffix_chain[1:] to prefix
        candidate_chain = prefix + new_suffix_chain[1:]
        
        # Validate self-avoidance (should be guaranteed by _greedy_grow, but double check is cheap for small N)
        # Actually, _greedy_grow ensures no collision with prefix_occupied.
        # And it ensures self-avoidance within the new suffix.
        # So it should be valid.
        
        new_energy = _compute_energy(sequence, candidate_chain)
        
        # Acceptance criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < math.exp(-delta_e / temp):
            current_chain = candidate_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        temp *= cool_rate
        
        if temp < 0.01:
            # Reheat occasionally to escape local minima
            if rng.random() < 0.1:
                temp = start_temp * 0.5
            else:
                break

    # Final local improvement: try single residue moves on the best chain
    # This is a quick polish step
    if time.time() < _DEADLINE - 2:
        best_chain, best_energy = _simple_local_improve(sequence, best_chain, rng, max_moves=500)

    return [list(p) for p in best_chain]


def _simple_local_improve(sequence, chain, rng, max_moves=500):
    """
    Try to improve a conformation by small local moves.
    Move types:
    1. Move a single interior residue to a neighbouring empty site if connectivity is preserved.
    """
    n = len(chain)
    if n < 3:
        return chain, _compute_energy(sequence, chain)

    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    for _ in range(max_moves):
        if time.time() > _DEADLINE - 1:
            break

        i = rng.randint(1, n - 2)  # interior residues only
        p = chain[i]
        moved = False

        for np in _neighbours(p):
            # Check if np is occupied by another residue
            if np in occupied and occupied[np] != i:
                continue
            
            # Check connectivity: i-1 and i+1 must be adjacent to np
            # dist(np, chain[i-1]) == 1 AND dist(np, chain[i+1]) == 1
            # Note: if np is chain[i-1] or chain[i+1], the distance is 0, which is invalid for "adjacent" in the sense of a step.
            # Actually, if we move i to np, the chain segment is ... - (i-1) - np - (i+1) - ...
            # So np must be adjacent to i-1 and i+1.
            
            dist1 = abs(np[0] - chain[i - 1][0]) + abs(np[1] - chain[i - 1][1])
            dist2 = abs(np[0] - chain[i + 1][0]) + abs(np[1] - chain[i + 1][1])
            
            if dist1 != 1 or dist2 != 1:
                continue

            # Perform the move
            del occupied[p]
            chain[i] = np
            occupied[np] = i
            new_energy = _compute_energy(sequence, chain)

            if new_energy < current_energy:
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
            else:
                # Revert
                del occupied[np]
                chain[i] = p
                occupied[p] = i

            moved = True
            break

    return best_chain, best_energy
# EVOLVE-BLOCK-END
