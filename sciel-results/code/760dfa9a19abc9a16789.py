"""HP lattice protein folding using Simulated Annealing with Pivot and Reversal moves."""
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


def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    for i in range(1, n):
        neighbors = [s for s in _neighbours(chain[-1]) if s not in occupied]
        if not neighbors:
            return None
        site = rng.choice(neighbors)
        chain.append(site)
        occupied[site] = i
    return chain


def _calc_energy_delta_pivot(sequence, chain, k, rot_choice, px, py):
    """
    Calculate the change in energy for a pivot move.
    Pivot at k, rotating suffix k+1..n-1.
    rot_choice: 1 (90), 2 (180), 3 (270).
    px, py: pivot point coordinates.
    Returns: (delta_energy, new_suffix_positions)
    """
    n = len(chain)
    if k >= n - 1:
        return 0, []
    
    # Determine rotation function
    if rot_choice == 1:
        def rot(x, y): return (-y, x)
    elif rot_choice == 2:
        def rot(x, y): return (-x, -y)
    else:
        def rot(x, y): return (y, -x)
        
    old_suffix = chain[k+1:]
    new_suffix = []
    
    # Map of occupied positions for collision check and energy calc
    # We only need to check collisions with prefix [0..k-1]
    # And energy changes involve:
    # 1. Loss of contacts in old_suffix
    # 2. Loss of contacts between old_suffix and prefix
    # 3. Gain of contacts in new_suffix
    # 4. Gain of contacts between new_suffix and prefix
    
    # To be efficient, we can just compute the energy of the whole chain if the move is valid,
    # but incremental is faster.
    # Let's identify the indices involved.
    # Indices in suffix: k+1 to n-1
    # Indices in prefix: 0 to k
    
    # 1. Identify old contacts to be removed
    removed_contacts = 0
    # Contacts within old suffix
    for i in range(k+1, n):
        if sequence[i] == 'H':
            p_old = old_suffix[i - (k+1)]
            for q in _neighbours(p_old):
                # Check if q is in old suffix (index > i) or prefix (index < k+1)
                # We need to map position to index.
                # This is getting complex. For N<=100, full recalculation is ~100*4 ops = 400 ops.
                # 10000 iterations * 400 ops = 4,000,000 ops. This is very fast in Python (<0.1s).
                # So, we can just do a validity check, then full energy calc.
                pass
        
    # Strategy:
    # 1. Generate new suffix positions.
    # 2. Check collision with prefix [0..k-1].
    # 3. If valid, compute full energy of new chain.
    
    prefix_occupied = set(chain[:k]) # Positions 0 to k-1
    
    valid = True
    for j in range(k + 1, n):
        ox = chain[j][0] - px
        oy = chain[j][1] - py
        nx, ny = rot(ox, oy)
        new_pos = (px + nx, py + ny)
        new_suffix.append(new_pos)
        if new_pos in prefix_occupied:
            valid = False
            break
            
    if not valid:
        return None, None
        
    # Construct full new chain to calculate energy
    # Note: We assume the move is valid with respect to self-avoidance.
    # We still need to calculate energy.
    new_chain = chain[:k+1] + new_suffix
    new_energy = _energy(sequence, new_chain)
    old_energy = _energy(sequence, chain)
    
    return new_energy - old_energy, new_chain


def _calc_energy_delta_reversal(sequence, chain, i, j):
    """
    Calculate energy change for a reversal move of segment [i, j].
    Reversal maps P_m -> P_{i+j-m} for m in [i, j].
    """
    n = len(chain)
    if i >= j:
        return 0, chain
    
    # Check validity: The new segment [i, j] must not collide with [0, i-1] or [j+1, n-1].
    # Note: The internal structure of the reversed segment is a mirror image, so it is self-avoiding
    # if the original was. The only collisions can be with the outside.
    
    prefix_occupied = set(chain[:i])
    suffix_occupied = set(chain[j+1:])
    outside_occupied = prefix_occupied.union(suffix_occupied)
    
    # The new positions for indices i..j are chain[j], chain[j-1], ..., chain[i].
    # We need to check if any of these positions (except the endpoints which connect to outside)
    # collide with the outside.
    # Actually, the endpoints i and j stay at the same "interfaces" but the positions change.
    # Position of new P_i is old P_j.
    # Position of new P_j is old P_i.
    # Position of new P_m is old P_{i+j-m}.
    
    # Collision check:
    # For m in [i, j]:
    #   new_pos = chain[i+j-m]
    #   If m != i and m != j, new_pos must not be in outside_occupied.
    #   If m == i, new_pos = chain[j]. Must not be in outside_occupied (it was in inside, but now it's at boundary i).
    #   Wait, the "outside" set contains positions of indices < i and > j.
    #   The new position of index i is chain[j]. Is chain[j] in outside_occupied?
    #   chain[j] was originally at index j, which is not in outside_occupied (it was inside).
    #   However, it might collide with a position that is in outside_occupied?
    #   No, outside_occupied is a set of coordinates.
    #   So we just check if the new coordinate is in the set of coordinates of the outside indices.
    
    for m in range(i, j + 1):
        new_pos = chain[i + j - m]
        # The new position is occupied by index m.
        # It must not be occupied by any index < i or > j.
        if new_pos in outside_occupied:
            return None, None
            
    # If valid, compute energy
    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
    new_energy = _energy(sequence, new_chain)
    old_energy = _energy(sequence, chain)
    
    return new_energy - old_energy, new_chain


def _pivot_and_reversal_search(sequence, chain, max_iters, rng, time_limit):
    """
    Simulated Annealing with Pivot and Reversal moves.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Simulated Annealing parameters
    T_start = 1.5
    T_end = 0.01
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
    T = T_start
    
    start_time = time.time()
    
    for iteration in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        # Choose move type: 50% Pivot, 50% Reversal
        if rng.random() < 0.5:
            # Pivot Move
            k = rng.randint(0, n - 2)
            rot_choice = rng.choice([1, 2, 3])
            px, py = current_chain[k]
            
            delta_E, new_chain = _calc_energy_delta_pivot(sequence, current_chain, k, rot_choice, px, py)
            if delta_E is None:
                continue
        else:
            # Reversal Move
            # Pick i, j such that i < j and j - i >= 1
            # Bias towards shorter segments? Or random?
            # Random uniform is fine.
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            
            delta_E, new_chain = _calc_energy_delta_reversal(sequence, current_chain, i, j)
            if delta_E is None:
                continue
        
        # Acceptance criterion
        if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
            current_chain = new_chain
            current_energy = current_energy + delta_E
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        T *= cooling
        
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using multiple restarts of SA with Pivot/Reversal."""
    
    global _fold_start_time
    if not hasattr(fold, '_start_time'):
        fold._start_time = time.time()
        fold._total_time_used = 0.0
    
    # Total budget is 120s. We are called 10 times.
    # We need to be careful not to exceed the total.
    # However, the prompt says "120 s CPU budget for all sequences together".
    # If we run 10 sequences, we have 12s each on average.
    # But if one sequence is hard, it might take longer.
    # Let's allocate a fixed time per sequence to ensure we finish all of them.
    # 12s per sequence is safe.
    
    seq_time_limit = 11.0
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
        
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Number of restarts depends on time and sequence length
    # For short sequences, more restarts are cheap.
    # For long sequences, fewer restarts but longer SA.
    
    num_restarts = 5 if n > 50 else 10
    
    start_time = time.time()
    
    for trial in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > seq_time_limit:
            break
            
        remaining = seq_time_limit - elapsed
        if remaining < 0.5:
            break
            
        # Generate initial chain
        # Mix of greedy and random
        if trial % 2 == 0:
            chain = _greedy_grow(sequence, dimension, rng)
        else:
            chain = _random_walk(sequence, dimension, rng)
            
        if chain is None:
            continue
            
        # Run SA
        # Allocate remaining time to this trial, but leave a bit for next if possible?
        # No, just use up to remaining time.
        max_iters = 50000 # High enough, time limit will cut it off
        refined_chain, refined_energy = _pivot_and_reversal_search(
            sequence, chain, 
            max_iters=max_iters,
            rng=rng,
            time_limit=remaining
        )
        
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain
            
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
