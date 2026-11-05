"""HP lattice protein folding using greedy growth with local search refinement."""
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


def _pivot_search(sequence, chain, max_iters=2000, rng=None):
    """
    Monte Carlo search using pivot moves.
    A pivot move rotates the suffix of the chain (from pivot to end) around the pivot point.
    This guarantees that if the new suffix does not collide with the prefix, the chain remains
    a valid self-avoiding walk with unit steps.
    """
    if rng is None:
        rng = random.Random()
    
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Simulated Annealing parameters
    T_start = 1.0
    T_end = 0.05
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
    T = T_start
    
    for iteration in range(max_iters):
        # Pick a pivot point (index of the residue before the one being rotated)
        # Pivot index k means we rotate the chain from k+1 to n-1 around chain[k]
        # k ranges from 0 to n-2
        k = rng.randint(0, n - 2)
        
        # The axis of rotation is the bond between k and k+1.
        # In 2D, we can rotate the suffix by 90, 180, or 270 degrees.
        # Actually, pivot moves usually pick a pivot residue and rotate the tail relative to the bond before it.
        # Let's define the pivot as the bond (k, k+1). We rotate the segment [k+1, ..., n-1] around node k.
        # The vector from k to k+1 defines the local frame.
        
        # Get the vector from k to k+1
        v_x = current_chain[k+1][0] - current_chain[k][0]
        v_y = current_chain[k+1][1] - current_chain[k][1]
        
        # Possible rotations for the next vector (k+1 relative to k)
        # Original: (v_x, v_y)
        # Rotate 90 deg CCW: (-v_y, v_x)
        # Rotate 180 deg: (-v_x, -v_y)
        # Rotate 270 deg CCW: (v_y, -v_x)
        
        # We choose one of the 3 non-identity rotations (or sometimes identity, but that's a no-op)
        # To be efficient, we pick a random rotation factor from {1, -1, i, -i} in complex plane terms, 
        # but since we must change the conformation, we pick from the 3 alternatives.
        
        # Let's just try all 3 non-trivial rotations and pick the best one (or use SA acceptance)
        # For speed, we will generate one random rotation per step.
        
        rot_choice = rng.choice([1, 2, 3]) # 1: 90deg, 2: 180deg, 3: 270deg
        
        if rot_choice == 1:
            new_dx, new_dy = -v_y, v_x
        elif rot_choice == 2:
            new_dx, new_dy = -v_x, -v_y
        else:
            new_dx, new_dy = v_y, -v_x
        
        # Apply the rotation to the entire suffix [k+1, ..., n-1]
        # We need to transform coordinates relative to current_chain[k]
        # Let P0 = current_chain[k]
        # For j from k+1 to n-1:
        #   new_P[j] = P0 + R * (old_P[j] - P0)
        
        # To check for collision, we need to see if any new_P[j] for j > k+1 
        # (note: new_P[k+1] is guaranteed to be a neighbor of P0 and distinct from P0, 
        # but it might collide with some other part of the chain? No, P0 is fixed. 
        # new_P[k+1] is a neighbor of P0. It is distinct from P0. 
        # It could be occupied by some P[i] where i < k. 
        # Also, any new_P[j] for j > k+1 could be occupied by P[i] where i < k or i is in the suffix? 
        # The suffix is being moved, so we only check against the prefix [0, ..., k].)
        
        # Optimization: 
        # 1. Compute new positions for suffix.
        # 2. Check if any new position is in the set of occupied positions of the prefix [0, ..., k-1].
        #    Note: current_chain[k] is the pivot, it stays. 
        #    The new positions are for indices k+1 to n-1.
        #    We must ensure no new position equals any old position in [0, ..., k-1].
        
        prefix_occupied = set(current_chain[:k]) # Positions 0 to k-1
        
        valid = True
        new_suffix = []
        
        # We need to apply rotation recursively or using complex numbers for precision?
        # Integer arithmetic is fine.
        # Let's store the old suffix relative to pivot to avoid floating point issues? 
        # No, integer coords are exact.
        
        # To apply rotation R to a vector (x,y) relative to pivot P0:
        # If R is 90deg CCW: (x,y) -> (-y, x)
        # If R is 180deg: (x,y) -> (-x, -y)
        # If R is 270deg CCW: (x,y) -> (y, -x)
        
        # We can compute new coords for each j in suffix.
        
        # However, doing this for every step in a loop of 2000 iterations for n=100 might be slow in Python.
        # n=100, max_iters=2000. 2000 * 100 operations = 200k ops. This is fine.
        
        # Let's pre-compute the rotation function based on rot_choice
        if rot_choice == 1:
            def rot(x, y): return (-y, x)
        elif rot_choice == 2:
            def rot(x, y): return (-x, -y)
        else:
            def rot(x, y): return (y, -x)
            
        px, py = current_chain[k]
        
        new_chain = current_chain[:k+1] # Copy prefix including pivot
        new_chain.extend([None] * (n - k - 1))
        
        # Calculate new positions for k+1 to n-1
        # Note: The relative vector from k to k+1 is (v_x, v_y).
        # The new relative vector is (new_dx, new_dy).
        # For subsequent points, the relative vectors also rotate.
        # So we can just take the old relative vectors and rotate them.
        
        # old_rel[j] = current_chain[j] - current_chain[k]
        # new_abs[j] = current_chain[k] + rot(old_rel[j])
        
        for j in range(k + 1, n):
            ox = current_chain[j][0] - px
            oy = current_chain[j][1] - py
            nx, ny = rot(ox, oy)
            new_pos = (px + nx, py + ny)
            
            # Check collision with prefix (0 to k-1)
            if new_pos in prefix_occupied:
                valid = False
                break
            
            # Also check if new_pos is (px, py)? No, j>=k+1, so relative vector is not 0.
            # But we should also check if new_pos collides with *other* new positions?
            # Since the rotation is a rigid body transformation, if the original suffix was self-avoiding,
            # the new suffix is self-avoiding. We only need to check against the prefix.
            
            new_chain[j] = new_pos
        
        if not valid:
            # Move rejected due to collision
            continue
        
        # Calculate new energy
        # We only need to recalculate energy for contacts involving the moved suffix.
        # However, _energy is O(N^2) or O(N) with hash map. For N=100, it's fast.
        # To be safe and simple, we recompute full energy.
        new_energy = _energy(sequence, new_chain)
        
        # Acceptance criterion: Simulated Annealing
        delta_E = new_energy - current_energy
        if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        T *= cooling
        
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using greedy growth followed by local search refinement."""
    start_time = time.time()
    deadline = start_time + 150  # Leave 60s margin for 210s wall-clock limit
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    n = len(sequence)
    
    # Phase 1: Greedy growth with multiple restarts
    # Use fewer restarts as pivot search is more powerful per iteration
    num_grow = 20 if n <= 30 else (10 if n <= 60 else 5)
    
    for trial in range(num_grow):
        if time.time() > deadline:
            break
        
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Phase 2: Pivot Search refinement
        # Allocate more iterations to pivot search as it is the main optimizer
        max_pivot_iters = 5000 if n <= 50 else 3000
        
        refined_chain, refined_energy = _pivot_search(
            sequence, chain, 
            max_iters=max_pivot_iters,
            rng=rng
        )
        
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain
            
            # If we have time, run a second pass of pivot search on the best chain found so far
            if time.time() < deadline - 20:
                further_chain, further_energy = _pivot_search(
                    sequence, best_chain,
                    max_iters=max_pivot_iters,
                    rng=rng
                )
                if further_energy < best_energy:
                    best_energy = further_energy
                    best_chain = further_chain
    
    if best_chain is None:
        # Fallback: a straight line is always self-avoiding
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
