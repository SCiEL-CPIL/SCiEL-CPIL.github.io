"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import heapq
import copy


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


def _local_search(sequence, chain, max_moves=5000, rng=None):
    """
    Perform local search using moves: 
    1. Rotate a segment by 90 degrees around an axis (if valid).
    2. Translate a segment if it creates space (complex, skip for now).
    3. Simple: pick a random index i (not first/last), pick a random neighbour of chain[i] that is not chain[i-1] or chain[i+1] and not occupied. 
       But this breaks the chain connectivity unless we move the whole tail.
    
    Better local search strategy for HP lattice:
    - "Pivot" move: pick a pivot index k. Rotate the sub-chain chain[k+1:] around chain[k] by 90 degrees around one of the 3 axes (or 180).
    - Check self-avoidance.
    - Calculate energy change.
    - Accept if better or with probability exp(-delta/T) (Simulated Annealing).
    
    Let's implement Simulated Annealing with pivot moves.
    """
    if rng is None:
        rng = random.Random()
        
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)

    current_chain = [tuple(p) for p in chain]
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Precompute occupied set for fast lookup
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    # Temperature schedule
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    
    # For efficiency, we only evaluate energy changes locally if possible, 
    # but _energy is O(N^2) roughly (N neighbors per H). For N=136, it's fast enough.
    # Actually, _energy is O(N * 6). So recalculating is cheap.
    
    for step in range(max_moves):
        # Choose pivot index k (1 to n-2)
        k = rng.randint(1, n - 2)
        
        # Choose rotation axis and angle
        # In 3D, we can rotate around any of the 3 axes.
        # A 90 degree rotation around axis 'a' at pivot point P:
        # The vector from P to a point X is v = X - P.
        # Rotating v by 90 deg around axis a.
        
        # Let's try all 6 possible 90-degree rotations (3 axes, 2 directions) + 3 180-degree rotations
        # Actually, let's just pick a random valid move.
        
        # To make it simpler and more effective, let's try "segment translation" is hard.
        # Pivot moves are standard.
        
        # Generate a new tail by rotating the tail around the pivot.
        # We need to map the coordinates.
        
        # Let's pick a random axis (0, 1, or 2) and a random direction (1 or -1)
        axis = rng.randint(0, 2)
        direction = rng.choice([-1, 1])
        
        # We also need to handle the fact that the tail must not intersect the head (chain[:k+1])
        # and not itself.
        
        # Let's construct the new tail.
        # Pivot is current_chain[k].
        # The rest of the chain is current_chain[k+1:].
        
        # We rotate each point in the tail around the pivot.
        # However, a simple rotation might not result in integer lattice points if the axis isn't aligned?
        # No, in a cubic lattice, rotating a vector (dx, dy, dz) by 90 degrees around an axis
        # results in integer coordinates.
        
        # Rotation matrix for 90 deg around axis 0 (x-axis):
        # (x, y, z) -> (x, -z, y) or (x, z, -y)
        # Around axis 1 (y-axis):
        # (x, y, z) -> (z, y, -x) or (-z, y, x)
        # Around axis 2 (z-axis):
        # (x, y, z) -> (-y, x, z) or (y, -x, z)
        
        # Let's implement a helper to rotate a point around a pivot.
        def rotate_point(p, pivot, axis, direction):
            v = [p[i] - pivot[i] for i in range(3)]
            if axis == 0:
                if direction == 1:
                    # (x, y, z) -> (x, -z, y)
                    v = [v[0], -v[2], v[1]]
                else:
                    # (x, y, z) -> (x, z, -y)
                    v = [v[0], v[2], -v[1]]
            elif axis == 1:
                if direction == 1:
                    # (x, y, z) -> (z, y, -x)
                    v = [v[2], v[1], -v[0]]
                else:
                    # (x, y, z) -> (-z, y, x)
                    v = [-v[2], v[1], v[0]]
            else: # axis 2
                if direction == 1:
                    # (x, y, z) -> (-y, x, z)
                    v = [-v[1], v[0], v[2]]
                else:
                    # (x, y, z) -> (y, -x, z)
                    v = [v[1], -v[0], v[2]]
            
            new_p = [pivot[i] + v[i] for i in range(3)]
            return tuple(new_p)
        
        new_tail = []
        valid = True
        # The head is chain[0..k]. The new tail will be chain[k+1..n-1] rotated.
        # We must ensure the first point of the new tail is a neighbour of the last point of the head.
        # Since we are rotating around the pivot (which is the last point of the head), 
        # the first point of the tail (which was adjacent to the pivot) will remain adjacent to the pivot 
        # because rotation preserves distance. 
        # However, we need to check self-avoidance against the head and the new tail itself.
        
        # Occupied set for the head (indices 0 to k)
        head_occupied = {current_chain[i] for i in range(k + 1)}
        
        for i in range(k + 1, n):
            new_p = rotate_point(current_chain[i], current_chain[k], axis, direction)
            if new_p in head_occupied:
                valid = False
                break
            if new_p in new_tail: # Check against previously added tail points
                valid = False
                break
            new_tail.append(new_p)
            
        if not valid:
            continue
            
        new_chain = current_chain[:k+1] + new_tail
        new_energy = _energy(sequence, new_chain)
        
        delta = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        T *= cooling
        if T < T_min:
            T = T_min
            
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
    """
    Use a hybrid approach:
    1. Generate several initial chains using randomized greedy growth.
    2. Refine each chain using Simulated Annealing with pivot moves.
    3. Return the best chain found.
    """
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0 # Energy is negative, so we want the minimum (most negative)
    
    # Number of initial chains to generate
    num_initial = 10
    
    for _ in range(num_initial):
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Refine with local search
        refined_chain, refined_energy = _local_search(sequence, chain, max_moves=2000, rng=rng)
        
        if best_chain is None or refined_energy < best_energy:
            best_chain = refined_chain
            best_energy = refined_energy
            
    # If no chain was generated (unlikely), use a straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
