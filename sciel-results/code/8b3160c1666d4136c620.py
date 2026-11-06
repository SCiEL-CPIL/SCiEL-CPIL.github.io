"""HP lattice protein folding on the cubic lattice.
Strategy: Simulated Annealing with pivot moves, plus initial growth heuristics.
"""
# EVOLVE-BLOCK-START
import random
import time
import math


def _neighbours(point):
    """Yield the 6 neighbours of a 3D lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _build_adjacency(sequence, chain):
    """Build a dictionary mapping each lattice point to its residue index."""
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    return where


def _compute_contacts(sequence, chain):
    """Count H-H contacts (non-consecutive) on the lattice."""
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return contacts


def _energy(sequence, chain):
    """Energy = -number of H-H contacts."""
    return -_compute_contacts(sequence, chain)


def _initial_growth(sequence, dimension, rng):
    """Grow a chain using a greedy heuristic with some randomness."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    # Start at origin
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score: prefer sites that create H-H contacts
            score = rng.random() * 0.5
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            score += 1.0
            # Slight preference for open space to avoid trapping
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.05 * free
            options.append((score, site))
        
        if not options:
            # Dead end: return what we have (will be incomplete)
            return chain
        
        options.sort(key=lambda x: -x[0])
        # Pick from top options with some randomness
        top_k = min(3, len(options))
        chosen = random.choice(options[:top_k])[1]
        chain.append(chosen)
        occupied[chosen] = i
    
    return chain


def _pivot_move(chain, rng):
    """Apply a single pivot move: select two non-adjacent residues, rotate the segment between them.
    Returns (new_chain, energy_delta) or (None, 0) if move invalid.
    """
    n = len(chain)
    if n < 4:
        return None, 0
    
    # Pick pivot points i and j such that j - i >= 2 (so segment has at least 1 residue)
    # and i+1 < j-1 to avoid trivial moves
    # Actually pivot: pick i, j with 1 <= i < j-1 <= n-2
    # The segment chain[i+1..j] is rotated 180 degrees around the midpoint of chain[i] and chain[j]
    
    i = rng.randint(0, n - 3)
    j = rng.randint(i + 2, n - 1)
    
    # The segment to rotate is chain[i+1] through chain[j]
    # After rotation, the new positions must not collide with the fixed parts
    
    # Check that chain[i] and chain[j] are not adjacent in the chain (they aren't by construction since j >= i+2)
    # But we need to ensure the rotated segment doesn't collide with chain[0..i] or chain[j+1..n-1]
    
    # For efficiency, let's just try it and check
    segment = chain[i+1:j+1]
    # Rotate 180 degrees around midpoint of chain[i] and chain[j]
    # New position of old chain[k] (where i+1 <= k <= j) is:
    # new_k = chain[i] + chain[j] - old_k
    # But wait, the pivot is around the bond between i and i+1, and between j-1 and j?
    # Standard pivot: pick i, j. Rotate segment [i+1, j] around the midpoint of positions i and j.
    # Actually, the standard pivot move rotates the subchain between two non-adjacent residues.
    # Let's use: pick i and j with j - i >= 2. Rotate segment chain[i+1..j] around the point
    # that is the midpoint of chain[i] and chain[j].
    
    # Actually, simpler: the pivot is around the midpoint of the two residues at the ends of the segment.
    # Let's define: the segment is from index i to index j (inclusive), where j - i >= 2.
    # We rotate this segment 180 degrees around the midpoint of chain[i] and chain[j].
    # But that might not be standard. Let's use the common definition:
    # Pick two indices i and j such that 0 <= i < j-1 < n-1.
    # The segment chain[i+1..j] is rotated 180 degrees around the midpoint of chain[i] and chain[j].
    
    # Let me re-implement more carefully.
    # A pivot move selects two non-consecutive residues i and j (|i-j| >= 2).
    # The subchain between them (exclusive of i and j) is rotated 180 degrees.
    # Actually, let's use: select i and j with j >= i + 2.
    # Rotate the segment from i+1 to j around the midpoint of chain[i] and chain[j].
    
    # New position of chain[k] for i+1 <= k <= j:
    # new_pos[k] = chain[i] + chain[j] - chain[k]
    
    # But we need to check that the new positions don't collide with the fixed parts.
    
    # For simplicity and speed, let's just do a full check.
    
    new_chain = chain[:]
    for k in range(i + 1, j + 1):
        new_pos = tuple(chain[i][a] + chain[j][a] - chain[k][a] for a in range(3))
        new_chain[k] = new_pos
    
    # Check self-avoidance
    seen = set()
    for p in new_chain:
        if p in seen:
            return None, 0
        seen.add(p)
    
    # Compute energy delta
    old_energy = _energy(sequence, chain)
    new_energy = _energy(sequence, new_chain)
    delta = new_energy - old_energy
    
    return new_chain, delta


def _energy_delta_pivot(sequence, chain, i, j):
    """Compute energy delta for a pivot move between indices i and j (rotating segment i+1..j).
    Returns delta or None if move is invalid (collision).
    """
    n = len(chain)
    if j - i < 2:
        return None
    
    # The segment to rotate is chain[i+1..j]
    # New position of chain[k] = chain[i] + chain[j] - chain[k]
    
    # First, check for collisions with fixed parts (0..i and j+1..n-1)
    # and within the rotated segment itself
    
    # Build set of fixed positions
    fixed = set(chain[0:i+1])
    fixed.update(chain[j+1:n])
    
    # Compute new positions for the segment
    new_positions = {}
    for k in range(i + 1, j + 1):
        new_pos = tuple(chain[i][a] + chain[j][a] - chain[k][a] for a in range(3))
        new_positions[k] = new_pos
        if new_pos in fixed:
            return None
        if new_pos in new_positions.values():
            # Check if it's a duplicate within the segment
            pass
    
    # Check for duplicates within the segment
    seen_seg = set()
    for k in range(i + 1, j + 1):
        p = new_positions[k]
        if p in seen_seg:
            return None
        seen_seg.add(p)
    
    # Now compute energy delta
    # Old contacts involving the segment
    # New contacts involving the segment
    
    # For efficiency, we can compute the total energy before and after,
    # but that's O(n). For now, let's just do it.
    
    new_chain = chain[:]
    for k in range(i + 1, j + 1):
        new_chain[k] = new_positions[k]
    
    old_e = _energy(sequence, chain)
    new_e = _energy(sequence, new_chain)
    return new_e - old_e


def fold(sequence, dimension):
    """Fold a protein sequence using simulated annealing with pivot moves."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    if n == 2:
        return [[0, 0, 0], [1, 0, 0]]
    
    start_time = time.time()
    deadline = start_time + 320  # Leave 10s margin
    
    # Generate initial conformations
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = float('inf')
    
    # Try multiple initial growths
    for attempt in range(50):
        if time.time() > deadline - 30:
            break
        chain = _initial_growth(sequence, dimension, rng)
        if len(chain) < n:
            # Incomplete, try to repair or skip
            continue
        e = _energy(sequence, chain)
        if e < best_energy:
            best_energy = e
            best_chain = chain[:]
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Simulated annealing
    current_chain = best_chain[:]
    current_energy = best_energy
    
    T_start = 5.0
    T_end = 0.01
    cooling_rate = 0.999
    
    T = T_start
    no_improve = 0
    max_no_improve = 1000
    
    while T > T_end:
        if time.time() > deadline:
            break
        
        # Apply a pivot move
        i = rng.randint(0, n - 3)
        j = rng.randint(i + 2, n - 1)
        
        # Compute new positions
        new_chain = current_chain[:]
        valid = True
        for k in range(i + 1, j + 1):
            new_pos = tuple(current_chain[i][a] + current_chain[j][a] - current_chain[k][a] for a in range(3))
            new_chain[k] = new_pos
        
        # Check self-avoidance
        seen = set()
        for p in new_chain:
            if p in seen:
                valid = False
                break
            seen.add(p)
        
        if not valid:
            T *= cooling_rate
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        T *= cooling_rate
        
        if no_improve > max_no_improve:
            # Restart with a new initial conformation if time allows
            if time.time() > deadline - 30:
                break
            chain = _initial_growth(sequence, dimension, rng)
            if len(chain) == n:
                e = _energy(sequence, chain)
                if e < best_energy:
                    best_energy = e
                    best_chain = chain[:]
                current_chain = chain[:]
                current_energy = e
                T = T_start * 0.5
                no_improve = 0
                max_no_improve = 1000
            else:
                break
    
    # Return the best chain found
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
