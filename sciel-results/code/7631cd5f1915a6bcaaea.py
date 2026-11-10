"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard: must finish well before 330s wall-clock.
_TOTAL_START = time.time()
_SAFETY_MARGIN = 30.0  # seconds to leave before hard kill

def _remaining_time():
    """Return remaining seconds before our safety deadline."""
    return _TOTAL_START + 300.0 - _SAFETY_MARGIN - time.time()

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

def _build_h_positions(sequence):
    """Precompute set of H residue indices for fast lookup."""
    return set(i for i, c in enumerate(sequence) if c == "H")

def _contact_delta(sequence, h_pos, i, old_site, new_site, occupied):
    """
    Compute energy delta for moving residue i from old_site to new_site.
    Uses precomputed H position set for fast membership tests.
    Returns: negative delta means energy improves (more contacts).
    """
    if i not in h_pos:
        return 0
    
    delta = 0
    # Lose contacts at old_site
    for q in _neighbours(old_site):
        j = occupied.get(q)
        if j is not None and j != i and j in h_pos and abs(i - j) > 1:
            delta += 1
    # Gain contacts at new_site
    for q in _neighbours(new_site):
        j = occupied.get(q)
        if j is not None and j != i and j in h_pos and abs(i - j) > 1:
            delta -= 1
    return delta

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    h_pos = _build_h_positions(sequence)
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if i in h_pos:
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and occupied[q] in h_pos)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _pivot_move(sequence, chain, occupied, h_pos, i, rng, dimension):
    """
    Attempt a pivot move at index i.
    A pivot move rotates the tail chain[i+1:] around the bond (chain[i-1], chain[i]) or similar.
    Actually, standard pivot: pick a cut point k. Rotate the part after k around the bond before k.
    Here we implement a specific valid move: pick a random interior index i.
    We try to rotate the segment starting at i around the bond (i-1, i).
    Wait, the standard pivot move in HP folding literature usually rotates the tail.
    Let's implement a simpler "slide" or "rotation" that preserves connectivity.
    
    Alternative: "Crankshaft" or "Pivot".
    Let's stick to the single-residue move from the current program as it is robust,
    but add a "segment rotation" if possible? No, that's complex.
    
    Let's refine the single-residue move to be more effective by trying all valid neighbors
    and accepting the best one (steepest descent) or using SA.
    The current program already does this.
    
    Instead, let's implement a "2-opt" like move? No, that breaks connectivity.
    
    Let's try a "segment flip" or "rotation" of a small segment (length 2 or 3).
    For a segment of length 2 (residues i, i+1), we can try to rotate them around the bond (i-1, i) or (i+1, i+2)?
    Actually, rotating a 2-residue segment is equivalent to moving one residue if the other stays?
    
    Let's implement a "Tail Rotation" move:
    Pick a random index k (1 to n-2).
    Rotate the tail chain[k+1:] around the bond (chain[k-1], chain[k]).
    This is a powerful move.
    """
    n = len(chain)
    if n < 4:
        return None, None, 0
    
    # Pick a pivot index k. The tail starts at k+1.
    # The bond is (chain[k-1], chain[k]).
    # We rotate the vector from chain[k] to chain[k+1] around the axis defined by the bond?
    # No, in cubic lattice, we can only rotate by 90, 180, 270 degrees.
    # The direction of the bond (k-1 -> k) is fixed.
    # The direction of the bond (k -> k+1) can be changed to any of the 5 other directions
    # (excluding the reverse of the previous bond and the current direction? No, just any valid neighbor).
    # But we must check self-avoidance for the entire tail.
    
    # This is expensive O(N) per try.
    # Let's try a simpler move: "Slide" a single residue, which we already have.
    
    # Let's try to improve the single residue move by allowing "long range" moves?
    # No, let's stick to the current robust single-residue move but optimize the loop.
    
    return None, None, 0

def _local_search(sequence, chain, max_iters, rng, time_limit, use_annealing=True, polish=False):
    """
    Local search with optional simulated annealing and optional polish mode.
    Returns the best chain found.
    """
    n = len(sequence)
    if n < 3:
        return chain
    
    h_pos = _build_h_positions(sequence)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    occupied = {p: i for i, p in enumerate(chain)}
    
    # Annealing parameters
    if use_annealing:
        if polish:
            T_start = 0.5
            T_end = 0.01
        else:
            T_start = 2.0
            T_end = 0.1
        cooling_rate = (T_end / T_start) ** (1.0 / max(1, max_iters))
    
    iters = 0
    no_improve_count = 0
    patience = 200 if polish else 500
    
    while iters < max_iters:
        if time.time() > time_limit:
            break
        iters += 1
        
        # Pick a random interior residue to try moving
        i = rng.randrange(1, n - 1)
        old_site = chain[i]
        
        improved = False
        # Try each neighbor of old_site as a new position
        for new_site in _neighbours(old_site):
            # Check if new_site is free (not occupied by any other residue)
            if new_site in occupied:
                continue
            
            # Check adjacency with neighbors
            if i > 0:
                prev = chain[i-1]
                if abs(new_site[0]-prev[0]) + abs(new_site[1]-prev[1]) + abs(new_site[2]-prev[2]) != 1:
                    continue
            if i < n - 1:
                nxt = chain[i+1]
                if abs(new_site[0]-nxt[0]) + abs(new_site[1]-nxt[1]) + abs(new_site[2]-nxt[2]) != 1:
                    continue
            
            # Compute energy change using fast contact delta
            delta = _contact_delta(sequence, h_pos, i, old_site, new_site, occupied)
            
            new_energy = current_energy + delta
            
            # Acceptance criterion
            accept = False
            if new_energy < current_energy:
                accept = True
            elif use_annealing and iters < max_iters:
                T = T_start * (cooling_rate ** iters)
                if T > 0:
                    prob = math.exp(-delta / T)
                    if rng.random() < prob:
                        accept = True
            
            if accept:
                # Apply the move
                occupied.pop(old_site)
                occupied[new_site] = i
                chain[i] = new_site
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                improved = True
                break  # Only one move per iteration
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= patience:
                break
    
    return best_chain

def fold(sequence, dimension):
    """Best conformation of a fixed number of randomized greedy growths, with local search and annealing."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    n = len(sequence)
    
    # Run growths with dynamic budget based on remaining time
    if n <= 48:
        num_growths = 250
    elif n <= 80:
        num_growths = 150
    else:
        num_growths = 100
    
    for attempt in range(num_growths):
        if _remaining_time() < 5.0:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if best is None or value < best_energy:
            best, best_energy = chain, value
    
    # Try local search on the best chain if time permits
    if best is not None and _remaining_time() > 10.0:
        remaining = _remaining_time()
        # Split time: 50% for first local search, 30% for second, 20% for polish
        ls_time1 = min(time.time() + remaining * 0.5, _TOTAL_START + 300.0 - _SAFETY_MARGIN)
        
        # First local search pass with annealing
        max_ls_iters = max(500, min(15000, 80 * len(sequence)))
        improved_chain = _local_search(sequence, best, max_ls_iters, rng, ls_time1, use_annealing=True)
        improved_energy = _energy(sequence, improved_chain)
        if improved_energy < best_energy:
            best, best_energy = improved_chain, improved_energy
        
        # Second local search pass (steepest descent) if time remains
        if _remaining_time() > 5.0:
            ls_time2 = min(time.time() + _remaining_time() * 0.3, _TOTAL_START + 300.0 - _SAFETY_MARGIN)
            max_ls_iters2 = max(300, min(10000, 50 * len(sequence)))
            improved_chain2 = _local_search(sequence, best, max_ls_iters2, rng, ls_time2, use_annealing=False)
            improved_energy2 = _energy(sequence, improved_chain2)
            if improved_energy2 < best_energy:
                best, best_energy = improved_chain2, improved_energy2
        
        # Final high-precision polish pass with tighter parameters
        if _remaining_time() > 3.0:
            polish_time = min(time.time() + _remaining_time() * 0.2, _TOTAL_START + 300.0 - _SAFETY_MARGIN)
            max_polish_iters = max(1000, min(20000, 100 * len(sequence)))
            polished_chain = _local_search(sequence, best, max_polish_iters, rng, polish_time, use_annealing=True, polish=True)
            polished_energy = _energy(sequence, polished_chain)
            if polished_energy < best_energy:
                best, best_energy = polished_chain, polished_energy
    
    if best is None:  # a straight line is always self-avoiding
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
