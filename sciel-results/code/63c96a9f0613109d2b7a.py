"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute the energy of a chain. Energy = -number of H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow_beam(sequence, dimension, rng, beam_width=10, time_limit=None):
    """Beam search chain growth: maintain multiple partial paths, prefer sites with new H-H contacts."""
    start = (0,) * dimension
    # Each beam entry: (chain, occupied_dict, score)
    # score = current energy contribution (negative is better)
    beams = [([start], {start: 0}, 0)]
    
    for i in range(1, len(sequence)):
        if time_limit is not None and time.time() > time_limit:
            break
        new_beams = []
        for chain, occupied, score in beams:
            options = []
            for site in _neighbours(chain[-1]):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == "H":
                    gain = sum(1 for q in _neighbours(site)
                               if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                # Prefer more contacts, then more free space
                options.append((gain + 0.1 * free + rng.random(), site))
            if not options:
                continue
            # Take the best option for this beam
            best_option = max(options)
            site = best_option[1]
            new_chain = chain + [site]
            new_occupied = dict(occupied)
            new_occupied[site] = i
            new_score = score - gain  # gain is positive, energy is negative
            new_beams.append((new_chain, new_occupied, new_score))
        
        if not new_beams:
            break
        
        # Keep only the best beam_width beams
        new_beams.sort(key=lambda x: x[2])  # lowest energy (most negative) first
        beams = new_beams[:beam_width]
    
    # Return the best complete chain
    if not beams:
        return None
    best_chain = min(beams, key=lambda x: x[2])[0]
    if len(best_chain) == len(sequence):
        return best_chain
    return None


def _local_refinement(sequence, chain, rng, max_iterations=100):
    """Local refinement via pivot moves: try to improve energy by reconfiguring segments."""
    current_chain = list(chain)
    current_energy = _compute_energy(sequence, current_chain)
    
    for _ in range(max_iterations):
        improved = False
        # Try pivot moves: pick two non-adjacent positions and try to flip a segment
        n = len(current_chain)
        if n < 4:
            break
        
        # Randomly pick a pivot point
        pivot_idx = rng.randint(1, n - 2)
        # Pick a segment length
        seg_len = rng.randint(2, min(6, n - pivot_idx - 1))
        
        # Extract the segment
        segment = current_chain[pivot_idx:pivot_idx + seg_len]
        
        # Try to reinsert the segment in a different orientation
        # Simple approach: try reversing the segment or shifting it
        for attempt in range(5):
            new_segment = segment[:]
            # Try random perturbations
            for pos in range(len(new_segment)):
                if pos > 0:
                    # Try moving this point to a random neighbour
                    for site in _neighbours(new_segment[pos]):
                        # Check if site is free and doesn't conflict
                        # This is a simplified check
                        pass
            break
        
        # For simplicity, try a different local move: single point displacement
        idx = rng.randint(1, n - 2)  # Don't move endpoints
        for site in _neighbours(current_chain[idx]):
            if site in current_chain:
                continue
            # Check if moving idx to site is valid
            # Must be adjacent to idx-1 and idx+1
            prev = current_chain[idx - 1]
            next_pos = current_chain[idx + 1]
            
            # Check adjacency
            prev_diff = [abs(a - b) for a, b in zip(site, prev)]
            next_diff = [abs(a - b) for a, b in zip(site, next_pos)]
            
            if sum(prev_diff) != 1 or sum(next_diff) != 1:
                continue
            
            # Check self-avoidance
            if site in current_chain:
                continue
            
            # Try the move
            old_pos = current_chain[idx]
            current_chain[idx] = site
            new_energy = _compute_energy(sequence, current_chain)
            
            if new_energy < current_energy:
                current_energy = new_energy
                improved = True
                break
            else:
                current_chain[idx] = old_pos
        
        if not improved:
            break
    
    return current_chain, current_energy


def _check_deadline(deadline):
    """Return True if we should stop searching."""
    return time.time() >= deadline


def fold(sequence, dimension):
    """Best conformation using beam search and local refinement."""
    start_time = time.time()
    deadline = start_time + 200  # Leave 40s safety margin for 240s budget
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 1  # Energy is negative, so start high
    
    # Try beam search with different widths
    for beam_width in [5, 10, 20]:
        if _check_deadline(deadline):
            break
        
        for restart in range(50):
            if _check_deadline(deadline):
                break
            
            chain = _grow_beam(sequence, dimension, rng, beam_width=beam_width,
                             time_limit=deadline)
            if chain is None or len(chain) != len(sequence):
                continue
            
            # Apply local refinement
            refined_chain, refined_energy = _local_refinement(sequence, chain, rng, max_iterations=50)
            
            if refined_energy < best_energy:
                best_energy = refined_energy
                best_chain = refined_chain
            
            # Also try the unrefined chain
            unrefined_energy = _compute_energy(sequence, chain)
            if unrefined_energy < best_energy:
                best_energy = unrefined_energy
                best_chain = chain
        
        if _check_deadline(deadline):
            break
    
    # If no valid chain found, use a straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
