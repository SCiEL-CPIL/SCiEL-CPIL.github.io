"""HP lattice protein folding on the cubic lattice.
Strategy: Beam search growth + simulated annealing with kink moves and deadline guard.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard
_TOTAL_DEADLINE = 330.0  # seconds wall-clock
_start_time = None


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _time_remaining():
    return _TOTAL_DEADLINE - _get_elapsed() - 10.0  # 10s safety margin


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


def _beam_growth(sequence, dimension, beam_width, rng):
    """Beam search growth: keep top beam_width partial chains at each step."""
    # Each beam entry: (chain, occupied_set, score)
    # Start with single chain
    start = (0,) * dimension
    beams = [([start], {start: 0}, 0)]
    
    for i in range(1, len(sequence)):
        new_beams = []
        for chain, occupied, score in beams:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                # Calculate gain
                gain = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        if q in occupied:
                            j = occupied[q]
                            if j < i - 1 and sequence[j] == "H":
                                gain += 1
                # New chain and occupied set
                new_chain = chain + [site]
                new_occupied = set(occupied)
                new_occupied.add(site)
                new_occupied_map = dict(occupied)
                new_occupied_map[site] = i
                new_score = score + gain
                new_beams.append((new_chain, new_occupied_map, new_score))
        
        if not new_beams:
            return None
        
        # Keep top beam_width by score
        new_beams.sort(key=lambda x: x[2], reverse=True)
        beams = new_beams[:beam_width]
    
    if not beams:
        return None
    # Return the best complete chain
    return beams[0][0]


def _simulated_annealing(sequence, chain, dimension, rng, time_limit):
    """Simulated annealing with kink moves."""
    n = len(sequence)
    current = list(chain)
    occupied = {p: i for i, p in enumerate(current)}
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    start_temp = 5.0
    end_temp = 0.01
    cooling_rate = 0.999
    
    temp = start_temp
    start_time = time.time()
    
    while temp > end_temp:
        if time.time() - start_time > time_limit:
            break
        
        # Try a kink move: pick a random internal residue and move it
        if n < 4:
            break
        i = rng.randint(1, n - 2)
        old_pos = current[i]
        
        # Find a valid new position adjacent to current[i-1] or current[i+1]
        # but not occupied and not creating self-intersection
        candidates = []
        for site in _neighbours(old_pos):
            if site == current[i - 1] or site == current[i + 1]:
                continue
            if site in occupied:
                continue
            # Check that moving i to site doesn't create overlap with i-1 or i+1
            # site is already checked not to be current[i-1] or current[i+1]
            # Also check that site is not adjacent to any other occupied site in a way that breaks self-avoidance
            # Actually, we just need to ensure the chain remains self-avoiding
            # The only new adjacency is between site and its neighbors
            # But since we're just moving one residue, we need to check that site is not occupied
            # and that the chain remains valid
            candidates.append(site)
        
        if not candidates:
            temp *= cooling_rate
            continue
        
        new_pos = rng.choice(candidates)
        
        # Check if the move is valid (self-avoiding)
        # Remove old_pos from occupied, add new_pos
        # But we need to check that new_pos is not occupied by any other residue
        # We already checked that, but let's be safe
        if new_pos in occupied:
            temp *= cooling_rate
            continue
        
        # Perform the move
        occupied.pop(old_pos)
        occupied[new_pos] = i
        current[i] = new_pos
        
        # Recalculate energy (or update incrementally for efficiency)
        new_energy = _energy(sequence, current)
        delta = new_energy - current_energy
        
        # Metropolis criterion
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best = list(current)
        else:
            # Revert the move
            occupied.pop(new_pos)
            occupied[old_pos] = i
            current[i] = old_pos
        
        temp *= cooling_rate
    
    return best, best_energy


def fold(sequence, dimension):
    """Best conformation using beam search + simulated annealing with deadline guard."""
    if _start_time is None:
        global _start_time
        _start_time = time.time()
    
    remaining = _time_remaining()
    if remaining <= 0:
        # Return a straight line as fallback
        return [list([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    # Phase 1: Beam search to get an initial good chain
    beam_width = min(50, max(10, n // 2))
    initial_chain = _beam_growth(sequence, dimension, beam_width, rng)
    
    if initial_chain is None:
        # Fallback: straight line
        return [list([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    best_chain = initial_chain
    best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Simulated annealing to improve
    remaining = _time_remaining()
    if remaining > 5.0:
        sa_time = remaining - 5.0
        best_chain, best_energy = _simulated_annealing(sequence, best_chain, dimension, rng, sa_time)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
