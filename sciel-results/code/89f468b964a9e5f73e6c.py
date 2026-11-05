"""HP lattice protein folding using beam search initialization and multi-strategy local search."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import defaultdict

# Global deadline guard
_START_TIME = None
_DEADLINE = None
_BUDGET_FRACTION = 0.90  # Use 90% of the 330s budget to be safe

def _get_deadline():
    global _DEADLINE, _START_TIME
    if _DEADLINE is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 330 * _BUDGET_FRACTION
    return _DEADLINE

def _is_deadline_exceeded():
    return time.time() > _get_deadline()

def _time_left():
    return _get_deadline() - time.time()

def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D (for dimension=3)."""
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])

def _energy(sequence, chain):
    """Compute energy as -number of H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(sequence)
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _beam_grow(sequence, dimension, rng, beam_width=10):
    """Beam search growth: keep top-K partial chains by contact score."""
    n = len(sequence)
    if n < 4:
        return _simple_grow(sequence, dimension, rng)
    
    is_h = [c == 'H' for c in sequence]
    start = (0, 0, 0)
    
    # Each beam entry: [score, chain, occupied_dict]
    beams = [[0.0, [start], {start: 0}]]
    
    for i in range(1, n):
        if _is_deadline_exceeded():
            break
            
        new_beams = []
        seen_suffixes = set()
        
        for score, chain, occupied in beams:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                
                # Diversity: use last 4 positions as key
                suffix = tuple(chain[-3:] + [site]) if len(chain) >= 3 else tuple(chain + [site])
                if suffix in seen_suffixes:
                    continue
                seen_suffixes.add(suffix)
                
                gain = 0
                if is_h[i]:
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and is_h[j]:
                            gain += 1
                
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                new_score = score + gain * 4.0 + 0.02 * free + rng.random() * 0.5
                
                new_chain = chain + [site]
                new_occ = dict(occupied)
                new_occ[site] = i
                new_beams.append([new_score, new_chain, new_occ])
        
        if not new_beams:
            break
        
        # Keep top beam_width beams by score
        new_beams.sort(key=lambda x: -x[0])
        beams = new_beams[:beam_width]
    
    if not beams:
        return None
    
    # Return the best beam
    best = max(beams, key=lambda x: x[0])
    return best[1]

def _simple_grow(sequence, dimension, rng):
    """Simple randomized greedy growth."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    is_h = [c == 'H' for c in sequence]
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if is_h[i]:
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and is_h[j]:
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 4.0 + 0.02 * free + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(reverse=True)
        site = options[0][1]
        chain.append(site)
        occupied[site] = i
    
    return chain

def _local_search(sequence, chain, dimension, rng, max_iterations=1000):
    """Local search with single-residue moves and segment rotations."""
    n = len(sequence)
    if n <= 2:
        return chain
    
    is_h = [c == 'H' for c in sequence]
    
    # Build position map
    pos_map = {p: i for i, p in enumerate(chain)}
    
    def compute_energy():
        contacts = 0
        for i, p in enumerate(chain):
            if is_h[i]:
                for q in _neighbours(p):
                    j = pos_map.get(q)
                    if j is not None and j > i + 1 and is_h[j]:
                        contacts += 1
        return -contacts
    
    current_energy = compute_energy()
    best_energy = current_energy
    best_chain = list(chain)
    
    for iteration in range(max_iterations):
        if _is_deadline_exceeded():
            break
        
        # Pick a random move type
        move_type = rng.random()
        
        if move_type < 0.6:
            # Single residue move
            i = rng.randint(1, n - 2)
            if rng.random() < 0.1:
                i = rng.randint(0, n - 1)
            
            p_i = chain[i]
            
            # Try all 6 neighbours
            for site in _neighbours(p_i):
                # Check adjacency constraints
                if i > 0 and _manhattan(site, chain[i-1]) != 1:
                    continue
                if i < n-1 and _manhattan(site, chain[i+1]) != 1:
                    continue
                if site in pos_map:
                    continue
                
                # Make the move
                old_pos = chain[i]
                chain[i] = site
                del pos_map[old_pos]
                pos_map[site] = i
                
                new_energy = compute_energy()
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                    current_energy = new_energy
                    break
                else:
                    # Revert
                    chain[i] = old_pos
                    del pos_map[site]
                    pos_map[old_pos] = i
        
        elif move_type < 0.9:
            # Segment rotation (pivot move)
            if n < 4:
                continue
            pivot = rng.randint(1, n - 3)
            
            # Try rotating the segment starting at pivot+1
            seg_start = pivot + 1
            seg_end = n
            
            # Extract segment
            segment = chain[seg_start:]
            
            # Try all 6 directions for the first step of the segment
            for axis in range(3):
                for step in (-1, 1):
                    # Calculate new position for first segment residue
                    new_first = list(chain[pivot])
                    new_first[axis] += step
                    
                    # Check if new position is valid
                    if new_first in pos_map:
                        continue
                    if _manhattan(new_first, chain[pivot]) != 1:
                        continue
                    
                    # Try to place the rest of the segment
                    new_segment = [tuple(new_first)]
                    valid = True
                    temp_pos = new_first
                    
                    for k in range(1, len(segment)):
                        orig_step = _manhattan(segment[k], segment[k-1])
                        # Calculate direction from previous
                        dx = segment[k][0] - segment[k-1][0]
                        dy = segment[k][1] - segment[k-1][1]
                        dz = segment[k][2] - segment[k-1][2]
                        
                        # Apply same relative movement
                        next_pos = (temp_pos[0] + dx, temp_pos[1] + dy, temp_pos[2] + dz)
                        
                        if next_pos in pos_map or next_pos in new_segment:
                            valid = False
                            break
                        new_segment.append(next_pos)
                        temp_pos = next_pos
                    
                    if not valid:
                        continue
                    
                    # Check if segment end connects properly (if not at end)
                    # For now, we just accept if placement is valid
                    
                    # Apply move
                    old_segment = chain[seg_start:]
                    chain[seg_start:] = new_segment
                    
                    # Update pos_map
                    for p in old_segment:
                        del pos_map[p]
                    for idx, p in enumerate(new_segment):
                        pos_map[p] = seg_start + idx
                    
                    new_energy = compute_energy()
                    
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                        current_energy = new_energy
                        break
                    else:
                        # Revert
                        chain[seg_start:] = old_segment
                        for p in new_segment:
                            del pos_map[p]
                        for idx, p in enumerate(old_segment):
                            pos_map[p] = seg_start + idx
        
        else:
            # Crankshaft move
            if n < 4:
                continue
            i = rng.randint(1, n - 3)
            
            # Try rotating residues i and i+1 around the axis defined by i-1 and i+2
            p_prev = chain[i-1]
            p_next = chain[i+1]
            p_i = chain[i]
            p_ip1 = chain[i+1]
            
            # Skip complex crankshaft for now, just do single moves
            continue
    
    return best_chain

def fold(sequence, dimension):
    """Fold a protein sequence on the simple cubic lattice."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Determine number of trials based on time and sequence length
    # Shorter sequences can afford more trials
    base_trials = 30
    if n < 50:
        base_trials = 50
    elif n < 100:
        base_trials = 30
    else:
        base_trials = 20
    
    for trial in range(base_trials):
        if _is_deadline_exceeded():
            break
        
        # Generate initial chain using beam search
        chain = _beam_grow(sequence, dimension, rng, beam_width=10)
        if chain is None:
            # Fallback to simple grow
            chain = _simple_grow(sequence, dimension, rng)
            if chain is None:
                continue
        
        energy = _energy(sequence, chain)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
        
        # Local search improvement
        # Use more iterations for better initial solutions
        max_iter = 200 if energy < best_energy else 100
        improved = _local_search(sequence, chain, dimension, rng, max_iterations=max_iter)
        improved_energy = _energy(sequence, improved)
        
        if improved_energy < best_energy:
            best_chain = improved
            best_energy = improved_energy
    
    # If no valid chain found, return a straight line
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
