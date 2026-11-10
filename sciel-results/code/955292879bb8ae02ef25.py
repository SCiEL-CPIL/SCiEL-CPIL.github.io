"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math

_GLOBAL_DEADLINE = None
_CALL_COUNT = 0

def _neighbours(point):
    """Yield 6 neighbors of a point in 3D lattice."""
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _energy(sequence, chain):
    """Compute energy as minus the number of non-adjacent H-H contacts."""
    n = len(chain)
    if n == 0:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _beam_search(sequence, dimension, beam_width=50, max_depth=None):
    """
    Beam search to generate initial conformations.
    Returns a list of top conformations (chains).
    """
    if max_depth is None:
        max_depth = len(sequence)
    
    n = len(sequence)
    if n == 0:
        return [[]]
    
    # State: (chain, occupied_set, energy)
    start = (0,) * dimension
    initial_chain = [start]
    initial_occ = {start}
    initial_energy = 0
    beam = [(initial_chain, initial_occ, initial_energy)]
    
    for step in range(1, max_depth):
        new_beam = []
        for chain, occ, energy in beam:
            last = chain[-1]
            for q in _neighbours(last):
                if q in occ:
                    continue
                # New chain
                new_chain = chain + [q]
                new_occ = occ | {q}
                
                # Calculate new energy incrementally or fully?
                # For beam search, full energy is safer and N is small enough in early steps.
                # However, for N=136, this might be slow.
                # Let's optimize: only check new contacts involving the new bead.
                # Contacts involving new bead (index step):
                # Check neighbors of new_chain[-1].
                # If neighbor is in new_chain[0:step-1] and is H, and current is H, add contact.
                # Note: we must exclude adjacent chain bonds (step-1 and step).
                
                new_energy = energy
                if sequence[step] == 'H':
                    # Check neighbors of the new bead
                    for q2 in _neighbours(q):
                        # Find index of q2 in new_chain
                        # Since new_chain is a list, linear search is O(N).
                        # But we can use a dict for occupied positions to indices?
                        # occ is a set. We need index.
                        # Let's just do a quick check.
                        # Actually, maintaining a pos_to_idx dict in beam search state is expensive.
                        # Given beam width 50 and N up to 136, 50 * 136 * 6 checks per step is ~40k ops.
                        # 136 steps * 40k = 5.4M ops. Acceptable in Python? Maybe.
                        
                        # Let's stick to full _energy for correctness and simplicity first,
                        # but if it's too slow, we'll optimize.
                        # Actually, _energy is O(N^2) if not careful, but with dict it's O(N).
                        # _energy builds a dict every time. That's O(N).
                        # So _energy is O(N).
                        # Calling _energy 50 times per step is 50 * O(N).
                        # Total O(N^2 * BeamWidth). 136^2 * 50 = 900k. Very fast.
                        
                        pass # Placeholder, we will call _energy below
                
                # Recalculate energy for simplicity and correctness
                # Optimization: We can pass the previous energy and just check new contacts.
                # But _energy is fast enough.
                new_energy = _energy(sequence, new_chain)
                new_beam.append((new_chain, new_occ, new_energy))
        
        if not new_beam:
            break
            
        # Sort by energy (lower is better)
        new_beam.sort(key=lambda x: x[2])
        
        # Keep top beam_width
        beam = new_beam[:beam_width]
        
    # Return the chains
    return [b[0] for b in beam]

def _rotate_vector(v, axis, angle):
    """
    Rotate a 3D vector v around a given axis (0,1,2) by 90, 180, or 270 degrees.
    """
    x, y, z = v
    if axis == 0: # X axis
        if angle == 1: # 90 deg
            return (x, -z, y)
        elif angle == 2: # 180 deg
            return (x, -y, -z)
        elif angle == 3: # 270 deg
            return (x, z, -y)
    elif axis == 1: # Y axis
        if angle == 1:
            return (z, y, -x)
        elif angle == 2:
            return (-x, y, z)
        elif angle == 3:
            return (-z, y, x)
    elif axis == 2: # Z axis
        if angle == 1:
            return (y, -x, z)
        elif angle == 2:
            return (-x, -y, z)
        elif angle == 3:
            return (-y, x, z)
    return v

def _pivot_move(sequence, chain, i, rng):
    """
    Pivot move: Rotate suffix chain[i:] around chain[i-1].
    Returns (new_chain, energy_delta) or None.
    """
    n = len(chain)
    if i <= 0 or i >= n:
        return None
    
    pivot = chain[i-1]
    suffix = chain[i:]
    
    # Choose rotation axis and angle
    axis = rng.randint(0, 2)
    angle = rng.randint(1, 3)
    
    new_suffix = []
    for p in suffix:
        v = (p[0] - pivot[0], p[1] - pivot[1], p[2] - pivot[2])
        nv = _rotate_vector(v, axis, angle)
        new_p = (pivot[0] + nv[0], pivot[1] + nv[1], pivot[2] + nv[2])
        new_suffix.append(new_p)
        
    # Check validity
    prefix_set = set(chain[:i])
    seen = set()
    for p in new_suffix:
        if p in seen:
            return None
        seen.add(p)
        
    for p in new_suffix:
        if p in prefix_set:
            return None
            
    new_chain = chain[:i] + new_suffix
    
    # Calculate energy delta
    # Only need to check contacts involving the moved part (suffix)
    # Internal contacts of suffix are preserved.
    # We need to check contacts between suffix and prefix.
    
    # Build a map of prefix points to indices
    prefix_map = {}
    for idx in range(i):
        prefix_map[chain[idx]] = idx
        
    old_delta = 0
    for idx in range(i, n):
        if sequence[idx] == 'H':
            p = chain[idx]
            for q in _neighbours(p):
                if q in prefix_map:
                    j = prefix_map[q]
                    if sequence[j] == 'H':
                        # Check if they are adjacent in chain
                        if abs(idx - j) > 1:
                            old_delta += 1

    new_delta = 0
    for idx in range(i, n):
        if sequence[idx] == 'H':
            p = new_chain[idx]
            for q in _neighbours(p):
                if q in prefix_map:
                    j = prefix_map[q]
                    if sequence[j] == 'H':
                        if abs(idx - j) > 1:
                            new_delta += 1

    delta = new_delta - old_delta
    
    return new_chain, delta

def _segment_move(sequence, chain, i, j, rng):
    """
    Segment move: Rotate chain[i+1:j+1] around chain[i].
    Returns (new_chain, energy_delta) or None.
    """
    n = len(chain)
    if i < 0 or j >= n or i >= j:
        return None
        
    pivot = chain[i]
    segment = chain[i+1:j+1]
    
    axis = rng.randint(0, 2)
    angle = rng.randint(1, 3)
    
    new_segment = []
    for p in segment:
        v = (p[0] - pivot[0], p[1] - pivot[1], p[2] - pivot[2])
        nv = _rotate_vector(v, axis, angle)
        new_p = (pivot[0] + nv[0], pivot[1] + nv[1], pivot[2] + nv[2])
        new_segment.append(new_p)
        
    # Check validity
    prefix_set = set(chain[:i+1]) # Includes pivot
    suffix_set = set(chain[j+1:])
    seen = set()
    for p in new_segment:
        if p in seen:
            return None
        seen.add(p)
        
    for p in new_segment:
        if p in prefix_set or p in suffix_set:
            return None
            
    new_chain = chain[:i+1] + new_segment + chain[j+1:]
    
    # Energy delta
    # Internal contacts of segment are preserved.
    # Check contacts between segment and (prefix + suffix).
    
    static_map = {}
    for idx in range(0, i+1):
        static_map[chain[idx]] = idx
    for idx in range(j+1, n):
        static_map[chain[idx]] = idx
        
    old_delta = 0
    for idx in range(i+1, j+1):
        if sequence[idx] == 'H':
            p = chain[idx]
            for q in _neighbours(p):
                if q in static_map:
                    j_idx = static_map[q]
                    if sequence[j_idx] == 'H':
                        if abs(idx - j_idx) > 1:
                            old_delta += 1

    new_delta = 0
    for idx in range(i+1, j+1):
        if sequence[idx] == 'H':
            p = new_chain[idx]
            for q in _neighbours(p):
                if q in static_map:
                    j_idx = static_map[q]
                    if sequence[j_idx] == 'H':
                        if abs(idx - j_idx) > 1:
                            new_delta += 1

    delta = new_delta - old_delta
    
    return new_chain, delta

def _reversal_move(sequence, chain, i, j, rng):
    """
    Reversal move: Reverse chain[i:j+1].
    Returns (new_chain, energy_delta) or None.
    """
    n = len(chain)
    if i < 0 or j >= n or i >= j:
        return None
        
    segment = chain[i:j+1]
    new_segment = segment[::-1]
    
    # Check validity
    prefix_set = set(chain[:i])
    suffix_set = set(chain[j+1:])
    seen = set()
    for p in new_segment:
        if p in seen:
            return None
        seen.add(p)
        
    for p in new_segment:
        if p in prefix_set or p in suffix_set:
            return None
            
    new_chain = chain[:i] + new_segment + chain[j+1:]
    
    # Energy delta
    # Internal contacts are preserved.
    # Check contacts between segment and static parts.
    
    static_map = {}
    for idx in range(0, i):
        static_map[chain[idx]] = idx
    for idx in range(j+1, n):
        static_map[chain[idx]] = idx
        
    old_delta = 0
    for idx in range(i, j+1):
        if sequence[idx] == 'H':
            p = chain[idx]
            for q in _neighbours(p):
                if q in static_map:
                    j_idx = static_map[q]
                    if sequence[j_idx] == 'H':
                        if abs(idx - j_idx) > 1:
                            old_delta += 1

    new_delta = 0
    for idx in range(i, j+1):
        if sequence[idx] == 'H':
            p = new_chain[idx]
            for q in _neighbours(p):
                if q in static_map:
                    j_idx = static_map[q]
                    if sequence[j_idx] == 'H':
                        if abs(idx - j_idx) > 1:
                            new_delta += 1

    delta = new_delta - old_delta
    
    return new_chain, delta

def _is_valid_chain(chain):
    """
    Check if the chain is self-avoiding and has unit steps.
    """
    n = len(chain)
    if n == 0:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            prev = chain[i-1]
            dist = abs(p[0]-prev[0]) + abs(p[1]-prev[1]) + abs(p[2]-prev[2])
            if dist != 1:
                return False
    return True

def _simulated_annealing(sequence, chain, rng, deadline):
    """
    Simulated Annealing with Pivot and Segment moves.
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    t_end = deadline
    initial_temp = 5.0
    
    temperature = initial_temp
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < t_end:
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.6:
            # Pivot move
            i = rng.randint(1, n - 1)
            result = _pivot_move(sequence, current_chain, i, rng)
        elif move_type < 0.8:
            # Segment move
            # Pick random segment length
            seg_len = rng.randint(3, min(10, n-1))
            i = rng.randint(0, n - 1 - seg_len)
            j = i + seg_len
            result = _segment_move(sequence, current_chain, i, j, rng)
        else:
            # Reversal move
            seg_len = rng.randint(3, min(10, n-1))
            i = rng.randint(0, n - 1 - seg_len)
            j = i + seg_len
            result = _reversal_move(sequence, current_chain, i, j, rng)
        
        if result is None:
            # Failed move (collision), continue
            continue
            
        new_chain, delta = result
        
        # Acceptance
        if delta > 0 or rng.random() < math.exp(delta / temperature):
            current_chain = new_chain
            current_energy -= delta
            no_improve_count = 0
            if current_energy < best_energy:
                best_chain = list(current_chain)
                best_energy = current_energy
        else:
            no_improve_count += 1
            
        # Cooling
        temperature *= 0.999
        
        if temperature < 0.01:
            if no_improve_count > max_no_improve:
                temperature = initial_temp * 0.5
                no_improve_count = 0
            else:
                temperature = initial_temp * 0.1
                no_improve_count = 0

    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Find a good conformation for the given HP sequence.
    Returns a list of lattice points (lists of integers).
    """
    global _GLOBAL_DEADLINE
    global _CALL_COUNT
    if _GLOBAL_DEADLINE is None:
        _GLOBAL_DEADLINE = time.time() + 330.0
    
    _CALL_COUNT += 1
    start_time = time.time()
    remaining = _GLOBAL_DEADLINE - start_time
    if remaining < 1.0:
        # Not enough time, return a simple valid conformation
        n = len(sequence)
        return [[i, 0, 0] for i in range(n)]
    
    # Heuristic: Assume we have ~1/26th of the total time left on average?
    # Total budget 330s. 26 sequences. ~12s per sequence.
    # Let's use 10s per sequence to be safe.
    budget = min(remaining, 10.0)
    deadline = start_time + budget
    
    n = len(sequence)
    rng = random.Random(12345 + len(sequence) + int(time.time() * 1000) % 10000)
    
    # 1. Beam Search for initial conformations
    # Allocate 20% of time to beam search
    beam_time = budget * 0.2
    beam_deadline = start_time + beam_time
    
    # Beam search is greedy, doesn't need deadline check inside if fast enough.
    # But to be safe, we can limit beam width or depth if time runs out.
    # For now, assume it's fast.
    beam_results = _beam_search(sequence, dimension, beam_width=50)
    
    if not beam_results:
        # Fallback: Straight line
        chain = [(i, 0, 0) for i in range(n)]
        best_chain = chain
    else:
        # Take top 3 candidates
        candidates = beam_results[:3]
        
        # 2. Simulated Annealing on each candidate
        # Allocate 80% of time to SA, split among candidates
        sa_budget = budget * 0.8
        sa_time_per_candidate = sa_budget / len(candidates)
        
        best_chain = None
        best_energy = 0
        
        for candidate in candidates:
            sa_deadline = start_time + (time.time() - start_time) + sa_time_per_candidate
            # Actually, we should track absolute time
            current_time = time.time()
            sa_deadline = current_time + sa_time_per_candidate
            
            result_chain, result_energy = _simulated_annealing(sequence, candidate, rng, sa_deadline)
            
            if best_chain is None or result_energy < best_energy:
                best_chain = result_chain
                best_energy = result_energy
                
        if best_chain is None:
            best_chain = candidates[0]
            
    # Validation: Ensure the chain is valid (connected and self-avoiding)
    if not _is_valid_chain(best_chain):
        # Fallback to a valid straight line if SA produced an invalid chain
        best_chain = [(i, 0, 0) for i in range(n)]
    
    # Convert to lists of ints
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
