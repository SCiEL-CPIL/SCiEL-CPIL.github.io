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
    """Compute HP contact energy: -number of non-adjacent H-H contacts."""
    if len(chain) < 2:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _is_valid_chain(chain):
    """Check if the chain is self-avoiding and has unit steps."""
    if not chain:
        return True
    occupied = set()
    for i, p in enumerate(chain):
        if p in occupied:
            return False
        occupied.add(p)
        if i > 0:
            prev = chain[i - 1]
            if abs(p[0] - prev[0]) + abs(p[1] - prev[1]) != 1:
                return False
    return True


def _try_reversal(sequence, chain, start_idx, end_idx):
    """Attempt a reversal move. Returns new_chain and new_energy if valid, else None."""
    n = len(chain)
    if start_idx == 0 or end_idx == n - 1:
        # Reversal at ends is just reflection, usually not useful or invalid for SAW unless it fits
        # Actually, reversing the whole chain or part of it at the boundary is a valid move type (reflection of tail/head)
        # But standard reversal keeps the connection at start-1 and end+1.
        # If start_idx == 0, the new chain starts with reversed[0]. The connection to 'before' doesn't exist.
        # So it's valid if the internal steps are valid (they are, reversal preserves unit steps) 
        # and no self-intersection.
        pass
    
    # Extract segment
    segment = chain[start_idx:end_idx + 1]
    reversed_segment = list(reversed(segment))
    
    new_chain = chain[:start_idx] + reversed_segment + chain[end_idx + 1:]
    
    # Check validity
    # 1. Self-avoidance
    occupied = set()
    for i, p in enumerate(new_chain):
        if p in occupied:
            return None
        occupied.add(p)
        
    # 2. Unit steps (reversal preserves unit steps internally, but check boundaries)
    # Internal steps are preserved in reverse order, so they are unit steps.
    # We only need to check the boundaries:
    # Connection between start_idx-1 and start_idx (new)
    if start_idx > 0:
        p1 = new_chain[start_idx - 1]
        p2 = new_chain[start_idx]
        if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) != 1:
            return None
            
    # Connection between end_idx and end_idx+1 (new)
    if end_idx < n - 1:
        p1 = new_chain[end_idx]
        p2 = new_chain[end_idx + 1]
        if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) != 1:
            return None
            
    # Calculate energy
    new_energy = _compute_energy(sequence, new_chain)
    return new_chain, new_energy


def _try_pivot(sequence, chain, pivot_idx, rot):
    """Attempt a pivot move. Returns new_chain and new_energy if valid, else None."""
    n = len(chain)
    if n <= 2:
        return None
        
    if pivot_idx <= 0 or pivot_idx >= n:
        return None
        
    pivot = chain[pivot_idx - 1]
    tail = chain[pivot_idx:]
    
    occupied_set = set(chain[:pivot_idx])
    
    rotated = []
    valid = True
    
    for pos in tail:
        dx = pos[0] - pivot[0]
        dy = pos[1] - pivot[1]
        
        if rot == 1: # 90 deg CW
            ndx, ndy = dy, -dx
        elif rot == 2: # 180 deg
            ndx, ndy = -dx, -dy
        else: # 3 rot # 270 deg CW
            ndx, ndy = -dy, dx
            
        new_pos = (pivot[0] + ndx, pivot[1] + ndy)
        
        if new_pos in occupied_set:
            valid = False
            break
        occupied_set.add(new_pos)
        
        if rotated:
            prev_rot = rotated[-1]
            if abs(new_pos[0] - prev_rot[0]) + abs(new_pos[1] - prev_rot[1]) != 1:
                valid = False
                break
        else:
            if abs(new_pos[0] - pivot[0]) + abs(new_pos[1] - pivot[1]) != 1:
                valid = False
                break
                
        rotated.append(new_pos)
        
    if not valid:
        return None
        
    new_chain = chain[:pivot_idx] + rotated
    new_energy = _compute_energy(sequence, new_chain)
    return new_chain, new_energy


def _hill_climb(sequence, chain, time_budget, rng):
    """Hill climbing local search with reversals and pivots."""
    n = len(chain)
    if n <= 2:
        return chain
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    start_time = time.time()
    max_time = time_budget
    stall_limit = 50 + n # Increase stall limit for larger chains
    stall_count = 0
    
    while time.time() - start_time < max_time:
        move_type = rng.random()
        
        if move_type < 0.7:
            # Reversal
            # Pick a random segment length between 2 and min(n, 20) or so
            # Short reversals are more likely to be valid and local
            max_len = min(n, 20)
            if n > 3:
                seg_len = rng.randint(2, max_len)
                if seg_len > 1:
                    start_idx = rng.randint(0, n - seg_len)
                    end_idx = start_idx + seg_len - 1
                    result = _try_reversal(sequence, current_chain, start_idx, end_idx)
                    if result:
                        new_chain, new_energy = result
                        if new_energy < current_energy:
                            current_chain = new_chain
                            current_energy = new_energy
                            if current_energy < best_energy:
                                best_chain = list(current_chain)
                                best_energy = current_energy
                                stall_count = 0
                            else:
                                stall_count += 1
                        else:
                            stall_count += 1
                    else:
                        stall_count += 1
                else:
                    stall_count += 1
            else:
                stall_count += 1
        else:
            # Pivot
            if n > 2:
                pivot_idx = rng.randint(1, n - 1)
                rot = rng.choice([1, 2, 3])
                result = _try_pivot(sequence, current_chain, pivot_idx, rot)
                if result:
                    new_chain, new_energy = result
                    if new_energy < current_energy:
                        current_chain = new_chain
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_chain = list(current_chain)
                            best_energy = current_energy
                            stall_count = 0
                        else:
                            stall_count += 1
                    else:
                        stall_count += 1
                else:
                    stall_count += 1
            else:
                stall_count += 1
                
        if stall_count > stall_limit:
            break
            
    return best_chain


def _generate_initial_chain(sequence, rng):
    """Generate a random self-avoiding walk as initial conformation."""
    n = len(sequence)
    if n <= 2:
        return [(0, 0), (1, 0)][:n]
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    direction = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
    
    for i in range(1, n):
        candidates = []
        last = chain[-1]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_pos = (last[0] + dx, last[1] + dy)
            if new_pos not in occupied:
                candidates.append(new_pos)
        
        if not candidates:
            return None
        
        # Prefer continuing in similar direction to avoid tight corners
        if direction in [(c[0] - last[0], c[1] - last[1]) for c in candidates]:
            new_pos = [c for c in candidates if (c[0] - last[0], c[1] - last[1]) == direction][0]
        else:
            new_pos = rng.choice(candidates)
        
        direction = (new_pos[0] - last[0], new_pos[1] - last[1])
        chain.append(new_pos)
        occupied.add(new_pos)
    
    return chain


def fold(sequence, dimension):
    """Fold HP sequence using Hill Climbing with Restarts."""
    n = len(sequence)
    if n <= 2:
        return [[0, 0], [1, 0]][:n]
    
    # Time budget distribution
    # 120s total. 
    # Give more time to longer sequences.
    # Linear scaling with a cap.
    time_per_seq = 5.0 + (n / 100.0) * 10.0 # 5s for short, 15s for long? 
    # Let's stick to a safe budget. 10 sequences.
    # If we use 10s each, total 100s. Safe.
    # Let's use 8s per sequence to be safe against overhead.
    time_per_seq = 8.0
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = float('inf')
    
    # Multi-start Hill Climbing
    # Number of starts depends on time. 
    # Each hill climb runs until stall or time slice.
    # Let's do a fixed number of starts, each with a time slice.
    num_starts = 5
    time_per_start = time_per_seq / num_starts
    
    for start in range(num_starts):
        initial = _generate_initial_chain(sequence, rng)
        if initial is None:
            initial = [(i, 0) for i in range(n)]
            
        improved = _hill_climb(sequence, initial, time_per_start, rng)
        energy = _compute_energy(sequence, improved)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = improved
            
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
