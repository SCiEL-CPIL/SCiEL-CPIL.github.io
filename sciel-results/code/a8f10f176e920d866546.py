"""
Snake-in-the-box in Q_9: construct longest induced path.

Approach: Simulated Annealing (SA) on the space of partial snakes.
Unlike the greedy approach, SA allows "worse" moves (shortening the path)
to escape local optima, and "pivot" moves to change direction without
changing length. This explores the solution space more broadly.

We use:
1. A fast greedy initializer to start near a local optimum.
2. Simulated Annealing with three move types:
   - Extend: Add a valid vertex (improves length).
   - Retract: Remove the last vertex (worsens length, accepted probabilistically).
   - Pivot: Replace the last vertex with a different valid neighbor of the
     second-to-last vertex (same length, explores alternative directions).
3. Time-bounded execution to ensure completion within limits.
"""

import random
import time

N = 9
MAX_VERTEX = 1 << N

# EVOLVE-BLOCK-START

def _is_valid_extension(candidate, path, used):
    """Check if candidate can be appended to path to keep it induced."""
    if candidate in used:
        return False
    # Must differ from last vertex by exactly 1 bit
    if (candidate ^ path[-1]).bit_count() != 1:
        return False
    # Must differ from all other vertices by at least 2 bits
    for p in path[:-1]:
        if (candidate ^ p).bit_count() == 1:
            return False
    return True

def _get_valid_neighbors(vertex, path, used):
    """Get all valid neighbors of vertex that can be appended to path."""
    neighbors = []
    for b in range(N):
        v = vertex ^ (1 << b)
        if v in used:
            continue
        if (v ^ path[-1]).bit_count() != 1:
            continue
        # Check against all other path vertices
        valid = True
        for p in path[:-1]:
            if (v ^ p).bit_count() == 1:
                valid = False
                break
        if valid:
            neighbors.append(v)
    return neighbors

def _greedy_init(start=0):
    """Quick greedy initialization to get a decent starting path."""
    path = [start]
    used = {start}
    while True:
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if _is_valid_extension(v, path, used):
                candidates.append(v)
        if not candidates:
            break
        # Pick random candidate to add variety
        next_v = random.choice(candidates)
        path.append(next_v)
        used.add(next_v)
    return path

def construct():
    """
    Construct the longest snake in Q_9 using Simulated Annealing.
    Returns a list of distinct integers representing the vertices in order.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer for safety
    
    # Initialize
    random.seed(42)
    current_path = _greedy_init(0)
    best_path = list(current_path)
    best_len = len(best_path)
    
    # SA Parameters
    T = 1.0
    cooling_rate = 0.9999
    min_T = 0.01
    
    # We keep a 'used' set for the current path for fast lookups
    current_used = set(current_path)
    
    while time.time() - start_time < time_limit:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to Extend
            cur = current_path[-1]
            candidates = _get_valid_neighbors(cur, current_path, current_used)
            if candidates:
                next_v = random.choice(candidates)
                new_path = current_path + [next_v]
                # This is an improvement
                current_path = new_path
                current_used.add(next_v)
                if len(current_path) > best_len:
                    best_path = list(current_path)
                    best_len = len(current_path)
        elif move_type < 0.8:
            # Try to Pivot (change last vertex)
            if len(current_path) >= 2:
                prev = current_path[-2]
                last = current_path[-1]
                # Find valid neighbors of prev, excluding those in path[:-2]
                # We need to check against path[:-2] and ensure not equal to last
                # Note: last is in current_used, so we remove it temporarily for check?
                # No, _get_valid_neighbors checks against path[:-1] which includes last.
                # We need a specific check for pivot.
                
                # Let's manually find candidates for prev
                candidates = []
                for b in range(N):
                    v = prev ^ (1 << b)
                    if v in current_used:
                        continue
                    if v == last:
                        continue # We are replacing last, so v != last is required for a change
                    # Check against path[:-2]
                    valid = True
                    for p in current_path[:-2]:
                        if (v ^ p).bit_count() == 1:
                            valid = False
                            break
                    if valid:
                        candidates.append(v)
                
                if candidates:
                    new_last = random.choice(candidates)
                    # Update path
                    current_path[-1] = new_last
                    current_used.remove(last)
                    current_used.add(new_last)
                    # Length unchanged, so no best update needed unless we want to track
                    # But this move allows future extensions.
        else:
            # Try to Retract
            if len(current_path) > 1:
                # Accept with probability e^(1/T) ? 
                # Delta E = +1 (worse). Prob = exp(-Delta E / T) = exp(-1/T)
                # Wait, standard SA: accept if exp(-Delta E / T) > random()
                # Here we maximize length. Energy = -Length.
                # Retract increases Energy by 1.
                # Prob = exp(-1 / T)
                if random.random() < (2.71828 ** (-1.0 / T)):
                    last = current_path.pop()
                    current_used.remove(last)
        
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T
    
    return best_path

# EVOLVE-BLOCK-END
