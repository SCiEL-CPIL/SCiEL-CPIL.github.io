"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 9
TIME_LIMIT = 110.0  # safety margin below 120s evaluator limit
SEED_LENGTH = 104

# Precompute neighbors for all vertices in Q_9
# neighbors[v] is a tuple of 9 integers
neighbors = []
for v in range(1 << N):
    nb = tuple(v ^ (1 << b) for b in range(N))
    neighbors.append(nb)

# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

# Precompute neighbor masks for fast bitmask operations
# For each vertex, we want to know which neighbors it has
# We'll use bitmasks to track used and excluded vertices
TOTAL_VERTS = 1 << N
ALL_MASK = (1 << TOTAL_VERTS) - 1

# For each vertex v, precompute the bitmask of its neighbors
neighbor_mask = [0] * TOTAL_VERTS
for v in range(TOTAL_VERTS):
    mask = 0
    for b in range(N):
        mask |= 1 << (v ^ (1 << b))
    neighbor_mask[v] = mask

def _warnsdorff_backtrack(rng, time_budget, max_depth=200):
    """
    Warnsdorff backtracking search using bitmask operations for speed.
    """
    path = [0]
    # used_mask: bits set for vertices in the path
    # excluded_mask: bits set for vertices that are neighbors of non-terminal path vertices
    # A vertex is valid if it's not in used_mask and not in excluded_mask
    
    used_mask = 1  # vertex 0 is used
    excluded_mask = 0
    
    # Precompute: for vertex 0, which neighbors are excluded (none yet, 0 is terminal)
    # Actually, excluded should be neighbors of non-terminal vertices.
    # Initially path = [0], so 0 is terminal, no excluded vertices.
    
    # For backtracking, we need to be able to undo steps
    # We'll track the history of (vertex, prev_used_mask, prev_excluded_mask)
    
    best_path = list(path)
    best_len = len(path)
    
    time_check = time.time
    deadline = time_check() + time_budget
    nbr_mask = neighbor_mask
    
    def backtrack(depth, cur, used_m, excluded_m, path_list):
        nonlocal best_path, best_len
        nonlocal deadline
        
        # Time check
        if time_check() > deadline:
            return False
        
        # Update best if current path is longer
        if len(path_list) > best_len:
            best_len = len(path_list)
            best_path = list(path_list)
        
        # Find valid candidates: neighbors of cur that are not used and not excluded
        valid_mask = nbr_mask[cur] & ~used_m & ~excluded_m
        
        if valid_mask == 0:
            # Dead end, backtrack
            return False
        
        # Get list of valid candidates
        candidates = []
        vm = valid_mask
        while vm:
            # Extract lowest set bit
            lsb = vm & (-vm)
            v = lsb.bit_length() - 1
            candidates.append(v)
            vm ^= lsb
        
        # Warnsdorff heuristic: sort by number of valid extensions
        # For each candidate v, count how many of its neighbors are valid
        # A neighbor w of v is valid if w is not in used_m and not in excluded_m and w != cur
        # But we also need to consider that after moving to v, cur becomes excluded
        # So the new excluded_mask will be excluded_m | nbr_mask[cur]
        # And the new used_mask will be used_m | (1 << v)
        
        new_excluded = excluded_m | nbr_mask[cur]
        new_used = used_m | (1 << candidates[0])  # placeholder
        
        # Score each candidate
        scored_candidates = []
        for v in candidates:
            # After moving to v:
            # - v becomes used
            # - cur becomes excluded (its neighbors are added to excluded)
            # - v's neighbors that are not used and not in new_excluded are potential next moves
            
            v_used = used_m | (1 << v)
            v_excluded = excluded_m | nbr_mask[cur]  # cur is now non-terminal
            
            # Count valid next moves from v
            v_valid_mask = nbr_mask[v] & ~v_used & ~v_excluded
            score = bin(v_valid_mask).count('1')
            scored_candidates.append((score, v))
        
        # Sort by score (ascending - Warnsdorff: go to the most constrained first)
        scored_candidates.sort(key=lambda x: x[0])
        
        # Add some randomness for diversity
        if rng.random() < 0.1 and len(scored_candidates) > 1:
            # Occasionally pick a random candidate for diversity
            v = rng.choice(scored_candidates)[1]
        else:
            v = scored_candidates[0][1]
        
        # Make the move
        path_list.append(v)
        v_used = used_m | (1 << v)
        v_excluded = excluded_m | nbr_mask[cur]
        
        # Try to extend
        if backtrack(depth + 1, v, v_used, v_excluded, path_list):
            return True
        
        # Backtrack
        path_list.pop()
        
        # Try other candidates
        for score, v2 in scored_candidates:
            if v2 == v:
                continue
            path_list.append(v2)
            v2_used = used_m | (1 << v2)
            v2_excluded = excluded_m | nbr_mask[cur]
            
            if backtrack(depth + 1, v2, v2_used, v2_excluded, path_list):
                return True
            
            path_list.pop()
        
        return False
    
    # Start backtracking
    path_list = [0]
    backtrack(0, 0, 1, 0, path_list)
    
    return best_path

def _quick_greedy(rng, time_budget):
    """
    Fast greedy search using bitmask operations.
    """
    path = [0]
    used_mask = 1
    excluded_mask = 0
    nbr_mask = neighbor_mask
    time_check = time.time
    deadline = time_check() + time_budget
    
    while True:
        cur = path[-1]
        valid_mask = nbr_mask[cur] & ~used_mask & ~excluded_mask
        
        if valid_mask == 0:
            break
        
        # Get candidates
        candidates = []
        vm = valid_mask
        while vm:
            lsb = vm & (-vm)
            v = lsb.bit_length() - 1
            candidates.append(v)
            vm ^= lsb
        
        if len(candidates) == 1:
            v = candidates[0]
        else:
            # Score by Warnsdorff
            new_excluded = excluded_mask | nbr_mask[cur]
            best_score = -1
            best_v = candidates[0]
            for v in candidates:
                v_used = used_mask | (1 << v)
                v_excluded = excluded_mask | nbr_mask[cur]
                v_valid_mask = nbr_mask[v] & ~v_used & ~v_excluded
                score = bin(v_valid_mask).count('1')
                if score > best_score:
                    best_score = score
                    best_v = v
            v = best_v
        
        path.append(v)
        used_mask |= (1 << v)
        excluded_mask |= nbr_mask[cur]
        
        if time_check() > deadline:
            break
    
    return path

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    best_path = None
    best_len = 0
    
    rng = random.Random(42)
    
    # First: quick greedy to get a baseline
    path = _quick_greedy(rng, 1.0)
    best_path = path
    best_len = len(path)
    
    # Now run Warnsdorff backtracking with different seeds
    search_count = 0
    while time.time() < deadline - 2.0:  # 2s safety margin
        # Use a portion of remaining time for each backtracking attempt
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        
        # Allocate time for this attempt: min(remaining - 1, some fraction)
        attempt_time = min(remaining - 1.0, 10.0)
        
        # Create a new RNG with different seed for diversity
        seed = 42 + search_count * 1000
        local_rng = random.Random(seed)
        
        p = _warnsdorff_backtrack(local_rng, attempt_time)
        
        search_count += 1
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Final: try one more quick greedy with remaining time
    remaining = deadline - time.time()
    if remaining > 0.5:
        p = _quick_greedy(rng, remaining - 0.5)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    return best_path
# EVOLVE-BLOCK-END
