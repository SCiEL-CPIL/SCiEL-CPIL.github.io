"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys
from collections import deque

N = 9
TIME_LIMIT = 115  # seconds, leave buffer before 120s

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a long induced path (snake) in Q_9 using a constraint satisfaction
    approach with efficient chord constraint propagation.
    """
    best_path = [0]
    best_len = 1
    start_time = time.time()
    
    # Precompute neighbors for all vertices
    neighbors = [[] for _ in range(1 << N)]
    for v in range(1 << N):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
            
    # Precompute neighbor sets for O(1) lookups
    neighbor_sets = [set(n) for n in neighbors]

    random.seed(42)

    def dfs_rand(path, used_set, blocked_set, depth):
        nonlocal best_path, best_len
        
        if time.time() - start_time > TIME_LIMIT - 1:
            return
        
        if depth > best_len:
            best_len = depth
            best_path = path[:]
        
        cur = path[-1]
        
        # Generate valid candidates
        candidates = []
        for v in neighbors[cur]:
            if v in used_set:
                continue
            if v in blocked_set:
                continue
            candidates.append(v)
        
        if not candidates:
            return
        
        # Heuristic: Sort candidates by the number of valid options they leave for the next step
        def score_candidate_rand(v):
            # The blocked set for the NEXT step will be blocked_set | neighbors[cur]
            next_blocked = blocked_set | neighbor_sets[cur]
            count = 0
            for w in neighbors[v]:
                if w in used_set:
                    continue
                if w in next_blocked:
                    continue
                count += 1
            # Add small random noise to break ties
            return (count, random.random())
        
        candidates.sort(key=score_candidate_rand, reverse=True)
        
        for v in candidates:
            if time.time() - start_time > TIME_LIMIT - 1:
                return
            
            new_path = path + [v]
            new_used = used_set | {v}
            new_blocked = blocked_set | neighbor_sets[cur]
            
            dfs_rand(new_path, new_used, new_blocked, depth + 1)

    # Run multiple randomized deep searches
    for attempt in range(10):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        dfs_rand([0], {0}, set(), 1)

    return best_path
# EVOLVE-BLOCK-END
