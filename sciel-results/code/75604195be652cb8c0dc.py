"""
Snake-in-the-box in Q_9 using constraint-based backtracking with heuristic ordering.
Goal: find an induced path (snake) longer than the greedy seed (104 edges).
"""
import numpy as np
import time
import random

N = 9
NUM_VERTICES = 1 << N  # 512

def construct():
    """
    Returns a list of distinct integers in [0, 2**N) representing an induced path
    in Q_N. Consecutive vertices differ in exactly one bit; non-consecutive
    vertices differ in at least two bits.
    """
    # Precompute neighbors for each vertex
    neighbors = [[] for _ in range(NUM_VERTICES)]
    for v in range(NUM_VERTICES):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    # Precompute adjacency as bitsets for fast chord checking
    # adj_mask[v] = bitmask of all vertices adjacent to v
    adj_mask = [0] * NUM_VERTICES
    for v in range(NUM_VERTICES):
        mask = 0
        for nb in neighbors[v]:
            mask |= (1 << nb)
        adj_mask[v] = mask
    
    start_time = time.time()
    TIME_LIMIT = 110  # Leave 10s buffer
    
    best_path = None
    best_len = 0
    
    def time_exceeded():
        return time.time() - start_time > TIME_LIMIT
    
    def backtrack(path, used_mask, blocked_mask):
        """
        path: list of vertices in current snake
        used_mask: bitmask of vertices in path
        blocked_mask: bitmask of vertices that cannot be added (adjacent to any path vertex except the last)
        """
        nonlocal best_path, best_len
        
        if time_exceeded():
            return
        
        cur_len = len(path)
        if cur_len > best_len:
            best_len = cur_len
            best_path = path[:]
        
        cur = path[-1]
        
        # Find candidates: neighbors of cur that are not used and not blocked
        # A vertex v is a valid candidate if:
        #   - v is not in used_mask
        #   - v is not in blocked_mask (i.e., not adjacent to any path[:-1])
        # Note: v is adjacent to cur=path[-1], which is allowed.
        
        candidates = []
        for nb in neighbors[cur]:
            if not (used_mask >> nb) & 1:
                if not (blocked_mask >> nb) & 1:
                    candidates.append(nb)
        
        if not candidates:
            return
        
        # Heuristic ordering: try candidates that lead to more constrained states first
        # Score = number of valid next steps after adding candidate
        def score_candidate(c):
            # After adding c, new path is path + [c]
            # New blocked = blocked_mask | adj_mask[c] (but we need to be careful)
            # Actually, when we add c to path, the new blocked mask should include
            # all vertices adjacent to any vertex in path[:-1] (the old path without cur)
            # Wait, let me reconsider.
            # 
            # blocked_mask represents vertices that are adjacent to any vertex in path[:-1]
            # When we extend path by c, the new path is path + [c].
            # The new "blocked" for the next step should be vertices adjacent to (path + [c])[:-1] = path
            # So new_blocked = blocked_mask | adj_mask[c] ... no wait.
            # 
            # Let me redefine: 
            # blocked_mask[v] = 1 if v is adjacent to any vertex in path[:-1]
            # When we add c to make new_path = path + [c]:
            #   new_path[:-1] = path
            #   So new_blocked should mark vertices adjacent to any vertex in path
            #   new_blocked = blocked_mask | adj_mask[cur] ... no
            #   Actually, blocked_mask already has vertices adjacent to path[:-1] (old)
            #   We need to add vertices adjacent to cur (which is path[-1])
            #   So new_blocked = blocked_mask | adj_mask[cur]
            #   But wait, that's not right either. Let me think again.
            #
            # Original: path = [v0, v1, ..., vk-1, vk] where vk = cur
            # blocked_mask marks vertices adjacent to any of v0, v1, ..., vk-1
            # When we add c: new_path = [v0, v1, ..., vk-1, vk, c]
            # New blocked should mark vertices adjacent to any of v0, v1, ..., vk-1, vk
            # = old_blocked | adj_mask[vk]
            # = blocked_mask | adj_mask[cur]
            #
            # So for candidate c, the new blocked mask would be:
            # new_blocked = blocked_mask | adj_mask[cur]
            # And the valid next steps from c are neighbors of c that are not in (used_mask | (1<<c)) and not in new_blocked
            
            new_blocked = blocked_mask | adj_mask[cur]
            new_used = used_mask | (1 << c)
            count = 0
            for nb2 in neighbors[c]:
                if not (new_used >> nb2) & 1:
                    if not (new_blocked >> nb2) & 1:
                        count += 1
            return count
        
        # Sort candidates: most constrained first (fewest future options)
        candidates.sort(key=score_candidate)
        
        for c in candidates:
            new_path = path + [c]
            new_used = used_mask | (1 << c)
            new_blocked = blocked_mask | adj_mask[cur]
            backtrack(new_path, new_used, new_blocked)
            
            if time_exceeded():
                return
    
    # Try multiple starting configurations to explore different regions
    # By symmetry, we can fix start=0, but vary the first few steps
    # to explore different "directions" in the hypercube
    
    random.seed(42)
    
    # Strategy: Try several different initial 3-vertex paths
    # to avoid getting stuck in one bad region
    
    initial_configs = []
    # Fix first vertex to 0
    # Try different second vertices (all equivalent by symmetry, but different third vertices)
    for second in range(1, NUM_VERTICES):
        if (0 ^ second).bit_count() == 1:
            for third in neighbors[second]:
                if third != 0 and (third ^ 0).bit_count() != 1:
                    # Valid initial path: 0 -> second -> third
                    initial_configs.append([0, second, third])
    
    # Shuffle and try a subset
    random.shuffle(initial_configs)
    
    # Try up to 50 different initial configurations
    max_configs = 50
    for config in initial_configs[:max_configs]:
        if time_exceeded():
            break
        
        path = config
        used_mask = 0
        for v in path:
            used_mask |= (1 << v)
        
        # Compute initial blocked mask: vertices adjacent to path[:-1] = [0]
        blocked_mask = adj_mask[path[0]]
        
        backtrack(path, used_mask, blocked_mask)
        
        if time_exceeded():
            break
    
    if best_path and best_len > 0:
        return best_path
    else:
        # Fallback to greedy
        path, used = [0], {0}
        while True:
            cur, nxt = path[-1], None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used or any((v ^ p).bit_count() == 1 for p in path[:-1]):
                    continue
                nxt = v
                break
            if nxt is None:
                return path
            path.append(nxt)
            used.add(nxt)
