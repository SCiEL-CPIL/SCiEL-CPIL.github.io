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
    Construct a long induced path (snake) in Q_9 using constraint satisfaction
    with clique constraints on non-consecutive vertices.
    
    Key insight: In an induced path, non-consecutive vertices must be at
    Hamming distance >= 2. We use this to:
    1. Precompute forbidden sets for each vertex given the current path
    2. Use constraint propagation to prune candidates early
    3. Employ multiple strategies: greedy with different heuristics, then local search
    """
    best_path = [0]
    best_len = 1
    
    # Precompute neighbors for all vertices
    neighbors = [[] for _ in range(1 << N)]
    for v in range(1 << N):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    def is_valid_extension(path, v):
        """Check if v can be added to path as the next vertex."""
        if v in path_set:
            return False
        # v must not be adjacent to any vertex in path except the last one
        for p in path[:-1]:
            if (v ^ p).bit_count() == 1:
                return False
        return True
    
    def greedy_extend(path, used_set, bit_order=None, randomize=False, seed=None):
        """Greedy extension with optional randomization."""
        if seed is not None:
            random.seed(seed)
        
        current_path = path[:]
        current_used = used_set.copy()
        
        while True:
            cur = current_path[-1]
            candidates = []
            
            for b in bit_order if bit_order else range(N):
                v = cur ^ (1 << b)
                if v in current_used:
                    continue
                # Check chord constraint: v not adjacent to any path vertex except cur
                valid = True
                for p in current_path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    candidates.append(v)
            
            if not candidates:
                break
            
            if randomize:
                random.shuffle(candidates)
                # Score and pick best among top candidates
                scored = []
                for v in candidates:
                    # Count how many of v's neighbors are still available
                    avail = 0
                    for nb in neighbors[v]:
                        if nb not in current_used and nb != cur:
                            avail += 1
                    scored.append((avail, v))
                scored.sort(reverse=True)
                # Pick from top 3 to add some randomness
                top_k = min(3, len(scored))
                chosen = random.choice(scored[:top_k])[1]
            else:
                # Deterministic: pick neighbor with most available extensions
                best_v = None
                best_score = -1
                for v in candidates:
                    avail = 0
                    for nb in neighbors[v]:
                        if nb not in current_used and nb != cur:
                            avail += 1
                    if avail > best_score:
                        best_score = avail
                        best_v = v
                chosen = best_v
            
            current_path.append(chosen)
            current_used.add(chosen)
        
        return current_path
    
    def backtracking_search(path, used_set, depth, start_time, max_depth_increase=10):
        """
        Backtracking search to extend the path. Uses constraint propagation
        to prune the search space.
        """
        nonlocal best_path, best_len
        
        if time.time() - start_time > TIME_LIMIT - 3:
            return
        
        cur = path[-1]
        
        # Generate valid candidates
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used_set:
                continue
            # Check chord constraint
            valid = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    valid = False
                    break
            if valid:
                candidates.append(v)
        
        if not candidates:
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
            return
        
        # Score candidates: prefer those that leave more options
        def score(v):
            cnt = 0
            for nb in neighbors[v]:
                if nb not in used_set and nb != cur:
                    cnt += 1
            # Penalize closeness to existing path vertices
            min_dist = N
            for p in path[:-1]:
                d = (v ^ p).bit_count()
                if d < min_dist:
                    min_dist = d
            return (cnt, min_dist)
        
        candidates.sort(key=score, reverse=True)
        
        for v in candidates:
            if time.time() - start_time > TIME_LIMIT - 3:
                return
            
            new_path = path + [v]
            new_used = used_set | {v}
            
            if len(new_path) > best_len:
                best_len = len(new_path)
                best_path = new_path[:]
            
            backtracking_search(new_path, new_used, depth + 1, start_time)
    
    def local_search_improve(path, start_time, num_iterations=100):
        """
        Try to improve the path by local search:
        - Try replacing segments with better alternatives
        - Try extending from different points
        """
        nonlocal best_path, best_len
        
        current_path = path[:]
        current_used = set(current_path)
        
        for _ in range(num_iterations):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            improved = False
            
            # Try to extend the current path
            extended = greedy_extend(current_path, current_used, randomize=True, seed=random.randint(0, 10000))
            if len(extended) > len(current_path):
                current_path = extended
                current_used = set(current_path)
                improved = True
            
            if not improved:
                # Try restarting from a random prefix
                prefix_len = random.randint(max(1, len(current_path) // 3), len(current_path) - 1)
                prefix = current_path[:prefix_len]
                prefix_used = set(prefix)
                
                # Extend the prefix greedily with randomization
                extended = greedy_extend(prefix, prefix_used, randomize=True, seed=random.randint(0, 10000))
                if len(extended) > len(current_path):
                    current_path = extended
                    current_used = set(current_path)
                    improved = True
            
            if not improved:
                # Try a completely new random start
                start_v = random.randint(0, (1 << N) - 1)
                new_path = greedy_extend([start_v], {start_v}, randomize=True, seed=random.randint(0, 10000))
                if len(new_path) > len(current_path):
                    current_path = new_path
                    current_used = set(current_path)
                    improved = True
            
            if len(current_path) > best_len:
                best_len = len(current_path)
                best_path = current_path[:]
        
        return current_path
    
    # Strategy 1: Deterministic greedy with different bit orderings
    start_time = time.time()
    
    # Try all bit orderings for the first few steps
    import itertools
    for perm in itertools.permutations(range(N)):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        path = greedy_extend([0], {0}, bit_order=list(perm))
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Strategy 2: Randomized greedy with multiple restarts
    random.seed(42)
    for _ in range(200):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        
        start_v = random.randint(0, (1 << N) - 1)
        path = greedy_extend([start_v], {start_v}, randomize=True, seed=random.randint(0, 100000))
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Strategy 3: Backtracking from the best path found so far
    if time.time() - start_time < TIME_LIMIT - 10:
        backtracking_search(best_path, set(best_path), 0, start_time)
    
    # Strategy 4: Local search improvement
    if time.time() - start_time < TIME_LIMIT - 5:
        local_search_improve(best_path, start_time, num_iterations=50)
    
    return best_path
# EVOLVE-BLOCK-END
