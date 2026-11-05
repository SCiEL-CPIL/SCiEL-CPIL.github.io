"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 9
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    best_path = None
    best_len = 0
    
    # Precompute neighbor masks
    bit_masks = [1 << b for b in range(N)]
    all_neighbors = [v ^ (1 << b) for v in range(1 << N) for b in range(N)]
    # Precompute for each vertex, the list of neighbors (9 each)
    neighbors = [[] for _ in range(1 << N)]
    for v in range(1 << N):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    # Precompute hamming distance lookup? No, too big (512x512 = 262144 entries, actually fine)
    # But we can use bit_count which is fast in Python 3.10+
    
    def greedy_extend_fast(start_path, rng, max_time):
        """Extend a starting path using greedy strategy with randomization."""
        path = list(start_path)
        used = [False] * (1 << N)
        for v in path:
            used[v] = True
        
        while time.time() < max_time:
            cur = path[-1]
            # Shuffle bit order for randomization
            bits = list(range(N))
            rng.shuffle(bits)
            
            next_v = None
            for b in bits:
                v = cur ^ (1 << b)
                if used[v]:
                    continue
                # Check against all previous vertices (except the last one, which is cur)
                valid = True
                for i in range(len(path) - 1):
                    if (v ^ path[i]).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    next_v = v
                    break
            
            if next_v is None:
                break
            path.append(next_v)
            used[next_v] = True
        
        return path
    
    def backtrack_extend_fast(path, used, depth, rng, max_time):
        """Try to extend with limited backtracking."""
        if time.time() > max_time:
            return path, used
        
        cur = path[-1]
        bits = list(range(N))
        rng.shuffle(bits)
        
        for b in bits:
            v = cur ^ (1 << b)
            if used[v]:
                continue
            valid = True
            for i in range(len(path) - 1):
                if (v ^ path[i]).bit_count() == 1:
                    valid = False
                    break
            if valid:
                path.append(v)
                used[v] = True
                # Try to extend further
                path, used = backtrack_extend_fast(path, used, depth + 1, rng, max_time)
                if len(path) > 0:
                    return path, used
                # Backtrack
                path.pop()
                used[v] = False
        
        return path, used
    
    # Multiple restarts with different random seeds
    for restart in range(1000):
        if time.time() > deadline:
            break
        
        rng = random.Random(restart * 7919 + 1)
        
        # Start from 0 or a random vertex
        if restart == 0:
            start = [0]
        else:
            start = [rng.randint(0, (1 << N) - 1)]
        
        # First, do a pure greedy extension
        path = greedy_extend_fast(start, rng, deadline - 2.0)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        # If we have time, try some backtracking from the current end
        if time.time() < deadline - 5.0 and len(path) > best_len - 10:
            # Try backtracking from the last few positions
            used = [False] * (1 << N)
            for v in path:
                used[v] = True
            # Try removing the last few vertices and exploring alternatives
            for trim in range(1, min(6, len(path))):
                if time.time() > deadline - 5.0:
                    break
                trimmed_path = path[:-trim]
                # Reset used for trimmed path
                for v in path[trim:]:
                    used[v] = False
                # Try to extend from the trimmed end
                new_path, new_used = backtrack_extend_fast(trimmed_path, used, 0, rng, deadline - 5.0)
                if len(new_path) > best_len:
                    best_len = len(new_path)
                    best_path = list(new_path)
                # Restore used for next trim iteration
                for v in new_path[len(trimmed_path):]:
                    used[v] = False
        
        # Early termination if we've found a very long snake
        if best_len >= 190:
            break
    
    if best_path is None:
        return [0]
    
    return best_path
# EVOLVE-BLOCK-END
