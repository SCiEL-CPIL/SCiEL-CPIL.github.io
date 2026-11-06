import time
import random
import sys
import numpy as np

N = 9
MAX_V = 1 << N

def construct():
    deadline = time.time() + 115.0  # 5s safety margin before 120s limit
    
    # Precompute neighbors for all vertices
    neighbors = [[] for _ in range(MAX_V)]
    for v in range(MAX_V):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    best_path = []
    best_len = 0
    
    def compute_block_counts(used_mask, path):
        """Compute block counts: for each vertex, how many path vertices are at distance 2."""
        block_count = [0] * MAX_V
        for v in path:
            for n in neighbors[v]:
                if not (used_mask >> n) & 1:
                    block_count[n] += 1
        return block_count
    
    def can_extend_from_end(path, used_mask, block_count, end_idx):
        """Check if we can extend path from given end (0 or 1 for last)."""
        cur = path[-1] if end_idx == 1 else path[0]
        for n in neighbors[cur]:
            if not (used_mask >> n) & 1 and block_count[n] == 1:
                return n
        return -1
    
    def greedy_construct(start, rng):
        """Greedy construction with Warnsdorff-like heuristic."""
        path = [start]
        used_mask = 1 << start
        block_count = [0] * MAX_V
        for n in neighbors[start]:
            block_count[n] += 1
        
        while True:
            cur = path[-1]
            candidates = []
            for n in neighbors[cur]:
                if not (used_mask >> n) & 1 and block_count[n] == 1:
                    # Count valid extensions from n
                    deg = 0
                    for nn in neighbors[n]:
                        if not (used_mask >> nn) & 1 and nn != cur:
                            if block_count[nn] == 0:
                                deg += 1
                    candidates.append((deg, n))
            
            if not candidates:
                break
            
            # Sort by degree (prefer lower degree)
            candidates.sort(key=lambda x: x[0])
            
            # Randomized choice from top candidates
            k = min(3, len(candidates))
            idx = rng.randint(0, k - 1)
            chosen = candidates[idx][1]
            
            path.append(chosen)
            used_mask |= (1 << chosen)
            for nn in neighbors[chosen]:
                block_count[nn] += 1
        
        return path
    
    def extend_path(path):
        """Try to extend path from both ends repeatedly."""
        used_mask = 0
        for v in path:
            used_mask |= (1 << v)
        block_count = compute_block_counts(used_mask, path)
        
        extended = True
        while extended:
            extended = False
            
            # Try extending from end
            n = can_extend_from_end(path, used_mask, block_count, 1)
            if n >= 0:
                path.append(n)
                used_mask |= (1 << n)
                for nn in neighbors[n]:
                    if not (used_mask >> nn) & 1:
                        block_count[nn] += 1
                extended = True
                continue
            
            # Try extending from beginning
            n = can_extend_from_end(path, used_mask, block_count, 0)
            if n >= 0:
                path.insert(0, n)
                used_mask |= (1 << n)
                for nn in neighbors[n]:
                    if not (used_mask >> nn) & 1:
                        block_count[nn] += 1
                extended = True
        
        return path
    
    def try_swap_improve(path):
        """Try to improve path by swapping adjacent elements in the middle."""
        # This is a simple local search: try swapping consecutive pairs
        # and see if it allows further extension
        n = len(path)
        if n < 4:
            return path
        
        # Try a few random swap positions
        rng = random.Random()
        for _ in range(min(20, n // 2)):
            i = rng.randint(1, n - 3)  # Avoid ends
            # Check if swapping path[i] and path[i+1] maintains validity
            # This is complex; skip for now and just return
            pass
        
        return path
    
    # Strategy: Many fast greedy restarts with extension
    random.seed(42)
    rng = random.Random(42)
    
    # Generate diverse starting vertices
    start_vertices = list(range(MAX_V))
    rng.shuffle(start_vertices)
    
    # Run greedy + extend for many starts
    max_starts = 500  # Fast operations, should fit in time
    
    for start_idx in range(min(max_starts, len(start_vertices))):
        if time.time() > deadline - 5:
            break
        
        start = start_vertices[start_idx]
        
        # Greedy construction
        path = greedy_construct(start, rng)
        
        # Extend from both ends
        path = extend_path(path)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Final extension pass on best path
    if best_path:
        final_path = extend_path(best_path[:])
        if len(final_path) > best_len:
            best_path = final_path
            best_len = len(best_path)
    
    # Ensure we return a valid path
    if not best_path:
        return [0]
        
    return best_path
