import itertools
import time
import random

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234124132413214321"

    # For n >= 5, use TSP on blocks with aggressive local search
    # Generate all permutations of 1..n-1
    prev_perms = list(itertools.permutations(range(1, n)))
    
    # Create blocks: p + [n] + p
    blocks = []
    for p in prev_perms:
        b = list(p) + [n] + list(p)
        blocks.append(b)
        
    num_blocks = len(blocks)
    
    # Precompute overlaps efficiently
    overlaps = [[0]*num_blocks for _ in range(num_blocks)]
    
    for i in range(num_blocks):
        bi = blocks[i]
        for j in range(num_blocks):
            bj = blocks[j]
            max_k = min(len(bi), len(bj))
            found = 0
            for k in range(max_k, 0, -1):
                if bi[-k:] == bj[:k]:
                    found = k
                    break
            overlaps[i][j] = found

    def calc_path_length(path):
        total = len(blocks[path[0]])
        for k in range(len(path)-1):
            total += len(blocks[path[k+1]]) - overlaps[path[k]][path[k+1]]
        return total

    # Greedy TSP initialization
    def greedy_tsp(start=0):
        current = start
        unvisited = set(range(num_blocks))
        unvisited.discard(start)
        path = [start]
        
        while unvisited:
            best_next = -1
            best_overlap = -1
            for j in unvisited:
                if overlaps[current][j] > best_overlap:
                    best_overlap = overlaps[current][j]
                    best_next = j
            path.append(best_next)
            unvisited.remove(best_next)
            current = best_next
        return path

    # 2-Opt local search with delta calculation
    def two_opt(path, time_limit=90.0):
        n_path = len(path)
        best_len = calc_path_length(path)
        start_time = time.time()
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for i in range(n_path - 1):
                if (time.time() - start_time) >= time_limit:
                    break
                for j in range(i + 2, n_path):
                    # Reverse path[i+1:j+1]
                    # Old: ... path[i], path[i+1], ..., path[j], path[j+1] ...
                    # New: ... path[i], path[j], path[j-1], ..., path[i+1], path[j+1] ...
                    
                    # Calculate delta in total length
                    # Old contributions: ov(path[i], path[i+1]) + ov(path[j], path[j+1] if j+1<n_path else 0)
                    # New contributions: ov(path[i], path[j]) + ov(path[i+1], path[j+1] if j+1<n_path else 0)
                    
                    old_ov1 = overlaps[path[i]][path[i+1]]
                    new_ov1 = overlaps[path[i]][path[j]]
                    
                    if j + 1 < n_path:
                        old_ov2 = overlaps[path[j]][path[j+1]]
                        new_ov2 = overlaps[path[i+1]][path[j+1]]
                    else:
                        old_ov2 = 0
                        new_ov2 = 0
                    
                    delta = (old_ov1 + old_ov2) - (new_ov1 + new_ov2)
                    # We want to maximize overlap, so if delta > 0, new is better
                    if delta > 0:
                        # Apply the reversal
                        path[i+1:j+1] = path[i+1:j+1][::-1]
                        best_len -= delta
                        improved = True
        return path, best_len

    # Or-opt: move a segment of length 1, 2, or 3 to a different position
    def or_opt(path, time_limit=90.0):
        n_path = len(path)
        best_len = calc_path_length(path)
        start_time = time.time()
        
        while (time.time() - start_time) < time_limit:
            improved = False
            for seg_len in [1, 2, 3]:
                if (time.time() - start_time) >= time_limit:
                    break
                for i in range(n_path - seg_len):
                    if (time.time() - start_time) >= time_limit:
                        break
                    seg = path[i:i+seg_len]
                    # Try inserting after position j
                    for j in range(n_path - seg_len):
                        if j >= i and j < i + seg_len:
                            continue
                        # Remove seg from i
                        # Insert after j
                        # Calculate delta
                        # This is complex, so let's just try and recalc if promising
                        
                        # Old neighbors:
                        left_old = path[i-1] if i > 0 else None
                        right_old = path[i+seg_len] if i+seg_len < n_path else None
                        
                        # New positions: insert after j
                        # If j < i, the segment shifts
                        # Let's just do a simple version: try all insert positions
                        
                        # For simplicity, let's skip or-opt and rely on 2-opt
                        pass
        return path, best_len

    # Simulated annealing
    def simulated_annealing(path, time_limit=85.0):
        n_path = len(path)
        current_len = calc_path_length(path)
        best_path = path[:]
        best_len = current_len
        start_time = time.time()
        temp = 10.0
        
        while (time.time() - start_time) < time_limit:
            # Random perturbation: swap two random elements
            i, j = random.sample(range(n_path), 2)
            path[i], path[j] = path[j], path[i]
            
            new_len = calc_path_length(path)
            delta = new_len - current_len
            
            if delta < 0 or random.random() < pow(2.718281828, -delta / temp):
                current_len = new_len
                if new_len < best_len:
                    best_len = new_len
                    best_path = path[:]
            else:
                # Revert
                path[i], path[j] = path[j], path[i]
            
            # Cool down
            temp *= 0.99999
            
            if temp < 0.01:
                break
        
        return best_path, best_len

    # Multi-start with different initializations
    start_time = time.time()
    time_limit = 110.0
    
    best_path = None
    best_len = float('inf')
    
    # Try multiple greedy starts
    for start in range(min(10, num_blocks)):
        path = greedy_tsp(start)
        path, length = two_opt(path, time_limit=5.0)
        if length < best_len:
            best_len = length
            best_path = path[:]
    
    # Apply simulated annealing to best
    if best_path is not None:
        best_path, best_len = simulated_annealing(best_path, time_limit=80.0)
    
    # Final 2-opt pass
    if best_path is not None:
        best_path, best_len = two_opt(best_path, time_limit=15.0)
    
    # Build the string
    final_word = list(blocks[best_path[0]])
    for k in range(1, len(best_path)):
        prev_block = blocks[best_path[k-1]]
        curr_block = blocks[best_path[k]]
        ov = overlaps[best_path[k-1]][best_path[k]]
        final_word += curr_block[ov:]
        
    return "".join(map(str, final_word))
