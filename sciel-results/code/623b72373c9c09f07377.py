import time
import sys
from collections import deque

# Increase recursion limit for DFS
sys.setrecursionlimit(10000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute adjacency as bitmasks for fast operations
    adj_mask = [0] * n
    for u in range(n):
        for v in adj[u]:
            adj_mask[u] |= (1 << v)
    
    best_path = []
    
    # Helper to check time
    def time_check():
        return time.time() > deadline
    
    # Compute degrees for initial ordering
    degrees = [len(adj[i]) for i in range(n)]
    
    # Strategy 1: Greedy Warnsdorff to get a baseline - try multiple starts
    for start in range(n):
        if time_check():
            break
        path = [start]
        visited_mask = 1 << start
        while True:
            curr = path[-1]
            # Get unvisited neighbors using bitmask
            unvisited_neighbors = adj_mask[curr] & ~visited_mask
            if unvisited_neighbors == 0:
                break
            # Find all unvisited neighbor vertices
            candidates = []
            m = unvisited_neighbors
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                candidates.append(v)
                m ^= lsb
            # Sort by number of unvisited neighbors (Warnsdorff)
            candidates.sort(key=lambda v: (adj_mask[v] & ~visited_mask).bit_count())
            next_v = candidates[0]
            path.append(next_v)
            visited_mask |= (1 << next_v)
        
        if len(path) > len(best_path):
            best_path = path[:]
    
    if len(best_path) == n:
        return best_path
    
    # Strategy 2: Iterative deepening DFS with bitmask and Warnsdorff
    # Use a stack-based approach to avoid recursion overhead
    best_len = len(best_path)
    
    # Compute "matching slack" - vertices with fewer alternative connections
    # are more critical to visit early to avoid trapping
    # For each vertex, compute how many of its neighbors have degree <= 2
    criticality = [0] * n
    for v in range(n):
        for u in adj[v]:
            if degrees[u] <= 2:
                criticality[v] += 1
    
    def dfs_iterative(start):
        nonlocal best_path, best_len
        if time_check():
            return False
        
        # Stack entries: (vertex, path_so_far, visited_mask, next_candidate_idx)
        # We'll use a list as stack
        stack = [(start, [start], 1 << start, 0)]
        
        while stack:
            if time.time() > deadline:
                return False
            
            curr, path, visited_mask, cand_idx = stack.pop()
            
            cur_len = len(path)
            if cur_len > best_len:
                best_len = cur_len
                best_path = path[:]
            
            if cur_len == n:
                return True
            
            # Get unvisited neighbors
            unvisited_neighbors = adj_mask[curr] & ~visited_mask
            if unvisited_neighbors == 0:
                continue
            
            # Get all unvisited neighbor vertices
            candidates = []
            m = unvisited_neighbors
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                candidates.append(v)
                m ^= lsb
            
            # Sort candidates using Warnsdorff + criticality
            # Prefer vertices with fewer unvisited neighbors, and higher criticality
            candidates.sort(key=lambda v: ((adj_mask[v] & ~visited_mask).bit_count(), -criticality[v]))
            
            # Push candidates in reverse order so we process the best first
            for v in reversed(candidates):
                new_path = path + [v]
                new_mask = visited_mask | (1 << v)
                stack.append((v, new_path, new_mask, 0))
        
        return False
    
    # Try DFS from low-degree vertices first (more constrained)
    sorted_starts = sorted(range(n), key=lambda x: degrees[x])
    for start in sorted_starts:
        if time_check():
            break
        if len(best_path) == n:
            break
        dfs_iterative(start)
    
    if len(best_path) == n:
        return best_path
    
    # Strategy 3: Local search - iterative vertex insertion
    if best_path:
        in_path = set(best_path)
        not_in_path = [v for v in range(n) if v not in in_path]
        
        # Repeatedly try to insert missing vertices
        improved = True
        while improved and not time_check():
            improved = False
            in_path = set(best_path)
            not_in_path = [v for v in range(n) if v not in in_path]
            
            for missing in not_in_path:
                if time_check():
                    break
                if len(best_path) == n:
                    break
                
                # Try inserting missing at each position
                for i in range(len(best_path) + 1):
                    if time_check():
                        break
                    valid = False
                    if i == 0:
                        if best_path[0] in adj[missing]:
                            valid = True
                    elif i == len(best_path):
                        if best_path[-1] in adj[missing]:
                            valid = True
                    else:
                        if best_path[i-1] in adj[missing] and best_path[i] in adj[missing]:
                            valid = True
                    
                    if valid:
                        new_path = best_path[:i] + [missing] + best_path[i:]
                        if len(new_path) > len(best_path):
                            best_path = new_path
                            improved = True
                            break
        
        # Try 2-opt: reverse segments to create new insertion opportunities
        if len(best_path) < n and not time_check():
            for _ in range(50):  # Limited 2-opt passes
                if time_check():
                    break
                if len(best_path) == n:
                    break
                
                # Try reversing a random segment
                if len(best_path) < 2:
                    break
                i = 0
                j = len(best_path) - 1
                # Try a few segment reversals
                for attempt in range(10):
                    if i >= j:
                        break
                    new_path = best_path[:i] + list(reversed(best_path[i:j+1])) + best_path[j+1:]
                    # Check validity
                    valid = True
                    for k in range(len(new_path) - 1):
                        if new_path[k+1] not in adj[new_path[k]]:
                            valid = False
                            break
                    if valid:
                        # Try to insert missing vertices
                        in_path = set(new_path)
                        not_in_path = [v for v in range(n) if v not in in_path]
                        for missing in not_in_path:
                            for pos in range(len(new_path) + 1):
                                valid2 = False
                                if pos == 0:
                                    if new_path[0] in adj[missing]:
                                        valid2 = True
                                elif pos == len(new_path):
                                    if new_path[-1] in adj[missing]:
                                        valid2 = True
                                else:
                                    if new_path[pos-1] in adj[missing] and new_path[pos] in adj[missing]:
                                        valid2 = True
                                if valid2:
                                    candidate_path = new_path[:pos] + [missing] + new_path[pos:]
                                    if len(candidate_path) > len(best_path):
                                        best_path = candidate_path
                                    break
                        break
                    i += 1
                    j -= 1
    
    return best_path if best_path else [0]
