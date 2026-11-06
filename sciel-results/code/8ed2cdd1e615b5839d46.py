"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = P['n']
    
    # Build bitset adjacency for fast neighbor filtering
    adj_bits = [0] * n
    adj_list = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    best = []
    best_len = 0
    
    def try_extend(path, seen_bits, end_idx):
        """Try to extend path from end_idx (0 for front, -1 for back)."""
        if end_idx == 0:
            # Extend from front: path becomes [new, path[0], path[1], ...]
            cur = path[0]
            while True:
                avail = adj_bits[cur] & ~seen_bits
                if avail == 0:
                    break
                # Pick neighbor with fewest remaining options (MRV)
                best_v = -1
                best_count = n + 1
                x = avail
                while x:
                    low = x & (-x)
                    v = low.bit_length() - 1
                    x ^= low
                    cnt = bin(adj_bits[v] & ~seen_bits).count('1')
                    if cnt < best_count:
                        best_count = cnt
                        best_v = v
                    if cnt == 0:
                        break
                if best_v == -1:
                    break
                path.insert(0, best_v)
                seen_bits |= (1 << best_v)
                cur = best_v
        else:
            # Extend from back: path becomes [..., path[-1], new]
            cur = path[-1]
            while True:
                avail = adj_bits[cur] & ~seen_bits
                if avail == 0:
                    break
                best_v = -1
                best_count = n + 1
                x = avail
                while x:
                    low = x & (-x)
                    v = low.bit_length() - 1
                    x ^= low
                    cnt = bin(adj_bits[v] & ~seen_bits).count('1')
                    if cnt < best_count:
                        best_count = cnt
                        best_v = v
                    if cnt == 0:
                        break
                if best_v == -1:
                    break
                path.append(best_v)
                seen_bits |= (1 << best_v)
                cur = best_v
        return path, seen_bits
    
    def greedy_min(start, rng=None):
        """Greedy: always pick neighbor with fewest remaining options."""
        path = [start]
        seen = 1 << start
        cur = start
        while True:
            avail = adj_bits[cur] & ~seen
            if avail == 0:
                break
            best_v = -1
            best_count = n + 1
            best_rand = 1.0
            x = avail
            while x:
                low = x & (-x)
                v = low.bit_length() - 1
                x ^= low
                cnt = bin(adj_bits[v] & ~seen).count('1')
                r = rng.random() if rng else 0.0
                if cnt < best_count or (cnt == best_count and r < best_rand):
                    best_count = cnt
                    best_v = v
                    best_rand = r
                if cnt == 0:
                    break
            if best_v == -1:
                break
            path.append(best_v)
            seen |= (1 << best_v)
            cur = best_v
        return path, seen
    
    def greedy_max(start, rng=None):
        """Greedy: always pick neighbor with most remaining options."""
        path = [start]
        seen = 1 << start
        cur = start
        while True:
            avail = adj_bits[cur] & ~seen
            if avail == 0:
                break
            best_v = -1
            best_count = -1
            best_rand = 1.0
            x = avail
            while x:
                low = x & (-x)
                v = low.bit_length() - 1
                x ^= low
                cnt = bin(adj_bits[v] & ~seen).count('1')
                r = rng.random() if rng else 0.0
                if cnt > best_count or (cnt == best_count and r < best_rand):
                    best_count = cnt
                    best_v = v
                    best_rand = r
            if best_v == -1:
                break
            path.append(best_v)
            seen |= (1 << best_v)
            cur = best_v
        return path, seen
    
    def backtrack_search(start, max_depth, branch_factor, rng=None, time_limit=None):
        """Iterative DFS with backtracking."""
        nonlocal best, best_len
        path = [start]
        seen = 1 << start
        cur = start
        
        # Stack for iterative backtracking
        # Each entry: (node, seen_bits, candidates_list, candidate_index)
        stack = [(start, seen, None, 0)]
        
        while stack:
            if time.time() > time_limit:
                break
            node, seen_bits, cands, cidx = stack[-1]
            
            if cands is None:
                # Compute candidates
                avail = adj_bits[node] & ~seen_bits
                cand_list = []
                x = avail
                while x:
                    low = x & (-x)
                    v = low.bit_length() - 1
                    x ^= low
                    cnt = bin(adj_bits[v] & ~seen_bits).count('1')
                    r = rng.random() if rng else 0.0
                    cand_list.append((cnt, r, v))
                cand_list.sort(key=lambda t: (t[0], t[1], t[2]))
                cands = cand_list
                cidx = 0
                stack[-1] = (node, seen_bits, cands, cidx)
            
            if cidx >= len(cands):
                # No more candidates, backtrack
                stack.pop()
                continue
            
            cnt, r, v = cands[cidx]
            stack[-1] = (node, seen_bits, cands, cidx + 1)
            
            # Try v
            new_seen = seen_bits | (1 << v)
            path.append(v)
            
            if len(path) > best_len:
                best = path[:]
                best_len = len(path)
            
            # Check if we can extend further
            avail_next = adj_bits[v] & ~new_seen
            if avail_next:
                stack.append((v, new_seen, None, 0))
            else:
                path.pop()
        
        return best
    
    # Phase 1: Greedy from all starts (min and max)
    for start in range(n):
        if time.time() > deadline:
            break
        path, seen = greedy_min(start)
        if len(path) > best_len:
            best = path[:]
            best_len = len(path)
    
    for start in range(n):
        if time.time() > deadline:
            break
        path, seen = greedy_max(start)
        if len(path) > best_len:
            best = path[:]
            best_len = len(path)
    
    # Phase 2: Randomized greedy with many trials
    rng = random.Random(42)
    for trial in range(3000):
        if time.time() > deadline:
            break
        start = rng.randint(0, n - 1)
        if trial % 2 == 0:
            path, seen = greedy_min(start, rng)
        else:
            path, seen = greedy_max(start, rng)
        
        # Try to extend from both ends
        path2, seen2 = try_extend(path[:], seen, 0)
        path2, seen2 = try_extend(path2, seen2, -1)
        
        if len(path2) > best_len:
            best = path2[:]
            best_len = len(path2)
    
    # Phase 3: Backtracking search from promising starts
    # Start from vertices with moderate degree
    degrees = [len(adj_list[i]) for i in range(n)]
    # Sort by degree: prefer medium-degree vertices
    sorted_starts = sorted(range(n), key=lambda i: (abs(degrees[i] - 2), i))
    
    for start in sorted_starts:
        if time.time() > deadline:
            break
        time_limit = min(deadline, time.time() + 3)  # 3s per start
        backtrack_search(start, n, 8, rng, time_limit)
    
    # Phase 4: More randomized backtracking
    rng2 = random.Random(123)
    for trial in range(200):
        if time.time() > deadline:
            break
        start = rng2.randint(0, n - 1)
        time_limit = min(deadline, time.time() + 2)
        backtrack_search(start, n, 6, rng2, time_limit)
    
    # Phase 5: Final extension attempts on best path
    if best:
        seen = 0
        for v in best:
            seen |= (1 << v)
        path_ext, seen_ext = try_extend(best[:], seen, 0)
        path_ext, seen_ext = try_extend(path_ext, seen_ext, -1)
        if len(path_ext) > best_len:
            best = path_ext[:]
            best_len = len(path_ext)
    
    return best if best else [0]
# EVOLVE-BLOCK-END
