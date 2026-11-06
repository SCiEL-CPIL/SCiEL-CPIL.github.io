"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

# Pinned graph data from seed
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
EDGES = P['edges']

def build_adj():
    adj = [set() for _ in range(N)]
    for u, v in EDGES:
        adj[u].add(v)
        adj[v].add(u)
    return adj

ADJ = build_adj()
DEG = [len(adj) for adj in ADJ]

def path_is_valid(path):
    if not path:
        return False
    if len(path) != len(set(path)):
        return False
    for i in range(len(path) - 1):
        if path[i+1] not in ADJ[path[i]]:
            return False
    return True

def greedy_grow(start, adj, strategy='warnsdorff'):
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        candidates = adj[cur] - seen
        if not candidates:
            break
        if strategy == 'warnsdorff':
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
        else:
            v = random.choice(list(candidates))
        path.append(v)
        seen.add(v)
    return path

def try_extend_both_ends(path, adj):
    path = list(path)
    seen = set(path)
    changed = True
    while changed:
        changed = False
        # Extend start
        cur = path[0]
        candidates = adj[cur] - seen
        if candidates:
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            changed = True
        # Extend end
        cur = path[-1]
        candidates = adj[cur] - seen
        if candidates:
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.append(v)
            seen.add(v)
            changed = True
    return path

def try_or_opt(path, adj):
    """Try to move a segment of length 1, 2, or 3 to another position."""
    n = len(path)
    if n < 4:
        return path
    
    best_path = list(path)
    improved = False
    
    for seg_len in [1, 2, 3]:
        if n < seg_len + 2:
            continue
        for i in range(n - seg_len - 1):
            # Segment is path[i : i+seg_len]
            seg = path[i : i+seg_len]
            # Remove segment
            new_path = path[:i] + path[i+seg_len:]
            # Try to insert seg at every possible position
            for j in range(len(new_path) - seg_len + 1):
                if i == j:
                    continue
                # Check connectivity
                # Left neighbor of insertion point
                if j > 0:
                    left_node = new_path[j-1]
                    if seg[0] not in adj[left_node]:
                        continue
                # Right neighbor of insertion point
                if j + seg_len < len(new_path):
                    right_node = new_path[j+seg_len]
                    if seg[-1] not in adj[right_node]:
                        continue
                # Check if the gap created is valid (if not at ends)
                if i > 0:
                    gap_left = path[i-1]
                    gap_right = path[i+seg_len]
                    if gap_right not in adj[gap_left]:
                        continue
                # Construct new path
                candidate_path = new_path[:j] + seg + new_path[j:]
                # Verify validity (distinct vertices)
                if len(set(candidate_path)) != len(candidate_path):
                    continue
                # If valid, accept it (we assume any valid move is good for exploration, 
                # but for maximization, we might want to prefer moves that allow extension)
                # For now, just return the first valid improvement found in a random order?
                # Or try to extend?
                # Let's just return the new path if it's valid. 
                # The caller will try to extend it.
                return candidate_path
            
    return best_path

def try_3_opt(path, adj):
    """Try 3-opt moves."""
    n = len(path)
    if n < 5:
        return path
    
    # 3-opt involves removing 3 edges and reconnecting.
    # Edges are (0,1), (1,2), ..., (n-2, n-1).
    # We pick 3 distinct edges i, j, k.
    # This is complex. A simpler variant is 2-opt + 1-opt (Or-opt).
    # Let's stick to Or-opt and 2-opt for now to keep code robust.
    return path

def local_search_improve(path, adj, max_iter=100):
    best_path = list(path)
    best_len = len(best_path)
    
    for _ in range(max_iter):
        improved = False
        
        # 1. Try Or-opt
        new_path = try_or_opt(best_path, adj)
        if new_path != best_path:
            # Try to extend the new path
            extended = try_extend_both_ends(new_path, adj)
            if len(extended) > best_len:
                best_path = extended
                best_len = len(best_path)
                improved = True
        
        # 2. Try 2-opt (reverse segment)
        if not improved and len(best_path) >= 4:
            for i in range(len(best_path) - 3):
                for j in range(i + 2, len(best_path)):
                    a = best_path[i]
                    b = best_path[i+1]
                    c = best_path[j-1]
                    d = best_path[j]
                    if d in adj[a] and b in adj[c]:
                        new_path = best_path[:i+1] + best_path[i+1:j][::-1] + best_path[j:]
                        if len(set(new_path)) == len(new_path):
                            # Verify edges
                            valid = True
                            for k in range(len(new_path) - 1):
                                if new_path[k+1] not in adj[new_path[k]]:
                                    valid = False
                                    break
                            if valid:
                                extended = try_extend_both_ends(new_path, adj)
                                if len(extended) > best_len:
                                    best_path = extended
                                    best_len = len(best_path)
                                    improved = True
                                    break
                if improved:
                    break
        
        # 3. Try Insertion
        if not improved:
            seen = set(best_path)
            for v in range(N):
                if v in seen:
                    continue
                # Try to insert v
                for i in range(len(best_path) - 1):
                    if v in adj[best_path[i]] and v in adj[best_path[i+1]]:
                        new_path = best_path[:i+1] + [v] + best_path[i+1:]
                        extended = try_extend_both_ends(new_path, adj)
                        if len(extended) > best_len:
                            best_path = extended
                            best_len = len(best_path)
                            improved = True
                            break
                if improved:
                    break
                # Try ends
                if v in adj[best_path[0]]:
                    new_path = [v] + best_path
                    extended = try_extend_both_ends(new_path, adj)
                    if len(extended) > best_len:
                        best_path = extended
                        best_len = len(best_path)
                        improved = True
                if not improved and v in adj[best_path[-1]]:
                    new_path = best_path + [v]
                    extended = try_extend_both_ends(new_path, adj)
                    if len(extended) > best_len:
                        best_path = extended
                        best_len = len(best_path)
                        improved = True
        
        if not improved:
            break
            
    return best_path

def dfs_longest_path(start, adj, time_limit=5.0):
    start_time = time.time()
    best_path = [start]
    best_len = 1
    
    def dfs(path, seen):
        nonlocal best_path, best_len
        if time.time() - start_time > time_limit:
            return
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        cur = path[-1]
        candidates = list(adj[cur] - seen)
        if not candidates:
            return
            
        # Warnsdorff's rule: sort by remaining degree
        candidates.sort(key=lambda w: (len(adj[w] - seen), w))
        
        for v in candidates:
            dfs(path + [v], seen | {v})
            
    dfs([start], {start})
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_path = []
    best_len = 0
    
    # Phase 1: Greedy from all starts
    for start in range(N):
        if time.time() > deadline:
            break
        path = greedy_grow(start, ADJ, strategy='warnsdorff')
        path = try_extend_both_ends(path, ADJ)
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
            
    # Phase 2: Local Search
    if time.time() < deadline and best_len > 0:
        improved = local_search_improve(best_path, ADJ, max_iter=200)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved
            
    # Phase 3: DFS from promising starts
    if time.time() < deadline:
        promising_starts = [best_path[0], best_path[-1]]
        # Add some high-degree nodes
        promising_starts += sorted(range(N), key=lambda x: -DEG[x])[:5]
        promising_starts = list(set(promising_starts))
        
        for start in promising_starts:
            if time.time() > deadline:
                break
            remaining = deadline - time.time()
            if remaining < 2.0:
                break
            path = dfs_longest_path(start, ADJ, time_limit=min(remaining * 0.7, 10.0))
            path = try_extend_both_ends(path, ADJ)
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
            
            # Improve DFS result
            if time.time() < deadline:
                improved = local_search_improve(path, ADJ, max_iter=50)
                if len(improved) > best_len:
                    best_len = len(improved)
                    best_path = improved
                    
    # Phase 4: Final Polish
    if time.time() < deadline and best_len > 0:
        improved = local_search_improve(best_path, ADJ, max_iter=100)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved
            
    if not best_path:
        best_path = [0]
        
    if not path_is_valid(best_path):
        for u, v in EDGES:
            best_path = [u, v]
            break
        else:
            best_path = [0]
            
    return best_path
# EVOLVE-BLOCK-END
