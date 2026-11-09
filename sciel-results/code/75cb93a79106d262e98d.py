"""Rewritten program: stronger local search for longest simple path."""
import time

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _greedy_extend(path, adj, used):
    """Extend path greedily using Warnsdorff-like heuristic."""
    while True:
        cur = path[-1]
        candidates = [v for v in adj[cur] if v not in used]
        if not candidates:
            break
        # Choose candidate with fewest onward options (Warnsdorff)
        best_v = min(candidates, key=lambda v: len(adj[v] - used))
        path.append(best_v)
        used.add(best_v)
    return path

def _try_swap(path, adj, used, budget_fn):
    """Try to extend path by swapping a middle vertex with an unused neighbor."""
    n = P['n']
    # Try each position i (not 0 or last) and each unused neighbor of path[i]
    # that connects to path[i-1] or path[i+1]
    for i in range(1, len(path) - 1):
        if budget_fn():
            return path
        v = path[i]
        prev = path[i-1]
        nxt = path[i+1]
        # Try inserting an unused vertex w adjacent to v
        for w in sorted(adj[v]):
            if w in used:
                continue
            # Check if we can replace v with w: need prev-w or w-nxt edge
            # Actually we want to insert w, so path becomes ...prev, v, w, nxt... 
            # But that requires v-w and w-nxt edges.
            if w in adj[v] and w in adj[nxt]:
                # Insert w after v
                new_path = path[:i+1] + [w] + path[i+1:]
                new_used = used | {w}
                # Verify validity
                valid = True
                for j in range(len(new_path)-1):
                    if new_path[j+1] not in adj[new_path[j]]:
                        valid = False
                        break
                if valid and len(set(new_path)) == len(new_path):
                    return _greedy_extend(new_path, adj, new_used)
            # Try replacing v with w: prev-w and w-nxt
            if w in adj[prev] and w in adj[nxt]:
                new_path = path[:i] + [w] + path[i+1:]
                new_used = (used - {v}) | {w}
                valid = True
                for j in range(len(new_path)-1):
                    if new_path[j+1] not in adj[new_path[j]]:
                        valid = False
                        break
                if valid and len(set(new_path)) == len(new_path):
                    return _greedy_extend(new_path, adj, new_used)
    return path

def _local_search(path, adj, used, deadline):
    """Iterate greedy extend + swap until no improvement or deadline."""
    import time
    while time.time() < deadline:
        old_len = len(path)
        used = set(path)
        path = _greedy_extend(path, adj, used)
        if len(path) > old_len:
            old_len = len(path)
        path = _try_swap(path, adj, used, lambda: time.time() >= deadline)
        if len(path) == old_len:
            break
    return path

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin
    adj = _build_adj()
    n = P['n']
    
    best_path = []
    
    # Try all starting vertices
    for start in range(n):
        if time.time() >= deadline:
            break
        path = [start]
        used = {start}
        path = _greedy_extend(path, adj, used)
        path = _local_search(path, adj, used, deadline)
        if len(path) > len(best_path):
            best_path = path[:]
    
    # Also try starting from each edge endpoint in different orders
    for u, v in P['edges']:
        if time.time() >= deadline:
            break
        for start, second in [(u, v), (v, u)]:
            if second not in adj[start]:
                continue
            path = [start, second]
            used = {start, second}
            path = _greedy_extend(path, adj, used)
            path = _local_search(path, adj, used, deadline)
            if len(path) > len(best_path):
                best_path = path[:]
    
    if not best_path:
        best_path = [0]
    
    return best_path
