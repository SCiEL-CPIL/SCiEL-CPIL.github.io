import time
import random

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']

_edge_pos = {}
for _i, (_u, _v) in enumerate(EDGE_ORDER, 1):
    _edge_pos[(min(_u, _v), max(_u, _v))] = _i

def _edge_idx(u, v):
    return _edge_pos[(min(u, v), max(u, v))]

def _cycle_score(cycle):
    return max(_edge_idx(cycle[i], cycle[(i+1) % N]) for i in range(N))

def construct():
    t0 = time.time()
    deadline = t0 + 50.0  # Safety margin for 60s limit
    
    # Precompute edge indices matrix for fast lookup
    idx_mat = [[0]*N for _ in range(N)]
    for u in range(N):
        for v in range(N):
            if u != v:
                idx_mat[u][v] = _edge_idx(u, v)
    
    # Trivial cycle
    trivial = list(range(N))
    best_score = _cycle_score(trivial)
    best_cycle = trivial[:]
    
    # Fast DFS using bitsets for Warnsdorff's rule
    def find_cycle_warnsdorff(K, time_limit=2.0):
        """Find a Hamilton cycle using only edges with index <= K."""
        n = N
        adj = [[] for _ in range(n)]
        adj_bits = [0] * n
        for u in range(n):
            for v in range(n):
                if u != v and idx_mat[u][v] <= K:
                    adj[u].append(v)
                    adj_bits[u] |= (1 << v)
        
        # Check necessary conditions
        for u in range(n):
            if len(adj[u]) < 2:
                return None
        
        start = 0
        path = [0] * n
        path[0] = start
        used_bits = 1 << start
        depth = 1
        deadline_local = time.time() + time_limit
        steps = 0
        
        def backtrack(cur, depth, used_bits):
            nonlocal steps
            steps += 1
            if steps % 1000 == 0 and time.time() > deadline_local:
                return False
            
            if depth == n:
                if idx_mat[cur][start] <= K:
                    return True
                return False
            
            # Get unvisited neighbors
            unvisited_mask = adj_bits[cur] & ~used_bits
            if unvisited_mask == 0:
                return False
            
            # Extract candidates
            candidates = []
            m = unvisited_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                candidates.append(v)
                m ^= lsb
            
            if not candidates:
                return False
            
            # Sort by Warnsdorff's rule: fewest unvisited neighbors
            def warnsdorff_key(v):
                return (adj_bits[v] & ~used_bits).bit_count()
            
            candidates.sort(key=warnsdorff_key)
            
            for nxt in candidates:
                new_used = used_bits | (1 << nxt)
                path[depth] = nxt
                if backtrack(nxt, depth + 1, new_used):
                    return True
            
            return False
        
        if backtrack(start, 1, 1 << start):
            return path[:]
        return None
    
    # Binary search
    lo = 24
    hi = best_score
    
    while lo <= hi and time.time() < deadline - 15.0:
        mid = (lo + hi) // 2
        cycle = find_cycle_warnsdorff(mid, time_limit=2.0)
        
        if cycle is not None:
            sc = _cycle_score(cycle)
            if sc <= mid:
                best_cycle = cycle
                best_score = sc
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            lo = mid + 1
    
    # Best Improvement 2-opt + 3-opt hybrid
    def best_improvement_local_search(cycle, time_limit):
        """Apply best-improvement 2-opt and 3-opt to reduce max edge index."""
        n = N
        deadline_local = time.time() + time_limit
        best = cycle[:]
        best_sc = _cycle_score(best)
        
        def try_2opt_move(c, i, j):
            """Try reversing segment i..j and return new score."""
            # Compute new edges: (c[i-1], c[j]) and (c[i], c[j+1]) replace (c[i-1], c[i]) and (c[j], c[j+1])
            # But we need to check the whole cycle max, so just compute
            old_sc = _cycle_score(c)
            # Reverse in place
            c[i:j+1] = reversed(c[i:j+1])
            new_sc = _cycle_score(c)
            # Reverse back
            c[i:j+1] = reversed(c[i:j+1])
            return new_sc
        
        def try_3opt_move(c, i, j, k):
            """Try 3-opt: cut three edges and reconnect in a different way.
            For simplicity, we'll try a few 3-opt patterns."""
            # Pattern: reverse [i..j] then reverse [j+1..k]
            # This is equivalent to swapping the middle segment
            # Simpler: just try 2-opt on subranges
            return None  # Skip complex 3-opt for now, focus on 2-opt
        
        improved = True
        while improved and time.time() < deadline_local:
            improved = False
            best_move = None
            best_new_sc = best_sc
            
            # Try all 2-opt moves
            for i in range(n - 1):
                for j in range(i + 2, n):
                    new_sc = try_2opt_move(best, i, j)
                    if new_sc < best_new_sc:
                        best_new_sc = new_sc
                        best_move = (i, j)
            
            if best_move is not None:
                i, j = best_move
                best[i:j+1] = reversed(best[i:j+1])
                best_sc = best_new_sc
                improved = True
        
        return best, best_sc
    
    # Apply best-improvement local search on the best found so far
    time_left = deadline - time.time() - 5.0
    if time_left > 1.0:
        best_cycle, best_score = best_improvement_local_search(best_cycle, time_left)
    
    # Try multiple random restarts with local search if time permits
    rng = random.Random(42)
    restart_count = 0
    while time.time() < deadline - 2.0:
        perm = list(range(N))
        rng.shuffle(perm)
        
        # Short best-improvement 2-opt pass
        improved_cycle, improved_score = best_improvement_local_search(perm, 0.5)
        
        if improved_score < best_score:
            best_cycle = improved_cycle
            best_score = improved_score
        
        restart_count += 1
        # Limit restarts to avoid excessive time
        if restart_count >= 50:
            break
    
    return best_cycle
