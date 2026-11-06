import time
import random
import sys

sys.setrecursionlimit(100000)

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
    deadline = t0 + 85.0

    # Trivial cycle
    trivial = list(range(N))
    best_cycle = trivial
    best_score = _cycle_score(trivial)

    # Precompute sorted adjacency for each vertex by edge position
    full_adj = [[] for _ in range(N)]
    for i, (u, v) in enumerate(EDGE_ORDER, 1):
        full_adj[u].append((v, i))
        full_adj[v].append((u, i))
    for u in range(N):
        full_adj[u].sort(key=lambda x: x[1])

    # --- Phase 1: Bitmask DP to find minimum K with a Hamilton cycle ---
    # We binary search on K. For a given K, we check if a Hamilton cycle exists
    # in the subgraph with edges of position <= K.
    # Use DP: dp[mask][v] = True if there's a path from 0 through exactly the
    # vertices in mask ending at v, using only edges with pos <= K.
    # mask is over vertices 0..23. We fix start=0.
    # dp is a list of 2^24 booleans per vertex? That's 24 * 2^24 = ~400M booleans.
    # Too much memory. Instead, use a set of (mask, v) reachable states, or
    # use a bitset approach: for each v, a bitset of masks where dp[mask][v] is True.
    # Actually, 2^24 = 16,777,216. For 24 vertices, that's 24 * 16M = 400M bits = 50MB per bitset.
    # We can use Python integers as bitsets: each vertex v has an integer where bit m is set
    # if there's a path from 0 through mask m ending at v.
    
    # But iterating over all masks is 2^24 = 16M, and for each we check transitions.
    # That's too slow in pure Python.
    
    # Alternative: use a BFS/DFS with memoization but with early termination.
    # Or: use the fact that we just need ANY Hamilton cycle, so we can use a
    # randomized approach or a smarter search.
    
    # Let's try a different approach: for a given K, use a backtracking search
    # with strong pruning (degree constraints, connectivity).
    
    def has_ham_cycle(K, time_limit):
        """Check if Hamilton cycle exists using only edges with pos <= K.
        Returns (found, cycle) or (False, None)."""
        # Build adjacency with only edges <= K
        adj = [set() for _ in range(N)]
        for u in range(N):
            for v, pos in full_adj[u]:
                if pos <= K:
                    adj[u].add(v)
        
        # Check minimum degree
        for u in range(N):
            if len(adj[u]) < 2:
                return (False, None)
        
        # Check connectivity
        visited = [False] * N
        stack = [0]
        visited[0] = True
        count = 1
        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if not visited[nxt]:
                    visited[nxt] = True
                    count += 1
                    stack.append(nxt)
        if count < N:
            return (False, None)
        
        # Backtracking search with pruning
        start = 0
        path = [start]
        used = [False] * N
        used[start] = True
        
        # Precompute for each vertex, the list of neighbors sorted by position
        adj_list = []
        for u in range(N):
            nbrs = [(v, _edge_idx(u, v)) for v in adj[u]]
            nbrs.sort(key=lambda x: x[1])
            adj_list.append(nbrs)
        
        def backtrack(cur, depth):
            if time.time() > time_limit:
                return None
            if depth == N:
                # Check closing edge
                if _edge_idx(cur, start) <= K:
                    return list(path)
                return None
            for nxt, pos in adj_list[cur]:
                if not used[nxt]:
                    # Pruning: if nxt is not the last vertex, it must have at least
                    # one unused neighbor (other than cur) to continue
                    if depth < N - 1:
                        has_cont = False
                        for nnxt, npos in adj_list[nxt]:
                            if not used[nnxt] and nnxt != cur:
                                has_cont = True
                                break
                        if not has_cont:
                            continue
                    used[nxt] = True
                    path.append(nxt)
                    result = backtrack(nxt, depth + 1)
                    if result is not None:
                        return result
                    path.pop()
                    used[nxt] = False
            return None
        
        result = backtrack(start, 1)
        if result is not None:
            return (True, result)
        return (False, None)
    
    # Binary search for minimum K
    lo, hi = N, best_score - 1
    dp_found_cycle = None
    dp_found_K = best_score
    
    while lo <= hi and time.time() < deadline - 30.0:
        mid = (lo + hi) // 2
        found, cycle = has_ham_cycle(mid, deadline - 30.0)
        if found:
            sc = _cycle_score(cycle)
            if sc <= mid:
                dp_found_cycle = cycle
                dp_found_K = sc
                if sc < best_score:
                    best_cycle = list(cycle)
                    best_score = sc
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            lo = mid + 1
    
    # --- Phase 2: Local search to improve ---
    # Try 2-opt moves and random perturbations
    rng = random.Random(42)
    improved = True
    stall_count = 0
    max_stall = 50
    
    while time.time() < deadline - 3.0:
        if not improved:
            stall_count += 1
            if stall_count > max_stall:
                # Random restart: perturb the current best
                for _ in range(10):
                    i = rng.randrange(N)
                    j = rng.randrange(N)
                    best_cycle[i], best_cycle[j] = best_cycle[j], best_cycle[i]
                    new_sc = _cycle_score(best_cycle)
                    if new_sc < best_score:
                        best_score = new_sc
                        stall_count = 0
                        break
                if stall_count > max_stall * 2:
                    break
            else:
                # Try random swap
                i = rng.randrange(N)
                j = rng.randrange(N)
                if i != j:
                    best_cycle[i], best_cycle[j] = best_cycle[j], best_cycle[i]
                    new_sc = _cycle_score(best_cycle)
                    if new_sc < best_score:
                        best_score = new_sc
                        improved = True
                        stall_count = 0
                    else:
                        best_cycle[i], best_cycle[j] = best_cycle[j], best_cycle[i]
                continue
        improved = False
        
        # Try all 2-opt moves
        for i in range(N - 1):
            if time.time() > deadline - 3.0:
                break
            for j in range(i + 2, N):
                if j == N - 1 and i == 0:
                    continue  # same as reversing whole cycle
                # Reverse segment [i, j]
                best_cycle[i:j+1] = best_cycle[i:j+1][::-1]
                new_sc = _cycle_score(best_cycle)
                if new_sc < best_score:
                    best_score = new_sc
                    improved = True
                else:
                    best_cycle[i:j+1] = best_cycle[i:j+1][::-1]
    
    # Final check
    final_sc = _cycle_score(best_cycle)
    if final_sc < best_score:
        best_score = final_sc
    
    return best_cycle
