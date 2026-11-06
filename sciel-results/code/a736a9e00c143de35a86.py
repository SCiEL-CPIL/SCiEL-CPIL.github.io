import time
import sys
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

def _dp_feasible(K):
    """
    Check if a Hamilton cycle exists in the subgraph containing only edges with position <= K.
    Uses bitmask DP (Held-Karp) starting from vertex 0.
    Returns True if feasible, and if so, also returns the cycle.
    """
    n = N
    full_mask = (1 << n) - 1
    
    # dp[mask][last] = previous vertex in the path, or -1 if unreachable
    # To save memory and time, use a list of dicts or arrays
    # dp is a 2D array: dp[mask][last] stores the predecessor
    # For n=24, 2^24 * 24 is too large for a full array. 
    # We need a sparse representation or a different approach.
    
    # Actually, 2^24 * 24 integers is about 400 million integers, which is too much for Python lists of ints.
    # Let's use a dictionary-based DP or a more memory-efficient structure.
    # However, for feasibility, we can use a set of (mask, last) pairs that are reachable.
    # But we need to reconstruct the path.
    
    # Let's try a different approach: use a list of sets.
    # reachable[mask] is a set of vertices 'last' such that there is a path from 0 visiting exactly the vertices in 'mask' and ending at 'last'.
    # And pred[mask][last] = previous vertex.
    
    # Memory check: 2^24 masks, each with up to 24 vertices. 
    # Total states ~ 2^24 * 24 / 2 (average) ~ 200 million. This is likely too slow and memory-heavy in pure Python.
    
    # Alternative: Since we just need *a* cycle, and n=24 is small, maybe we can use a heuristic or a different DP formulation?
    # Or perhaps we can reduce the state space by fixing vertex 0 as the start and not including it in the mask until the end?
    
    # Let's try the standard Held-Karp but optimized.
    # dp[mask][v] = True if there is a path from 0 to v visiting exactly the set 'mask'.
    # We store predecessors for reconstruction.
    
    # To save memory, we can use a list of bytearrays or similar.
    # But 2^24 * 24 bytes = 400 MB. This is on the edge of what might work in 60s with 90s limit, but Python overhead is high.
    
    # Let's try a different strategy: 
    # Since the graph is dense (complete graph edges are just ordered), for small K, the graph is sparse.
    # For large K, it's dense.
    # The bottleneck K is likely small enough that the number of reachable states is manageable?
    # Or maybe we should use a heuristic search like A* or IDA*?
    
    # Let's stick to the DP but be careful. If it's too slow, we'll fall back to the previous method or a heuristic.
    
    # Actually, let's try to optimize the DP using numpy or arrays? No, standard library + numpy/scipy available.
    # But bitmask DP with numpy is tricky.
    
    # Let's try a recursive DP with memoization (top-down).
    # state: (mask, last)
    # returns: True if we can complete the cycle from 'last' given that 'mask' has been visited.
    # But we need to ensure we return to 0.
    
    # Let's use a bottom-up DP with dictionaries for sparse states.
    
    # dp[mask] is a set of 'last' vertices reachable.
    # pred[(mask, last)] is the previous vertex.
    
    dp = [None] * (1 << n)
    dp[1] = {0}  # mask with only vertex 0, last is 0
    pred = {}
    
    # Iterate over masks
    # This might be too slow if we iterate over all 2^24 masks.
    # We should iterate only over reachable masks.
    
    # Let's use a queue or list of active masks.
    # But the order of masks matters for DP. We must process masks in increasing order of size?
    # No, just increasing order of mask value works if we ensure transitions go from smaller to larger masks?
    # Adding a bit increases the mask value, so yes.
    
    # However, iterating 2^24 masks is 16 million. For each mask, we check which bits are set.
    # This is feasible in Python if optimized? 16 million iterations might take 10-20 seconds.
    
    # Let's try it.
    
    # Precompute neighbors for each vertex based on K
    neighbors = [[] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            if _edge_idx(u, v) <= K:
                neighbors[u].append(v)
    
    # DP
    # dp[mask] is a set of last vertices
    # We need to be careful with memory.
    
    # Let's try a different approach: 
    # Since n=24, we can use a recursive backtracking with pruning (branch and bound) for the feasibility check?
    # The previous program did DFS. The issue was it might not find the *optimal* K quickly or might get stuck.
    # But the previous program's score was 2.3017.
    
    # Let's try the DP approach but limit the time.
    
    # Actually, let's use a heuristic: 
    # 1. Find the minimum K such that the graph is connected and each vertex has degree >= 2. (Necessary condition)
    # 2. Use that as a lower bound.
    # 3. Use the DP to find the exact minimum K.
    
    # Let's implement the DP carefully.
    
    # To speed up, we can use bit operations.
    
    # Let's try a top-down memoized DFS for the DP.
    # f(mask, last) = True if we can reach a state where all vertices are visited and we can return to 0.
    # But we need to visit ALL vertices.
    
    # Let's stick to the bottom-up but only process reachable states.
    
    # State: (mask, last)
    # We'll use a dictionary: dp[(mask, last)] = prev_vertex
    
    # But there are potentially many states.
    
    # Let's try the previous program's approach but with a better local search?
    # The previous program did binary search on K and used DFS to check feasibility.
    # The DFS might have been too slow or not exploring enough.
    
    # Let's try to improve the DFS by using heuristics: 
    # - Order neighbors by degree or by edge index?
    # - Use Warnsdorff's rule (choose next vertex with fewest unvisited neighbors)?
    
    # Let's try Warnsdorff's rule in the DFS.
    
    def dfs_warnsdorff(K, deadline):
        # Build adjacency list for edges <= K
        adj = [[] for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u != v and _edge_idx(u, v) <= K:
                    adj[u].append(v)
        
        # Sort neighbors by degree (Warnsdorff's rule heuristic)
        # Actually, we should sort dynamically or pre-sort by degree in the induced subgraph?
        # Pre-sorting by degree in G_K might help.
        deg = [len(adj[u]) for u in range(n)]
        for u in range(n):
            adj[u].sort(key=lambda x: deg[x])
        
        start = 0
        path = [start]
        used = bytearray(n)
        used[start] = 1
        
        # Heuristic: check if any vertex has degree 0 in G_K? Then no cycle.
        # Check if graph is connected?
        
        # DFS
        def backtrack(cur, depth):
            if time.time() > deadline:
                return None
            if depth == n:
                if _edge_idx(cur, start) <= K:
                    return list(path)
                return None
            
            # Try neighbors
            # Heuristic: try neighbors with fewer unvisited neighbors first?
            # We can compute this on the fly.
            
            neighbors_to_try = []
            for nxt in adj[cur]:
                if not used[nxt]:
                    # Count unvisited neighbors of nxt
                    unvisited_count = 0
                    for nnxt in adj[nxt]:
                        if not used[nnxt]:
                            unvisited_count += 1
                    neighbors_to_try.append((unvisited_count, nxt))
            
            neighbors_to_try.sort()
            
            for _, nxt in neighbors_to_try:
                used[nxt] = 1
                path.append(nxt)
                res = backtrack(nxt, depth + 1)
                if res is not None:
                    return res
                path.pop()
                used[nxt] = 0
                
            return None
        
        return backtrack(start, 1)
    
    # Let's use this DFS with Warnsdorff's rule for the feasibility check.
    # It should be faster than plain DFS.
    
    return dfs_warnsdorff(K, time.time() + 5.0)

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    
    # Trivial cycle
    trivial = list(range(N))
    best_score = _cycle_score(trivial)
    best_cycle = trivial
    
    # Binary search for minimum K
    lo = 24
    hi = best_score
    
    # Lower bound: minimum K such that min degree >= 2 and graph is connected
    # We can just start from 24.
    
    while lo <= hi and time.time() < deadline - 10.0:
        mid = (lo + hi) // 2
        
        # Check feasibility using DFS with Warnsdorff's rule
        cycle = _dp_feasible(mid)
        
        if cycle is not None:
            # Verify score
            sc = _cycle_score(cycle)
            if sc <= mid:
                best_cycle = cycle
                best_score = sc
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            lo = mid + 1
    
    # Local search: 2-opt
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(N - 1):
            for j in range(i + 1, N):
                if j == i + 1:
                    continue
                # Reverse segment
                best_cycle[i:j+1] = reversed(best_cycle[i:j+1])
                new_score = _cycle_score(best_cycle)
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    best_cycle[i:j+1] = reversed(best_cycle[i:j+1])
    
    return best_cycle
