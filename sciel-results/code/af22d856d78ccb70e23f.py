import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edges = P['edge_order']

# Precompute adjacency for all edges? No, we need subgraphs.
# We will build adjacency masks on the fly for a given K.

def popcount(x):
    return bin(x).count('1')

def find_hamilton_cycle(K):
    """
    Returns a list of vertices representing a Hamilton cycle using only the first K edges, or None.
    """
    # Build adjacency masks
    adj = [0] * n
    for i in range(K):
        u, v = edges[i]
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Check degree constraint: every vertex must have degree >= 2
    for u in range(n):
        if popcount(adj[u]) < 2:
            return None

    full_mask = (1 << n) - 1
    start = 0
    start_mask = 1 << start
    
    # DFS with Warnsdorff
    # path is a list of vertices
    # visited is a bitmask
    path = [start]
    visited = start_mask
    
    # To allow backtracking, we use a recursive function or a stack.
    # Given n=24, recursion depth is fine.
    
    def dfs(u, path, visited):
        if len(path) == n:
            # Check if last vertex connects to start
            if adj[u] & start_mask:
                return path[:]
            return None
        
        # Get candidates
        # Candidates are neighbors of u that are not visited
        cand_mask = adj[u] & ~visited
        if cand_mask == 0:
            return None
            
        # Heuristic: Warnsdorff
        # Score each candidate v by the number of unvisited neighbors of v
        # Note: when checking neighbors of v, we consider the current state of 'visited'
        # plus u (which is visited) but v is not visited yet.
        # Actually, for the next step, v will be visited.
        # So we look at neighbors of v in (unvisited | {u})?
        # Standard Warnsdorff: choose v with min degree in the remaining graph.
        # Remaining graph vertices: unvisited.
        # But v is connected to u.
        
        candidates = []
        m = cand_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            
            # Count unvisited neighbors of v
            # Neighbors of v that are not visited and not u?
            # u is already visited.
            # We want to know how many options v will have after we move to it.
            # So count neighbors of v in (~visited).
            # Note: v itself is not in ~visited.
            # But v is a neighbor of u.
            # The "remaining" graph for the path extension includes unvisited vertices.
            # v will be visited.
            # So v's future neighbors are those in ~visited.
            # However, if we are at the last step (len(path)==n-1), v must connect to start.
            
            # Pruning check:
            # If any unvisited vertex w (w != u, w != start) has < 2 neighbors in (~visited | {u, start?}), it's bad.
            # Specifically, for any w in ~visited:
            # If w == start: skip (start is visited)
            # The available neighbors for w are those in (adj[w] & ~visited).
            # Plus, if w is adjacent to u, that edge is available? No, u is visited, so edge (w,u) is "used" or "available" depending on perspective.
            # Actually, the path is u -> v -> ...
            # w is unvisited. It can only connect to other unvisited vertices or the current vertex u (if adjacent) or the start vertex (if adjacent and it's the last step).
            # Wait, the path is a simple path.
            # w can only be connected to vertices in the path if it's the next step.
            # So for w to be reachable, it must have at least 1 neighbor in the "frontier".
            # But for a cycle, every internal vertex in the final path must have 2 neighbors in the path.
            # In the partial path, unvisited vertices will eventually be internal or the last vertex.
            # The last vertex connects to start.
            # So, for any unvisited w:
            # Let N_w = adj[w] & ~visited.
            # If w is not the last vertex, it needs 2 neighbors in the final path.
            # One neighbor will be the one it comes from, one it goes to.
            # Both must be in the set of vertices that are either unvisited or the current vertex u (if w is next) or start (if w is last).
            # This is complex.
            
            # Simple heuristic:
            # Score = popcount(adj[v] & ~visited)
            # If we are at step n-1 (last vertex), score should be infinity if v doesn't connect to start, or 0 if it does?
            # Actually, if len(path) == n-1, we MUST pick a v such that adj[v] & start_mask != 0.
            
            if len(path) == n - 1:
                if adj[v] & start_mask:
                    candidates.append((0, v)) # Priority 0
                else:
                    candidates.append((1, v)) # Priority 1 (will fail later, but let's sort)
            else:
                # Standard Warnsdorff
                score = popcount(adj[v] & ~visited)
                candidates.append((score, v))
                
        candidates.sort()
        
        for score, v in candidates:
            # Additional pruning:
            # Check if any unvisited vertex becomes "stuck"
            # A vertex w in ~visited (w != v) is stuck if it has < 2 neighbors in (~visited | {v, start})?
            # Actually, if w has < 2 neighbors in the set of "available" vertices for it.
            # Available for w:
            # - Other unvisited vertices
            # - The vertex we are moving to (v) if adjacent
            # - The start vertex if adjacent (only if w is the last vertex)
            
            # Let's check a simpler condition:
            # For any w in ~visited:
            # If w == v: skip
            # Let avail_w = adj[w] & (~visited | (1 << v) | start_mask)
            # If popcount(avail_w) < 2:
            #    If w is adjacent to start, it might be the last vertex.
            #    If it's the last vertex, it needs 1 neighbor in path (the one before it) and 1 to start.
            #    So if popcount(avail_w) == 1 and (adj[w] & start_mask) and (popcount(adj[w] & ~visited) == 0):
            #         It must be the last vertex.
            #    Else, if popcount(avail_w) < 2, it's stuck.
            
            # This check is O(n^2) per step, which is 24*24 = 576.
            # With 24 steps, it's ~14k ops. Very fast.
            
            is_valid = True
            # We move to v. So v becomes visited.
            # New unvisited mask: ~visited & ~(1<<v)
            # But for the check, we consider the state AFTER adding v to path?
            # No, we check if the choice of v leads to a dead end.
            # The "unvisited" set for the next step will be ~visited & ~(1<<v).
            # Let's call this next_unvisited.
            next_unvisited = ~visited & ~(1 << v) & full_mask
            
            # Check all w in next_unvisited
            m_w = next_unvisited
            while m_w:
                lsb_w = m_w & -m_w
                w = lsb_w.bit_length() - 1
                m_w ^= lsb_w
                
                # Neighbors of w that are "available" to be its predecessors/successors in the path
                # Available vertices for w are:
                # 1. Other unvisited vertices (in next_unvisited)
                # 2. The vertex v (which is now the last in path)
                # 3. The start vertex 0 (if w is the last vertex)
                
                # So avail_mask = adj[w] & (next_unvisited | (1 << v) | start_mask)
                avail = adj[w] & (next_unvisited | (1 << v) | start_mask)
                deg_avail = popcount(avail)
                
                # If w is the last vertex, it needs to connect to start.
                # If w is an internal vertex, it needs 2 neighbors in the path.
                # The path neighbors for w must be in avail.
                
                if deg_avail < 2:
                    # Can w be the last vertex?
                    if (adj[w] & start_mask) and deg_avail == 1:
                        # It has exactly 1 neighbor in avail.
                        # That neighbor must be its predecessor.
                        # And it connects to start.
                        # This is possible ONLY if w is the LAST vertex.
                        # But we don't know if w is the last vertex yet.
                        # However, if there are other unvisited vertices, w cannot be the last one?
                        # No, w *could* be the last one, but then all other unvisited vertices must be visited before it.
                        # This constraint is hard to check locally.
                        # Simple heuristic: if deg_avail < 2, prune.
                        # Exception: if deg_avail == 1 and w connects to start, allow it?
                        # Risky. Let's just prune if deg_avail < 2.
                        # Wait, if w is the last vertex, it needs 1 neighbor in the path (predecessor) and 1 to start.
                        # So total 2 "connections" needed.
                        # In avail, it has 1 (predecessor) + 1 (start) = 2.
                        # So if deg_avail == 1 and (adj[w] & start_mask), it has 2 total connections (1 in avail, 1 to start).
                        # So it's NOT stuck.
                        if not (adj[w] & start_mask):
                            is_valid = False
                            break
                        # If it connects to start, it's potentially the last vertex.
                        # We allow it.
                    else:
                        is_valid = False
                        break
                
            if not is_valid:
                continue
            
            # Recursive step
            res = dfs(v, path + [v], visited | (1 << v))
            if res:
                return res
                
        return None

    return dfs(start, path, visited)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    # Binary search for K
    lo = 24
    hi = len(edges)
    best_cycle = None
    
    # First, find the minimal K
    # We can just iterate K from 24 upwards if the range is small?
    # 24 to 50 is 27 checks.
    # Binary search is safer.
    
    # Check if hi works? It should.
    # But we want the MINIMAL K.
    
    # Let's do binary search.
    # If has_hamilton_cycle(K) is True, try smaller.
    
    # Optimization: Pre-check degree for K.
    # If any vertex has deg < 2, return False immediately.
    
    while lo < hi:
        if time.time() > deadline:
            break
            
        mid = (lo + hi) // 2
        # Check mid
        cycle = find_hamilton_cycle(mid)
        if cycle:
            best_cycle = cycle
            hi = mid
        else:
            lo = mid + 1
            
    # If binary search finished, check lo
    if time.time() < deadline:
        if best_cycle is None or lo < len(edges):
            # We found a cycle at 'hi' (which is now 'lo' if converged)
            # Actually, if loop exits with lo==hi, we know lo works?
            # Not necessarily, if we broke due to time.
            # But if we converged, hi was the last successful one.
            # Let's just return the best_cycle found.
            # If best_cycle is None, try lo one last time?
            if best_cycle is None:
                best_cycle = find_hamilton_cycle(lo)
                
    if best_cycle:
        return best_cycle
    else:
        # Fallback: return identity (should not happen)
        return list(range(n))

# EVOLVE-BLOCK-END
