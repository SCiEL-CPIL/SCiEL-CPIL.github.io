"""Rewritten program: binary search on threshold + optimized backtracking + 2-opt polish."""
import time
import sys

# Fixed edge order from the pinned shuffle
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    start_time = time.time()
    n = 24
    edge_order = P['edge_order']
    num_edges = len(edge_order)  # 276
    
    # Precompute edge positions
    edge_pos = {}
    for i, (u, v) in enumerate(edge_order):
        edge_pos[(u, v)] = i
        edge_pos[(v, u)] = i
    
    # Precompute adjacency lists sorted by edge position for each threshold
    # We'll build these on-the-fly for speed
    
    def get_score(cycle):
        """Compute the score (max one-based edge position) of a cycle."""
        max_pos = 0
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            pos = edge_pos[(u, v)]
            if pos > max_pos:
                max_pos = pos
        return max_pos + 1
    
    def find_cycle_backtrack(threshold, start_vertex, time_budget):
        """
        Find a Hamiltonian cycle using only edges with position < threshold.
        Uses backtracking with connectivity pruning.
        Returns cycle list or None.
        """
        deadline = time.time() + time_budget
        
        # Build adjacency for edges with pos < threshold
        adj = [[] for _ in range(n)]
        for i in range(threshold):
            u, v = edge_order[i]
            adj[u].append(v)
            adj[v].append(u)
        
        # Sort neighbors by edge position (prefer early edges)
        for u in range(n):
            adj[u].sort(key=lambda v: edge_pos[(u, v)])
        
        start = start_vertex
        path = [start]
        visited = [False] * n
        visited[start] = True
        
        steps = [0]
        max_steps = 500000
        
        def check_connectivity(remaining):
            """Check if remaining unvisited vertices form a connected component
            (considering only edges among them) and can be connected to path ends."""
            if not remaining:
                return True
            # BFS/DFS from the first remaining vertex
            rem_set = set(remaining)
            start_rem = remaining[0]
            seen = {start_rem}
            stack = [start_rem]
            while stack:
                cur = stack.pop()
                for nb in adj[cur]:
                    if nb in rem_set and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            if len(seen) != len(rem_set):
                return False
            return True
        
        def backtrack():
            if time.time() > deadline:
                return False
            steps[0] += 1
            if steps[0] > max_steps:
                return False
            
            if len(path) == n:
                last = path[-1]
                if (last, start) in edge_pos and edge_pos[(last, start)] < threshold:
                    return True
                return False
            
            last = path[-1]
            # Get unvisited neighbors
            candidates = [v for v in adj[last] if not visited[v]]
            
            # Pruning: if this is the last vertex to visit, it must connect back to start
            if len(path) == n - 1:
                if not candidates:
                    return False
                # The only candidate must be able to connect to start
                for v in candidates:
                    if edge_pos.get((v, start), 9999) >= threshold:
                        return False
            
            # Additional pruning: check if remaining vertices are connected
            if len(path) >= 2 and len(path) <= n - 3:
                remaining = [v for v in range(n) if not visited[v] and v != start]
                # Only do this check occasionally to save time
                if steps[0] % 100 == 0:
                    if not check_connectivity(remaining):
                        return False
            
            for next_v in candidates:
                # Don't visit start until the very end
                if next_v == start and len(path) < n - 1:
                    continue
                path.append(next_v)
                visited[next_v] = True
                if backtrack():
                    return True
                path.pop()
                visited[next_v] = False
            return False
        
        if backtrack():
            return path[:]
        return None
    
    def two_opt_refine(cycle, time_limit):
        """
        Refine a cycle using 2-opt moves to minimize the maximum edge index.
        """
        best_cycle = cycle[:]
        best_score = get_score(best_cycle)
        improved = True
        
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n):
                if time.time() > time_limit:
                    break
                for j in range(i + 2, n):
                    if time.time() > time_limit:
                        break
                    u1 = best_cycle[i]
                    v1 = best_cycle[(i + 1) % n]
                    u2 = best_cycle[j]
                    v2 = best_cycle[(j + 1) % n]
                    
                    # Skip if edges would be the same
                    if {u1, v1} == {u2, v2}:
                        continue
                    if {u1, v1} == {u2, v2}:
                        continue
                    
                    pos_old1 = edge_pos[(u1, v1)]
                    pos_old2 = edge_pos[(u2, v2)]
                    pos_new1 = edge_pos[(u1, u2)]
                    pos_new2 = edge_pos[(v1, v2)]
                    
                    old_max = max(pos_old1, pos_old2)
                    new_max = max(pos_new1, pos_new2)
                    
                    if new_max < old_max:
                        segment = best_cycle[i+1:j+1]
                        best_cycle[i+1:j+1] = segment[::-1]
                        new_score = get_score(best_cycle)
                        if new_score < best_score:
                            best_score = new_score
                            improved = True
                        else:
                            best_cycle[i+1:j+1] = best_cycle[i+1:j+1][::-1]
        return best_cycle
    
    def three_opt_refine(cycle, time_limit):
        """
        Additional 3-opt refinement: try all triples of edge removals.
        This is expensive so we use it sparingly.
        """
        best_cycle = cycle[:]
        best_score = get_score(best_cycle)
        
        # 3-opt: remove 3 edges, reconnect in 4 possible ways
        # For n=24, this is O(n^3) per iteration which is 13824, manageable
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n):
                if time.time() > time_limit:
                    break
                for j in range(i + 2, n):
                    if time.time() > time_limit:
                        break
                    for k in range(j + 2, n):
                        if time.time() > time_limit:
                            break
                        # Three edges: (i, i+1), (j, j+1), (k, k+1)
                        a1 = best_cycle[i]
                        b1 = best_cycle[(i+1) % n]
                        a2 = best_cycle[j]
                        b2 = best_cycle[(j+1) % n]
                        a3 = best_cycle[k]
                        b3 = best_cycle[(k+1) % n]
                        
                        # Try different reconnections
                        # Option 1: reverse segment between i+1 and j
                        # This is already covered by 2-opt, skip
                        # For true 3-opt, we'd try all 4 reconnections of 3 segments
                        # Given time constraints, we'll just do a few key patterns
                        
                        # Pattern: reverse [i+1..j] and [j+1..k]
                        # Edges removed: (a1,b1), (a2,b2), (a3,b3)
                        # Edges added: (a1,a2), (b1,a3), (b2,b3)
                        # But this requires careful index handling
                        # For simplicity and time, we'll rely on 2-opt primarily
                        pass
        return best_cycle
    
    # Main search strategy
    best_cycle = None
    best_score = float('inf')
    
    # Phase 1: Binary search for the minimum threshold where a Hamiltonian cycle exists
    # The minimum possible score is n=24 (all edges in first 24 positions)
    # But that's extremely unlikely. Let's start with a reasonable range.
    
    # First, find any valid cycle (full edge set)
    full_cycle = find_cycle_backtrack(num_edges, 0, 5.0)
    if full_cycle:
        best_cycle = full_cycle
        best_score = get_score(full_cycle)
    else:
        # Fallback: identity permutation
        best_cycle = list(range(n))
        best_score = get_score(best_cycle)
    
    # Binary search
    lo = n  # 24
    hi = best_score  # Current best score as upper bound
    
    # Try to narrow down
    while lo < hi and time.time() < start_time + 50:
        mid = (lo + hi) // 2
        cycle_found = None
        
        # Try multiple starting vertices
        for start_v in range(n):
            if time.time() > start_time + 50:
                break
            cycle = find_cycle_backtrack(mid, start_v, 2.0)
            if cycle:
                cycle_found = cycle
                break
        
        if cycle_found:
            score = get_score(cycle_found)
            if score < best_score:
                best_score = score
                best_cycle = cycle_found
            hi = mid
        else:
            lo = mid + 1
    
    # Phase 2: Refine with 2-opt
    if best_cycle and time.time() < start_time + 70:
        refined = two_opt_refine(best_cycle, start_time + 75)
        refined_score = get_score(refined)
        if refined_score < best_score:
            best_score = refined_score
            best_cycle = refined
    
    # Phase 3: Try random restarts with 2-opt if time permits
    if time.time() < start_time + 75:
        import random
        rng = random.Random(12345)
        num_restarts = 30
        for r in range(num_restarts):
            if time.time() > start_time + 80:
                break
            perm = list(range(n))
            rng.shuffle(perm)
            score = get_score(perm)
            if score < best_score:
                best_score = score
                best_cycle = perm
            # Refine
            if time.time() < start_time + 80:
                refined = two_opt_refine(perm, start_time + 80)
                refined_score = get_score(refined)
                if refined_score < best_score:
                    best_score = refined_score
                    best_cycle = refined
    
    # Phase 4: Final polish - try to reduce score by 1 more
    # For each edge in the cycle with the maximum position, try to replace it
    if best_cycle and time.time() < start_time + 82:
        # Find the edge(s) with max position
        max_pos_in_cycle = 0
        max_edge_idx = -1
        for i in range(n):
            u = best_cycle[i]
            v = best_cycle[(i + 1) % n]
            pos = edge_pos[(u, v)]
            if pos > max_pos_in_cycle:
                max_pos_in_cycle = pos
                max_edge_idx = i
        
        if max_edge_idx >= 0:
            u = best_cycle[max_edge_idx]
            v = best_cycle[(max_edge_idx + 1) % n]
            
            # Try to find an alternative path from u to v that avoids the direct edge
            # and uses only edges with position < max_pos_in_cycle
            # This is a shortest path problem with constraints
            # For simplicity, try 2-opt moves that specifically target this edge
            for i in range(n):
                if time.time() > start_time + 85:
                    break
                for j in range(i + 2, n):
                    if time.time() > start_time + 85:
                        break
                    a1 = best_cycle[i]
                    b1 = best_cycle[(i+1) % n]
                    a2 = best_cycle[j]
                    b2 = best_cycle[(j+1) % n]
                    
                    if {a1, b1} == {a2, b2}:
                        continue
                    
                    pos_old1 = edge_pos[(a1, b1)]
                    pos_old2 = edge_pos[(a2, b2)]
                    pos_new1 = edge_pos[(a1, a2)]
                    pos_new2 = edge_pos[(b1, b2)]
                    
                    # Only accept if the new max is strictly less than current best
                    new_max = max(pos_new1, pos_new2)
                    old_max = max(pos_old1, pos_old2)
                    
                    if new_max < old_max and max(pos_new1, pos_new2) < max_pos_in_cycle:
                        segment = best_cycle[i+1:j+1]
                        best_cycle[i+1:j+1] = segment[::-1]
                        new_score = get_score(best_cycle)
                        if new_score < best_score:
                            best_score = new_score
                        else:
                            best_cycle[i+1:j+1] = best_cycle[i+1:j+1][::-1]
    
    return best_cycle

# EVOLVE-BLOCK-END
