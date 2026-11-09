"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    start_time = time.time()
    n = P['n']
    edge_order = P['edge_order']
    num_edges = len(edge_order)
    
    # Precompute edge positions
    edge_pos = {}
    for i, (u, v) in enumerate(edge_order):
        edge_pos[(u, v)] = i
        edge_pos[(v, u)] = i

    def get_cycle_score(cycle):
        """Return the one-based max edge index in the cycle."""
        max_idx = 0
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            idx = edge_pos[(u, v)]
            if idx + 1 > max_idx:
                max_idx = idx + 1
        return max_idx

    def two_opt(cycle, threshold):
        """
        Try to improve the cycle by 2-opt moves, keeping all edges < threshold.
        Returns improved cycle or original if no improvement.
        """
        cycle = list(cycle)
        improved = True
        while improved:
            if time.time() - start_time > 80:
                break
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n - 1):
                    a, b = cycle[i], cycle[i + 1]
                    c, d = cycle[j], cycle[j + 1]
                    if a == c or a == d or b == c or b == d:
                        continue
                    
                    old_score = max(edge_pos[(a, b)], edge_pos[(c, d)])
                    new_score = max(edge_pos[(a, c)], edge_pos[(b, d)])
                    
                    if new_score < old_score and new_score < threshold:
                        cycle[i + 1:j + 1] = reversed(cycle[i + 1:j + 1])
                        improved = True
        return cycle

    def greedy_build(threshold):
        """
        Build a Hamiltonian path greedily, then try to close it.
        Returns a cycle or None.
        """
        adj = [[] for _ in range(n)]
        for i in range(threshold):
            u, v = edge_order[i]
            adj[u].append(v)
            adj[v].append(u)
        
        for u in range(n):
            adj[u].sort(key=lambda v: edge_pos[(u, v)])
        
        for start in range(n):
            path = [start]
            visited = [False] * n
            visited[start] = True
            
            for _ in range(n - 1):
                last = path[-1]
                best_next = None
                best_pos = float('inf')
                for next_v in adj[last]:
                    if not visited[next_v] and edge_pos[(last, next_v)] < threshold:
                        if edge_pos[(last, next_v)] < best_pos:
                            best_pos = edge_pos[(last, next_v)]
                            best_next = next_v
                
                if best_next is None:
                    break
                
                path.append(best_next)
                visited[best_next] = True
            
            if len(path) == n:
                last = path[-1]
                if edge_pos.get((last, start), float('inf')) < threshold:
                    return path
            
        return None

    def find_hamiltonian_cycle(threshold):
        """
        Find a Hamiltonian cycle using only edges with position < threshold.
        Returns the cycle (list of vertices) or None if not found.
        Uses greedy first, then backtracking if needed.
        """
        cycle = greedy_build(threshold)
        if cycle:
            return cycle
        
        adj = [[] for _ in range(n)]
        for i in range(threshold):
            u, v = edge_order[i]
            adj[u].append(v)
            adj[v].append(u)
        
        for u in range(n):
            adj[u].sort(key=lambda v: edge_pos[(u, v)])
            
        start = 0
        path = [start]
        visited = [False] * n
        visited[start] = True
        
        def backtrack():
            if time.time() - start_time > 82:
                return False
            if len(path) == n:
                last = path[-1]
                if edge_pos.get((last, start), float('inf')) < threshold:
                    return True
                return False
            
            last = path[-1]
            for next_v in adj[last]:
                if not visited[next_v]:
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
            return path
        return None

    def assignment_based_cycle(threshold):
        """
        Use minimum-cost perfect matching (assignment) to find a 2-factor,
        then try to merge cycles into a Hamiltonian cycle.
        Returns a cycle or None.
        """
        # Build cost matrix for assignment: cost[u][v] = edge position if < threshold, else inf
        INF = float('inf')
        cost = np.full((n, n), INF, dtype=float)
        for i in range(threshold):
            u, v = edge_order[i]
            cost[u][v] = i
            cost[v][u] = i
        # Set diagonal to high cost (no self-loops)
        np.fill_diagonal(cost, 1e9)
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Check if all assigned edges are valid
        for r, c in zip(row_ind, col_ind):
            if cost[r][c] >= threshold:
                return None
        
        # Build the 2-factor: each vertex has one outgoing and one incoming edge
        # next[v] = assigned successor of v
        next_v = {}
        for r, c in zip(row_ind, col_ind):
            next_v[r] = c
        
        # Find cycles in the 2-factor
        visited = [False] * n
        cycles = []
        for start in range(n):
            if not visited[start]:
                cycle = []
                cur = start
                while not visited[cur]:
                    visited[cur] = True
                    cycle.append(cur)
                    cur = next_v[cur]
                cycles.append(cycle)
        
        # If we already have a single cycle, return it
        if len(cycles) == 1:
            return cycles[0]
        
        # Try to merge cycles using 2-opt style exchanges
        # For each pair of cycles, try all edge swaps to merge them
        merged = True
        while merged and time.time() - start_time < 75:
            merged = False
            for i in range(len(cycles)):
                for j in range(i + 1, len(cycles)):
                    c1 = cycles[i]
                    c2 = cycles[j]
                    # Try swapping edges: remove (a,b) from c1 and (c,d) from c2,
                    # add (a,c) and (b,d) or (a,d) and (b,c)
                    for ai in range(len(c1)):
                        a = c1[ai]
                        b = c1[(ai + 1) % len(c1)]
                        for ci in range(len(c2)):
                            c = c2[ci]
                            d = c2[(ci + 1) % len(c2)]
                            # Check if new edges exist in threshold
                            if (a, c) in edge_pos and edge_pos[(a, c)] < threshold and \
                               (b, d) in edge_pos and edge_pos[(b, d)] < threshold:
                                # Merge: reverse c2 segment from c to d, splice
                                # New cycle: c1[0..ai] + reversed(c2[ci..end + 0..ci-1]) 
                                # Actually: take c1 up to a, then follow c2 from c backwards... 
                                # Simpler: create new cycle by splicing
                                # c1: ... a -> b ... (rest of c1)
                                # c2: ... c -> d ... (rest of c2)
                                # Remove a->b and c->d, add a->c and b->d
                                # This merges into one cycle
                                new_cycle = []
                                # Start from b in c1, go around c1 until a
                                for k in range(ai + 1, len(c1)):
                                    new_cycle.append(c1[k])
                                for k in range(ai + 1):
                                    new_cycle.append(c1[k])
                                # Now insert c2: from c go around c2 until d, but we removed c->d
                                # So from c: c, ..., d's predecessor in c2
                                # Then from d: d, ..., c's predecessor in c2
                                # Wait, let me think more carefully.
                                # After removing a->b and c->d:
                                # c1 path: b -> ... -> a (broken)
                                # c2 path: d -> ... -> c (broken)
                                # New edges: a->c and b->d
                                # So: b -> ... -> a -> c -> ... -> d -> b? No.
                                # Let me reconstruct:
                                # c1 = [x0, x1, ..., x_ai=a, x_ai+1=b, ..., x_m-1]
                                # c2 = [y0, y1, ..., y_ci=c, y_ci+1=d, ..., y_k-1]
                                # Remove a->b: c1 becomes path b, ..., a
                                # Remove c->d: c2 becomes path d, ..., c
                                # Add a->c: a connects to c
                                # Add b->d: b connects to d
                                # New cycle: b -> ... -> a -> c -> ... -> d -> b
                                # That's: [b, ..., a] + [c, ..., d]
                                seg1 = c1[ai+1:] + c1[:ai+1]  # b, ..., a
                                seg2 = c2[ci:] + c2[:ci]       # c, ..., d
                                new_cycle = seg1 + seg2
                                # Verify it's a valid cycle
                                if len(new_cycle) == len(c1) + len(c2):
                                    cycles[i] = new_cycle
                                    del cycles[j]
                                    merged = True
                                    break
                        if merged:
                            break
                    if merged:
                        break
                if merged:
                    break
        
        if len(cycles) == 1:
            return cycles[0]
        return None

    # Binary search for the minimum threshold
    lo = n
    hi = num_edges
    best_cycle = None
    best_threshold = num_edges
    
    # Try assignment-based approach for initial solution
    for t in range(n, min(n + 50, num_edges + 1)):
        if time.time() - start_time > 20:
            break
        cyc = assignment_based_cycle(t)
        if cyc:
            best_cycle = cyc
            best_threshold = t
            break
    
    # If assignment didn't find one, fall back to greedy/backtrack
    if not best_cycle:
        full_cycle = find_hamiltonian_cycle(num_edges)
        if full_cycle:
            best_cycle = full_cycle
            best_threshold = num_edges
        else:
            return list(range(n))
    
    # Binary search for the minimum threshold
    lo = n
    hi = best_threshold
    while lo < hi:
        mid = (lo + hi) // 2
        if time.time() - start_time > 70:
            break
        
        # Try assignment first, then greedy/backtrack
        cycle = assignment_based_cycle(mid)
        if not cycle:
            cycle = find_hamiltonian_cycle(mid)
        if cycle:
            best_cycle = cycle
            best_threshold = mid
            hi = mid
        else:
            lo = mid + 1
            
    # Polish: apply 2-opt to improve the specific cycle
    if best_cycle:
        improved = two_opt(best_cycle, best_threshold)
        if get_cycle_score(improved) < get_cycle_score(best_cycle):
            best_cycle = improved
        
        # Try to find an even better threshold by checking just below
        for t in range(best_threshold - 1, max(n - 1, best_threshold - 10), -1):
            if time.time() - start_time > 80:
                break
            cycle = assignment_based_cycle(t)
            if not cycle:
                cycle = find_hamiltonian_cycle(t)
            if cycle:
                improved = two_opt(cycle, t)
                if get_cycle_score(improved) < get_cycle_score(best_cycle):
                    best_cycle = improved
                    best_threshold = t
                break
    
    if best_cycle:
        return best_cycle
    else:
        return list(range(n))

# EVOLVE-BLOCK-END
