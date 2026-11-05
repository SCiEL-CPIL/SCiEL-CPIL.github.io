"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_bitmasks(edges, adj):
        """Build conflict as list of bitmasks for fast checking."""
        m = len(edges)
        if m == 0:
            return []
        
        conflict = [0] * m
        
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                # Check shared vertices
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                    continue
                # Check if any endpoint of e1 is adjacent to any endpoint of e2
                if (u2 in adj.get(u1, set()) or v2 in adj.get(u1, set()) or
                    u2 in adj.get(v1, set()) or v2 in adj.get(v1, set())):
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
        
        return conflict

    def get_neighbors(conflict, i, m):
        """Get list of neighbors of vertex i in conflict graph."""
        neighbors = []
        mask = conflict[i]
        while mask:
            lsb = mask & (-mask)
            j = lsb.bit_length() - 1
            neighbors.append(j)
            mask ^= lsb
        return neighbors

    def dsatur_greedy(conflict, m):
        """
        DSATUR-inspired greedy coloring:
        Pick the uncolored edge with the most distinct conflicting colors (saturation),
        breaking ties by degree in the conflict graph.
        """
        color = [-1] * m
        num_colors = 0
        
        for _ in range(m):
            best_edge = -1
            best_sat = -1
            best_deg = -1
            
            for i in range(m):
                if color[i] != -1:
                    continue
                # Count distinct colors among neighbors
                sat_colors = set()
                for j in get_neighbors(conflict, i, m):
                    if color[j] != -1:
                        sat_colors.add(color[j])
                sat = len(sat_colors)
                deg = bin(conflict[i]).count('1')
                
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_edge = i
            
            if best_edge == -1:
                break
            
            # Assign the smallest available color
            used_colors = set()
            for j in get_neighbors(conflict, best_edge, m):
                if color[j] != -1:
                    used_colors.add(color[j])
            
            c = 0
            while c in used_colors:
                c += 1
            
            color[best_edge] = c
            if c >= num_colors:
                num_colors = c + 1
        
        return color, num_colors

    def arc_consistency(color, conflict, m, max_color):
        """
        Apply arc consistency (AC-3) to prune domains.
        Returns True if consistent, False if domain wipeout.
        Also prunes colors > max_color from domains.
        """
        # Domain for each vertex: set of available colors
        domains = [set() for _ in range(m)]
        
        for i in range(m):
            for c in range(max_color + 1):
                # Check if color c is available for vertex i
                available = True
                for j in get_neighbors(conflict, i, m):
                    if color[j] == c:
                        available = False
                        break
                if available:
                    domains[i].add(c)
                elif color[i] == c:
                    # Already assigned, keep it
                    domains[i].add(c)
        
        # AC-3 queue
        queue = deque()
        for i in range(m):
            for j in get_neighbors(conflict, i, m):
                queue.append((i, j))
        
        while queue:
            i, j = queue.popleft()
            # Remove values from domain[i] that have no support in domain[j]
            to_remove = []
            for c in domains[i]:
                # Check if there's a value in domain[j] different from c
                # (Since they're adjacent, they must have different colors)
                if not any(dj != c for dj in domains[j]):
                    to_remove.append(c)
            
            for c in to_remove:
                domains[i].discard(c)
            
            if not domains[i]:
                return False
            
            # If domain[i] changed, add (k, i) for all neighbors k of i
            if to_remove:
                for k in get_neighbors(conflict, i, m):
                    if k != j:
                        queue.append((k, i))
        
        return True

    def forward_checking_search(color, conflict, m, max_color, time_limit):
        """
        Backtracking search with forward checking.
        Tries to find a valid coloring using at most max_color+1 colors.
        Returns a valid coloring or None.
        """
        # Start from current assignment, unassigned vertices are -1
        assignment = color[:]
        unassigned = [i for i in range(m) if assignment[i] == -1]
        
        # Order by degree (most constrained first)
        unassigned.sort(key=lambda i: -bin(conflict[i]).count('1'))
        
        # Try to assign colors to unassigned vertices
        def backtrack(idx):
            if time.time() > time_limit:
                return None
            
            if idx == len(unassigned):
                return assignment[:]
            
            i = unassigned[idx]
            
            # Get available colors
            used_colors = set()
            for j in get_neighbors(conflict, i, m):
                if assignment[j] != -1:
                    used_colors.add(assignment[j])
            
            # Try colors in order of least constraining
            available = [c for c in range(max_color + 1) if c not in used_colors]
            
            # Sort by least constraining (fewest options removed from neighbors)
            def constraining(c):
                count = 0
                for j in get_neighbors(conflict, i, m):
                    if assignment[j] == -1:
                        count += 1
                return count
            
            available.sort(key=lambda c: (constraining(c), c))
            
            for c in available:
                assignment[i] = c
                
                # Forward check: verify no neighbor has empty domain
                consistent = True
                for j in get_neighbors(conflict, i, m):
                    if assignment[j] == -1:
                        # Check if j still has available colors
                        neighbor_used = set()
                        for k in get_neighbors(conflict, j, m):
                            if assignment[k] != -1:
                                neighbor_used.add(assignment[k])
                        if not any(cc not in neighbor_used for cc in range(max_color + 1)):
                            consistent = False
                            break
                
                if consistent:
                    result = backtrack(idx + 1)
                    if result is not None:
                        return result
                
                assignment[i] = -1
            
            return None
        
        return backtrack(0)

    def min_sum_coloring(conflict, m, initial_color, time_limit):
        """
        Minimum Sum Coloring via iterative deepening with constraint propagation.
        """
        best_color = initial_color[:]
        best_sum = sum(best_color)
        
        # Get current max color
        current_max = max(best_color) if best_color else 0
        
        # Try to reduce the number of colors using backtracking
        for max_c in range(current_max, -1, -1):
            if time.time() > time_limit:
                break
            
            # Create a partial assignment: keep vertices with color <= max_c
            # Unassign vertices with color > max_c
            partial = best_color[:]
            for i in range(m):
                if partial[i] > max_c:
                    partial[i] = -1
            
            # Try to complete the assignment
            result = forward_checking_search(partial, conflict, m, max_c, time_limit)
            
            if result is not None:
                best_color = result
                best_sum = sum(best_color)
                # Continue trying to reduce further
                continue
            else:
                # Can't reduce further with this max_c, try local improvements
                break
        
        # Local search to reduce sum
        for iteration in range(50):
            if time.time() > time_limit:
                break
            
            improved = False
            cur_sum = sum(best_color)
            
            # Try moving each vertex to a lower color if possible
            for i in range(m):
                if time.time() > time_limit:
                    break
                
                current_c = best_color[i]
                if current_c == 0:
                    continue
                
                # Try moving to color 0, then 1, etc.
                for c in range(current_c):
                    # Check if color c is available
                    available = True
                    for j in get_neighbors(conflict, i, m):
                        if best_color[j] == c:
                            available = False
                            break
                    
                    if available:
                        best_color[i] = c
                        improved = True
                        break
            
            if not improved:
                break
        
        return best_color

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        m = len(edges)
        
        if m == 0:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflict = build_conflict_bitmasks(edges, adj)
        
        # DSATUR greedy initialization
        color, nc = dsatur_greedy(conflict, m)
        
        # Use minimum sum coloring with constraint propagation
        remaining_time = deadline - time.time() - 10
        if remaining_time > 5:
            color = min_sum_coloring(conflict, m, color, time.time() + remaining_time)
        
        return color

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline - 45:
            break
        
        n, d, seed = spec
        out.append(solve_instance(n, d, seed))
    
    while len(out) < 3:
        out.append([])
    
    return out
# EVOLVE-BLOCK-END
