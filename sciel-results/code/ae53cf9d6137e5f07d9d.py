"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque
import sys

# Increase recursion limit for DFS
sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_bicolored_forest_full(colors, n, edges):
    """Full check for validity. Used for final verification."""
    if len(colors) != n:
        return False
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
        
    # Group edges by color pair
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        key = (min(cu, cv), max(cu, cv))
        pair_edges[key].append((u, v))
    
    # For each pair, check if it forms a forest using union-find
    for pair, elist in pair_edges.items():
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        for u, v in elist:
            if not union(u, v):
                return False
    return True

def _solve_3_coloring():
    """
    Attempt to find a 3-coloring using Backtracking with Constraint Propagation (AC-3).
    The constraint is:
    1. Adjacent vertices must have different colors.
    2. For any pair of colors (c1, c2), the subgraph induced by vertices with colors in {c1, c2} must be a forest.
    
    This is equivalent to saying: There is no cycle in the graph where all vertices are colored with exactly two colors.
    """
    n = P['n']
    edges = P['edges']
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Pre-compute neighbors for speed
    neighbors = [list(adj[i]) for i in range(n)]
    
    # State: colors[i] = -1 if unassigned, else 0, 1, 2
    colors = [-1] * n
    domains = [set([0, 1, 2]) for _ in range(n)]
    
    # AC-3 implementation
    # Queue of arcs (i, j) to check
    queue = deque()
    
    # Initialize queue with all arcs
    for u in range(n):
        for v in neighbors[u]:
            queue.append((u, v))
            
    def revise(i, j):
        """
        Check if domain of i can be reduced given domain of j.
        Returns True if domain of i was reduced.
        """
        modified = False
        for ci in list(domains[i]):
            # Check if there exists a valid cj in domains[j]
            # Constraint 1: ci != cj
            # Constraint 2: Adding edge (i,j) with colors (ci, cj) must not create a bicolored cycle.
            
            # If we assign i=ci and j=cj, we need to check if this creates a cycle in the subgraph of colors {ci, cj}.
            # The subgraph consists of all edges (u,v) where {colors[u], colors[v]} == {ci, cj}.
            # This is complex to check dynamically in AC-3 because the "cycle" depends on the entire assignment of other vertices.
            
            # Simplified heuristic for AC-3 in this specific problem:
            # Actually, AC-3 is hard here because the "bicolored cycle" constraint is global (non-binary in a simple sense).
            # However, we can use a necessary condition:
            # If we fix i=ci and j=cj, does it force a cycle?
            # A simpler propagation: Just enforce ci != cj.
            # The cycle constraint will be handled by the search (backtracking) and local checks.
            
            # Let's stick to simple arc consistency for ci != cj.
            if ci in domains[j]:
                # If domains[j] only contains ci, then ci is not a valid value for i (since i!=j)
                # Wait, if domains[j] = {ci}, then j MUST be ci. If i tries to be ci, conflict.
                # So if ci in domains[j] and len(domains[j]) == 1:
                #   remove ci from domains[i]
                pass
            
            # Actually, let's just do the basic inequality constraint for propagation.
            # The cycle constraint is too strong for simple AC-3 without more state.
            
            # Revised strategy: Use AC-3 only for inequality. Use Backtracking for the rest.
            # This is essentially a standard graph coloring search, but with a specific validity check at leaves/intermediate nodes.
            
            # To make it smarter, we can check: if we assign i=ci, does it create a monochromatic edge? No, domains handle that.
            # Does it create a bicolored cycle?
            # We can check: If i is assigned ci, and j is assigned cj (ci!=cj),
            # is there a path between i and j in the graph induced by colors {ci, cj} that does not use edge (i,j)?
            # If yes, then (i,j) creates a cycle.
            
            # This check is O(N+E). Doing it for every arc revision is too slow for AC-3.
            # So we will only use AC-3 for inequality, and rely on DFS with a fast "local cycle check" when assigning.
            
        return modified

    # Since AC-3 for the cycle constraint is complex, we will use a different approach:
    # Backtracking Search with:
    # 1. MRV Heuristic (Minimum Remaining Values) - pick vertex with smallest domain.
    # 2. Forward Checking: When assigning v=c, remove c from neighbors' domains.
    # 3. Cycle Check: When assigning v=c, check if this creates a bicolored cycle involving v.
    
    # Let's implement a specialized backtracker.
    
    # Pre-compute: For each vertex v and color c, what are the "critical" edges?
    # Actually, let's just do the check on the fly.
    
    def is_valid_assignment(v, c, current_colors):
        """
        Check if assigning v=c is valid given current_colors (where -1 is unassigned).
        1. No neighbor has color c.
        2. No bicolored cycle is formed.
        
        A bicolored cycle is formed if:
        - There exists a neighbor u with color c' (c' != c).
        - And there is a path from v to u in the graph induced by colors {c, c'} that does not use the edge (v,u)?
        - Actually, if we add edge (v,u) with colors (c, c'), a cycle exists if there is already a path between v and u in the subgraph of colors {c, c'} excluding edge (v,u).
        
        Note: current_colors has -1 for unassigned.
        The subgraph induced by {c, c'} consists of:
        - Edges (x,y) where current_colors[x] and current_colors[y] are in {c, c'} and not -1.
        - Plus the new edge (v,u) if we assign v=c.
        
        So we need to check connectivity in the graph of vertices with colors in {c, c'} (including v, which is now c).
        """
        # Check 1: Neighbor conflict
        for u in neighbors[v]:
            if current_colors[u] == c:
                return False
        
        # Check 2: Bicolored cycle
        # For each neighbor u with a fixed color c' != c:
        # We need to check if v and u are connected in the subgraph of colors {c, c'} WITHOUT using edge (v,u).
        # If they are connected, adding (v,u) creates a cycle.
        
        # To do this efficiently, we can run BFS/DFS from v in the subgraph of colors {c, c'}
        # and see if we can reach any neighbor u (with color c') via a path of length >= 2.
        # Or simpler: Check if v and u are in the same connected component of the subgraph G[{c, c'}] 
        # where G[{c, c'}] includes v (color c) and all fixed vertices with colors c or c'.
        # If v and u are connected in this component (before adding edge v-u? No, v-u is the edge we are adding).
        # Actually, if there is a path from v to u in the existing fixed subgraph of colors {c, c'}, then adding edge (v,u) creates a cycle.
        
        # Let's build the set of "active" vertices for colors {c, c'}.
        # Active vertices: v (color c) + all u' where current_colors[u'] in {c, c'}.
        # Edges: All edges (x,y) where x,y are active and current_colors[x], current_colors[y] in {c, c'}.
        # Wait, if current_colors[x] is -1, it's not in the subgraph.
        
        # So for a specific c':
        # 1. Identify neighbors of v that have color c'. Let's call them U_cprime.
        # 2. If U_cprime is empty, no cycle formed with color pair {c, c'}.
        # 3. If U_cprime is not empty, check if v is connected to any u in U_cprime in the graph induced by vertices with colors in {c, c'} (where v is treated as color c).
        
        # Optimization: We can do a single BFS/DFS from v in the graph induced by colors {c, c'} for each c'.
        # Since there are only 2 other colors, we do this for c'=0,1,2 (where c'!=c).
        
        for c_prime in range(3):
            if c_prime == c:
                continue
            
            # Check if any neighbor has color c_prime
            has_neighbor_cp = False
            for u in neighbors[v]:
                if current_colors[u] == c_prime:
                    has_neighbor_cp = True
                    break
            
            if not has_neighbor_cp:
                continue
            
            # Check if v and any neighbor with color c_prime are connected in the subgraph of colors {c, c_prime}
            # Subgraph vertices: v (color c) + all fixed vertices with color c or c_prime.
            # We want to know if there is a path from v to any neighbor u (with color c_prime) that does not use the direct edge (v,u).
            # This is equivalent to: Are v and u in the same connected component of the subgraph formed by vertices with colors {c, c_prime} (including v)?
            # If yes, then there is a path. Adding edge (v,u) creates a cycle.
            
            # BFS from v
            visited = {v}
            q = deque([v])
            found_cycle = False
            
            while q:
                node = q.popleft()
                for nb in neighbors[node]:
                    if nb in visited:
                        continue
                    
                    # Can we move to nb?
                    # nb must be in the subgraph: current_colors[nb] must be c or c_prime.
                    # Note: v is c. Other nodes must be fixed to c or c_prime.
                    cnb = current_colors[nb]
                    
                    if cnb == c or cnb == c_prime:
                        # If we reach a neighbor u of v with color c_prime, we found a path.
                        # But we must ensure the path doesn't just go v->u.
                        # Since we start at v, the first step is to a neighbor.
                        # If the first step is to a neighbor with color c_prime, that's the direct edge.
                        # We need a path of length >= 2.
                        # So if we reach a neighbor u (of v) with color c_prime via a node other than v, it's a cycle.
                        
                        # Actually, if we reach u directly from v, that's the edge we are adding.
                        # So we should check if we reach u from a node x != v.
                        
                        # Let's modify BFS to track parent or distance.
                        # Or simpler: Remove v from the graph temporarily? No, v is the start.
                        
                        # If we reach a node nb that is a neighbor of v AND has color c_prime,
                        # and the path length is > 1, then cycle.
                        
                        # Let's use a distance array.
                        pass
                    
                    # Let's restart BFS logic with distance
            # Redo BFS with distance
            dist = {v: 0}
            q = deque([v])
            
            while q:
                node = q.popleft()
                for nb in neighbors[node]:
                    if nb in dist:
                        continue
                    cnb = current_colors[nb]
                    if cnb == c or cnb == c_prime:
                        dist[nb] = dist[node] + 1
                        # If nb is a neighbor of v and has color c_prime and dist[nb] > 1:
                        # Then there is a path v -> ... -> nb of length > 1.
                        # Since nb is a neighbor of v, edge (v, nb) exists.
                        # So v-...-nb-v is a cycle.
                        if nb in neighbors[v] and cnb == c_prime and dist[nb] > 1:
                            return False
                        q.append(nb)
                            
        return True

    # Backtracking Search
    # Heuristic: MRV (Minimum Remaining Values)
    # Since domains are implicitly {0,1,2} minus assigned neighbors, we can just pick the unassigned vertex with the most assigned neighbors?
    # Or just static ordering? Let's try static ordering first for simplicity, then optimize if needed.
    # Actually, for 50 nodes, static ordering with good pruning might be enough.
    
    # Let's use a dynamic variable ordering: Pick unassigned vertex with highest degree to assigned neighbors?
    # Or just pick the first unassigned.
    
    def backtrack():
        # Find unassigned vertex
        # Use MRV: vertex with smallest number of available colors
        # Available colors for v: {0,1,2} - {colors[u] for u in neighbors[v] if colors[u] != -1}
        
        min_domain_size = 4
        best_v = -1
        
        for v in range(n):
            if colors[v] == -1:
                # Count available colors
                avail = 3
                for u in neighbors[v]:
                    if colors[u] != -1:
                        avail -= 1
                        # Note: This assumes all neighbors have distinct colors, which is not true.
                        # Better: count distinct colors in neighbors.
                        pass
                
                # Let's just compute distinct colors
                used_colors = set()
                for u in neighbors[v]:
                    if colors[u] != -1:
                        used_colors.add(colors[u])
                
                domain_size = 3 - len(used_colors)
                
                if domain_size < min_domain_size:
                    min_domain_size = domain_size
                    best_v = v
                    if domain_size == 0:
                        return False # Dead end
                    if domain_size == 1:
                        break # Can't get better than 1
        
        if best_v == -1:
            return True # All assigned
        
        v = best_v
        used_colors = set()
        for u in neighbors[v]:
            if colors[u] != -1:
                used_colors.add(colors[u])
        
        # Try colors
        # Order colors by least constraining value? Or just 0,1,2
        for c in [0, 1, 2]:
            if c in used_colors:
                continue
            
            # Check validity (cycle constraint)
            if is_valid_assignment(v, c, colors):
                colors[v] = c
                if backtrack():
                    return True
                colors[v] = -1
        
        return False

    # Start search
    if backtrack():
        return colors
    return None

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 85  # Safety margin
    
    # Try 3 colors first with the sophisticated solver
    result = _solve_3_coloring()
    if result is not None:
        if _check_bicolored_forest_full(result, n, edges):
            return [int(c) for c in result]
    
    # Fallback: Random search for 3, 4, 5 colors
    rng = random.Random(42)
    
    for num_colors in [3, 4, 5]:
        if time.time() > deadline:
            break
            
        for _ in range(2000):
            if time.time() > deadline:
                break
            colors = [rng.randint(0, num_colors - 1) for _ in range(n)]
            if _check_bicolored_forest_full(colors, n, edges):
                return [int(c) for c in colors]
                
    # Fallback: All different
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
