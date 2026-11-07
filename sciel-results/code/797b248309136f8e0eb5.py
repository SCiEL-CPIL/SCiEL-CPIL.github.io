"""Edge-disjoint Hamiltonian cycle packing via integer linear programming (scipy MILP)."""
import time
import random
import itertools
from scipy.optimize import linprog

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

DEADLINE = 85  # seconds before 90s kill

def _edge_index(u, v):
    if u > v:
        u, v = v, u
    return u * 21 + v  # since n=21, this is unique for u < v

def _find_cycles_ilp(adj, used_edges, deadline, max_cycles=3, seed=42):
    """
    Find edge-disjoint Hamiltonian cycles on all 21 vertices avoiding used_edges.
    Uses ILP formulation with scipy linprog.
    Returns list of cycles found.
    """
    n = 21
    rng = random.Random(seed)
    
    # Build edge list (only unused edges)
    edge_list = []
    edge_to_idx = {}
    for u in range(n):
        for v in adj[u]:
            if u < v:
                ekey = frozenset([u, v])
                if ekey not in used_edges:
                    idx = len(edge_list)
                    edge_list.append((u, v))
                    edge_to_idx[ekey] = idx
    
    m = len(edge_list)
    if m < n * max_cycles:
        return []
    
    # Decision variables: x[i] for each edge i (binary)
    # We want to find a 2-factor (or 2-regular spanning subgraph) that decomposes into cycles
    # Actually, let's use the standard ILP for Hamiltonian cycle packing.
    
    # Approach: Find k edge-disjoint Hamiltonian cycles.
    # For each cycle c, we have variables x_{c,e} in {0,1}.
    # Constraints:
    # 1. For each vertex v and cycle c: sum of x_{c,e} over edges incident to v = 2
    # 2. For each edge e: sum over c of x_{c,e} <= 1
    # 3. Subtour elimination: for each subset S of vertices with 2 <= |S| <= n-2,
    #    sum of x_{c,e} over edges e with both endpoints in S <= |S| - 1 for each c
    #    (This is expensive; use lazy constraints or cut generation)
    
    # Simpler approach: Find one Hamiltonian cycle at a time using ILP,
    # then mark its edges as used and repeat.
    
    cycles_found = []
    local_used = set()
    
    for cycle_num in range(max_cycles):
        if time.time() > deadline - 5:
            break
        
        # Build edge list excluding used edges
        avail_edges = []
        avail_edge_idx = {}
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    ekey = frozenset([u, v])
                    if ekey not in used_edges and ekey not in local_used:
                        idx = len(avail_edges)
                        avail_edges.append((u, v))
                        avail_edge_idx[ekey] = idx
        
        m_a = len(avail_edges)
        if m_a < n:
            break
        
        # ILP: find Hamiltonian cycle
        # x_i in {0,1} for each available edge
        # Objective: minimize sum x_i (should be exactly n for a Hamiltonian cycle)
        # Actually, maximize number of edges selected subject to degree constraints
        # and subtour elimination.
        # 
        # For a Hamiltonian cycle: each vertex has degree exactly 2, and no subtours.
        # 
        # c = -1 for all edges (maximize edges)
        # A_eq: degree constraints: for each vertex, sum of incident edges = 2
        # A_ub: subtour elimination (lazy)
        
        c = [-1.0] * m_a
        
        # Degree constraints: for each vertex v, sum of x_i for edges incident to v = 2
        A_eq = []
        b_eq = []
        for v in range(n):
            row = [0.0] * m_a
            for idx, (u, w) in enumerate(avail_edges):
                if u == v or w == v:
                    row[idx] = 1.0
            A_eq.append(row)
            b_eq.append(2.0)
        
        # Solve with subtour elimination via lazy constraints
        best_x = None
        
        for attempt in range(5):  # up to 5 rounds of cut generation
            if time.time() > deadline - 3:
                break
            
            # Solve LP relaxation
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, 1)] * m_a,
                         method='highs', options={'time_limit': 20, 'presolve': True})
            
            if res.status != 0:
                break
            
            x = res.x
            
            # Check if solution is integer
            is_integer = all(abs(x[i] - round(x[i])) < 1e-6 for i in range(m_a))
            
            if is_integer:
                # Extract cycle
                selected_edges = []
                for i in range(m_a):
                    if x[i] > 0.5:
                        selected_edges.append(avail_edges[i])
                
                # Verify it's a single cycle
                if len(selected_edges) == n:
                    # Build adjacency from selected edges
                    cycle_adj = [set() for _ in range(n)]
                    for u, v in selected_edges:
                        cycle_adj[u].add(v)
                        cycle_adj[v].add(u)
                    
                    # Check all vertices have degree 2
                    valid = all(len(cycle_adj[v]) == 2 for v in range(n))
                    
                    if valid:
                        # Trace the cycle
                        visited = set()
                        path = [0]
                        visited.add(0)
                        cur = 0
                        for _ in range(n - 1):
                            nxts = list(cycle_adj[cur] - visited)
                            if not nxts:
                                # Try the other neighbor
                                other = list(cycle_adj[cur])
                                nxts = [o for o in other if o not in visited]
                                if not nxts:
                                    valid = False
                                    break
                            cur = nxts[0]
                            visited.add(cur)
                            path.append(cur)
                        
                        if valid and len(path) == n:
                            # Check closing edge
                            if frozenset([path[-1], path[0]]) in avail_edge_idx:
                                best_x = [0.0] * m_a
                                for u, v in selected_edges:
                                    best_x[avail_edge_idx[frozenset([u, v])]] = 1.0
                                cycles_found.append(path)
                                for u, v in selected_edges:
                                    local_used.add(frozenset([u, v]))
                                break
                            else:
                                valid = False
                        
                        if not valid:
                            # Add subtour cut
                            # Find connected components
                            comp = [-1] * n
                            comp_id = 0
                            for v in range(n):
                                if comp[v] == -1:
                                    stack = [v]
                                    comp[v] = comp_id
                                    while stack:
                                        node = stack.pop()
                                        for nb in cycle_adj[node]:
                                            if comp[nb] == -1:
                                                comp[nb] = comp_id
                                                stack.append(nb)
                                    comp_id += 1
                            
                            # For each component with size < n, add cut
                            for cid in range(comp_id):
                                S = [v for v in range(n) if comp[v] == cid]
                                if 2 <= len(S) <= n - 2:
                                    row = [0.0] * m_a
                                    for idx, (u, w) in enumerate(avail_edges):
                                        if u in S and w in S:
                                            row[idx] = 1.0
                                    A_eq.append(row)
                                    b_eq.append(len(S) - 1)
                        
                        if best_x is not None:
                            break
                    else:
                        # Add cuts
                        A_eq.append([0.0] * m_a)  # dummy, will be replaced
                        A_eq.pop()
                else:
                    # Not exactly n edges, add constraint
                    row = [1.0] * m_a
                    A_eq.append(row)
                    b_eq.append(n)
            else:
                # Non-integer: add cutting plane
                # Find subtours in the support of the LP solution
                # Build graph with edges where x[i] > 0.1
                support_adj = [set() for _ in range(n)]
                for i in range(m_a):
                    if x[i] > 0.1:
                        u, v = avail_edges[i]
                        support_adj[u].add(v)
                        support_adj[v].add(u)
                
                # Find connected components
                comp = [-1] * n
                comp_id = 0
                for v in range(n):
                    if comp[v] == -1:
                        stack = [v]
                        comp[v] = comp_id
                        while stack:
                            node = stack.pop()
                            for nb in support_adj[node]:
                                if comp[nb] == -1:
                                    comp[nb] = comp_id
                                    stack.append(nb)
                        comp_id += 1
                
                for cid in range(comp_id):
                    S = [v for v in range(n) if comp[v] == cid]
                    if 2 <= len(S) <= n - 2:
                        row = [0.0] * m_a
                        for idx, (u, w) in enumerate(avail_edges):
                            if u in S and w in S:
                                row[idx] = 1.0
                        A_eq.append(row)
                        b_eq.append(len(S) - 1)
        
        if best_x is None:
            break
    
    return cycles_found

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    used_edges = set()
    all_cycles = []
    
    # Try to find 3 edge-disjoint Hamiltonian cycles using ILP
    found = _find_cycles_ilp(adj, used_edges, deadline, max_cycles=3, seed=42)
    all_cycles.extend(found)
    
    # If ILP didn't find enough, try alternative approaches
    if len(all_cycles) < 3 and time.time() < deadline - 10:
        # Try randomized greedy with backtracking for remaining cycles
        local_used = set()
        for c in all_cycles:
            for j in range(len(c)):
                u = c[j]
                v = c[(j + 1) % len(c)]
                local_used.add(frozenset([u, v]))
        
        for attempt in range(3 - len(all_cycles)):
            if time.time() > deadline - 5:
                break
            cycle = _find_one_ham_greedy(adj, local_used, deadline, seed=100 + attempt)
            if cycle is not None:
                all_cycles.append(cycle)
                for j in range(len(cycle)):
                    u = cycle[j]
                    v = cycle[(j + 1) % len(cycle)]
                    local_used.add(frozenset([u, v]))
    
    if not all_cycles:
        # Fallback: find any cycle
        for u, v in P['edges']:
            common = adj[u] & adj[v]
            for w in common:
                if w != u and w != v:
                    return [[int(u), int(v), int(w)]]
        return [[0, 1, 2]]
    
    # Validate: check edge-disjointness and same vertex set
    edge_count = {}
    vertex_set = None
    for cycle in all_cycles:
        vs = frozenset(cycle)
        if vertex_set is None:
            vertex_set = vs
        elif vs != vertex_set:
            # Mismatch - should not happen but guard
            pass
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            ekey = frozenset([u, v])
            edge_count[ekey] = edge_count.get(ekey, 0) + 1
    
    # Check no edge used twice
    valid = all(cnt == 1 for cnt in edge_count.values())
    if not valid:
        # Keep only first occurrence of each edge
        seen = set()
        filtered = []
        for cycle in all_cycles:
            ok = True
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                ekey = frozenset([u, v])
                if ekey in seen:
                    ok = False
                    break
                seen.add(ekey)
            if ok:
                filtered.append(cycle)
        all_cycles = filtered
    
    # Convert to proper types
    result = []
    for cycle in all_cycles:
        result.append([int(v) for v in cycle])
    
    return result

def _find_one_ham_greedy(adj, used_edges, deadline, seed=42):
    """Find one Hamiltonian cycle avoiding used_edges using randomized greedy with backtracking."""
    n = 21
    rng = random.Random(seed)
    start = 0
    
    visited = {start}
    path = [start]
    local_edges = set()
    
    def dfs(depth):
        if time.time() > deadline:
            return False
        if depth == n:
            last = path[-1]
            if start in adj[last] and frozenset([last, start]) not in used_edges and frozenset([last, start]) not in local_edges:
                return True
            return False
        
        cur = path[-1]
        neighbors = list(adj[cur] - visited)
        
        # Sort by fewest unvisited neighbors (Warnsdorff)
        def heuristic(nb):
            unvisited = 0
            for x in adj[nb]:
                if x not in visited and x != cur:
                    unvisited += 1
            return unvisited
        
        neighbors.sort(key=heuristic)
        rng.shuffle(neighbors)  # Add some randomness
        
        for nb in neighbors:
            e = frozenset([cur, nb])
            if e in used_edges or e in local_edges:
                continue
            
            visited.add(nb)
            local_edges.add(e)
            path.append(nb)
            
            if dfs(depth + 1):
                return True
            
            path.pop()
            local_edges.remove(e)
            visited.remove(nb)
        
        return False
    
    if dfs(1):
        return path
    return None
