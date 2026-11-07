"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def construct():
    import random
    from random import Random

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_2sat(num_vars, clauses):
        """Solve 2-SAT using implication graph + topological sort.
        num_vars: number of boolean variables (0..num_vars-1)
        clauses: list of (var_i, val_i, var_j, val_j) meaning (x_i == val_i) OR (x_j == val_j)
        Returns assignment list or None if unsatisfiable.
        """
        if num_vars == 0:
            return []
        
        # Build implication graph
        # Variable encoding: var i, value 0 -> node 2*i, value 1 -> node 2*i+1
        # Negation: node 2*i <-> node 2*i+1
        n_nodes = 2 * num_vars
        graph = [[] for _ in range(n_nodes)]
        rev_graph = [[] for _ in range(n_nodes)]
        
        def node_of(var, val):
            return 2 * var + val
        
        def neg_node(node):
            return node ^ 1
        
        for (vi, vali, vj, valj) in clauses:
            ni = node_of(vi, vali)
            nj = node_of(vj, valj)
            # Clause: (xi == vali) OR (xj == valj)
            # Equivalent to: NOT(xi == vali) -> (xj == valj) AND NOT(xj == valj) -> (xi == vali)
            # NOT(xi == vali) = (xi == 1-vali)
            ni_neg = node_of(vi, 1 - vali)
            nj_neg = node_of(vj, 1 - valj)
            
            graph[ni_neg].append(nj)
            rev_graph[nj].append(ni_neg)
            graph[nj_neg].append(ni)
            rev_graph[ni].append(nj_neg)
        
        # Kosaraju's algorithm for SCC
        visited = [False] * n_nodes
        order = []
        
        def dfs1(v):
            visited[v] = True
            for u in graph[v]:
                if not visited[u]:
                    dfs1(u)
            order.append(v)
        
        for v in range(n_nodes):
            if not visited[v]:
                dfs1(v)
        
        comp = [-1] * n_nodes
        comp_id = 0
        
        def dfs2(v, cid):
            comp[v] = cid
            for u in rev_graph[v]:
                if comp[u] == -1:
                    dfs2(u, cid)
        
        for v in reversed(order):
            if comp[v] == -1:
                dfs2(v, comp_id)
                comp_id += 1
        
        # Check for contradiction
        for v in range(num_vars):
            if comp[node_of(v, 0)] == comp[node_of(v, 1)]:
                return None
        
        # Determine assignment: variable v = 1 if comp[node(v,1)] > comp[node(v,0)]
        # (higher component ID means it's assigned first in reverse topological order,
        #  which means it's "more true" in the 2-SAT assignment)
        assignment = []
        for v in range(num_vars):
            # In Kosaraju, components are found in reverse topological order.
            # A variable should be True if its True-literal is in a later component.
            c0 = comp[node_of(v, 0)]
            c1 = comp[node_of(v, 1)]
            if c1 > c0:
                assignment.append(1)
            else:
                assignment.append(0)
        
        return assignment

    def solve_instance(n, edges, lists, deadline):
        """Solve using constraint propagation + 2-SAT reduction with backtracking on remaining choices."""
        if time.time() > deadline:
            return None

        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Try multiple strategies:
        # Strategy 1: Full constraint propagation, then 2-SAT if all domains <= 2
        # Strategy 2: If 2-SAT not directly applicable, do limited backtracking with propagation

        def try_solve_with_propagation_2sat(initial_domains):
            """Apply AC-3 style propagation, then reduce to 2-SAT if possible."""
            domains = [set(d) for d in initial_domains]
            color = [-1] * n
            
            # AC-3 style propagation
            # Queue of (arc, color) pairs that need checking
            # Actually, let's do iterative domain reduction
            changed = True
            while changed:
                changed = False
                for v in range(n):
                    if len(domains[v]) == 0:
                        return None
                    if len(domains[v]) == 1:
                        c = next(iter(domains[v]))
                        if color[v] != c:
                            color[v] = c
                            for u in adj[v]:
                                if c in domains[u]:
                                    domains[u].discard(c)
                                    changed = True
                                    if len(domains[u]) == 0:
                                        return None
            
            # Check if all unassigned vertices have domain size <= 2
            unassigned = []
            for v in range(n):
                if color[v] == -1:
                    if len(domains[v]) > 2:
                        return None  # Can't reduce to 2-SAT
                    unassigned.append(v)
            
            if not unassigned:
                return color
            
            # Build 2-SAT instance
            # Map unassigned vertices to variable indices
            var_map = {v: i for i, v in enumerate(unassigned)}
            num_vars = len(unassigned)
            
            clauses = []
            for u in unassigned:
                for v in adj[u]:
                    if v not in var_map:
                        # v is already assigned
                        if color[v] in domains[u]:
                            return None  # Contradiction
                        continue
                    # Both u and v are unassigned with 2-color domains
                    if len(domains[u]) == 2 and len(domains[v]) == 2:
                        du = sorted(domains[u])
                        dv = sorted(domains[v])
                        # u gets du[0] -> var_u = 0, u gets du[1] -> var_u = 1
                        # v gets dv[0] -> var_v = 0, v gets dv[1] -> var_v = 1
                        # Constraint: not (color_u == color_v)
                        # For each pair (c_u, c_v) with c_u == c_v, add clause forbidding (u=c_u AND v=c_v)
                        # That is: NOT(u=c_u) OR NOT(v=c_v)
                        # u=c_u means var_u = idx where du[idx] = c_u
                        # NOT(u=c_u) means var_u != idx, i.e., var_u = 1-idx
                        for c in du:
                            if c in dv:
                                idx_u = du.index(c)
                                idx_v = dv.index(c)
                                # Clause: (var_u == 1-idx_u) OR (var_v == 1-idx_v)
                                clauses.append((var_map[u], 1 - idx_u, var_map[v], 1 - idx_v))
            
            # Solve 2-SAT
            sat_assignment = solve_2sat(num_vars, clauses)
            if sat_assignment is None:
                return None
            
            # Extract colors
            for v in unassigned:
                vi = var_map[v]
                dv = sorted(domains[v])
                color[v] = dv[sat_assignment[vi]]
            
            # Verify
            for v in range(n):
                if color[v] not in lists[v]:
                    return None
                for u in adj[v]:
                    if u > v and color[u] == color[v]:
                        return None
            
            return color

        # First, try pure propagation + 2-SAT
        result = try_solve_with_propagation_2sat(lists)
        if result is not None:
            return result
        
        # If that fails, do backtracking with propagation
        # Use MRV + forward checking but with stronger propagation
        domains = [set(lst) for lst in lists]
        color = [-1] * n
        uncoloured = set(range(n))
        
        def propagate(domains, color, uncoloured):
            """Apply unit propagation. Returns False if contradiction."""
            queue = deque()
            # Find all vertices with domain size 1
            for v in range(n):
                if v in uncoloured and len(domains[v]) == 1:
                    queue.append(v)
            
            while queue:
                v = queue.popleft()
                if v not in uncoloured:
                    continue
                if len(domains[v]) != 1:
                    continue
                c = next(iter(domains[v]))
                if color[v] != -1:
                    if color[v] != c:
                        return False
                    continue
                color[v] = c
                uncoloured.discard(v)
                for u in adj[v]:
                    if u in uncoloured and c in domains[u]:
                        domains[u].discard(c)
                        if len(domains[u]) == 0:
                            return False
                        if len(domains[u]) == 1:
                            queue.append(u)
            return True

        def backtrack():
            if time.time() > deadline:
                return False
            
            if not uncoloured:
                return True
            
            # Check for domain wipeout
            for v in uncoloured:
                if len(domains[v]) == 0:
                    return False
            
            # MRV with tie-break by degree
            best_v = -1
            min_size = float('inf')
            best_deg = -1
            for v in uncoloured:
                size = len(domains[v])
                if size < min_size or (size == min_size and len(adj[v]) > best_deg):
                    min_size = size
                    best_v = v
                    best_deg = len(adj[v])
            
            v = best_v
            uncoloured.discard(v)
            
            # Try each color in domain
            for c in sorted(domains[v]):
                # Save state
                saved_domains = [set(d) for d in domains]
                saved_color = color[v]
                
                # Assign
                domains[v] = {c}
                color[v] = c
                
                # Propagate
                if propagate(domains, color, uncoloured):
                    # Try 2-SAT on remaining
                    remaining = [v for v in uncoloured]
                    if all(len(domains[v]) <= 2 for v in remaining):
                        # Try 2-SAT reduction
                        temp_result = try_solve_with_propagation_2sat(
                            [domains[v] if v in uncoloured else {color[v]} for v in range(n)]
                        )
                        if temp_result is not None:
                            # Verify it works with current assignments
                            valid = True
                            for i in range(n):
                                if color[i] != -1 and temp_result[i] != color[i]:
                                    valid = False
                                    break
                            if valid:
                                # Update global state
                                for i in range(n):
                                    color[i] = temp_result[i]
                                return True
                        # Restore and continue backtracking
                        for i in range(n):
                            domains[i] = saved_domains[i]
                        color[v] = saved_color
                        if v in uncoloured:
                            uncoloured.discard(v)
                        # Re-add v to uncoloured for next iteration
                        uncoloured.add(v)
                        continue
                    
                    if backtrack():
                        return True
                
                # Undo
                for i in range(n):
                    domains[i] = saved_domains[i]
                color[v] = saved_color
                uncoloured.add(v)
            
            return False

        if propagate(domains, color, uncoloured):
            if backtrack():
                # Final verification
                for v in range(n):
                    if color[v] not in lists[v]:
                        return None
                    for u in adj[v]:
                        if color[u] == color[v]:
                            return None
                return color
        
        return None

    deadline = time.time() + 115  # 115 second safety margin (150s limit)

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        result = solve_instance(n, edges, lists, deadline)
        if result is not None:
            # Ensure all values are Python int
            out.append([int(c) for c in result])
        else:
            # Fallback: greedy with conflict resolution
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [0] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colour[u] for u in adj[v]}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            out.append([int(c) for c in colour])

    return out
# EVOLVE-BLOCK-END
