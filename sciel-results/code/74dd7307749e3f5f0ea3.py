import time
import random
import math

def construct():
    n = 15
    deadline = time.time() + 58.0  # Safety margin
    
    # Initialize with K_{7,8}
    # Part A: 0..6, Part B: 7..14
    adj = [0] * n
    deg = [0] * n
    edges = []
    
    for u in range(7):
        for v in range(7, 15):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            edges.append([u, v])
            
    def is_valid(adj, deg):
        """
        Checks if the graph defined by adj and deg satisfies the constraint.
        Constraint: For every edge (u,v), the set of degrees of neighbors of u (excluding v)
        and the set of degrees of neighbors of v (excluding u) must be disjoint.
        """
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    continue
                
                # Get degrees of neighbors of u, excluding v
                set_u = set()
                mask_u = adj[u] & ~(1 << v)
                # Iterate set bits
                m = mask_u
                while m:
                    lsb = m & -m
                    s = lsb.bit_length() - 1
                    set_u.add(deg[s])
                    m ^= lsb
                    
                if not set_u:
                    continue
                
                # Get degrees of neighbors of v, excluding u
                mask_v = adj[v] & ~(1 << u)
                m = mask_v
                while m:
                    lsb = m & -m
                    t = lsb.bit_length() - 1
                    if deg[t] in set_u:
                        return False
                    m ^= lsb
                    
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1
        edges.append([u, v])

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        deg[u] -= 1
        deg[v] -= 1
        # Remove from edges list
        for i, e in enumerate(edges):
            if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                edges.pop(i)
                break

    def get_missing_edges():
        missing = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    missing.append((u, v))
        return missing

    # Conflict graph approach: model edges as nodes, conflicts as edges
    # Two edges e1=(a,b) and e2=(c,d) conflict if adding both would create a violation
    
    def build_conflict_graph(adj, deg, candidate_edges):
        """
        Build a conflict graph where nodes are candidate edges.
        Two candidate edges conflict if they cannot both be present without causing a violation.
        We use a simplified model: edges conflict if they share a vertex and would create
        a path with equal-degree endpoints.
        """
        m = len(candidate_edges)
        # conflict[i] is a bitmask of which other edges conflict with edge i
        conflict = [0] * m
        
        for i in range(m):
            a, b = candidate_edges[i]
            for j in range(i + 1, m):
                c, d = candidate_edges[j]
                # Check if these two edges could create a violation
                # Simplified: if they share a vertex, check if the degrees would match
                if a == c or a == d or b == c or b == d:
                    # They share a vertex, potential conflict
                    # Check if adding both would create equal-degree path endpoints
                    # This is a heuristic - we mark as conflicting if they share a vertex
                    # and the non-shared vertices have the same degree
                    if a == c:
                        if deg[b] == deg[d]:
                            conflict[i] |= (1 << j)
                            conflict[j] |= (1 << i)
                    elif a == d:
                        if deg[b] == deg[c]:
                            conflict[i] |= (1 << j)
                            conflict[j] |= (1 << i)
                    elif b == c:
                        if deg[a] == deg[d]:
                            conflict[i] |= (1 << j)
                            conflict[j] |= (1 << i)
                    elif b == d:
                        if deg[a] == deg[c]:
                            conflict[i] |= (1 << j)
                            conflict[j] |= (1 << i)
        
        return conflict

    def greedy_max_independent_set(conflict, m):
        """
        Greedy algorithm for maximum independent set on a conflict graph.
        Returns a bitmask of selected edges.
        """
        selected = 0
        available = (1 << m) - 1
        
        while available:
            # Find the node with minimum degree in the remaining graph
            best_node = -1
            best_deg = m + 1
            for i in range(m):
                if (available >> i) & 1:
                    d = bin(conflict[i] & available).count('1')
                    if d < best_deg:
                        best_deg = d
                        best_node = i
            
            if best_node == -1:
                break
            
            # Select this node
            selected |= (1 << best_node)
            # Remove it and all its neighbors from available
            available &= ~(1 << best_node)
            available &= ~conflict[best_node]
        
        return selected

    def conflict_graph_heuristic(adj, deg, current_edges):
        """
        Use conflict graph to find a better set of edges.
        Returns a list of edges.
        """
        missing = get_missing_edges()
        if not missing:
            return current_edges[:]
        
        # Try to add missing edges using conflict graph
        all_candidate = list(current_edges) + missing
        m = len(all_candidate)
        
        if m == 0:
            return current_edges[:]
        
        conflict = build_conflict_graph(adj, deg, all_candidate)
        selected_mask = greedy_max_independent_set(conflict, m)
        
        # Extract selected edges
        selected_edges = []
        for i in range(m):
            if (selected_mask >> i) & 1:
                u, v = all_candidate[i]
                selected_edges.append([u, v])
        
        return selected_edges

    # 1. Greedy Improvement
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        missing = get_missing_edges()
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 10:
                break
            add_edge(u, v)
            if is_valid(adj, deg):
                improved = True
            else:
                remove_edge(u, v)
                
    # 2. Conflict Graph Heuristic
    if time.time() < deadline - 5:
        new_edges = conflict_graph_heuristic(adj, deg, edges)
        if len(new_edges) > len(edges):
            # Rebuild the graph with new edges
            adj = [0] * n
            deg = [0] * n
            edges = []
            for u, v in new_edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                edges.append([u, v])
    
    # 3. Simulated Annealing
    current_edges_count = len(edges)
    best_edges_count = current_edges_count
    best_adj = [x for x in adj]
    best_deg = [x for x in deg]
    best_edges = [e[:] for e in edges]
    
    # Count violations function
    def count_violations(adj, deg):
        cnt = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    continue
                
                set_u = set()
                mask_u = adj[u] & ~(1 << v)
                m = mask_u
                while m:
                    lsb = m & -m
                    s = lsb.bit_length() - 1
                    set_u.add(deg[s])
                    m ^= lsb
                    
                if not set_u:
                    continue
                
                mask_v = adj[v] & ~(1 << u)
                m = mask_v
                while m:
                    lsb = m & -m
                    t = lsb.bit_length() - 1
                    if deg[t] in set_u:
                        cnt += 1
                    m ^= lsb
        return cnt

    current_viol = count_violations(adj, deg)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline - 2 and T > T_min:
        # Pick a random move
        if random.random() < 0.6:
            # Add a random missing edge
            missing = get_missing_edges()
            if not missing:
                break
            u, v = random.choice(missing)
            add_edge(u, v)
            new_viol = count_violations(adj, deg)
            delta_viol = new_viol - current_viol
            delta_edges = 1
            
            # Cost function: we prefer 0 violations. If violations > 0, we penalize heavily.
            # If violations == 0, we prefer more edges.
            # Let's define energy E = violations * 1000 - edges
            # We want to minimize E.
            old_E = current_viol * 1000 - len(edges)
            new_E = new_viol * 1000 - len(edges)
            delta_E = new_E - old_E
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                current_viol = new_viol
                if new_viol == 0 and len(edges) > best_edges_count:
                    best_edges_count = len(edges)
                    best_adj = [x for x in adj]
                    best_deg = [x for x in deg]
                    best_edges = [e[:] for e in edges]
            else:
                remove_edge(u, v)
        else:
            # Remove a random edge
            if not edges:
                break
            idx = random.randint(0, len(edges) - 1)
            u, v = edges[idx]
            remove_edge(u, v)
            new_viol = count_violations(adj, deg)
            delta_viol = new_viol - current_viol
            delta_edges = -1
            
            old_E = current_viol * 1000 - len(edges)
            new_E = new_viol * 1000 - len(edges)
            delta_E = new_E - old_E
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                current_viol = new_viol
            else:
                add_edge(u, v)
                
        T *= cooling

    # Final Polish: Greedy add on best solution found
    if time.time() < deadline - 1:
        adj = [x for x in best_adj]
        deg = [x for x in best_deg]
        edges = [e[:] for e in best_edges]
        
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            missing = get_missing_edges()
            random.shuffle(missing)
            for u, v in missing:
                if time.time() > deadline - 0.5:
                    break
                add_edge(u, v)
                if is_valid(adj, deg):
                    improved = True
                    if len(edges) > best_edges_count:
                        best_edges_count = len(edges)
                        best_adj = [x for x in adj]
                        best_deg = [x for x in deg]
                        best_edges = [e[:] for e in edges]
                else:
                    remove_edge(u, v)

    # Return the best valid edges found
    # Ensure types are int
    out = [[int(u), int(v)] for u, v in best_edges]
    return out
