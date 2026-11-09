import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = 15
    # Partition: A = {0,1,2,3,4,5,6}, B = {7,8,9,10,11,12,13,14}
    A = list(range(7))
    B = list(range(7, 15))
    
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def build_adj_and_deg(edge_set):
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg
    
    def check_violations(edge_set):
        """
        Returns count of violating (s,t) pairs where deg(s)==deg(t) and a 3-path exists.
        """
        adj, deg = build_adj_and_deg(edge_set)
        
        deg_groups = {}
        for i in range(n):
            deg_groups.setdefault(deg[i], []).append(i)
        
        violations = 0
        for d, vertices in deg_groups.items():
            if len(vertices) < 2:
                continue
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    s = vertices[i]
                    t = vertices[j]
                    # Check for simple path s-u-v-t
                    found = False
                    for u in adj[s]:
                        if u == t:
                            continue
                        for v in adj[u]:
                            if v == s or v == t:
                                continue
                            if t in adj[v]:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        violations += 1
        return violations
    
    def find_violating_path(edge_set, s, t):
        """
        Find one violating 3-path between s and t, or None.
        """
        adj, _ = build_adj_and_deg(edge_set)
        for u in adj[s]:
            if u == t:
                continue
            for v in adj[u]:
                if v == s or v == t:
                    continue
                if t in adj[v]:
                    return (s, u, v, t)
        return None
    
    # Step 1: Build K_{7,8} bipartite base (provably valid)
    base_edges = set()
    for a in A:
        for b in B:
            base_edges.add((a, b))
    
    # Verify base is valid
    assert check_violations(base_edges) == 0, "K_{7,8} base should be valid"
    
    best_edges = set(base_edges)
    best_count = len(base_edges)
    
    # Step 2: Greedy edge insertion - try to add intra-part edges
    # Intra-part edges: within A and within B
    intra_A = [(i, j) for i in range(7) for j in range(i+1, 7)]
    intra_B = [(i, j) for i in range(7, 15) for j in range(i+1, 15)]
    
    # Try adding each intra-part edge greedily
    for edge in intra_A + intra_B:
        if time.time() > deadline:
            break
        test_edges = best_edges | {edge}
        if check_violations(test_edges) == 0:
            best_edges = test_edges
            best_count += 1
    
    # Step 3: Local search - try adding more edges (including cross-part if any were 
    # removed, but we haven't removed any yet)
    # We'll do a more systematic local search:
    # Try all missing edges, and if adding one creates violations, try to fix by removing another edge
    
    missing_edges = [e for e in all_edges if e not in best_edges]
    random.shuffle(missing_edges)
    
    # Simulated annealing / local search
    T = 10.0
    T_min = 0.1
    alpha = 0.9995
    
    current_edges = set(best_edges)
    current_violations = check_violations(current_edges)
    
    steps = 0
    max_steps = 10000
    
    while steps < max_steps and time.time() < deadline:
        steps += 1
        
        if current_violations == 0:
            # Valid graph. Try to add an edge to increase density.
            # Pick a random non-edge
            non_edges = [e for e in all_edges if e not in current_edges]
            if not non_edges:
                break  # Complete graph and valid? Impossible.
            e = random.choice(non_edges)
            new_edges = set(current_edges)
            new_edges.add(e)
            new_violations = check_violations(new_edges)
            if new_violations == 0:
                current_edges = new_edges
                current_violations = 0
            else:
                # Accept with probability exp(-delta/T)
                delta = new_violations - 0  # We want to stay at 0, but adding edge breaks it
                # Actually, if we are at 0 violations, we only accept if new is 0.
                # Otherwise we might get stuck.
                # Let's allow temporary violations if we are far from optimal?
                # No, if we have a valid graph, we want to keep it valid if possible.
                # If adding an edge creates violations, we can still accept it if T is high,
                # hoping to remove them later.
                if random.random() < pow(2.718, -delta / T):
                    current_edges = new_edges
                    current_violations = new_violations
        else:
            # We have violations. Try to remove an edge involved in a violation.
            # Find all violating pairs
            adj, deg = build_adj_and_deg(current_edges)
            deg_groups = {}
            for i in range(n):
                deg_groups.setdefault(deg[i], []).append(i)
            
            violating_pairs = []
            for d, vertices in deg_groups.items():
                if len(vertices) < 2:
                    continue
                for i in range(len(vertices)):
                    for j in range(i+1, len(vertices)):
                        s = vertices[i]
                        t = vertices[j]
                        # Check if there is a 3-path
                        found = False
                        for u in adj[s]:
                            if u == t:
                                continue
                            for v in adj[u]:
                                if v == s or v == t:
                                    continue
                                if t in adj[v]:
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            violating_pairs.append((s, t))
            
            if not violating_pairs:
                # Inconsistent state? Should not happen if check_violations > 0
                break
            
            # Pick a random violating pair
            s, t = random.choice(violating_pairs)
            # Find a violating path
            path = find_violating_path(current_edges, s, t)
            if path is None:
                # Should not happen
                break
            
            s_path, u, v, t_path = path
            # Pick a random edge from this path to remove
            candidate_edges = [(s_path, u), (u, v), (v, t_path)]
            # Normalize edges
            cand_set = set()
            for a, b in candidate_edges:
                if a > b:
                    a, b = b, a
                cand_set.add((a, b))
            
            e = random.choice(list(cand_set))
            
            if e in current_edges:
                new_edges = set(current_edges)
                new_edges.discard(e)
                new_violations = check_violations(new_edges)
                
                delta = new_violations - current_violations
                # We want to decrease violations, so delta should be negative.
                if delta < 0 or random.random() < pow(2.718, -delta / T):
                    current_edges = new_edges
                    current_violations = new_violations
        
        # Update temperature
        T *= alpha
        
        # Track best
        if current_violations == 0:
            score = len(current_edges)
            if score > best_count:
                best_count = score
                best_edges = set(current_edges)
    
    # If no valid graph found, return empty or best effort?
    # The task says "Reject if...". So we must return a valid graph.
    # If we didn't find one, we need a fallback.
    # A star graph is valid?
    # Center 0, leaves 1..14.
    # Deg(0) = 14. Deg(leaves) = 1.
    # Pairs with same degree: all leaves.
    # Path between two leaves s, t: s-0-t is length 2.
    # Is there a path of length 3? s-0-x-t? No, x would have to be connected to 0 and t.
    # In a star, leaves are only connected to 0. So no path of length 3 between leaves.
    # So a star graph is valid.
    # Edges: 14.
    
    if best_count < 56:
        # Fallback: K_{7,8} base
        best_edges = set(base_edges)
    else:
        # Ensure types are correct
        best_edges = set(best_edges)
        
        # Final validation check just in case
        if check_violations(best_edges) != 0:
             # If our SA produced an invalid result due to bugs, fallback to K_{7,8}
             best_edges = set(base_edges)
    
    # Convert to list of [u, v] pairs
    result = []
    for u, v in sorted(best_edges):
        result.append([int(u), int(v)])
    
    return result
