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
    
    # Step 3: Local search - try adding more edges (including cross-part if any were skipped)
    # Actually all cross-part edges are already in. Try removing and re-adding different edges.
    
    # Try a more aggressive local search
    remaining_edges = [e for e in all_edges if e not in best_edges]
    random.shuffle(remaining_edges)
    
    for edge in remaining_edges:
        if time.time() > deadline:
            break
        test_edges = best_edges | {edge}
        if check_violations(test_edges) == 0:
            best_edges = test_edges
            best_count += 1
    
    # Step 4: Remove-and-rebuild local search
    # Try removing one edge and adding two others to increase count
    for _ in range(200):
        if time.time() > deadline:
            break
        
        # Try removing a random edge and adding a different one
        current_list = list(best_edges)
        if not current_list:
            break
        
        removed = random.choice(current_list)
        test_edges = best_edges - {removed}
        
        # Try to add a random edge not in test_edges
        candidates = [e for e in all_edges if e not in test_edges]
        if not candidates:
            continue
        random.shuffle(candidates)
        
        added = False
        for add_edge in candidates[:20]:  # Try up to 20 candidates
            trial = test_edges | {add_edge}
            if check_violations(trial) == 0 and len(trial) > len(best_edges):
                best_edges = trial
                best_count = len(trial)
                added = True
                break
    
    # Final validation
    if check_violations(best_edges) != 0:
        # Fallback to star graph
        best_edges = set((0, i) for i in range(1, n))
    
    result = [[int(u), int(v)] for u, v in best_edges]
    return result
