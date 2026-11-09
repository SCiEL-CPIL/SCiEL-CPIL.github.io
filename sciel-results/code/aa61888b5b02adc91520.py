import time
import random
from collections import defaultdict

def construct():
    n = 15
    start = time.time()
    deadline = start + 55.0

    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)

    def build_adj_deg(edge_set):
        adj = [0] * n
        deg = [0] * n
        for (u, v) in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg

    def check_violation_for_add(adj, deg, u, v):
        """
        Check if adding edge (u, v) to the graph defined by adj/deg creates a violation.
        adj/deg represent the graph BEFORE adding (u, v).
        """
        deg_u_new = deg[u] + 1
        deg_v_new = deg[v] + 1

        neighbors_u = adj[u]
        neighbors_v = adj[v]

        # Case 1: (u, v) is the middle edge. Path s-u-v-t.
        for s in range(n):
            if not (neighbors_u & (1 << s)):
                continue
            deg_s_new = deg[s] + (1 if s == u or s == v else 0)
            for t in range(n):
                if t == s:
                    continue
                if not (neighbors_v & (1 << t)):
                    continue
                deg_t_new = deg[t] + (1 if t == u or t == v else 0)
                if deg_s_new == deg_t_new:
                    return True

        # Case 2: (u, v) is the first edge. Path u-v-b-t.
        for b in range(n):
            if not (neighbors_v & (1 << b)):
                continue
            neighbors_b = adj[b]
            for t in range(n):
                if not (neighbors_b & (1 << t)):
                    continue
                if t == u or t == v:
                    continue
                deg_t_new = deg[t] + (1 if t == u or t == v else 0)
                if deg_u_new == deg_t_new:
                    return True

        # Case 3: (u, v) is the last edge. Path s-a-u-v.
        for a in range(n):
            if not (neighbors_u & (1 << a)):
                continue
            neighbors_a = adj[a]
            for s in range(n):
                if not (neighbors_a & (1 << s)):
                    continue
                if s == u or s == v:
                    continue
                deg_s_new = deg[s] + (1 if s == u or s == v else 0)
                if deg_v_new == deg_s_new:
                    return True

        return False

    def count_violations_for_edge(adj, deg, u, v):
        """Count how many violations edge (u,v) participates in if added."""
        deg_u_new = deg[u] + 1
        deg_v_new = deg[v] + 1
        neighbors_u = adj[u]
        neighbors_v = adj[v]
        violations = 0

        # Case 1: (u, v) is the middle edge
        for s in range(n):
            if not (neighbors_u & (1 << s)):
                continue
            deg_s_new = deg[s] + (1 if s == u or s == v else 0)
            for t in range(n):
                if t == s:
                    continue
                if not (neighbors_v & (1 << t)):
                    continue
                deg_t_new = deg[t] + (1 if t == u or t == v else 0)
                if deg_s_new == deg_t_new:
                    violations += 1

        # Case 2: (u, v) is the first edge
        for b in range(n):
            if not (neighbors_v & (1 << b)):
                continue
            neighbors_b = adj[b]
            for t in range(n):
                if not (neighbors_b & (1 << t)):
                    continue
                if t == u or t == v:
                    continue
                deg_t_new = deg[t] + (1 if t == u or t == v else 0)
                if deg_u_new == deg_t_new:
                    violations += 1

        # Case 3: (u, v) is the last edge
        for a in range(n):
            if not (neighbors_u & (1 << a)):
                continue
            neighbors_a = adj[a]
            for s in range(n):
                if not (neighbors_a & (1 << s)):
                    continue
                if s == u or s == v:
                    continue
                deg_s_new = deg[s] + (1 if s == u or s == v else 0)
                if deg_v_new == deg_s_new:
                    violations += 1

        return violations

    def find_violating_edges(edge_set):
        """Find all edges that are part of at least one violation."""
        adj, deg = build_adj_deg(edge_set)
        violating = set()
        
        for (u, v) in edge_set:
            if check_violation_for_add(adj, deg, u, v):
                violating.add((u, v))
        
        return violating

    def count_edge_violations(edge_set):
        """For each edge, count how many violations it participates in."""
        adj, deg = build_adj_deg(edge_set)
        edge_viol_count = defaultdict(int)
        
        for (u, v) in edge_set:
            if check_violation_for_add(adj, deg, u, v):
                edge_viol_count[(u, v)] += 1
        
        return edge_viol_count

    def greedy_add_edges(edge_set, max_attempts=100):
        """Greedily add edges that don't cause violations."""
        current_edges = set(edge_set)
        current_adj, current_deg = build_adj_deg(current_edges)
        
        for _ in range(max_attempts):
            if time.time() > deadline - 3:
                break
            
            # Find all edges that can be added
            addable = []
            for e in all_edges:
                if e in current_edges:
                    continue
                u, v = e
                if not check_violation_for_add(current_adj, current_deg, u, v):
                    addable.append(e)
            
            if not addable:
                break
            
            # Pick a random addable edge
            random.shuffle(addable)
            e = addable[0]
            u, v = e
            current_edges.add(e)
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            current_deg[u] += 1
            current_deg[v] += 1
        
        return current_edges

    def targeted_removal(edge_set):
        """Remove edges that participate in the most violations."""
        current_edges = set(edge_set)
        
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            
            # Find violating edges
            violating = find_violating_edges(current_edges)
            if not violating:
                break
            
            # Count violations per edge
            adj, deg = build_adj_deg(current_edges)
            edge_viol_count = defaultdict(int)
            for e in violating:
                u, v = e
                edge_viol_count[e] = count_violations_for_edge(adj, deg, u, v)
            
            # Remove the edge with the most violations
            worst_edge = max(edge_viol_count.keys(), key=lambda e: edge_viol_count[e])
            current_edges.remove(worst_edge)
        
        return current_edges

    # Strategy 1: Start from K_{7,8} and greedily add
    base_edges = [(i, j) for i in range(7) for j in range(7, 15)]
    best_edges = set(base_edges)
    best_count = len(best_edges)

    # Multiple random restarts with greedy add
    for attempt in range(300):
        if time.time() > deadline - 10:
            break
        
        current = set(base_edges)
        # Add some random internal edges to start with
        internal_edges = [(i, j) for i in range(7) for j in range(i+1, 7)] + \
                         [(i, j) for i in range(7, 15) for j in range(i+1, 15)]
        random.shuffle(internal_edges)
        for e in internal_edges[:5]:
            current.add(e)
        
        current = greedy_add_edges(current, max_attempts=50)
        
        if len(current) > best_count:
            best_count = len(current)
            best_edges = current

    # Strategy 2: Start from denser candidates and use targeted removal
    candidates = [
        # K_{7,8} plus some internal edges
        set(base_edges) | {(0, 1), (2, 3), (4, 5), (7, 8), (9, 10), (11, 12)},
        # Turan T(15,3)
        set(),
        # Complete minus matching
        set(all_edges) - {(i, i+1) for i in range(0, 14, 2)},
    ]
    
    # Generate Turan T(15,3)
    parts = [[0,3,6,9,12], [1,4,7,10,13], [2,5,8,11,14]]
    turan_edges = set()
    for a in range(3):
        for b in range(a+1, 3):
            for i in parts[a]:
                for j in parts[b]:
                    turan_edges.add((min(i,j), max(i,j)))
    candidates.append(turan_edges)
    
    for cand in candidates:
        if time.time() > deadline - 15:
            break
        
        # Targeted removal to fix violations
        fixed = targeted_removal(cand)
        
        # Then greedily add more
        fixed = greedy_add_edges(fixed, max_attempts=100)
        
        if len(fixed) > best_count:
            best_count = len(fixed)
            best_edges = fixed

    # Strategy 3: Local search - swap edges
    for ls_attempt in range(200):
        if time.time() > deadline - 5:
            break
        
        current_edges = set(best_edges)
        current_adj, current_deg = build_adj_deg(current_edges)
        
        # Try removing a random edge and adding a different one
        if not current_edges:
            break
        
        edges_list = list(current_edges)
        random.shuffle(edges_list)
        
        improved = False
        for edge_to_remove in edges_list[:20]:  # Try up to 20 removals
            if time.time() > deadline - 3:
                break
            
            test_edges = set(current_edges)
            test_edges.remove(edge_to_remove)
            test_adj, test_deg = build_adj_deg(test_edges)
            
            # Try to add a different edge
            for e in all_edges:
                if e in test_edges:
                    continue
                u, v = e
                if not check_violation_for_add(test_adj, test_deg, u, v):
                    test_edges.add(e)
                    test_adj[u] |= (1 << v)
                    test_adj[v] |= (1 << u)
                    test_deg[u] += 1
                    test_deg[v] += 1
                    break
            
            if len(test_edges) > len(current_edges):
                current_edges = test_edges
                current_adj, current_deg = test_adj, test_deg
                improved = True
                break
        
        if improved:
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = current_edges

    out = [[u, v] for u, v in best_edges]
    return out
