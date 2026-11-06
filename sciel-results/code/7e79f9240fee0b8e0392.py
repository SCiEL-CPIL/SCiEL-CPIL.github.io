"""Constraint propagation with precomputed 6-vertex subset regularity and conflict-maximizing flips."""
import itertools
import random
import time

def construct():
    """Construct a graph with no regular induced subgraph on 6+ vertices."""
    
    def is_regular(vertices, adj):
        """Check if induced subgraph on vertices is regular."""
        k = len(vertices)
        if k < 2:
            return True
        degs = []
        for v in vertices:
            d = 0
            for u in vertices:
                if u != v and adj[v][u]:
                    d += 1
            degs.append(d)
        return all(d == degs[0] for d in degs[1:])
    
    def full_verify(adj, n):
        """Verify no subset of size >= 6 induces a regular graph."""
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                if is_regular(subset, adj):
                    return False
        return True
    
    def solve(n, time_budget):
        """Solve for n vertices within time_budget seconds."""
        start = time.time()
        subsets6 = list(itertools.combinations(range(n), 6))
        num_subsets = len(subsets6)
        
        # Precompute for each edge, which 6-subsets contain it
        # edge -> list of subset indices
        edge_to_subsets = {}
        for idx, subset in enumerate(subsets6):
            for i in range(6):
                for j in range(i + 1, 6):
                    u, v = subset[i], subset[j]
                    key = (min(u, v), max(u, v))
                    if key not in edge_to_subsets:
                        edge_to_subsets[key] = []
                    edge_to_subsets[key].append(idx)
        
        # All possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Initialize adj
        adj = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        # Compute regularity status for all 6-subsets
        regular = [False] * num_subsets
        for idx, subset in enumerate(subsets6):
            regular[idx] = is_regular(subset, adj)
        
        # Count violations
        viol_count = sum(1 for r in regular if r)
        
        # Track which edges are in which violating subsets for quick updates
        # When we flip an edge, we need to update regularity of all subsets containing it
        
        def flip_edge(u, v):
            """Flip edge (u,v) and update regularity status of affected subsets."""
            nonlocal viol_count
            adj[u][v] ^= 1
            adj[v][u] ^= 1
            key = (min(u, v), max(u, v))
            if key in edge_to_subsets:
                for idx in edge_to_subsets[key]:
                    old_reg = regular[idx]
                    new_reg = is_regular(subsets6[idx], adj)
                    regular[idx] = new_reg
                    if old_reg and not new_reg:
                        viol_count -= 1
                    elif not old_reg and new_reg:
                        viol_count += 1
            return viol_count
        
        def get_violating_edges():
            """Get edges that appear in violating subsets, with counts."""
            edge_counts = {}
            for idx, r in enumerate(regular):
                if r:
                    subset = subsets6[idx]
                    for i in range(6):
                        for j in range(i + 1, 6):
                            u, v = subset[i], subset[j]
                            key = (min(u, v), max(u, v))
                            edge_counts[key] = edge_counts.get(key, 0) + 1
            return edge_counts
        
        # Main loop: iteratively flip edges to reduce violations
        best_viol = viol_count
        best_adj = [row[:] for row in adj]
        best_regular = regular[:]
        
        for iteration in range(50000):
            if time.time() - start > time_budget:
                break
            
            if viol_count == 0:
                # Check if full verification passes
                if full_verify(adj, n):
                    return adj, True
                # If 6-subsets are clean but larger ones aren't, try to fix
                # Find a violating large subset
                found = False
                for k in range(7, n + 1):
                    if time.time() - start > time_budget:
                        break
                    for subset in itertools.combinations(range(n), k):
                        if is_regular(subset, adj):
                            # Flip a random edge within this subset
                            edges_in = [(subset[i], subset[j]) for i in range(k) for j in range(i+1, k)]
                            u, v = random.choice(edges_in)
                            flip_edge(u, v)
                            found = True
                            break
                    if found:
                        break
                if not found and time.time() - start > time_budget:
                    break
                continue
            
            # Get violating edges sorted by frequency
            edge_counts = get_violating_edges()
            if not edge_counts:
                break
            
            # Sort by frequency descending
            sorted_edges = sorted(edge_counts.items(), key=lambda x: -x[1])
            
            improved = False
            # Try top candidates
            for (u, v), _ in sorted_edges[:20]:
                old_viol = viol_count
                flip_edge(u, v)
                if viol_count < old_viol:
                    improved = True
                    if viol_count < best_viol:
                        best_viol = viol_count
                        best_adj = [row[:] for row in adj]
                        best_regular = regular[:]
                    break
                else:
                    # Revert
                    flip_edge(u, v)
            
            if not improved:
                # Try random flips to escape local optimum
                for _ in range(3):
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u != v:
                        flip_edge(u, v)
        
        # Return best found
        if best_viol == 0 and full_verify(best_adj, n):
            return best_adj, True
        return None, False
    
    # Try n=21 first, then 20
    for n in [21, 20]:
        adj, success = solve(n, time_budget=55)
        if success:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i][j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
    
    # Fallback: return a valid small graph
    # 5-vertex empty graph is valid
    return {'n': 5, 'edges': []}
