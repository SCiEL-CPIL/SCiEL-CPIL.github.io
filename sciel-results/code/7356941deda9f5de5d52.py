import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    n = 20
    
    def is_k33_free(adj):
        """Check if graph is K3,3-free using the common-neighbor condition."""
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    # Count common neighbors of {a,b,c}
                    count = 0
                    for w in range(n):
                        if adj[a][w] and adj[b][w] and adj[c][w]:
                            count += 1
                            if count >= 3:
                                return False
        return True
    
    def count_edges(adj):
        return sum(sum(row) for row in adj) // 2
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps the graph K3,3-free.
        
        Adding edge (u,v) creates a K3,3 iff there exist vertices a,b,c
        such that {a,b,c} and {u,v} form a K3,3. This happens when:
        - a,b,c are all adjacent to both u and v (after adding u-v)
        - OR u,v are both common neighbors of some triple {a,b,c}
        
        More precisely: adding (u,v) is forbidden if there exists a triple
        {x,y,z} not containing u or v such that both u and v are common
        neighbors of {x,y,z}. Also forbidden if there exist a,b such that
        {u,a,b} has v as a 3rd+ common neighbor, etc.
        
        The correct check: after adding (u,v), for every triple {a,b,c},
        the number of common neighbors must be <= 2. Adding (u,v) only
        changes the common-neighbor count for triples that include u or v.
        For triple {u, a, b} (a,b != u,v): v becomes a common neighbor of
        {u,a,b} iff v is adjacent to both a and b. The count increases by 1.
        For triple {v, a, b}: similarly u becomes common neighbor iff u adj a,b.
        For triple {u, v, a}: u and v are already in the triple, so adding
        the edge between them doesn't change common-neighbor count (they're
        not neighbors of themselves in the standard sense, but we need to be
        careful). Actually for triple {u,v,a}, the common neighbors are
        vertices w adjacent to u, v, and a. Adding edge (u,v) doesn't add
        new common neighbors for this triple since u and v are in the triple.
        """
        if adj[u][v]:
            return False
        
        # Check triples {u, a, b} where a, b != u, v
        # v becomes a common neighbor of {u,a,b} iff adj[a][v] and adj[b][v]
        # In that case, count of common neighbors of {u,a,b} increases by 1
        # We need the new count <= 2, so current count must be <= 1
        for a in range(n):
            if a == u or a == v:
                continue
            for b in range(a+1, n):
                if b == u or b == v:
                    continue
                if adj[a][v] and adj[b][v]:
                    # v is a common neighbor of a and b, so it will be a
                    # common neighbor of {u,a,b} after adding (u,v)
                    # Count current common neighbors of {u,a,b}
                    count = 0
                    for w in range(n):
                        if w != u and w != a and w != b:
                            if adj[u][w] and adj[a][w] and adj[b][w]:
                                count += 1
                                if count >= 2:
                                    return False
        
        # Check triples {v, a, b} where a, b != u, v
        for a in range(n):
            if a == u or a == v:
                continue
            for b in range(a+1, n):
                if b == u or b == v:
                    continue
                if adj[a][u] and adj[b][u]:
                    # u is a common neighbor of a and b
                    count = 0
                    for w in range(n):
                        if w != v and w != a and w != b:
                            if adj[v][w] and adj[a][w] and adj[b][w]:
                                count += 1
                                if count >= 2:
                                    return False
        
        return True
    
    def greedy_build(seed):
        """Build a graph using greedy edge addition with random ordering."""
        random.seed(seed)
        adj = [[0]*n for _ in range(n)]
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            if can_add_edge(adj, u, v):
                adj[u][v] = 1
                adj[v][u] = 1
        
        return adj
    
    def complete_graph_prune(seed):
        """Start from complete graph and remove edges to eliminate K3,3 violations."""
        random.seed(seed)
        adj = [[1]*n for _ in range(n)]
        for i in range(n):
            adj[i][i] = 0
        
        # Repeatedly find K3,3 violations and remove edges
        max_iter = 500
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Find a K3,3 violation: three vertices with 3+ common neighbors
            found = False
            for a in range(n):
                if found:
                    break
                for b in range(a+1, n):
                    if found:
                        break
                    for c in range(b+1, n):
                        if found:
                            break
                        # Find common neighbors of a, b, c
                        common = []
                        for w in range(n):
                            if w != a and w != b and w != c:
                                if adj[a][w] and adj[b][w] and adj[c][w]:
                                    common.append(w)
                        if len(common) >= 3:
                            # Remove an edge to break this violation
                            # Strategy: remove edge from a common neighbor to one of a,b,c
                            # Try to remove the edge that appears in the most violations
                            best_removal = None
                            best_score = -1
                            for w in common:
                                for target in [a, b, c]:
                                    # Score: how many other K3,3 violations would be broken
                                    score = 0
                                    # This is expensive, so just pick randomly
                                    # But let's try a slightly smarter approach:
                                    # remove the edge that is "central" - appears in many triples
                                    # For now, just pick randomly
                                    pass
                            # Random choice
                            w = random.choice(common)
                            target = random.choice([a, b, c])
                            adj[w][target] = 0
                            adj[target][w] = 0
                            found = True
                            break
            
            if not found:
                break
        
        return adj
    
    def local_search_add(adj):
        """Try to add more edges to the graph."""
        adj = [row[:] for row in adj]
        pairs = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
        random.shuffle(pairs)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for u, v in pairs:
                if not adj[u][v] and can_add_edge(adj, u, v):
                    adj[u][v] = 1
                    adj[v][u] = 1
                    improved = True
        
        return adj
    
    def perturbation_search(adj, num_removals=2, seed=None):
        """Perturbation: remove random edges, then re-add greedily."""
        if seed is not None:
            random.seed(seed)
        
        adj = [row[:] for row in adj]
        existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
        
        if not existing_edges:
            return adj
        
        # Remove random edges
        to_remove = min(num_removals, len(existing_edges))
        removed = random.sample(existing_edges, to_remove)
        for u, v in removed:
            adj[u][v] = 0
            adj[v][u] = 0
        
        # Now try to add back edges greedily
        pairs = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            if can_add_edge(adj, u, v):
                adj[u][v] = 1
                adj[v][u] = 1
        
        return adj
    
    def edge_swap_search(adj, max_iterations=500):
        """
        Edge swap local search: try removing one edge and adding up to two new edges.
        This allows escaping local optima of add-only search.
        """
        adj = [row[:] for row in adj]
        best_adj = [row[:] for row in adj]
        best_count = count_edges(adj)
        
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        for _ in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Try removing a random edge
            if best_count == 0:
                break
            
            existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
            if not existing_edges:
                break
            
            rem_u, rem_v = random.choice(existing_edges)
            
            # Try removing this edge
            trial_adj = [row[:] for row in adj]
            trial_adj[rem_u][rem_v] = 0
            trial_adj[rem_v][rem_u] = 0
            
            # Now try to add edges to the trial graph
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not trial_adj[i][j]]
            random.shuffle(missing)
            
            added = 0
            for u, v in missing:
                if time.time() > deadline - 10:
                    break
                if can_add_edge(trial_adj, u, v):
                    trial_adj[u][v] = 1
                    trial_adj[v][u] = 1
                    added += 1
                    if added >= 2:  # We removed 1, so adding 2 gives net +1
                        break
            
            trial_count = count_edges(trial_adj)
            if trial_count > best_count:
                best_count = trial_count
                best_adj = [row[:] for row in trial_adj]
                adj = [row[:] for row in trial_adj]
            elif trial_count == best_count:
                # Accept equal moves with some probability to explore
                if random.random() < 0.3:
                    adj = [row[:] for row in trial_adj]
        
        return best_adj, best_count
    
    def two_opt_search(adj, max_iterations=300):
        """
        2-opt style: remove 2 edges, add up to 3 for net +1 improvement.
        """
        adj = [row[:] for row in adj]
        best_adj = [row[:] for row in adj]
        best_count = count_edges(adj)
        
        for _ in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
            if len(existing_edges) < 2:
                break
            
            # Remove 2 random edges
            rem1, rem2 = random.sample(existing_edges, 2)
            
            trial_adj = [row[:] for row in adj]
            trial_adj[rem1[0]][rem1[1]] = 0
            trial_adj[rem1[1]][rem1[0]] = 0
            trial_adj[rem2[0]][rem2[1]] = 0
            trial_adj[rem2[1]][rem2[0]] = 0
            
            # Now try to add edges
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not trial_adj[i][j]]
            random.shuffle(missing)
            
            added = 0
            for u, v in missing:
                if time.time() > deadline - 10:
                    break
                if can_add_edge(trial_adj, u, v):
                    trial_adj[u][v] = 1
                    trial_adj[v][u] = 1
                    added += 1
                    if added >= 3:  # Removed 2, add 3 for net +1
                        break
            
            trial_count = count_edges(trial_adj)
            if trial_count > best_count:
                best_count = trial_count
                best_adj = [row[:] for row in trial_adj]
                adj = [row[:] for row in trial_adj]
            elif trial_count == best_count:
                if random.random() < 0.2:
                    adj = [row[:] for row in trial_adj]
        
        return best_adj, best_count
    
    def spectral_init(seed):
        """
        Spectral initialization: create a dense random-like starting point.
        Use a random symmetric matrix, threshold at a level that gives roughly 60-70% edge density,
        then prune to make it K3,3-free.
        """
        random.seed(seed)
        
        # Start with a random dense graph (roughly 65% edge probability)
        adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.65:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        # Prune to make it K3,3-free: repeatedly find violations and remove edges
        for _ in range(200):
            if time.time() > deadline:
                break
            
            # Find a K3,3 violation
            found_violation = False
            for a in range(n):
                if found_violation:
                    break
                for b in range(a+1, n):
                    if found_violation:
                        break
                    for c in range(b+1, n):
                        if found_violation:
                            break
                        # Count common neighbors
                        common = []
                        for w in range(n):
                            if adj[a][w] and adj[b][w] and adj[c][w]:
                                common.append(w)
                        if len(common) >= 3:
                            # Remove an edge to break this violation
                            # Remove a random edge incident to one of the common neighbors
                            w = random.choice(common)
                            # Remove edge from w to one of a, b, c
                            target = random.choice([a, b, c])
                            adj[w][target] = 0
                            adj[target][w] = 0
                            found_violation = True
                            break
            
            if not found_violation:
                break
        
        return adj
    
    # Main search
    best_adj = None
    best_edges = 0
    
    # Phase 1: Complete graph pruning (dense starting points)
    num_complete = 20
    for i in range(num_complete):
        if time.time() > deadline - 40:
            break
        
        adj = complete_graph_prune(i)
        adj = local_search_add(adj)
        edges = count_edges(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj
    
    # Phase 2: Spectral initialization
    num_spectral = 20
    for i in range(num_spectral):
        if time.time() > deadline - 40:
            break
        
        adj = spectral_init(100 + i)
        adj = local_search_add(adj)
        edges = count_edges(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj
    
    # Phase 3: Greedy construction with local search
    num_greedy = 100
    for i in range(num_greedy):
        if time.time() > deadline - 40:
            break
        
        adj = greedy_build(1000 + i)
        adj = local_search_add(adj)
        edges = count_edges(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj
    
    # Phase 4: Edge swap search on the best solution
    if best_adj is not None and time.time() < deadline - 30:
        adj, edges = edge_swap_search(best_adj, max_iterations=200)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj
    
    # Phase 5: 2-opt search
    if best_adj is not None and time.time() < deadline - 20:
        adj, edges = two_opt_search(best_adj, max_iterations=150)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj
    
    # Phase 6: Perturbation-based search
    if best_adj is not None and time.time() < deadline - 15:
        for i in range(30):
            if time.time() > deadline - 10:
                break
            adj = perturbation_search(best_adj, num_removals=2, seed=5000 + i)
            adj = local_search_add(adj)
            edges = count_edges(adj)
            if edges > best_edges:
                best_edges = edges
                best_adj = adj
    
    # Phase 7: Another round of edge swaps and 2-opt
    if best_adj is not None and time.time() < deadline - 10:
        adj, edges = edge_swap_search(best_adj, max_iterations=150)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj
        
        if time.time() < deadline - 8:
            adj, edges = two_opt_search(best_adj, max_iterations=100)
            if edges > best_edges:
                best_edges = edges
                best_adj = adj
    
    # Phase 8: Final local search polish
    if best_adj is not None and time.time() < deadline - 5:
        adj = local_search_add(best_adj)
        edges = count_edges(adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj
    
    # Convert to edge list
    if best_adj is None:
        return []
    
    result_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                result_edges.append([i, j])
    
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
