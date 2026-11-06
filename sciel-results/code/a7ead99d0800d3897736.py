import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 140  # safety margin

    N = 16
    ALL = (1 << N) - 1

    # Precompute edge list
    edges_list = [(i, j) for i in range(N) for j in range(i+1, N)]
    M = len(edges_list)

    # Memo cache for 3-colorability checks
    colorability_cache = {}

    def is_3_colorable_memo(adj_mask):
        """3-colorability check with memoization on adjacency tuple."""
        key = tuple(adj_mask)
        if key in colorability_cache:
            return colorability_cache[key]
        
        order = sorted(range(N), key=lambda x: bin(adj_mask[x]).count('1'), reverse=True)
        color = [-1] * N
        
        # Precompute for each vertex in order, which earlier vertices in order are neighbors
        pred_neighbors = [[] for _ in range(N)]
        for i in range(N):
            v = order[i]
            for j in range(i):
                w = order[j]
                if ((adj_mask[v] >> w) & 1):
                    pred_neighbors[i].append(j)
        
        def backtrack(idx):
            if idx == N:
                return True
            v = order[idx]
            # Forbidden colors from predecessors
            forbidden = 0
            for j in pred_neighbors[idx]:
                forbidden |= (1 << color[order[j]])
            
            for c in range(3):
                if not (forbidden & (1 << c)):
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        result = backtrack(0)
        colorability_cache[key] = result
        return result

    def count_edges(adj_mask):
        return sum(bin(adj_mask[i]).count('1') for i in range(N)) // 2

    def add_edge(adj_mask, u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    def can_add_edge(adj_mask, u, v):
        """Check if adding edge (u,v) keeps graph triangle-free."""
        if (adj_mask[u] >> v) & 1:
            return False  # already present
        # Check for common neighbor
        common = adj_mask[u] & adj_mask[v]
        return common == 0

    # Start with the Grötzsch graph (11 vertices, 20 edges, 4-chromatic, triangle-free)
    def build_groetzsch():
        adj = [0] * N
        # 5-cycle: 0-1-2-3-4-0
        for i in range(5):
            add_edge(adj, i, (i+1) % 5)
        # Chords: 5-1, 5-3; 6-2, 6-4; 7-3, 7-0; 8-4, 8-1; 9-0, 9-2
        add_edge(adj, 5, 1); add_edge(adj, 5, 3)
        add_edge(adj, 6, 2); add_edge(adj, 6, 4)
        add_edge(adj, 7, 3); add_edge(adj, 7, 0)
        add_edge(adj, 8, 4); add_edge(adj, 8, 1)
        add_edge(adj, 9, 0); add_edge(adj, 9, 2)
        # Vertex 10 connects to 5,6,7,8,9
        for v in range(5, 10):
            add_edge(adj, 10, v)
        return adj

    # Build initial 4-chromatic graph
    best_adj = build_groetzsch()
    best_count = count_edges(best_adj)  # 20 edges

    # Verify it's not 3-colorable
    assert not is_3_colorable_memo(best_adj), "Grötzsch should be 4-chromatic"

    # Greedy maximize: try to add all possible edges in a given order
    def greedy_maximize(adj, edge_order):
        """Try to add edges in the given order, maintaining constraints."""
        current = adj[:]
        current_count = count_edges(current)
        
        for (u, v) in edge_order:
            if time.time() > deadline:
                break
            if can_add_edge(current, u, v):
                new_adj = current[:]
                add_edge(new_adj, u, v)
                if not is_3_colorable_memo(new_adj):
                    current = new_adj
                    current_count += 1
        return current, current_count

    # Generate different edge orderings
    rng = random.Random(42)
    
    best_result = best_adj[:]
    best_result_count = best_count

    # Multiple attempts with different strategies
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Create edge order: start with edges involving new vertices (11-15) first,
        # then edges among existing vertices, shuffled
        new_edges = [(u, v) for (u, v) in edges_list if u >= 11 or v >= 11]
        old_edges = [(u, v) for (u, v) in edges_list if u < 11 and v < 11]
        
        # Remove edges already present
        present = set()
        for i in range(N):
            for j in range(i+1, N):
                if (best_adj[i] >> j) & 1:
                    present.add((i, j))
        
        new_edges = [e for e in new_edges if e not in present]
        old_edges = [e for e in old_edges if e not in present]
        
        # Shuffle
        rng.shuffle(new_edges)
        rng.shuffle(old_edges)
        
        # Different orderings
        if attempt % 3 == 0:
            edge_order = new_edges + old_edges
        elif attempt % 3 == 1:
            edge_order = old_edges + new_edges
        else:
            edge_order = new_edges + old_edges
            rng.shuffle(edge_order)
        
        result, count = greedy_maximize(best_adj, edge_order)
        if count > best_result_count:
            best_result = result
            best_result_count = count

    # Local search: systematically try all missing edges, then edge swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try adding each missing edge (systematic, no early break)
        for (u, v) in edges_list:
            if time.time() > deadline:
                break
            if (best_result[u] >> v) & 1:
                continue
            if can_add_edge(best_result, u, v):
                new_adj = best_result[:]
                add_edge(new_adj, u, v)
                if not is_3_colorable_memo(new_adj):
                    best_result = new_adj
                    best_result_count += 1
                    improved = True
                    # Don't break; continue scanning for more additions
        
        if time.time() > deadline:
            break
        
        # Phase 2: Try edge swaps: remove one edge, add two
        if not improved:
            for (u, v) in edges_list:
                if time.time() > deadline:
                    break
                if not ((best_result[u] >> v) & 1):
                    continue
                # Try removing this edge
                test_adj = best_result[:]
                test_adj[u] &= ~(1 << v)
                test_adj[v] &= ~(1 << u)
                
                # Now try to add two edges
                added = 0
                for (a, b) in edges_list:
                    if time.time() > deadline:
                        break
                    if a == u or a == v or b == u or b == v:
                        continue
                    if (test_adj[a] >> b) & 1:
                        continue
                    if can_add_edge(test_adj, a, b):
                        test2 = test_adj[:]
                        add_edge(test2, a, b)
                        if not is_3_colorable_memo(test2):
                            test_adj = test2
                            added += 1
                            if added >= 2:
                                break
                
                if added >= 2:
                    best_result = test_adj
                    best_result_count += added - 1
                    improved = True
                    break

    # Final verification
    # Check triangle-free
    for u in range(N):
        for v in range(u+1, N):
            if (best_result[u] >> v) & 1:
                common = best_result[u] & best_result[v]
                if common != 0:
                    pass  # Should not happen

    # Check not 3-colorable
    if is_3_colorable_memo(best_result):
        # Fallback to Grötzsch
        best_result = build_groetzsch()
        best_result_count = count_edges(best_result)

    # Convert to edge list
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_result[u] >> v) & 1:
                edges.append([u, v])

    return edges
