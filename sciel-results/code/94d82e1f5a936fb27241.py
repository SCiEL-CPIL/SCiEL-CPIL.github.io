import time
import random

def construct():
    deadline = time.time() + 110
    n = 16
    
    # Helper: Check if graph is 3-colorable using bitmasks with better ordering
    def is_3_colorable(adj):
        colors = [-1] * n
        # Order vertices by degree descending, then by index for determinism
        order = sorted(range(n), key=lambda x: (-bin(adj[x]).count('1'), x))
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            used = 0
            for u in range(n):
                if colors[u] != -1 and (adj[v] >> u) & 1:
                    used |= (1 << colors[u])
            
            # Try colors in order of least constraining first
            for c in range(3):
                if not (used & (1 << c)):
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)

    # Helper: Check if adding edge (u,v) creates a triangle
    def creates_triangle(adj, u, v):
        return (adj[u] & adj[v]) != 0

    # 1. Generate Clebsch Graph
    vectors = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            vectors.append(i)
            
    clebsch_adj = [0] * n
    clebsch_edges = []
    for u_idx in range(n):
        for v_idx in range(u_idx + 1, n):
            xor_val = vectors[u_idx] ^ vectors[v_idx]
            if bin(xor_val).count('1') == 4:
                clebsch_adj[u_idx] |= (1 << v_idx)
                clebsch_adj[v_idx] |= (1 << u_idx)
                clebsch_edges.append([u_idx, v_idx])
                
    best_edges = clebsch_edges
    best_count = len(clebsch_edges)  # 40
    best_adj = clebsch_adj[:]

    # Verify baseline
    if is_3_colorable(best_adj):
        # Should not happen, but if it does, we have a problem
        best_edges = []
        best_count = 0
        best_adj = [0] * n

    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_pairs_set = set(all_pairs)

    def get_non_edges(adj):
        """Return list of non-edges that can be added without creating triangles."""
        result = []
        for u, v in all_pairs:
            if not ((adj[u] >> v) & 1) and not creates_triangle(adj, u, v):
                result.append((u, v))
        return result

    def edge_swap_search(adj, edges, max_swaps=2000):
        """Try to improve by swapping edges: remove one, add another."""
        adj = adj[:]
        edge_set = set(tuple(e) for e in edges)
        current_count = len(edges)
        improved = False
        
        for _ in range(max_swaps):
            if time.time() > deadline:
                break
            
            # Find non-edges that can be added
            non_edges = get_non_edges(adj)
            if not non_edges:
                break
            
            # Try each possible swap: remove an edge, add a non-edge
            found_improvement = False
            for u, v in non_edges:
                if time.time() > deadline:
                    break
                
                # Try removing each edge that shares a vertex with (u,v)
                for eu, ev in list(edge_set):
                    if eu == u or eu == v or ev == u or ev == v:
                        # Remove edge (eu, ev)
                        adj[eu] &= ~(1 << ev)
                        adj[ev] &= ~(1 << eu)
                        edge_set.discard((eu, ev))
                        edge_set.discard((ev, eu))
                        
                        # Try to add (u, v)
                        if not creates_triangle(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            edge_set.add((u, v))
                            
                            new_count = len(edge_set)
                            if new_count > current_count and not is_3_colorable(adj):
                                current_count = new_count
                                improved = True
                                found_improvement = True
                                break
                            else:
                                # Revert
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                edge_set.discard((u, v))
                        
                        # Revert removal
                        adj[eu] |= (1 << ev)
                        adj[ev] |= (1 << eu)
                        edge_set.add((eu, ev))
                
                if found_improvement:
                    break
            
            if not found_improvement:
                # No improvement from single swaps, try double swaps or stop
                break
        
        if improved:
            new_edges = [list(e) for e in edge_set]
            return adj, new_edges, current_count
        return None, None, None

    random.seed(42)
    
    # Try edge swap search from Clebsch
    adj_swapped, edges_swapped, count_swapped = edge_swap_search(best_adj, best_edges, max_swaps=500)
    if adj_swapped is not None and count_swapped > best_count:
        best_adj = adj_swapped
        best_edges = edges_swapped
        best_count = count_swapped

    # Also try random maximal TF graphs
    def random_maximal_tf():
        adj = [0] * n
        edges = []
        possible_edges = all_pairs[:]
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if not creates_triangle(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        return adj, edges

    attempts = 0
    while time.time() < deadline and attempts < 300:
        attempts += 1
        adj, edges = random_maximal_tf()
        
        if len(edges) > best_count:
            if not is_3_colorable(adj):
                best_edges = edges
                best_count = len(edges)
                best_adj = adj
                break

    return best_edges
