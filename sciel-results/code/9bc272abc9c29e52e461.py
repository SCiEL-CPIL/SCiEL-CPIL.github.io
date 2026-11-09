import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin from 150s limit

    n = 40
    target_edges = 401

    def build_initial_graph(seed_val):
        """
        Builds a graph with a bipartite core plus sparse intra-part edges.
        Returns (edges_set, adj_masks_list, max_book, max_edges_list)
        """
        random.seed(seed_val)
        edges = set()

        # Partition into two sets
        # We want roughly balanced partitions to minimize max degree in bipartite part
        # n=40, so 20-20 split is natural.
        # However, the current best uses 9-31. Let's try a few variations.
        # Actually, for minimizing book size, we want to avoid large cliques or dense subgraphs.
        # A complete bipartite graph K_{a,b} has book size 0 for all edges? No.
        # In K_{a,b}, an edge (u,v) with u in A, v in B.
        # Common neighbors of u and v:
        # u is connected to all in B. v is connected to all in A.
        # Intersection of N(u) and N(v) is empty?
        # N(u) = B \ {v} (if no intra-A edges)
        # N(v) = A \ {u} (if no intra-B edges)
        # Intersection is empty. So book size is 0.
        # So if we could build a bipartite graph with 401 edges, book size would be 0.
        # But we have 40 vertices. Max edges in bipartite graph is 20*20 = 400.
        # We need 401 edges. So we need at least 1 edge within a partition.
        # Let's try to make the graph as "bipartite-like" as possible.
        
        # Strategy:
        # 1. Create a bipartite graph K_{20,20} (400 edges).
        # 2. Add 1 edge within one partition.
        # This adds 1 edge. Total 401.
        # What is the book size?
        # The new edge (u1, u2) within A.
        # Common neighbors of u1 and u2:
        # u1 is connected to all 20 in B.
        # u2 is connected to all 20 in B.
        # So they share all 20 vertices in B as common neighbors.
        # Book size = 20.
        # This is very bad.
        
        # So we need to spread out the intra-part edges or reduce the bipartite density.
        
        # Alternative:
        # Start with a random graph with 401 edges?
        # Or start with a structure that has low book size.
        
        # Let's try the 9-31 split again, but optimize the intra-B edges more carefully.
        # A (9 vertices), B (31 vertices).
        # Complete bipartite A-B: 9*31 = 279 edges.
        # Need 401 - 279 = 122 edges within B.
        # To minimize book size of intra-B edges, we want them to have few common neighbors.
        # Common neighbors of an intra-B edge (u,v) are:
        # 1. Vertices in A that are connected to both u and v. Since A is complete to B, ALL 9 vertices in A are common neighbors.
        # 2. Vertices in B that are connected to both u and v.
        # So book size >= 9.
        # To minimize, we want the number of common neighbors within B to be small.
        # Ideally 0. Then book size = 9.
        # Can we add 122 edges in B (31 vertices) such that no two vertices in B have a common neighbor in B?
        # This means the graph induced by B should have no triangles? No, that's not it.
        # It means for any edge (u,v) in B, N_B(u) and N_B(v) are disjoint.
        # This implies the graph on B is a collection of matchings? No.
        # If (u,v) is an edge, and (u,w) is an edge, then v and w must not be adjacent to each other? No.
        # N_B(u) contains v. N_B(v) contains u.
        # If there is a path u-w-v, then w is in N_B(u) and w is in N_B(v). So w is a common neighbor.
        # So we cannot have paths of length 2 between endpoints of an edge.
        # This means the graph on B must have girth > 4? No.
        # If (u,v) is an edge, and there is a vertex w connected to both, book size increases.
        # So we want a graph on B where every edge is "isolated" from other edges in terms of sharing neighbors.
        # This is a very sparse structure.
        
        # Let's just use a random greedy approach for the intra-B edges, trying to minimize common neighbors.
        
        A = list(range(9))
        B = list(range(9, 40))
        
        # 1. Add all edges between A and B
        for a in A:
            for b in B:
                edges.add((a, b))
                
        # 2. Add 122 edges within B
        # We will use a greedy algorithm to pick edges that minimize the increase in max book size.
        
        # Precompute all potential intra-B edges
        potential_edges = [(u, v) for u in B for v in B if u < v]
        random.shuffle(potential_edges)
        
        # We need 122 edges.
        # Current book size for any intra-B edge is 9 (from A).
        # We want to keep it at 9.
        # So we should pick edges such that they don't create common neighbors within B.
        
        # Let's maintain the graph on B as we add edges.
        # adj_B = {v: set() for v in B}
        
        # Greedy:
        # For each candidate edge (u,v), check if adding it increases the book size of any existing edge or the new edge.
        # Actually, simpler: just check if u and v have any common neighbors in B already.
        # If yes, book size of (u,v) will be > 9.
        # If no, book size of (u,v) is 9.
        # But adding (u,v) might increase the book size of other edges?
        # No, because book size of (x,y) is |N(x) ∩ N(y)|.
        # Adding (u,v) adds u to N(v) and v to N(u).
        # This can only increase the intersection for edges involving u or v.
        
        # So we want to pick (u,v) such that:
        # 1. N_B(u) ∩ N_B(v) is empty (so book size of (u,v) is 9).
        # 2. For any existing edge (x,y), adding (u,v) doesn't increase |N(x) ∩ N(y)|.
        #    This happens if u is not in N(y) or v is not in N(x), etc.
        
        # Let's just try to pick edges with no common neighbors in B.
        
        adj_B = {v: set() for v in B}
        added = 0
        needed = 122
        
        for u, v in potential_edges:
            if added == needed:
                break
            # Check if u and v have common neighbors in B
            if adj_B[u] & adj_B[v]:
                continue
            # Also check if adding this edge increases book size of existing edges
            # This is expensive to check fully. Let's just try the simple condition first.
            # If we get stuck, we can relax.
            
            edges.add((u, v))
            adj_B[u].add(v)
            adj_B[v].add(u)
            added += 1
            
        if added < needed:
            # Relax condition: allow some common neighbors, but prefer those with fewer
            # Sort remaining potential edges by number of common neighbors
            remaining = [(u, v) for u, v in potential_edges if (u, v) not in edges]
            def common_count(e):
                u, v = e
                return len(adj_B[u] & adj_B[v])
            remaining.sort(key=common_count)
            
            for u, v in remaining:
                if added == needed:
                    break
                edges.add((u, v))
                adj_B[u].add(v)
                adj_B[v].add(u)
                added += 1

        # Build adjacency masks
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            
        # Calculate max book
        max_b = -1
        max_edges = []
        for u, v in edges:
            b = (adj_masks[u] & adj_masks[v]).bit_count()
            if b > max_b:
                max_b = b
                max_edges = [(u, v)]
            elif b == max_b:
                max_edges.append((u, v))
                
        return edges, adj_masks, max_b, max_edges

    # --- Phase 1: Multi-start Initialization ---
    best_state = None
    best_max_book = float('inf')
    
    # Try a few seeds
    seeds = [42, 123, 999, 555, 777, 111, 222, 333, 444, 555]
    for s in seeds:
        edges, adj_masks, max_b, max_edges = build_initial_graph(s)
        if max_b < best_max_book:
            best_max_book = max_b
            best_state = (edges, adj_masks, max_edges)
            
    edges, adj_masks, current_max_edges = best_state
    current_max_book = best_max_book

    # --- Phase 2: Local Search Refinement ---
    
    def update_adj(u, v, add):
        if add:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        else:
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
            
    def get_book_size(u, v):
        if u == v:
            return 0
        return (adj_masks[u] & adj_masks[v]).bit_count()
        
    def get_max_book_info():
        max_b = -1
        max_edges = []
        for u, v in edges:
            b = get_book_size(u, v)
            if b > max_b:
                max_b = b
                max_edges = [(u, v)]
            elif b == max_b:
                max_edges.append((u, v))
        return max_b, max_edges

    no_improve_count = 0
    max_no_improve = 5000 # Allow more exploration
    
    while time.time() < deadline:
        if current_max_book <= 7:
            # Theoretical lower bound is 40/6 = 6.66. So 7 is the smallest integer possible?
            # Wait, Edwards-Khadzhiivanov-Nikiforov: for n=40, m=401, book size >= 7?
            # The prompt says "score >= 40/6". 40/6 = 6.66.
            # Since book size is an integer, it must be at least 7.
            # So 7 is the theoretical minimum.
            # If we reach 7, we are at the theoretical optimum.
            # We should stop searching for lower, but we can try to stabilize or just return.
            # Actually, if we are at 7, we can't get lower.
            # So we can break immediately if we hit 7?
            # Yes, 7 is the minimum possible integer value >= 6.66.
            if current_max_book == 7:
                break
            elif current_max_book <= 6:
                # This shouldn't happen if the theorem holds, but just in case.
                break

        if not current_max_edges:
            break
            
        # Pick a random edge from the max-book set to remove
        u, v = random.choice(current_max_edges)
        
        # Find the best non-edge to add
        # We want to minimize the book size of the new edge.
        # Heuristic: Find non-edge (x,y) with min common neighbors.
        min_book_new = float('inf')
        best_x, best_y = -1, -1
        
        # Iterate over all possible non-edges
        for x in range(n):
            for y in range(x + 1, n):
                if (1 << y) & adj_masks[x]:
                    continue
                
                b = (adj_masks[x] & adj_masks[y]).bit_count()
                if b < min_book_new:
                    min_book_new = b
                    best_x, best_y = x, y
                    if b == 0:
                        # Can't get better than 0
                        break
            if min_book_new == 0:
                break
        
        if best_x == -1:
            break
            
        # Perform the swap
        old_max = current_max_book
        old_max_count = len(current_max_edges)
        
        edges.remove((u, v))
        update_adj(u, v, False)
        
        edges.add((best_x, best_y))
        update_adj(best_x, best_y, True)
        
        new_max, new_max_edges = get_max_book_info()
        
        accepted = False
        if new_max < old_max:
            accepted = True
            no_improve_count = 0
        elif new_max == old_max:
            if len(new_max_edges) < old_max_count:
                accepted = True
                no_improve_count = 0
            else:
                # Small probability to accept to escape local minima
                # Use a decaying probability? Or fixed?
                # Fixed 0.05 seems reasonable.
                if random.random() < 0.05:
                    accepted = True
                else:
                    accepted = False
        else:
            accepted = False

        if accepted:
            current_max_book = new_max
            current_max_edges = new_max_edges
        else:
            # Rollback
            edges.add((u, v))
            update_adj(u, v, True)
            
            edges.remove((best_x, best_y))
            update_adj(best_x, best_y, False)
            
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Random restart?
                # We can try to perturb the graph by doing a few random swaps that don't increase max book too much?
                # Or just break if we've been stuck for a long time.
                # Given the tight time limit, breaking is safer than risky perturbations.
                # However, if we are stuck at 8, we might be able to reach 7 with a better perturbation.
                # Let's try a "kick": remove a random edge (not necessarily max book) and add a random non-edge
                # if it doesn't increase the max book.
                
                # Actually, let's just break. The multi-start already provided diversity.
                break

    result = [[u, v] for u, v in edges]
    return result
