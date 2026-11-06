import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s deadline for safety
    
    n = 18
    all_vertices_mask = (1 << n) - 1
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def count_bits(x):
        return bin(x).count('1')
    
    def is_k4_free(adj_mask):
        """Check if graph is K4-free."""
        for combo in itertools.combinations(range(n), 4):
            u, v, w, x = combo
            if (adj_mask[u] & (1 << v)) and (adj_mask[u] & (1 << w)) and (adj_mask[u] & (1 << x)) and \
               (adj_mask[v] & (1 << w)) and (adj_mask[v] & (1 << x)) and (adj_mask[w] & (1 << x)):
                return False
        return True

    def can_add_edge(adj_mask, u, v):
        """Check if adding edge (u,v) keeps graph K4-free."""
        if adj_mask[u] & (1 << v):
            return False
        # Adding (u,v) creates K4 if there exists an edge (w,x) where w,x are common neighbors of u and v
        common = adj_mask[u] & adj_mask[v] & ~((1 << u) | (1 << v))
        # Check if any pair in common has an edge
        temp = common
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            # For each w in common, check neighbors of w in common
            if adj_mask[w] & common:
                return False
            temp -= lsb
        return True

    def is_triangle_free_mask(adj_mask, mask):
        """Check if induced subgraph on mask is triangle-free."""
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            neigh = adj_mask[i] & mask
            # Check if neigh is independent
            temp = neigh
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                if adj_mask[v] & neigh:
                    return False
                temp -= lsb
        return True

    def exact_largest_triangle_free(adj_mask):
        """Find largest triangle-free induced subgraph using branch and bound."""
        best = 0
        
        def branch_and_bound(remaining, current_mask, current_size):
            nonlocal best
            # Upper bound
            if current_size + count_bits(remaining) <= best:
                return
            if remaining == 0:
                if current_size > best:
                    best = current_size
                return
            
            # Heuristic: pick vertex with max degree in remaining
            max_deg = -1
            v = None
            for i in range(n):
                if remaining & (1 << i):
                    deg = count_bits(adj_mask[i] & remaining)
                    if deg > max_deg:
                        max_deg = deg
                        v = i
                        
            if v is None:
                return
            
            bit_v = 1 << v
            
            # Option 1: Exclude v
            branch_and_bound(remaining & ~bit_v, current_mask, current_size)
            
            # Option 2: Include v
            # Check if adding v to current_mask creates a triangle
            neigh_in_current = adj_mask[v] & current_mask
            # Check if neigh_in_current is independent
            creates_triangle = False
            temp = neigh_in_current
            while temp:
                lsb = temp & -temp
                a = lsb.bit_length() - 1
                if adj_mask[a] & neigh_in_current:
                    creates_triangle = True
                    break
                temp -= lsb
            
            if not creates_triangle:
                branch_and_bound(remaining & ~bit_v, current_mask | bit_v, current_size + 1)
        
        branch_and_bound(all_vertices_mask, 0, 0)
        return best

    def heuristic_finds_triangle_free_11(adj_mask, num_restarts=50):
        """
        Try to find a triangle-free induced subgraph of size 11.
        Returns True if found, False otherwise.
        """
        for _ in range(num_restarts):
            current_mask = 0
            current_size = 0
            remaining = all_vertices_mask
            
            # Random order of vertices
            vertices = list(range(n))
            random.shuffle(vertices)
            
            for v in vertices:
                if not (remaining & (1 << v)):
                    continue
                # Check if adding v keeps it triangle-free
                neigh_in_current = adj_mask[v] & current_mask
                # Check if neigh_in_current is independent
                creates_triangle = False
                temp = neigh_in_current
                while temp:
                    lsb = temp & -temp
                    a = lsb.bit_length() - 1
                    if adj_mask[a] & neigh_in_current:
                        creates_triangle = True
                        break
                    temp -= lsb
                
                if not creates_triangle:
                    current_mask |= (1 << v)
                    current_size += 1
                    if current_size >= 11:
                        return True
            # If we finished the loop without reaching 11, try again
        return False

    def local_search(adj_mask):
        """
        Perform local search to minimize largest triangle-free subgraph.
        Returns the best adj_mask found.
        """
        best_adj = adj_mask[:]
        best_score = exact_largest_triangle_free(best_adj)
        
        if time.time() > deadline:
            return best_adj, best_score
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_adj = best_adj[:]
            current_score = best_score
            
            # Try all possible edge additions/removals
            # To be faster, we can prioritize edges that might break large triangle-free sets
            # But for now, try random subset or all?
            # All pairs is 153. 153 * (K4 check + Heuristic) is fast.
            
            moves = []
            for u, v in all_pairs:
                moves.append((u, v, 0)) # 0: try flip (add if absent, remove if present)
            
            random.shuffle(moves)
            
            for u, v, _ in moves:
                if time.time() > deadline:
                    break
                    
                new_adj = current_adj[:]
                
                # Check if edge exists
                exists = bool(new_adj[u] & (1 << v))
                
                if exists:
                    # Remove edge
                    new_adj[u] &= ~(1 << v)
                    new_adj[v] &= ~(1 << u)
                else:
                    # Add edge
                    if not can_add_edge(new_adj, u, v):
                        continue
                    new_adj[u] |= (1 << v)
                    new_adj[v] |= (1 << u)
                
                # Heuristic check: Does it have a triangle-free set of size 11?
                # If yes, it's likely score >= 11.
                # If no, it might be < 11.
                # We want to minimize score.
                # If current best is 11, and this one is "maybe 10", we should investigate.
                # If current best is 10, and this one is "maybe 10", we should investigate.
                
                # Strategy: 
                # 1. If heuristic finds 11, reject (unless we have no better option, but we have 11).
                # 2. If heuristic fails to find 11, run exact B&B.
                
                if heuristic_finds_triangle_free_11(new_adj, num_restarts=20):
                    # Likely >= 11. Skip.
                    continue
                
                # Candidate for < 11. Verify.
                new_score = exact_largest_triangle_free(new_adj)
                
                if new_score < best_score:
                    best_score = new_score
                    best_adj = new_adj
                    improved = True
                    current_adj = new_adj
                    current_score = new_score
                    # Restart local search from this new best?
                    # Or continue with this as base?
                    # Let's restart the move loop to explore around the new best.
                    break
                elif new_score == best_score:
                    # Same score. Maybe accept with some probability to escape local minima?
                    # For now, skip to save time, or accept if it looks "denser"?
                    # Let's just skip to focus on finding < 11.
                    pass

        return best_adj, best_score

    # Initial Graph: Start with K_{6,6,6}
    # Parts: 0-5, 6-11, 12-17
    adj_mask = [0] * n
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            part_u = u // 6
            part_v = v // 6
            if part_u != part_v:
                adj_mask[u] |= (1 << v)
    
    # Verify it's K4 free (it is) and get score
    # K_{6,6,6} has max triangle-free size 12 (e.g., Part 0 + Part 1)
    
    # Run local search
    best_adj, best_score = local_search(adj_mask)
    
    # If we still haven't found < 11, try a few random restarts
    if best_score > 10 and time.time() < deadline:
        for restart in range(5):
            if time.time() > deadline:
                break
            
            # Random K4-free graph
            rand_adj = [0] * n
            for u, v in all_pairs:
                if random.random() < 0.5:
                    if can_add_edge(rand_adj, u, v):
                        rand_adj[u] |= (1 << v)
                        rand_adj[v] |= (1 << u)
            
            # Quick heuristic check
            if not heuristic_finds_triangle_free_11(rand_adj, num_restarts=10):
                score = exact_largest_triangle_free(rand_adj)
                if score < best_score:
                    best_score = score
                    best_adj = rand_adj
                    # Continue local search from here
                    if time.time() < deadline:
                        best_adj, best_score = local_search(best_adj)
                        
            # Also try perturbing the best_adj
            perturbed = best_adj[:]
            # Remove 1-2 random edges
            edges_present = [(u, v) for u, v in all_pairs if perturbed[u] & (1 << v)]
            if edges_present:
                for _ in range(random.randint(1, 2)):
                    if not edges_present: break
                    u, v = random.choice(edges_present)
                    edges_present.remove((u, v))
                    perturbed[u] &= ~(1 << v)
                    perturbed[v] &= ~(1 << u)
                
                if not heuristic_finds_triangle_free_11(perturbed, num_restarts=10):
                    score = exact_largest_triangle_free(perturbed)
                    if score < best_score:
                        best_score = score
                        best_adj = perturbed
                        if time.time() < deadline:
                            best_adj, best_score = local_search(best_adj)

    # Convert best_adj to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return edges
