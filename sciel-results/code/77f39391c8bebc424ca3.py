import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    n = 18
    all_vertices_mask = (1 << n) - 1
    
    def count_bits(x):
        return bin(x).count('1')
    
    def is_triangle_free(adj_mask, mask):
        """Check if the induced subgraph on mask is triangle-free."""
        # For each vertex i in mask, check if its neighbors within mask form an independent set
        # i.e., no edge between any two neighbors of i within mask
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            # Get neighbors of i that are in mask
            neigh = adj_mask[i] & mask
            # Check if neigh is independent
            # If neigh has an edge, return False
            # Iterate through bits in neigh
            temp = neigh
            while temp:
                lsb = temp & -temp
                v = (lsb.bit_length() - 1)
                # Check if v has any neighbor in neigh
                if adj_mask[v] & neigh:
                    return False
                temp -= lsb
        return True

    def largest_triangle_free(adj_mask):
        """Find largest triangle-free induced subgraph using branch and bound."""
        best = 0
        
        def branch_and_bound(remaining, current_mask, current_size):
            nonlocal best
            if current_size + count_bits(remaining) <= best:
                return
            if remaining == 0:
                if current_size > best:
                    best = current_size
                return
            
            # Pick a vertex from remaining
            # Heuristic: pick vertex with fewest connections to current_mask
            v = None
            min_connections = n + 1
            for i in range(n):
                if remaining & (1 << i):
                    # Count how many vertices in current_mask are adjacent to i
                    conns = count_bits(adj_mask[i] & current_mask)
                    if conns < min_connections:
                        min_connections = conns
                        v = i
            
            if v is None:
                return
            
            # Option 1: Don't include v
            branch_and_bound(remaining & ~(1 << v), current_mask, current_size)
            
            # Option 2: Include v if it doesn't create a triangle
            # Check if adding v to current_mask creates a triangle
            # A triangle is created if there's an edge (a,b) in current_mask where both a and b are neighbors of v
            neighbors_in_current = adj_mask[v] & current_mask
            # Check if any two neighbors in current are connected
            creates_triangle = False
            for a in range(n):
                if not (neighbors_in_current & (1 << a)):
                    continue
                for b in range(a+1, n):
                    if (neighbors_in_current & (1 << b)) and (adj_mask[a] & (1 << b)):
                        creates_triangle = True
                        break
                if creates_triangle:
                    break
            
            if not creates_triangle:
                branch_and_bound(remaining & ~(1 << v), current_mask | (1 << v), current_size + 1)
        
        branch_and_bound(all_vertices_mask, 0, 0)
        return best
    
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
        for i in range(n):
            if common & (1 << i):
                for j in range(i+1, n):
                    if (common & (1 << j)) and (adj_mask[i] & (1 << j)):
                        return False
        return True
    
    best_score = n
    best_adj_mask = [0] * n
    
    # Random restarts with local search
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Initialize with random sparse graph
        adj_mask = [0] * n
        edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges_to_try)
        
        # Add edges randomly with some probability
        for u, v in edges_to_try:
            if random.random() < 0.5 and can_add_edge(adj_mask, u, v):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        
        current_score = largest_triangle_free(adj_mask)
        
        if current_score < best_score:
            best_score = current_score
            best_adj_mask = adj_mask[:]
        
        # Local search: try to improve by adding edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_score = largest_triangle_free(adj_mask)
            
            # Find a triangle-free set of size current_score
            # Try all subsets of size current_score
            target_set = None
            target_mask = 0
            for combo in itertools.combinations(range(n), current_score):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                if is_triangle_free(adj_mask, mask):
                    target_set = list(combo)
                    target_mask = mask
                    break
            
            if target_set is None:
                break
            
            # Try adding edges within target_set
            # Optimization: Only run exact B&B if the fast check passes
            for i in range(len(target_set)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(target_set)):
                    u = target_set[i]
                    v = target_set[j]
                    if not (adj_mask[u] & (1 << v)) and can_add_edge(adj_mask, u, v):
                        # Fast check: Does adding (u,v) create a triangle in target_set?
                        # It creates a triangle if there is a w in target_set such that (u,w) and (v,w) are edges.
                        # Since target_set is triangle-free, no such w exists in the *current* graph.
                        # Wait, the fast check is: if we add (u,v), does it break the triangle-free property of target_set?
                        # Yes, if there is any w in target_set \ {u,v} such that w is connected to u AND w is connected to v.
                        # If such w exists, then {u,v,w} becomes a triangle.
                        # If no such w exists, the set remains triangle-free, so the max size is still >= current_score.
                        # But could it be larger? No, we are adding an edge, which can only reduce or maintain the max triangle-free set size.
                        # So if the specific set remains triangle-free, the score is still current_score.
                        # We only care if the score *decreases*.
                        # The score decreases ONLY IF ALL maximum triangle-free sets are broken.
                        # This is hard to check fast.
                        # However, if we break *this* specific set, the score might drop.
                        # Let's just do the exact check but only if the fast check says "might break".
                        
                        # Actually, simpler:
                        # If we add an edge, the score can only go down or stay same.
                        # It goes down if the *maximum* size drops.
                        # If we break one max set, there might be another.
                        # So we MUST run exact check to know for sure.
                        # BUT, we can skip the exact check if we know the score cannot drop below current_score - 1?
                        # No.
                        
                        # Let's stick to the exact check but optimize the loop structure.
                        # The previous code ran exact check for EVERY edge.
                        # Let's try to reduce the number of exact checks.
                        
                        # Tentatively add
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        
                        # Exact check
                        new_score = largest_triangle_free(adj_mask)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            # Keep the edge
                            break
                        else:
                            # Revert
                            adj_mask[u] &= ~(1 << v)
                            adj_mask[v] &= ~(1 << u)
                if improved:
                    break
            
            if current_score < best_score:
                best_score = current_score
                best_adj_mask = adj_mask[:]
        
        # Also try removing edges to escape local minima
        if time.time() > deadline:
            break
        
        # Random perturbation: remove some edges
        adj_mask_perturbed = adj_mask[:]
        for _ in range(random.randint(1, 5)):
            # Pick a random existing edge and remove it
            existing_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj_mask_perturbed[i] & (1 << j):
                        existing_edges.append((i, j))
            if existing_edges:
                u, v = random.choice(existing_edges)
                adj_mask_perturbed[u] &= ~(1 << v)
                adj_mask_perturbed[v] &= ~(1 << u)
        
        # Continue local search from perturbed state
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_score = largest_triangle_free(adj_mask_perturbed)
            
            # Find a triangle-free set of size current_score
            target_set = None
            for combo in itertools.combinations(range(n), current_score):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                if is_triangle_free(adj_mask_perturbed, mask):
                    target_set = list(combo)
                    break
            
            if target_set is None:
                break
            
            # Try adding edges within target_set
            for i in range(len(target_set)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(target_set)):
                    u = target_set[i]
                    v = target_set[j]
                    if not (adj_mask_perturbed[u] & (1 << v)) and can_add_edge(adj_mask_perturbed, u, v):
                        # Add edge
                        adj_mask_perturbed[u] |= (1 << v)
                        adj_mask_perturbed[v] |= (1 << u)
                        
                        new_score = largest_triangle_free(adj_mask_perturbed)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            break
                        else:
                            # Revert
                            adj_mask_perturbed[u] &= ~(1 << v)
                            adj_mask_perturbed[v] &= ~(1 << u)
                if improved:
                    break
            
            if current_score < best_score:
                best_score = current_score
                best_adj_mask = adj_mask_perturbed[:]
    
    # Convert best_adj_mask to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj_mask[i] & (1 << j):
                edges.append([i, j])
    
    return edges
