"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill

    n = 40
    all_mask = (1 << n) - 1

    def would_create_c6(adj, u, v):
        """
        Check if adding edge (u, v) to the graph represented by adjacency bitmasks `adj`
        creates a C6. Optimized with early termination and bitmask operations.
        """
        Nu = adj[u]
        Nv = adj[v]
        
        # For a C6: u-a-b-c-d-v-u, we need:
        # a in N(u), d in N(v), a!=d, b in N(a)\{u,v,d}, c in N(d)\{u,v,a}, b in N(c) or c in N(b)
        # Actually: u-a-b-c-d-v means a~u, a~b, b~c, c~d, d~v
        # So: a in N(u), d in N(v), b in N(a)∩N(c), c in N(d)∩N(b)
        # More precisely: need a~u, d~v, a~b, b~c, c~d
        # So: pick a from N(u)\{v}, d from N(v)\{u}, then need b,c such that
        # b in N(a)\{u,v,d}, c in N(d)\{u,v,a}, and b~c
        
        a_mask = Nu & ~(1 << v)
        while a_mask:
            lsb_a = a_mask & -a_mask
            a = lsb_a.bit_length() - 1
            a_mask ^= lsb_a
            
            d_mask = Nv & ~(1 << u) & ~(1 << a)
            while d_mask:
                lsb_d = d_mask & -d_mask
                d = lsb_d.bit_length() - 1
                d_mask ^= lsb_d
                
                Na = adj[a]
                Nd = adj[d]
                
                # b must be in N(a) \ {u, v, d}
                b_mask = Na & ~(1 << u) & ~(1 << v) & ~(1 << d)
                if not b_mask:
                    continue
                    
                # c must be in N(d) \ {u, v, a}
                c_mask = Nd & ~(1 << u) & ~(1 << v) & ~(1 << a)
                if not c_mask:
                    continue
                
                # Need b in b_mask and c in c_mask such that b~c
                # i.e., exists b in b_mask with adj[b] & c_mask != 0
                b_temp = b_mask
                while b_temp:
                    lsb_b = b_temp & -b_temp
                    b = lsb_b.bit_length() - 1
                    b_temp ^= lsb_b
                    
                    if adj[b] & c_mask:
                        return True
        
        return False

    def build_adj(edges_set):
        adj = [0] * n
        for u, v in edges_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def dense_random_init(seed_val):
        """
        Create a dense random initialization and prune C6s.
        This provides a higher quality starting point than sparse K5 seeds.
        Uses a more efficient pruning strategy with multiple density levels.
        """
        random.seed(seed_val)
        edges = set()
        
        # Try multiple density levels and keep the best
        best_density_edges = set()
        best_count = 0
        
        for density in [0.60, 0.65, 0.70, 0.55]:
            if time.time() > deadline - 30:
                break
                
            random.seed(seed_val + int(density * 100))
            candidate_edges = set()
            
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < density:
                        candidate_edges.add((i, j))
            
            # Prune edges that create C6s
            adj = build_adj(candidate_edges)
            
            # Iterate until stable, removing all bad edges in one pass
            changed = True
            while changed:
                changed = False
                to_remove = []
                for u, v in list(candidate_edges):
                    # Temporarily remove
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
                    # Check if re-adding would create C6
                    if would_create_c6(adj, u, v):
                        to_remove.append((u, v))
                    else:
                        # Keep it
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                
                for u, v in to_remove:
                    candidate_edges.remove((u, v))
                    changed = True
            
            if len(candidate_edges) > best_count:
                best_density_edges = candidate_edges
                best_count = len(candidate_edges)
        
        return best_density_edges

    def k5_seed():
        edges = set()
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u, v = 5 * k + a, 5 * k + b
                    edges.add((u, v))
        return edges

    def run_greedy(start_edges, seed_val):
        random.seed(seed_val)
        
        adj = build_adj(start_edges)
        edges = set(start_edges)
        
        possible_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    possible_edges.append((i, j))
        
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if time.time() > deadline - 2:
                return edges
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
        
        return edges

    def local_search(edges_set, seed_val):
        """
        Improve the edge set by adding edges and swapping.
        Uses aggressive 1-for-1 and 1-for-2 swap strategies with neighbor-based pruning.
        """
        random.seed(seed_val)
        
        adj = build_adj(edges_set)
        current_edges = set(edges_set)
        
        def get_non_edges():
            ne = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        ne.append((i, j))
            return ne
        
        # Pass 1: Add any possible non-edge (greedy)
        non_edges = get_non_edges()
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() > deadline - 3:
                break
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.add((u, v))
        
        # Pass 2: Try 1-for-1 swaps (remove e, add f) - same edge count but may enable further additions
        # Pass 3: Try 1-for-2 swaps (remove e, add 2 edges) - increases edge count by 1
        
        removable = list(current_edges)
        random.shuffle(removable)
        non_edges = get_non_edges()
        random.shuffle(non_edges)
        
        consecutive_failures = 0
        max_failures = 50
        
        while time.time() < deadline - 3:
            if consecutive_failures > max_failures:
                break
                
            if not removable or not non_edges:
                break
            
            # Choose a random edge to remove
            e_rem = removable[random.randint(0, len(removable) - 1)]
            u_rem, v_rem = e_rem
            
            # Save state
            old_adj = adj[:]
            old_edges = set(current_edges)
            
            # Remove edge
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            current_edges.remove(e_rem)
            
            # Try to add non-edges (up to 2)
            added = 0
            local_non_edges = non_edges[:]
            random.shuffle(local_non_edges)
            
            for u, v in local_non_edges:
                if added >= 2:
                    break
                if time.time() > deadline - 3:
                    break
                if (u, v) in current_edges:
                    continue
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    added += 1
            
            if added >= 2:
                # Success: Edge count increased by (added - 1) >= 1
                consecutive_failures = 0
                removable = list(current_edges)
                random.shuffle(removable)
                non_edges = get_non_edges()
                random.shuffle(non_edges)
            elif added == 1:
                # Same edge count - keep it if it might help (probabilistic)
                if random.random() < 0.3:
                    consecutive_failures = 0
                    removable = list(current_edges)
                    random.shuffle(removable)
                    non_edges = get_non_edges()
                    random.shuffle(non_edges)
                else:
                    adj = old_adj
                    current_edges = old_edges
                    consecutive_failures += 1
            else:
                # Revert
                adj = old_adj
                current_edges = old_edges
                consecutive_failures += 1

        # Final greedy pass
        non_edges = get_non_edges()
        random.shuffle(non_edges)
        for u, v in non_edges:
            if time.time() > deadline - 2:
                break
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.add((u, v))

        return current_edges

    def perturb_and_refine(edges_set, seed_val, num_remove):
        """
        Remove some edges randomly, then run greedy + local search to refine.
        """
        random.seed(seed_val)
        edges = set(edges_set)
        
        # Remove num_remove random edges
        to_remove = random.sample(list(edges), min(num_remove, len(edges)))
        for e in to_remove:
            edges.remove(e)
        
        # Rebuild adj
        adj = build_adj(edges)
        
        # Run greedy
        possible_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    possible_edges.append((i, j))
        
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if time.time() > deadline - 5:
                break
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
        
        # Run local search
        edges = local_search(edges, seed_val)
        
        return edges

    # Main search loop
    # Strategy: Use dense random initialization as the primary seed,
    # with K5 as a fallback if dense init is too slow or fails.
    
    best_edges = set()
    best_count = 0
    
    # Try multiple dense random initializations with different densities
    if time.time() < deadline - 40:
        for seed_val in [42, 123, 777, 999, 2024]:
            if time.time() > deadline - 35:
                break
            dense_edges = dense_random_init(seed_val)
            if len(dense_edges) > best_count:
                best_edges = dense_edges
                best_count = len(best_edges)
    
    # If dense init didn't work well, try K5
    if best_count < 90:
        k5 = k5_seed()
        if len(k5) > best_count:
            best_edges = k5
            best_count = len(best_edges)
    
    # Initial greedy
    best_edges = run_greedy(best_edges, 42)
    
    # Local search
    best_edges = local_search(best_edges, 42)
    
    best_count = len(best_edges)
    
    # Multiple restarts with perturbation - more aggressive
    restart_seed = 100
    for i in range(80):
        if time.time() > deadline - 8:
            break
        
        # Perturb the best solution with varying amounts
        num_remove = random.randint(3, 12)
        perturbed = perturb_and_refine(best_edges, restart_seed + i, num_remove)
        
        if len(perturbed) > best_count:
            best_edges = perturbed
            best_count = len(perturbed)
        
        restart_seed += 1
    
    # Final polish: try to add any remaining non-edges
    adj_final = build_adj(best_edges)
    final_non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj_final[i] & (1 << j)):
                final_non_edges.append((i, j))
    
    random.shuffle(final_non_edges)
    for u, v in final_non_edges:
        if time.time() > deadline - 1:
            break
        if not would_create_c6(adj_final, u, v):
            adj_final[u] |= (1 << v)
            adj_final[v] |= (1 << u)
            best_edges.add((u, v))
    
    # Convert to list of lists
    return [list(e) for e in best_edges]

# EVOLVE-BLOCK-END
