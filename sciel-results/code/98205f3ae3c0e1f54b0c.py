"""Improved K2,3-free graph construction for n=20 vertices."""
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    total_pairs = len(all_pairs)
    
    def is_valid_addition(adj, common, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        # Check 1: u and v must not have 3+ common neighbors
        if common[u][v] >= 3:
            return False
        # Check 2: For each neighbor w of v, common[u][w] must be < 2
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w] and common[u][w] >= 2:
                return False
        # Check 3: For each neighbor w of u, common[v][w] must be < 2
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[u][w] and common[v][w] >= 2:
                return False
        return True

    def add_edge(adj, common, u, v):
        """Add edge (u,v) and update common neighbor counts."""
        adj[u][v] = True
        adj[v][u] = True
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w]:
                common[u][w] += 1
                common[w][u] += 1
            if adj[u][w]:
                common[v][w] += 1
                common[w][v] += 1

    def remove_edge(adj, common, u, v):
        """Remove edge (u,v) and update common neighbor counts."""
        adj[u][v] = False
        adj[v][u] = False
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w]:
                common[u][w] -= 1
                common[w][u] -= 1
            if adj[u][w]:
                common[v][w] -= 1
                common[w][v] -= 1

    def get_edge_list(adj):
        """Extract edge list from adjacency matrix."""
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    def get_missing_edges(adj):
        """Get list of missing edges."""
        missing = []
        for u, v in all_pairs:
            if not adj[u][v]:
                missing.append((u, v))
        return missing

    def greedy_build(seed, order_strategy='random'):
        """Build graph using greedy edge addition with specified order strategy."""
        rng = random.Random(seed)
        adj = [[False]*n for _ in range(n)]
        common = [[0]*n for _ in range(n)]
        
        pairs = list(all_pairs)
        rng.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            if is_valid_addition(adj, common, u, v):
                add_edge(adj, common, u, v)
        
        return adj, common

    def fast_local_search(adj, common, max_iterations=1000):
        """
        Fast local search: try removing one edge and adding one or more others.
        Uses precomputed missing edges for speed.
        """
        if time.time() > deadline:
            return adj, common
        
        current_edges = get_edge_list(adj)
        current_count = len(current_edges)
        
        best_adj = [row[:] for row in adj]
        best_common = [row[:] for row in common]
        best_count = current_count
        
        rng = random.Random(42)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Pick a random existing edge to potentially remove
            if not current_edges:
                break
            rem_u, rem_v = rng.choice(current_edges)
            rem_u, rem_v = min(rem_u, rem_v), max(rem_u, rem_v)
            
            # Try removing this edge
            remove_edge(adj, common, rem_u, rem_v)
            
            # Get missing edges (excluding the one we just removed)
            missing = [(u, v) for u, v in all_pairs if not adj[u][v] and (u, v) != (rem_u, rem_v)]
            rng.shuffle(missing)
            
            added_count = 0
            added_edges = []
            
            for u, v in missing:
                if is_valid_addition(adj, common, u, v):
                    add_edge(adj, common, u, v)
                    added_count += 1
                    added_edges.append((u, v))
                    if added_count >= 3:
                        break
            
            # Evaluate: if we added more edges than we removed (1), we improved
            new_count = current_count - 1 + added_count
            
            if new_count > best_count:
                best_count = new_count
                best_adj = [row[:] for row in adj]
                best_common = [row[:] for row in common]
                current_count = new_count
                current_edges = get_edge_list(adj)
                adj = [row[:] for row in best_adj]
                common = [row[:] for row in best_common]
            else:
                # Revert: remove the added edges and re-add the removed edge
                for u, v in added_edges:
                    remove_edge(adj, common, u, v)
                add_edge(adj, common, rem_u, rem_v)
        
        return best_adj, best_common

    def swap_local_search(adj, common, max_iterations=500):
        """
        1-swap local search: try removing one edge and adding one different edge.
        This can enable further additions in subsequent iterations.
        """
        if time.time() > deadline:
            return adj, common
        
        current_edges = get_edge_list(adj)
        current_count = len(current_edges)
        
        best_adj = [row[:] for row in adj]
        best_common = [row[:] for row in common]
        best_count = current_count
        
        rng = random.Random(99)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Pick a random existing edge to remove
            if not current_edges:
                break
            rem_u, rem_v = rng.choice(current_edges)
            rem_u, rem_v = min(rem_u, rem_v), max(rem_u, rem_v)
            
            # Get missing edges
            missing = [(u, v) for u, v in all_pairs if not adj[u][v] and (u, v) != (rem_u, rem_v)]
            if not missing:
                break
            
            # Try a random missing edge to add
            add_u, add_v = rng.choice(missing)
            
            # Remove the edge, try to add the new one
            remove_edge(adj, common, rem_u, rem_v)
            
            if is_valid_addition(adj, common, add_u, add_v):
                add_edge(adj, common, add_u, add_v)
                # Now try to add more edges
                missing2 = [(u, v) for u, v in all_pairs if not adj[u][v]]
                rng.shuffle(missing2)
                for u, v in missing2[:10]:  # Try up to 10 more
                    if is_valid_addition(adj, common, u, v):
                        add_edge(adj, common, u, v)
                
                new_count = len(get_edge_list(adj))
                if new_count > best_count:
                    best_count = new_count
                    best_adj = [row[:] for row in adj]
                    best_common = [row[:] for row in common]
                    current_count = new_count
                    current_edges = get_edge_list(adj)
                    adj = [row[:] for row in best_adj]
                    common = [row[:] for row in best_common]
                else:
                    # Revert
                    for u, v in missing2[:10]:
                        if adj[u][v] and (u, v) != (add_u, add_v):
                            remove_edge(adj, common, u, v)
                    remove_edge(adj, common, add_u, add_v)
                    add_edge(adj, common, rem_u, rem_v)
            else:
                # Revert
                add_edge(adj, common, rem_u, rem_v)
        
        return best_adj, best_common

    def multi_start_local_search(adj, common, num_rounds=5):
        """Run multiple rounds of local search with perturbations."""
        best_adj = [row[:] for row in adj]
        best_common = [row[:] for row in common]
        best_count = len(get_edge_list(adj))
        
        for rnd in range(num_rounds):
            if time.time() > deadline:
                break
            
            # Perturb: remove 2-4 random edges
            adj_pert = [row[:] for row in adj]
            common_pert = [row[:] for row in common]
            edges = get_edge_list(adj_pert)
            
            rng = random.Random(200 + rnd)
            num_remove = rng.randint(2, 4)
            for _ in range(min(num_remove, len(edges))):
                if not edges:
                    break
                u, v = rng.choice(edges)
                u, v = min(u, v), max(u, v)
                if adj_pert[u][v]:
                    remove_edge(adj_pert, common_pert, u, v)
                    edges = [e for e in edges if not (min(e[0],e[1])==u and max(e[0],e[1])==v)]
            
            # Run fast local search on perturbed graph
            adj_ls, common_ls = fast_local_search(adj_pert, common_pert, max_iterations=300)
            
            count_ls = len(get_edge_list(adj_ls))
            if count_ls > best_count:
                best_count = count_ls
                best_adj = [row[:] for row in adj_ls]
                best_common = [row[:] for row in common_ls]
                adj = best_adj
                common = best_common
        
        return best_adj, best_common

    # Main search loop
    best_adj = None
    best_common = None
    best_count = 0
    
    # Strategy 1: Multiple greedy restarts with fast local search
    num_restarts = 40
    for seed in range(num_restarts):
        if time.time() > deadline:
            break
        
        adj, common = greedy_build(seed, order_strategy='random')
        count = len(get_edge_list(adj))
        
        # Apply fast local search
        adj, common = fast_local_search(adj, common, max_iterations=500)
        count = len(get_edge_list(adj))
        
        if count > best_count:
            best_count = count
            best_adj = [row[:] for row in adj]
            best_common = [row[:] for row in common]
    
    # Strategy 2: Multi-start with perturbations on the best found
    if best_adj is not None and time.time() < deadline:
        best_adj, best_common = multi_start_local_search(best_adj, best_common, num_rounds=8)
        best_count = len(get_edge_list(best_adj))
    
    # Strategy 3: Swap local search on the best
    if best_adj is not None and time.time() < deadline:
        best_adj, best_common = swap_local_search(best_adj, best_common, max_iterations=300)
        best_count = len(get_edge_list(best_adj))
    
    # Strategy 4: More greedy restarts if time permits
    if time.time() < deadline:
        for seed in range(num_restarts, num_restarts + 40):
            if time.time() > deadline:
                break
            adj, common = greedy_build(seed, order_strategy='random')
            count = len(get_edge_list(adj))
            
            if count + 3 > best_count:  # Only do local search if promising
                adj, common = fast_local_search(adj, common, max_iterations=300)
                count = len(get_edge_list(adj))
            
            if count > best_count:
                best_count = count
                best_adj = [row[:] for row in adj]
                best_common = [row[:] for row in common]
    
    if best_adj is not None:
        return get_edge_list(best_adj)
    
    # Fallback: disjoint K4s
    edges = []
    for k in range(5):
        for a in range(4):
            for b in range(a+1, 4):
                edges.append([4*k+a, 4*k+b])
    return edges
