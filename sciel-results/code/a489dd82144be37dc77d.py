import random
import time
import math

def construct():
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    idx_to_edge = {i: e for e, i in edge_to_idx.items()}
    
    # Precompute neighbors for each edge in the "line graph" sense? No.
    # Precompute for each edge (u,v), the set of edges that would conflict with it 
    # if we were building a matching? No.
    
    # We will use Simulated Annealing on the space of K2,3-free graphs.
    
    def count_common_neighbors(adj, u, v):
        return (adj[u] & adj[v]).bit_count()
        
    def is_valid(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if count_common_neighbors(adj, u, v) >= 3:
                    return False
        return True
        
    def can_add(u, v, adj):
        # Check if adding (u,v) violates K2,3
        # New common neighbors for (u, w) for w in N(v)
        # New common neighbors for (v, w) for w in N(u)
        
        # For w in N(v):
        # Common neighbors of u and w: (adj[u] & adj[w]).bit_count()
        # Adding (u,v) adds v to N(u).
        # v is in N(w) (since w in N(v)).
        # So v becomes a common neighbor.
        # Old count + 1 must be <= 2 => Old count <= 1.
        
        w = adj[v]
        while w:
            lsb = w & -w
            w ^= lsb
            idx = lsb.bit_length() - 1
            if (adj[u] & adj[idx]).bit_count() > 1:
                return False
                
        w = adj[u]
        while w:
            lsb = w & -w
            w ^= lsb
            idx = lsb.bit_length() - 1
            if (adj[v] & adj[idx]).bit_count() > 1:
                return False
                
        return True

    def get_missing_edges(adj, edge_set):
        missing = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in edge_set:
                    missing.append((u, v))
        return missing

    def simulated_annealing(initial_edges, adj, edge_set, t_start, t_end, steps):
        current_edges = list(initial_edges)
        current_adj = adj[:]
        current_edge_set = set(edge_set)
        current_score = len(current_edges)
        
        best_edges = current_edges[:]
        best_score = current_score
        
        # Temperature schedule
        # We will perform 'steps' iterations
        # Temperature decays linearly or exponentially
        
        for step in range(steps):
            t = t_start * (t_end / t_start) ** (step / steps)
            
            # Generate a move
            # 1. Try to add an edge
            # 2. Try to remove an edge
            # 3. Try to swap
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Add
                missing = get_missing_edges(current_adj, current_edge_set)
                if missing:
                    u, v = random.choice(missing)
                    if can_add(u, v, current_adj):
                        # Accept
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
                        current_edges.append((u, v))
                        current_edge_set.add((u, v))
                        current_score += 1
                        if current_score > best_score:
                            best_score = current_score
                            best_edges = current_edges[:]
                    else:
                        # Reject (or accept with prob exp(-1/t) if we allowed violations, but we don't)
                        pass
            elif move_type < 0.7:
                # Remove
                if current_edges:
                    idx = random.randint(0, len(current_edges) - 1)
                    u, v = current_edges[idx]
                    # Remove
                    current_adj[u] ^= (1 << v)
                    current_adj[v] ^= (1 << u)
                    current_edges.pop(idx)
                    current_edge_set.remove((u, v))
                    current_score -= 1
                    # Always accept removal to escape local minima? 
                    # Or accept with prob exp(-1/t)? 
                    # In SA for maximization with constraints, we usually only stay in valid space.
                    # Removing an edge is always valid.
                    # But we want to maximize edges. So we should only remove if it helps add more later.
                    # SA allows "bad" moves (fewer edges) to escape.
                    # So we accept removal.
            else:
                # Swap: Remove one, try to add another
                if current_edges:
                    idx = random.randint(0, len(current_edges) - 1)
                    u, v = current_edges[idx]
                    
                    # Temporarily remove
                    current_adj[u] ^= (1 << v)
                    current_adj[v] ^= (1 << u)
                    current_edges.pop(idx)
                    current_edge_set.remove((u, v))
                    
                    added = False
                    missing = get_missing_edges(current_adj, current_edge_set)
                    if missing:
                        # Try a few random missing edges
                        random.shuffle(missing)
                        for cu, cv in missing[:10]: # Limit attempts
                            if can_add(cu, cv, current_adj):
                                current_adj[cu] |= (1 << cv)
                                current_adj[cv] |= (1 << cu)
                                current_edges.append((cu, cv))
                                current_edge_set.add((cu, cv))
                                current_score += 1
                                added = True
                                break
                    
                    if not added:
                        # Revert swap (just re-add the removed edge)
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
                        current_edges.append((u, v))
                        current_edge_set.add((u, v))
                        # Score unchanged
                    else:
                        # Score increased by 1 (removed 1, added 1, net 0? No, removed 1, added 1. Net 0.)
                        # Wait, if we remove 1 and add 1, score is same.
                        # We only improve if we add 1 without removing? That's the "Add" move.
                        # Swap is useful to change structure to allow future Adds.
                        # So we accept swap even if score is same, based on SA?
                        # Since score is same, delta E = 0. Accept with prob 1.
                        pass

            # Update best
            if current_score > best_score:
                best_score = current_score
                best_edges = current_edges[:]
                
        return best_edges, best_score

    def run_sa(seed):
        rng = random.Random(seed)
        # Initialize with a greedy graph
        adj = [0] * n
        edges = []
        edge_set = set()
        
        order = all_edges[:]
        rng.shuffle(order)
        
        for u, v in order:
            if can_add(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
                edge_set.add((u, v))
                
        # Run SA
        # Time limit per SA run? Or total?
        # We have 120s total.
        # Let's run multiple short SA runs.
        
        t_start = 1.0
        t_end = 0.01
        steps = 5000
        
        best_edges, best_score = simulated_annealing(edges, adj, edge_set, t_start, t_end, steps)
        
        return best_edges, best_score

    start_time = time.time()
    time_limit = 110.0
    
    best_overall = []
    best_overall_score = 0
    
    seed_counter = 0
    while time.time() - start_time < time_limit:
        seed_counter += 1
        edges, score = run_sa(seed_counter)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = edges
        
        # Check time
        if time.time() - start_time > time_limit:
            break

    # Verification
    def is_valid(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] & adj[v]).bit_count() >= 3:
                    return False
        return True

    if not is_valid(best_overall):
        return [[i, (i+1)%20] for i in range(20)]
        
    return [[u, v] for u, v in best_overall]
