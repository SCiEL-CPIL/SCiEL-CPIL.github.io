"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
import itertools

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a K3,3-free graph on 20 vertices maximizing edges.
    
    Strategy:
    1. Greedy construction with randomized edge ordering.
    2. Local search with edge additions and swaps.
    3. Bitmask-based violation checks for efficiency.
    4. Multiple restarts with different random seeds.
    5. Additional polish pass with exhaustive local search.
    """
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    N = 20
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    def popcount(x):
        return bin(x).count('1')
    
    def can_add_edge(adj, u, v, n):
        """Check if adding edge (u,v) maintains K3,3-freeness."""
        # For triple (u, x, y) where x,y ≠ v:
        # New common adds v if v ∈ N(x) ∩ N(y)
        # Violation iff |old_common| = 2 AND v ∈ N(x) ∩ N(y)
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x+1, n):
                if y == u or y == v:
                    continue
                # Old common neighbors of u, x, y
                old_common = adj[u] & adj[x] & adj[y]
                old_common &= ~(1 << u) & ~(1 << x) & ~(1 << y)
                # Check if v becomes a new common neighbor
                if (adj[x] & (1 << v)) and (adj[y] & (1 << v)):
                    if popcount(old_common) >= 2:
                        return False
        # For triple (v, x, y) where x,y ≠ u:
        # New common adds u if u ∈ N(x) ∩ N(y)
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x+1, n):
                if y == u or y == v:
                    continue
                old_common = adj[v] & adj[x] & adj[y]
                old_common &= ~(1 << v) & ~(1 << x) & ~(1 << y)
                if (adj[x] & (1 << u)) and (adj[y] & (1 << u)):
                    if popcount(old_common) >= 2:
                        return False
        return True
    
    def count_edges(adj, n):
        total = 0
        for i in range(n):
            total += popcount(adj[i])
        return total // 2
    
    def greedy_construct(seed):
        """Build a dense K3,3-free graph using greedy edge addition."""
        rng = random.Random(seed)
        adj = [0] * N
        edge_count = 0
        
        # Order edges randomly
        edges = all_pairs[:]
        rng.shuffle(edges)
        
        for u, v in edges:
            if can_add_edge(adj, u, v, N):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
            if time.time() > deadline:
                break
        
        return adj, edge_count
    
    def local_search(adj, edge_count, seed, time_limit=10):
        """Improve graph via edge additions and swaps."""
        rng = random.Random(seed)
        end_time = time.time() + time_limit
        best_adj = adj[:]
        best_count = edge_count
        no_improvement_count = 0
        
        while time.time() < end_time:
            # Try to add a random missing edge
            missing = []
            for u in range(N):
                for v in range(u+1, N):
                    if not (adj[u] & (1 << v)):
                        missing.append((u, v))
            
            if not missing:
                break
            
            # Try random missing edges
            found_improvement = False
            attempts = 0
            max_attempts = 100
            while attempts < max_attempts and time.time() < end_time:
                u, v = rng.choice(missing)
                if can_add_edge(adj, u, v, N):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_count += 1
                    found_improvement = True
                    break
                attempts += 1
            
            if found_improvement:
                if edge_count > best_count:
                    best_adj = adj[:]
                    best_count = edge_count
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count > 20:
                    # Try edge swap: remove an edge, then try to add more
                    present = []
                    for u in range(N):
                        for v in range(u+1, N):
                            if adj[u] & (1 << v):
                                present.append((u, v))
                    
                    if present:
                        ru, rv = rng.choice(present)
                        # Remove edge
                        adj[ru] &= ~(1 << rv)
                        adj[rv] &= ~(1 << ru)
                        edge_count -= 1
                        
                        # Try to add edges from all possible missing edges
                        added_edges = []
                        for u in range(N):
                            for v in range(u+1, N):
                                if not (adj[u] & (1 << v)):
                                    if can_add_edge(adj, u, v, N):
                                        adj[u] |= (1 << v)
                                        adj[v] |= (1 << u)
                                        edge_count += 1
                                        added_edges.append((u, v))
                        
                        if edge_count > best_count:
                            best_adj = adj[:]
                            best_count = edge_count
                            no_improvement_count = 0
                        else:
                            # Revert if not improved
                            adj[ru] |= (1 << rv)
                            adj[rv] |= (1 << ru)
                            edge_count += 1
                            for u, v in added_edges:
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                edge_count -= 1
                            no_improvement_count += 5
        
        return best_adj, best_count
    
    def polish_pass(adj, edge_count, time_limit=5):
        """Final high-precision polish with exhaustive local search."""
        end_time = time.time() + time_limit
        improved = True
        while improved and time.time() < end_time:
            improved = False
            # Try all possible single edge additions
            for u in range(N):
                for v in range(u+1, N):
                    if not (adj[u] & (1 << v)):
                        if can_add_edge(adj, u, v, N):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            edge_count += 1
                            improved = True
        return adj, edge_count
    
    # Multiple restarts
    best_edges = None
    best_count = 0
    
    # First, try several greedy constructions with different seeds
    for seed in range(30):
        if time.time() > deadline - 15:
            break
        adj, ec = greedy_construct(seed)
        if ec > best_count:
            best_count = ec
            best_edges = adj[:]
        
        # Local search on this result
        if time.time() > deadline - 20:
            break
        remaining = deadline - time.time() - 10
        if remaining > 3:
            adj2, ec2 = local_search(adj, ec, seed, time_limit=min(5, remaining))
            if ec2 > best_count:
                best_count = ec2
                best_edges = adj2[:]
    
    # Final local search on the best result
    if best_edges is not None and time.time() < deadline - 10:
        remaining = deadline - time.time() - 5
        if remaining > 3:
            adj3, ec3 = local_search(best_edges, best_count, 99999, time_limit=min(8, remaining))
            if ec3 > best_count:
                best_count = ec3
                best_edges = adj3[:]
    
    # Polish pass
    if best_edges is not None and time.time() < deadline - 3:
        remaining = deadline - time.time() - 2
        if remaining > 1:
            adj4, ec4 = polish_pass(best_edges, best_count, time_limit=min(3, remaining))
            if ec4 > best_count:
                best_count = ec4
                best_edges = adj4[:]
    
    if best_edges is None:
        # Fallback: K2,18
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    # Convert adjacency to edge list
    result = []
    for u in range(N):
        for v in range(u+1, N):
            if best_edges[u] & (1 << v):
                result.append([u, v])
    
    return result

# EVOLVE-BLOCK-END
