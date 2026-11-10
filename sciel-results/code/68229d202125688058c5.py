"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = 20
    
    # Start with a random graph and optimize via simulated annealing
    # Use bitmask representation for fast feasibility checks
    
    def popcount(x):
        return bin(x).count('1')
    
    def can_add(adj, u, v):
        # Check if adding edge (u,v) keeps graph K2,3-free
        if (adj[u] >> v) & 1:
            return False
            
        # 1. Pair (u,v) common neighbors must be <= 2
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) > 2:
            return False
            
        # 2. For all w in N(v), pair (u,w) gains v as common neighbor
        #    So we need: for each w in N(v) \ {u}, popcount(N(u) & N(w) & ~{v}) <= 2
        #    But since v is not yet in N(u), N(u) & N(w) doesn't include v
        #    Actually, after adding (u,v), common neighbors of (u,w) = (N(u) & N(w)) + {v} if v in N(w)
        #    So we need popcount(N(u) & N(w)) <= 1 for all w in N(v) where v in N(w)
        #    Wait, let me reconsider. After adding (u,v):
        #    For any pair (x,y), common neighbors = N(x) & N(y)
        #    Adding (u,v) affects pairs that include u or v.
        #    Pairs affected: (u, w) for w in N(v), and (v, w) for w in N(u), and (u,v) itself.
        #    For (u,v): common neighbors = N(u) & N(v), must be <= 2.
        #    For (u, w) where w in N(v): new common neighbors = (N(u) & N(w)) | {v} if v in N(u)... 
        #    Actually N(u) doesn't contain v yet. After adding, N'(u) = N(u) ∪ {v}, N'(w) = N(w) (unchanged if w != v)
        #    So common neighbors of (u,w) after = (N(u) ∪ {v}) & N(w) = (N(u) & N(w)) ∪ ({v} if v in N(w) else ∅)
        #    Since w in N(v) means v in N(w), so common = (N(u) & N(w)) ∪ {v}
        #    So we need popcount(N(u) & N(w)) <= 1 for all w in N(v)
        #    Similarly for (v, w) where w in N(u): need popcount(N(v) & N(w)) <= 1 for all w in N(u)
        
        # Check pairs (u, w) for w in N(v)
        nv_mask = adj[v]
        w = 0
        while w < n:
            if (nv_mask >> w) & 1 and w != u:
                common_uw = adj[u] & adj[w]
                if popcount(common_uw) > 1:
                    return False
            w += 1
        
        # Check pairs (v, w) for w in N(u)
        nu_mask = adj[u]
        w = 0
        while w < n:
            if (nu_mask >> w) & 1 and w != v:
                common_vw = adj[v] & adj[w]
                if popcount(common_vw) > 1:
                    return False
            w += 1
        
        return True
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    # Initialize with a random graph
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                add_edge(adj, i, j)
    
    # Simulated annealing
    def get_edge_count(adj):
        count = 0
        for i in range(n):
            count += popcount(adj[i])
        return count // 2
    
    def is_valid(adj):
        # Check all pairs have at most 2 common neighbors
        for i in range(n):
            for j in range(i + 1, n):
                if popcount(adj[i] & adj[j]) > 2:
                    return False
        return True
    
    # Ensure initial state is valid
    if not is_valid(adj):
        # Remove edges until valid
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1 and not is_valid(adj):
                    remove_edge(adj, i, j)
    
    current_edges = get_edge_count(adj)
    best_adj = adj[:]
    best_edges = current_edges
    
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Try adding a random edge
        if random.random() < 0.7:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if not ((adj[u] >> v) & 1):
                if can_add(adj, u, v):
                    add_edge(adj, u, v)
                    new_edges = current_edges + 1
                    if new_edges > best_edges:
                        best_edges = new_edges
                        best_adj = adj[:]
                    current_edges = new_edges
                else:
                    # Edge addition not possible, try swap
                    pass
        else:
            # Try swapping: remove one edge, add another
            # Find a random existing edge
            existing_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj[i] >> j) & 1:
                        existing_edges.append((i, j))
            
            if not existing_edges:
                continue
            
            rem_u, rem_v = random.choice(existing_edges)
            
            # Try to find an edge to add
            added = False
            for _ in range(10):  # Try up to 10 times
                add_u = random.randint(0, n - 1)
                add_v = random.randint(0, n - 1)
                if add_u == add_v:
                    continue
                if (add_u, add_v) == (rem_u, rem_v) or (add_u, add_v) == (rem_v, rem_u):
                    continue
                if (adj[add_u] >> add_v) & 1:
                    continue
                
                # Temporarily remove the edge
                remove_edge(adj, rem_u, rem_v)
                
                if can_add(adj, add_u, add_v):
                    add_edge(adj, add_u, add_v)
                    current_edges = current_edges  # Same count
                    # Accept with some probability if it helps future moves
                    if random.random() < 0.3:
                        added = True
                        break
                    else:
                        # Revert
                        remove_edge(adj, add_u, add_v)
                        add_edge(adj, rem_u, rem_v)
                        added = False
                        break
                else:
                    # Revert
                    add_edge(adj, rem_u, rem_v)
                    added = False
                    break
            
            if not added:
                # Ensure the removed edge is back
                if not ((adj[rem_u] >> rem_v) & 1):
                    add_edge(adj, rem_u, rem_v)
        
        # Cooling
        T *= cooling
        if T < 0.1:
            T = 0.1
    
    # Extract edges from best_adj
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
# EVOLVE-BLOCK-END
