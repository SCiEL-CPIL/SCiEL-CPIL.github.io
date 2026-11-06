import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 20
    NPAIRS = n * (n - 1) // 2
    
    # Precompute all pairs
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_index = {}
    for idx, (a, b) in enumerate(pairs):
        pair_index[(a, b)] = idx
    
    # Precompute for each pair (a,b), the list of other vertices c (c != a, c != b)
    # This helps in checking: for edge (u,v) to be added, we need to check
    # for all pairs (a,b) where a,b are both neighbors of v (and a,b != u):
    #   the number of common neighbors of {u, a, b} must be <= 1
    # Actually, let's think about this differently.
    # When adding edge (u,v), the K3,3 constraint is violated iff there exists
    # a triple T such that |N(T)| >= 3 after the addition.
    # The only triples whose N(T) can change are those containing u or v.
    # For triple {u, a, b}: N(T) = N(u) ∩ N(a) ∩ N(b)
    #   After adding (u,v): v is in N(u). If v ∈ N(a) and v ∈ N(b), then v ∈ N(T).
    #   So we need: |N(u) ∩ N(a) ∩ N(b)| + [v ∈ N(a) ∧ v ∈ N(b)] <= 2
    #   Since v is NOT in N(u) before addition, v is not in N(u)∩N(a)∩N(b) before.
    #   So: |N(u) ∩ N(a) ∩ N(b)| + [v ∈ N(a) ∧ v ∈ N(b)] <= 2
    #   i.e., |N(u) ∩ N(a) ∩ N(b)| <= 2 - [v ∈ N(a) ∧ v ∈ N(b)]
    
    # Similarly for triple {v, a, b}: need |N(v) ∩ N(a) ∩ N(b)| + [u ∈ N(a) ∧ u ∈ N(b)] <= 2
    
    # For efficiency, precompute for each vertex v, the list of vertices adjacent to v.
    # And for each pair (a,b), precompute the bitmask of common neighbors.
    
    # Actually, let's use a different approach. Precompute cn[a][b] = N(a) ∩ N(b) as bitmask.
    # Then for adding (u,v):
    #   For all a < b, a,b ≠ u,v:
    #     If v ∈ N(a) and v ∈ N(b): check |N(u) ∩ N(a) ∩ N(b)| <= 1
    #     If u ∈ N(a) and u ∈ N(b): check |N(v) ∩ N(a) ∩ N(b)| <= 1
    #   N(u) ∩ N(a) ∩ N(b) = (N(u) ∩ N(a)) ∩ N(b) = cn[u][a] ∩ N(b)
    #   So we can compute this as (cn[u][a] & adj[b]).bit_count()
    
    # But we still need to iterate over all pairs (a,b). That's O(n^2) per check.
    # With n=20, that's ~190 pairs. And bit_count is fast.
    
    # Let me just write a clean, optimized version.
    
    def is_valid(adj, u, v):
        """Check if adding edge (u,v) keeps the graph K3,3-free."""
        if adj[u] & (1 << v):
            return False
        
        # Check triples containing u: {u, a, b} where a, b ≠ u
        # v becomes a new common neighbor if v ∈ N(a) and v ∈ N(b)
        # Need: |N(u) ∩ N(a) ∩ N(b)| + [v∈N(a) ∧ v∈N(b)] <= 2
        for a in range(n):
            if a == u:
                continue
            if not (adj[a] & (1 << v)):
                continue
            for b in range(a + 1, n):
                if b == u:
                    continue
                if not (adj[b] & (1 << v)):
                    continue
                # v is common neighbor of a and b, so it will be in N({u,a,b})
                # Need |N(u) ∩ N(a) ∩ N(b)| <= 1
                cnt = (adj[u] & adj[a] & adj[b]).bit_count()
                if cnt > 1:
                    return False
        
        # Check triples containing v: {v, a, b} where a, b ≠ v
        for a in range(n):
            if a == v:
                continue
            if not (adj[a] & (1 << u)):
                continue
            for b in range(a + 1, n):
                if b == v:
                    continue
                if not (adj[b] & (1 << u)):
                    continue
                cnt = (adj[v] & adj[a] & adj[b]).bit_count()
                if cnt > 1:
                    return False
        
        return True
    
    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2
    
    def greedy_construct(seed):
        random.seed(seed)
        adj = [0] * n
        shuffled = pairs[:]
        random.shuffle(shuffled)
        for u, v in shuffled:
            if time.time() > deadline - 30:
                break
            if is_valid(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    def try_add_all(adj):
        """Try to add all possible missing edges."""
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        if is_valid(adj, i, j):
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                            changed = True
        return adj
    
    def kick_and_fill(adj, kick_size, seed):
        """Remove kick_size random edges, then greedily add back."""
        random.seed(seed)
        existing = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i] & (1 << j)]
        if len(existing) < kick_size:
            return adj
        to_remove = random.sample(existing, kick_size)
        for u, v in to_remove:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        # Now greedily add edges
        shuffled = pairs[:]
        random.shuffle(shuffled)
        for u, v in shuffled:
            if not (adj[u] & (1 << v)):
                if is_valid(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj
    
    def local_search(adj, time_budget):
        """Perform local search with kick-and-fill."""
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        end_time = time.time() + time_budget
        
        # First, try to add all possible edges
        current_adj = try_add_all(current_adj)
        current_edges = count_edges(current_adj)
        
        # Kick-and-fill iterations
        seed = random.randint(0, 2**31)
        while time.time() < end_time:
            kick_size = random.choice([1, 2, 3, 5, 8, 12])
            new_adj = kick_and_fill(current_adj[:], kick_size, seed)
            seed += 1
            new_edges = count_edges(new_adj)
            if new_edges > current_edges:
                current_adj = new_adj
                current_edges = new_edges
                # Try to add more
                current_adj = try_add_all(current_adj)
                current_edges = count_edges(current_adj)
        
        return current_adj, current_edges
    
    best_adj = [0] * n
    best_edges = 0
    
    # Phase 1: Multiple greedy restarts
    n_restarts = 200
    for i in range(n_restarts):
        if time.time() > deadline - 40:
            break
        adj = greedy_construct(i)
        adj = try_add_all(adj)
        edges = count_edges(adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
    
    # Phase 2: Local search with kick-and-fill on best
    if time.time() < deadline - 10:
        remaining = deadline - time.time() - 5
        adj, edges = local_search(best_adj, remaining)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
    
    # Phase 3: Final polish - try all additions
    if time.time() < deadline - 3:
        best_adj = try_add_all(best_adj)
        best_edges = count_edges(best_adj)
    
    result_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                result_edges.append([i, j])
                
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
