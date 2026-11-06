import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 20
    
    # adj[u] is a bitmask of neighbors of u
    adj = [0] * n
    
    # pairs_with_common[v] is a list of (a, b) such that a < b, 
    # and both a and b are neighbors of v.
    # This helps in checking triples containing v.
    # When adding edge (u, v), we need to check triples {u, a, b} 
    # where a, b are neighbors of v (so v is a common neighbor of a, b).
    # And triples {v, a, b} where a, b are neighbors of u.
    
    def update_pairs_with_common(adj, v):
        """Recompute pairs_with_common for all vertices based on current adj."""
        # This is O(n^3) if done naively, but n=20 is small.
        # Or we can update incrementally.
        # Let's just recompute for simplicity in the helper, or maintain a structure.
        # Since we add edges one by one, incremental is better.
        pass

    # Let's stick to the bitmask check but optimize the loop.
    
    def is_valid_addition(adj, u, v):
        if adj[u] & (1 << v):
            return False
        
        # We need to check if adding (u, v) creates a triple with >= 3 common neighbors.
        # This happens if there exists a triple T such that v becomes a new common neighbor of T,
        # and T already had 2 common neighbors.
        # T must contain u or v? No, T's common neighbors change only if T contains u or v?
        # No. N(T) = intersection of N(x) for x in T.
        # If T does not contain u or v, N(T) is unchanged.
        # If T contains u, N(T) gains v if v is neighbor of all other members of T.
        # If T contains v, N(T) gains u if u is neighbor of all other members of T.
        
        # So we check triples containing u: {u, a, b}
        # v is in N({u, a, b}) iff v in N(a) and v in N(b).
        # We need |N(u) & N(a) & N(b)| <= 1 (before adding u-v, v is not in N(u)).
        # Wait, if we add u-v, v IS in N(u).
        # So new N({u, a, b}) = (N(u) & N(a) & N(b)) U {v} (if v in N(a) and N(b)).
        # Actually, if v in N(a) and N(b), then v is in the intersection.
        # So new count = old count + 1.
        # We need new count <= 2 => old count <= 1.
        
        # Same for triples containing v: {v, a, b}
        # u is in N({v, a, b}) iff u in N(a) and u in N(b).
        # New count = old count + 1.
        # We need old count <= 1.
        
        # Optimization:
        # For u: iterate over pairs (a, b) of neighbors of v.
        # For each pair, check bit_count of adj[u] & adj[a] & adj[b].
        
        # Let's precompute neighbors of v as a list.
        # But adj is bitmask.
        
        # Iterate a from 0 to n-1
        # If v in adj[a]:
        #   Iterate b from a+1 to n-1
        #   If v in adj[b]:
        #      count = (adj[u] & adj[a] & adj[b]).bit_count()
        #      if count > 1: return False
        
        # This is O(n^2) per check.
        
        # We can do this for u and for v.
        
        # Check triples containing u
        # Pairs (a, b) such that v is neighbor of a and b.
        for a in range(n):
            if a == u: continue
            if not (adj[a] & (1 << v)): continue
            for b in range(a + 1, n):
                if b == u: continue
                if not (adj[b] & (1 << v)): continue
                count = (adj[u] & adj[a] & adj[b]).bit_count()
                if count > 1:
                    return False
        
        # Check triples containing v
        # Pairs (a, b) such that u is neighbor of a and b.
        for a in range(n):
            if a == v: continue
            if not (adj[a] & (1 << u)): continue
            for b in range(a + 1, n):
                if b == v: continue
                if not (adj[b] & (1 << u)): continue
                count = (adj[v] & adj[a] & adj[b]).bit_count()
                if count > 1:
                    return False
        
        return True

    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2

    def greedy_construct(random_seed):
        random.seed(random_seed)
        adj = [0] * n
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj):
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        
        # Phase 1: Add edges
        pairs = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
        random.shuffle(pairs)
        for u, v in pairs:
            if time.time() > deadline:
                break
            if is_valid_addition(current_adj, u, v):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges += 1
                
        # Phase 2: Swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            existing = [(i, j) for i in range(n) for j in range(i+1, n) if current_adj[i] & (1 << j)]
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
            random.shuffle(existing)
            random.shuffle(missing)
            
            for u, v in existing:
                if time.time() > deadline:
                    break
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                
                added = 0
                for a, b in missing:
                    if a == u or a == v or b == u or b == v:
                        continue
                    if is_valid_addition(current_adj, a, b):
                        current_adj[a] |= (1 << b)
                        current_adj[b] |= (1 << a)
                        added += 1
                        if added >= 2:
                            break
                            
                if added >= 1:
                    current_edges += added - 1
                    improved = True
                    # Rebuild missing list? Or just update?
                    # Rebuilding is O(n^2), acceptable.
                    missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
                else:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    
        # Phase 3: Final add
        pairs = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
        random.shuffle(pairs)
        for u, v in pairs:
            if time.time() > deadline:
                break
            if is_valid_addition(current_adj, u, v):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges += 1
                
        return current_adj, current_edges

    best_adj = [0] * n
    best_edges = 0
    
    # Run multiple restarts
    for i in range(500):
        if time.time() > deadline - 5:
            break
        adj = greedy_construct(i)
        adj, edges = local_search(adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
    # Polish
    if time.time() < deadline - 5:
        adj, edges = local_search(best_adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj

    result_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result_edges.append([i, j])
                
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
