import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140 # 140s budget, 10s safety
    
    # 1. Generate Polarity Graph of PG(2,6)
    # Points are non-zero vectors in F_6^3 modulo scaling.
    # We need a specific enumeration.
    q = 6
    # Generate all non-zero vectors
    vecs = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize to ensure unique representation
                # Find first non-zero coordinate, scale it to 1
                # Since q=6 is not prime, we must be careful with inverses.
                # 1,5 are units. 2,3,4 are not.
                # Actually, we just need a set of representatives.
                # We can just take all vectors and deduplicate by checking if one is a multiple of another.
                pass
    
    # Better enumeration for PG(2,q)
    # Representatives:
    # Type 1: (1, a, b) for a,b in F_q
    # Type 2: (0, 1, a) for a in F_q
    # Type 3: (0, 0, 1)
    points = []
    for a in range(q):
        for b in range(q):
            points.append((1, a, b))
    for a in range(q):
        points.append((0, 1, a))
    points.append((0, 0, 1))
    
    n = len(points) # Should be 43
    
    # Define Polarity
    # We use the quadratic form Q(x,y,z) = xz - y^2
    # The polar form B(u,v) = u_x v_z + u_z v_x - 2 u_y v_y
    # In F_6, 2 is not invertible. This might be degenerate.
    # Let's use a standard non-degenerate polarity for q even.
    # For q even, the standard symplectic form is not symmetric.
    # We need a symmetric bilinear form B(u,v) such that B(u,u) is a quadratic form.
    # Let B(u,v) = u_x v_z + u_z v_x + u_y v_y.
    # B(u,u) = 2 u_x u_z + u_y^2.
    # In F_6, 2 u_x u_z + u_y^2 is a valid quadratic form.
    # Two points u, v are adjacent if B(u,v) == 0.
    
    def dot_product(u, v):
        # B(u,v) = u[0]*v[2] + u[2]*v[0] + u[1]*v[1]
        return (u[0]*v[2] + u[2]*v[0] + u[1]*v[1]) % q
        
    # Build Adjacency List
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if dot_product(points[i], points[j]) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Verify C4-free
    # For any pair u,v, |N(u) & N(v)| <= 1
    # This should hold by construction for polarity graphs of projective planes.
    # But let's verify to be safe.
    is_c4_free = True
    for u in range(n):
        for v in range(u+1, n):
            common = adj[u] & adj[v]
            # Count bits
            # Python int bit_count is fast
            if common.bit_count() > 1:
                is_c4_free = False
                break
        if not is_c4_free:
            break
            
    if not is_c4_free:
        # Fallback to random search if construction fails (should not happen)
        return construct_fallback()

    # Check Independent Set of size 11
    # Use bitmask backtracking
    def has_independent_set_k(adj_list, n, k):
        # adj_list is list of ints
        # We want to find a set of k vertices with no edges between them.
        # This is equivalent to finding a clique of size k in the complement graph.
        # Or just backtracking on independent set.
        
        # Optimization: Order vertices by degree?
        # For bitmask, we can use the following:
        # candidates = (1 << n) - 1
        # current_set_size = 0
        
        # We use a recursive helper
        def backtrack(candidates, depth):
            if depth == k:
                return True
            if candidates == 0:
                return False
            # Pruning: if popcount(candidates) < k - depth: return False
            if candidates.bit_count() < k - depth:
                return False
                
            # Pick a vertex v from candidates
            # Heuristic: pick the one with fewest neighbors in candidates?
            # Or just the lowest set bit.
            v = (candidates & -candidates).bit_length() - 1
            v_bit = 1 << v
            
            # Branch 1: Include v
            # New candidates = candidates & ~adj_list[v] & ~v_bit
            # Note: adj_list[v] contains neighbors. We must exclude them.
            # Also exclude v itself.
            new_cands = candidates & ~adj_list[v] & ~v_bit
            if backtrack(new_cands, depth + 1):
                return True
                
            # Branch 2: Exclude v
            new_cands2 = candidates & ~v_bit
            if backtrack(new_cands2, depth):
                return True
                
            return False
            
        return backtrack((1 << n) - 1, 0)

    if has_independent_set_k(adj, n, 11):
        # The 43-vertex graph has an independent set of 11.
        # We need to reduce n to 39.
        # Try removing 4 vertices.
        # We want to remove a set S of 4 vertices such that G-S has no independent set of 11.
        # This is hard to search directly.
        # Instead, let's try to find a subgraph of size 39 that is valid.
        # We can try random subsets of 39 vertices.
        
        found = False
        # Try 1000 random subsets of 39 vertices
        for _ in range(1000):
            if time.time() > deadline:
                break
                
            # Select 39 vertices randomly
            # We can sample 4 vertices to remove
            removed = random.sample(range(n), 4)
            removed_set = set(removed)
            
            # Build new adjacency for the remaining 39 vertices
            # Map old indices to new indices 0..38
            mapping = {}
            new_n = 0
            for i in range(n):
                if i not in removed_set:
                    mapping[i] = new_n
                    new_n += 1
            
            new_adj = [0] * new_n
            for old_u in range(n):
                if old_u in removed_set:
                    continue
                new_u = mapping[old_u]
                # Iterate over neighbors of old_u
                # adj[old_u] is a bitmask
                # We need to filter neighbors not in removed_set
                mask = adj[old_u]
                # Iterate through set bits
                m = mask
                while m:
                    low_bit = m & -m
                    old_v = low_bit.bit_length() - 1
                    m -= low_bit
                    if old_v not in removed_set:
                        new_v = mapping[old_v]
                        new_adj[new_u] |= (1 << new_v)
                        
            # Check C4-free (should be inherited)
            # Check Independent Set 11
            if not has_independent_set_k(new_adj, new_n, 11):
                # Valid graph found
                # Construct edge list
                edges = []
                for u in range(new_n):
                    m = new_adj[u]
                    while m:
                        low_bit = m & -m
                        v = low_bit.bit_length() - 1
                        m -= low_bit
                        if u < v:
                            edges.append([u, v])
                return {'n': new_n, 'edges': edges}
                
        # If no random subset worked, try a greedy approach?
        # Or just return the best known smaller graph?
        # The task says best known is 39. If we can't find a 39-vertex one quickly, 
        # we might return a 38 or smaller one.
        # But 39 is the target.
        
        # Let's try a more systematic removal: 
        # Remove vertices that are in many maximum independent sets?
        # This is complex.
        
        # For now, if we fail to find a 39-vertex subgraph, we fall back to a known valid smaller graph.
        # However, the 43-vertex graph itself is C4-free. 
        # If we just return a subgraph of size 39, it is C4-free.
        # The only risk is the independent set.
        
    else:
        # The 43-vertex graph does NOT have an independent set of 11.
        # Then we can return the 43-vertex graph?
        # Wait, the task says "best known construction has 39 vertices".
        # This implies R(C4, K11) <= 43 is known, but maybe > 39 is the max we can certify?
        # If the 43-vertex graph has no independent set of 11, then R(C4, K11) > 43.
        # This would be a new record!
        # But it's unlikely. Polarity graphs usually have large independent sets.
        
        # Construct edge list for 43 vertices
        edges = []
        for u in range(n):
            m = adj[u]
            while m:
                low_bit = m & -m
                v = low_bit.bit_length() - 1
                m -= low_bit
                if u < v:
                    edges.append([u, v])
        return {'n': n, 'edges': edges}

def construct_fallback():
    # Return a valid graph, e.g., n=30 seed
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

# Let's refine the independent set check to be faster.
# The recursive call `backtrack` creates new sets every time. This might be slow in Python.
# Bitmasks are faster for $n \le 64$.
