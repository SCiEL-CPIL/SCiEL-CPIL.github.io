import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 115  # Safety margin
    
    # 1. Try PG(2,5) Polarity Graph
    # Points of PG(2,5)
    # Represent points as (x, y, z) in F5^3 \ {0}, canonical form: first non-zero is 1
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Canonicalize
                if x != 0:
                    inv_x = pow(x, -1, 5)
                    points.append((1, (y * inv_x) % 5, (z * inv_x) % 5))
                elif y != 0:
                    inv_y = pow(y, -1, 5)
                    points.append((0, 1, (z * inv_y) % 5))
                else:
                    points.append((0, 0, 1))
    
    n = len(points) # Should be 31
    
    # Build adjacency based on dot product == 0
    # u . v = u[0]*v[0] + u[1]*v[1] + u[2]*v[2] % 5
    adj = [set() for _ in range(n)]
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            u = points[i]
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % 5
            if dot == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
    
    # Verify C4-free
    # For every pair (u, v), common neighbors <= 1
    c4_free = True
    for i in range(n):
        for j in range(i + 1, n):
            common = adj[i] & adj[j]
            if len(common) > 1:
                c4_free = False
                break
        if not c4_free:
            break
    
    if not c4_free:
        # Fallback to seed
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    # Check Independence Number
    # We need alpha <= 10
    # Use branch and bound
    # Order vertices by degree descending for better pruning
    degs = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: degs[x], reverse=True)
    # Remap indices? No, just use original indices but iterate in order.
    # Actually, remapping helps for bitmask if n is small, but n=31 is fine with sets.
    
    def max_independent_set_size(adj, n, limit=11):
        # Returns True if there is an independent set of size >= limit
        # Uses backtracking
        best = 0
        # We want to know if alpha >= 11
        # If we find one, we return False (invalid for our purpose)
        # If we exhaust search without finding 11, alpha < 11, return True (valid)
        
        # Heuristic: pick vertex with highest degree in remaining graph
        # But for fixed n=31, we can just do DFS.
        
        # To speed up, we can use the fact that if we have a set S,
        # the candidates are neighbors of S? No, non-neighbors.
        
        # Simple DFS
        # State: current_set, candidates (vertices not in current_set, not adjacent to any in current_set)
        
        def dfs(current_size, candidates):
            nonlocal best
            if current_size + len(candidates) <= best:
                return
            if current_size >= limit:
                best = limit
                return
            
            if not candidates:
                if current_size > best:
                    best = current_size
                return
            
            # Pick a vertex
            # Heuristic: pick vertex with highest degree in induced subgraph on candidates
            # Or just pick the first one
            # Let's pick one with max degree in candidates to prune faster
            # deg_in_cand = {v: len(adj[v] & candidates) for v in candidates}
            # v = max(deg_in_cand, key=deg_in_cand.get)
            
            # For n=31, picking the first is okay, but let's be smart.
            # Convert candidates to list
            cand_list = list(candidates)
            
            # Sort by degree in candidates
            # This is expensive if done every step.
            # Given small n, it's fine.
            
            # Let's just pick the first one for simplicity in this tight loop, 
            # but we can improve if it's slow.
            v = cand_list[0]
            
            # Branch 1: Include v
            new_candidates = candidates - {v} - adj[v]
            dfs(current_size + 1, new_candidates)
            
            if best >= limit:
                return
            
            # Branch 2: Exclude v
            new_candidates = candidates - {v}
            dfs(current_size, new_candidates)
            
        dfs(0, set(range(n)))
        return best < limit

    # Check if alpha < 11
    # If alpha >= 11, invalid.
    # We run the check.
    valid = max_independent_set_size(adj, n, limit=11)
    
    if valid:
        return {'n': n, 'edges': edges}
    else:
        # PG(2,5) has alpha >= 11.
        # Try to find a subgraph? No, we want max n.
        # Try a greedy construction for n=31 or higher?
        # For now, fallback to seed if this fails, but it likely succeeds.
        # If it fails, we might try to remove vertices? No, that reduces n.
        # We can try a random graph search for n=31.
        
        # Let's try a simple greedy extension of a smaller C4-free graph.
        # But given time, fallback to seed is safer if algebraic fails.
        # However, PG(2,5) is very likely to have small alpha.
        
        # Just in case, return seed
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

# EVOLVE-BLOCK-START
def construct():
    # Use the function defined above
    return _construct_impl()

def _construct_impl():
    start_time = time.time()
    
    # 1. Try PG(2,5) Polarity Graph
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, -1, 5)
                    points.append((1, (y * inv_x) % 5, (z * inv_x) % 5))
                elif y != 0:
                    inv_y = pow(y, -1, 5)
                    points.append((0, 1, (z * inv_y) % 5))
                else:
                    points.append((0, 0, 1))
    
    n = len(points)
    
    adj = [set() for _ in range(n)]
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            u = points[i]
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % 5
            if dot == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
    
    # Verify C4-free
    c4_free = True
    for i in range(n):
        for j in range(i + 1, n):
            if len(adj[i] & adj[j]) > 1:
                c4_free = False
                break
        if not c4_free:
            break
    
    if not c4_free:
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    # Check Independence Number
    # We need alpha < 11
    # Fast check
    def has_independent_set_size_k(adj, n, k):
        # Returns True if there is an independent set of size k
        # Uses DFS with pruning
        # Order vertices by degree
        degs = [len(adj[i]) for i in range(n)]
        # Simple heuristic: if sum of degrees is low, alpha is high?
        # No, let's just run DFS.
        
        # To make it faster, we can use bitmasks if n <= 64.
        # n=31 fits in 32-bit int.
        
        # Precompute neighbor masks
        masks = [0] * n
        for i in range(n):
            m = 0
            for j in adj[i]:
                m |= (1 << j)
            masks[i] = m
        
        all_mask = (1 << n) - 1
        
        def dfs(candidates, current_size):
            if current_size == k:
                return True
            if not candidates:
                return False
            # Pruning: if current_size + popcount(candidates) < k, return False
            # Python int.bit_count() is fast
            if current_size + bin(candidates).count('1') < k:
                return False
            
            # Pick a vertex
            # Heuristic: pick vertex with max degree in candidates
            # To avoid iterating all, just pick the lowest set bit?
            # Or iterate and find max degree?
            # For n=31, iterating is fine.
            
            best_v = -1
            max_deg = -1
            temp = candidates
            while temp:
                # Get lowest set bit
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                # Count neighbors in candidates
                # neighbors = masks[v] & candidates
                # deg = bin(neighbors).count('1')
                deg = (masks[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp -= lsb
            
            v = best_v
            bit_v = 1 << v
            
            # Branch 1: Include v
            new_cands = candidates & ~(bit_v | masks[v])
            if dfs(new_cands, current_size + 1):
                return True
            
            # Branch 2: Exclude v
            new_cands = candidates & ~bit_v
            if dfs(new_cands, current_size):
                return True
            
            return False
        
        return dfs(all_mask, 0)
    
    if not has_independent_set_size_k(adj, n, 11):
        return {'n': n, 'edges': edges}
    else:
        # Fallback
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
# EVOLVE-BLOCK-END
