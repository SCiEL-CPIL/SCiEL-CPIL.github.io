import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    # 1. Construct the q=5 Polarity Graph (n=31)
    # Points in PG(2, 5)
    # Field F5 = {0, 1, 2, 3, 4}
    # Points:
    # 0-24: (x, y, 1) for x,y in 0..4. idx = x*5 + y
    # 25-29: (x, 1, 0) for x in 0..4. idx = 25 + x
    # 30: (1, 0, 0)
    
    points = []
    for x in range(5):
        for y in range(5):
            points.append((x, y, 1))
    for x in range(5):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    n_base = 31
    # Adjacency list and bitmask
    adj = [set() for _ in range(n_base)]
    adj_mask = [0] * n_base
    
    for i in range(n_base):
        p1 = points[i]
        for j in range(i + 1, n_base):
            p2 = points[j]
            # Polarity: x1x2 + y1y2 + z1z2 = 0
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if dot == 0:
                adj[i].add(j)
                adj[j].add(i)
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    # Helper to check C4-free using bitmasks
    def is_c4_free(adj, n):
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    # Common neighbors of u and v
                    common = adj_mask[u] & adj_mask[v]
                    if common != 0:
                        # Check if more than 1
                        # Popcount
                        if common & (common - 1):
                            return False
        return True

    # Helper to check if independent set of size k exists
    def has_independent_set(adj, adj_mask, n, k, time_limit=None):
        if k <= 0:
            return True
        if n < k:
            return False
        
        sys.setrecursionlimit(10000)
        
        def search(v, remaining_mask, size):
            if time_limit is not None and time.time() > time_limit:
                return False
            if size == 0:
                return True
            if v == n:
                return False
            # If remaining bits < size, fail
            if bin(remaining_mask).count('1') < size:
                return False
            
            # Option 1: Include v
            if (remaining_mask & (1 << v)) != 0:
                new_remaining = remaining_mask & ~(adj_mask[v] | (1 << v))
                if search(v + 1, new_remaining, size - 1):
                    return True
            
            # Option 2: Exclude v
            if search(v + 1, remaining_mask, size):
                return True
            
            return False
        
        all_mask = (1 << n) - 1
        return search(0, all_mask, k)

    # Verify base graph
    if not is_c4_free(adj, n_base):
        # Fallback: 30 vertex graph (10 triangles)
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    if has_independent_set(adj, adj_mask, n_base, 11, time_limit=deadline - 10):
        # Fallback: 30 vertex graph
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    # Base graph is valid. n=31.
    best_n = 31
    best_edges = []
    for u in range(n_base):
        for v in adj[u]:
            if u < v:
                best_edges.append([u, v])

    # Try to extend the graph to n=32..39
    # Strategy: Add one vertex at a time.
    # A new vertex can be connected to a clique S in the current graph.
    # We try to find such a clique and add the vertex.
    # We must also ensure the new vertex doesn't break C4-freeness with other new vertices.
    # To keep it simple, we only add vertices connected to cliques of the *original* base graph?
    # No, as we add new vertices, the graph changes.
    
    # Let's try a randomized greedy extension.
    # We will try to add vertices 31..42.
    # For each new vertex, we try to connect it to a random subset of existing vertices
    # that forms a clique and satisfies the C4 condition.
    
    # To make this efficient, we pre-compute cliques of size 2,3,4 in the base graph?
    # Max clique size is 6.
    # We'll just try random subsets of existing neighbors.
    
    for target_n in range(32, 40):
        if time.time() > deadline - 20:
            break
            
        found = False
        for seed in range(20):
            if time.time() > deadline - 15:
                break
                
            random.seed(seed + target_n)
            
            # Copy current state
            cur_adj = [set(s) for s in adj]
            cur_mask = list(adj_mask)
            cur_n = len(cur_adj)
            
            # Add new vertex
            new_v = cur_n
            cur_adj.append(set())
            cur_mask.append(0)
            
            # Try to connect new_v to some existing vertices
            # Candidate neighbors: must form a clique.
            # We pick a random clique from the existing graph.
            # Let's just pick a random vertex u, and try to add its neighbors that form a clique with u.
            # Or simpler: try to connect to a random single vertex? No, that's too sparse.
            # We want to reduce alpha.
            
            # Let's try to connect to a random clique of size k.
            # Find all cliques of size 2..6 in cur_adj.
            # This is expensive.
            # Alternative: Randomly select a subset of existing vertices, check if it's a clique,
            # and if adding new_v connected to it is valid.
            
            # Let's try to connect new_v to a random subset of 3-5 vertices.
            # Pick 5 random vertices. Check if they form a clique.
            # If yes, tentatively add edges. Check C4-free.
            
            candidates = list(range(cur_n))
            random.shuffle(candidates)
            
            # Try subsets of size 2, 3, 4, 5
            for size in [5, 4, 3, 2, 1]:
                if not candidates:
                    break
                subset = candidates[:size]
                
                # Check if subset is a clique
                is_clique = True
                for i in range(len(subset)):
                    for j in range(i+1, len(subset)):
                        if subset[j] not in cur_adj[subset[i]]:
                            is_clique = False
                            break
                    if not is_clique:
                        break
                
                if not is_clique:
                    continue
                
                # Tentatively add edges
                for u in subset:
                    cur_adj[new_v].add(u)
                    cur_adj[u].add(new_v)
                    cur_mask[new_v] |= (1 << u)
                    cur_mask[u] |= (1 << new_v)
                
                # Check C4-free
                if not is_c4_free(cur_adj, cur_n + 1):
                    # Rollback
                    for u in subset:
                        cur_adj[new_v].remove(u)
                        cur_adj[u].remove(new_v)
                        cur_mask[new_v] &= ~(1 << u)
                        cur_mask[u] &= ~(1 << new_v)
                    continue
                
                # Check alpha
                if has_independent_set(cur_adj, cur_mask, cur_n + 1, 11, time_limit=deadline - 5):
                    # Rollback
                    for u in subset:
                        cur_adj[new_v].remove(u)
                        cur_adj[u].remove(new_v)
                        cur_mask[new_v] &= ~(1 << u)
                        cur_mask[u] &= ~(1 << new_v)
                    continue
                
                # Success!
                found = True
                break
            
            if found:
                # Update best
                best_n = cur_n + 1
                best_edges = []
                for u in range(cur_n + 1):
                    for v in cur_adj[u]:
                        if u < v:
                            best_edges.append([u, v])
                adj = cur_adj
                adj_mask = cur_mask
                break # Move to next target_n
        
        if not found:
            # Try next target_n? Or break?
            # If we can't extend to target_n, we might not be able to extend further.
            # But maybe a different seed works?
            # We already tried 20 seeds.
            # Let's try next target_n anyway, maybe we can skip? No, n must be contiguous.
            # So if we fail to get to 32, we can't get to 33.
            # So we break.
            break

    return {'n': best_n, 'edges': best_edges}
