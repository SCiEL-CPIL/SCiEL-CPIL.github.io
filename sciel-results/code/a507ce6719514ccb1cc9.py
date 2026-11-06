import time
import random
import numpy as np

def construct():
    """
    Construct a C4-free graph with alpha < 11, maximizing n.
    Strategy:
    1. Deterministically construct the polarity graph of PG(2, 5). This gives n=31.
    2. Verify it is C4-free and has alpha < 11.
    3. Try to extend this graph to n=32, ..., 39 by adding vertices and edges
       while maintaining C4-freeness and alpha < 11.
    4. Use a local search to maximize the number of vertices.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def time_left():
        return deadline - time.time()

    def popcount(x):
        return bin(x).count('1')

    def max_independent_set_size(n, adj):
        """
        Find the size of the maximum independent set using branch and bound.
        adj is a list of integers (bitmasks).
        Returns the size.
        """
        best = [0]
        
        def branch(candidates_mask, current_size):
            if time_left() < 5:
                return
            if candidates_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Upper bound
            if current_size + popcount(candidates_mask) <= best[0]:
                return
            
            # Heuristic: Pick vertex with highest degree in the candidate subgraph
            best_v = -1
            best_deg = -1
            m = candidates_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = popcount(candidates_mask & adj[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                m -= lsb
            
            if best_v == -1:
                return

            v_bit = 1 << best_v
            
            # Option 1: Include best_v
            new_candidates = candidates_mask & ~adj[best_v] & ~v_bit
            branch(new_candidates, current_size + 1)
            
            # Option 2: Exclude best_v
            branch(candidates_mask & ~v_bit, current_size)

        branch((1 << n) - 1, 0)
        return best[0]

    def is_c4_free(n, adj):
        """Check if graph is C4-free."""
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    if popcount(common) > 1:
                        return False
        return True

    def construct_polarity_graph():
        """
        Construct the polarity graph of PG(2, 5).
        This gives a C4-free graph on 31 vertices.
        """
        # Points in PG(2, 5)
        points = []
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    # Normalize
                    if x != 0:
                        inv_x = pow(x, -1, 5)
                        points.append((1, y * inv_x % 5, z * inv_x % 5))
                    elif y != 0:
                        inv_y = pow(y, -1, 5)
                        points.append((0, 1, z * inv_y % 5))
                    else:
                        points.append((0, 0, 1))
        points = list(set(points))
        n = len(points)
        
        # Bilinear form for polarity
        def bilinear(u, v):
            return (u[0]*v[2] + u[1]*v[1] + u[2]*v[0]) % 5
        
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if bilinear(points[i], points[j]) == 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        return n, adj

    def can_add_edge(adj, i, j, n):
        """Check if adding edge (i,j) maintains C4-freeness."""
        if adj[i] & (1 << j):
            return True  # Already an edge
        
        # For C4-freeness, we need |N(i) ∩ N(j)| <= 1
        common = adj[i] & adj[j]
        if popcount(common) > 1:
            return False
        
        # Also need to check that no new C4 is formed through other paths
        # The condition |N(a) ∩ N(b)| <= 1 for all pairs a,b is necessary and sufficient
        # But we need to verify that adding (i,j) doesn't violate this for other pairs
        
        # Check for all k in N(i): |N(k) ∩ N(j)| <= 1 after adding (i,j)
        # Currently |N(k) ∩ N(j)| <= 1, but after adding (i,j), i becomes in N(j)
        # So we need |N(k) ∩ N(j) ∪ {i}| <= 1
        # This means i should not be in N(k) ∩ N(j) unless N(k) ∩ N(j) is empty
        
        # Actually, the simpler check: for any two vertices a,b that are both neighbors of some vertex,
        # they should share at most one common neighbor. Adding edge (i,j) creates new potential C4s
        # of the form i-a-j-b-i where a,b are common neighbors of i and j.
        # We already checked |N(i) ∩ N(j)| <= 1, so no such C4 exists.
        
        # But we also need to check that for any vertex k, the neighbors of k still satisfy the C4-free condition.
        # The edge (i,j) affects the neighborhoods of i and j only.
        # For any other vertex k, N(k) is unchanged, so the C4-free condition for pairs not involving i or j is fine.
        # For pairs involving i: (i, k) for k in N(i) or k = j
        # For pairs involving j: (j, k) for k in N(j) or k = i
        
        # Let's check pairs (i, k) for k in N(i):
        # After adding (i,j), N(j) gains i. So for pair (i, k), we need |N(i) ∩ N(k)| <= 1.
        # N(i) is unchanged, N(k) is unchanged (unless k=j, but j is not in N(i) initially if we're adding it).
        # Wait, j might be in N(i)? No, we're checking if we can add (i,j), so j is not in N(i) yet.
        # So N(i) is unchanged. For k in N(i), N(k) is unchanged. So |N(i) ∩ N(k)| is unchanged.
        # But wait, we're adding edge (i,j), so N(i) gains j and N(j) gains i.
        # For pair (i, k) where k in N(i): N(i) now includes j, N(k) is unchanged.
        # We need |N(i) ∩ N(k)| <= 1. Since j is not in N(k) (unless k and j are adjacent), 
        # we need to check if j is in N(k). If j is in N(k), then j is a common neighbor of i and k,
        # but i and k were already adjacent, so this would mean |N(i) ∩ N(k)| was already >= 1
        # (from j), and we need it to be <= 1. But j was not in N(i) before, so j wasn't counted.
        # After adding, j is in N(i) and possibly in N(k). So we need to check.
        
        # This is getting complex. Let's use a simpler approach:
        # Tentatively add the edge and check all pairs involving i or j.
        
        # For efficiency, let's just check the critical condition:
        # After adding (i,j), for every pair (a,b) that shares a common neighbor, they should share at most one.
        # The only new common neighbors are i (for pairs involving j's neighbors) and j (for pairs involving i's neighbors).
        
        # Check: for each k in N(j) (before adding), i is now in N(j). So for pair (k, i), 
        # j is a common neighbor. We need |N(k) ∩ N(i)| <= 1.
        # N(i) gains j, so if j is in N(k), then j is a common neighbor of i and k.
        # But we need the total count to be <= 1.
        
        # Let's just do the full check for pairs involving i or j.
        
        ni = adj[i]  # Neighbors of i (before adding j)
        nj = adj[j]  # Neighbors of j (before adding i)
        
        # After adding: N(i) = ni ∪ {j}, N(j) = nj ∪ {i}
        
        # Check pair (i, j): |N(i) ∩ N(j)| = |(ni ∪ {j}) ∩ (nj ∪ {i})|
        # = |ni ∩ nj ∪ {i if i in nj} ∪ {j if j in ni} ∪ {i,j if both}|
        # Since i not in nj and j not in ni (we're adding the edge), 
        # = |ni ∩ nj|
        # We already checked this is <= 1.
        
        # Check pair (i, k) for k in ni:
        # |N(i) ∩ N(k)| = |(ni ∪ {j}) ∩ N(k)|
        # = |ni ∩ N(k)| + (1 if j in N(k) else 0)
        # We need this <= 1.
        # ni ∩ N(k) was already <= 1 (C4-free before).
        # If j in N(k), we need |ni ∩ N(k)| = 0.
        
        for k_bit in ni:
            k = k_bit.bit_length() - 1
            if nj & k_bit:  # j is in N(k)
                if ni & adj[k]:  # ni ∩ N(k) is non-empty
                    return False
        
        # Check pair (j, k) for k in nj:
        # |N(j) ∩ N(k)| = |(nj ∪ {i}) ∩ N(k)|
        # = |nj ∩ N(k)| + (1 if i in N(k) else 0)
        # We need this <= 1.
        # nj ∩ N(k) was already <= 1.
        # If i in N(k), we need |nj ∩ N(k)| = 0.
        
        for k_bit in nj:
            k = k_bit.bit_length() - 1
            if ni & k_bit:  # i is in N(k)
                if nj & adj[k]:  # nj ∩ N(k) is non-empty
                    return False
        
        # Check pair (i, k) for k in nj (k != i, j):
        # |N(i) ∩ N(k)| = |(ni ∪ {j}) ∩ N(k)|
        # = |ni ∩ N(k)| + (1 if j in N(k) else 0)
        # Since k in nj, j is in N(k). So we need |ni ∩ N(k)| = 0.
        
        for k_bit in nj:
            k = k_bit.bit_length() - 1
            if k == i or k == j:
                continue
            if ni & adj[k]:  # ni ∩ N(k) is non-empty
                return False
        
        # Check pair (j, k) for k in ni (k != i, j):
        # |N(j) ∩ N(k)| = |(nj ∪ {i}) ∩ N(k)|
        # = |nj ∩ N(k)| + (1 if i in N(k) else 0)
        # Since k in ni, i is in N(k). So we need |nj ∩ N(k)| = 0.
        
        for k_bit in ni:
            k = k_bit.bit_length() - 1
            if k == i or k == j:
                continue
            if nj & adj[k]:  # nj ∩ N(k) is non-empty
                return False
        
        return True

    def try_extend(n, adj, target_n):
        """
        Try to extend the graph to target_n vertices by adding new vertices.
        Returns (new_n, new_adj) if successful, else (n, adj).
        """
        new_adj = adj[:]
        new_n = n
        
        for new_v in range(n, target_n):
            if time_left() < 10:
                break
            
            # Try to find a valid position for new_v
            # new_v should be connected to some existing vertices
            # such that C4-freeness is maintained and alpha doesn't increase
            
            # Strategy: Try all subsets of existing vertices to connect to
            # Use a heuristic: start with empty set, then add edges greedily
            
            new_adj.append(0)
            
            # Try to add edges from new_v to existing vertices
            # in an order that maintains C4-freeness
            added = False
            for i in range(new_n):
                if new_adj[new_v] & (1 << i):
                    continue
                # Check if adding edge (new_v, i) creates a C4
                if can_add_edge(new_adj, new_v, i, new_n):
                    new_adj[new_v] |= (1 << i)
                    new_adj[i] |= (1 << new_v)
                    added = True
            
            if added:
                new_n += 1
            else:
                # Could not add new_v
                new_adj.pop()
                break
        
        return new_n, new_adj

    def maximize_edges(n, adj):
        """
        Add edges that do not create C4s to reduce alpha.
        """
        adj = adj[:]
        improved = True
        while improved and time_left() > 10:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        # Check if adding (i, j) creates a C4
                        if can_add_edge(adj, i, j, n):
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                            improved = True
        return adj

    # Main construction
    n, adj = construct_polarity_graph()
    
    # Verify initial graph
    if not is_c4_free(n, adj):
        # Fallback: return empty graph
        return {'n': 1, 'edges': []}
    
    alpha = max_independent_set_size(n, adj)
    if alpha >= 11:
        # Fallback
        return {'n': 1, 'edges': []}
    
    best_n = n
    best_adj = adj[:]
    
    # Try to extend
    for target_n in range(n + 1, 40):
        if time_left() < 20:
            break
        
        # Try multiple random seeds for extension
        for seed in range(20):
            if time_left() < 10:
                break
            
            random.seed(seed * 1000 + target_n)
            
            # Shuffle the order of vertices to try different extensions
            shuffled_adj = best_adj[:]
            
            new_n, new_adj = try_extend(best_n, shuffled_adj, target_n)
            
            if new_n > best_n:
                # Verify
                if is_c4_free(new_n, new_adj):
                    alpha = max_independent_set_size(new_n, new_adj)
                    if alpha < 11:
                        best_n = new_n
                        best_adj = new_adj[:]
                        break
    
    # Final edge maximization
    if time_left() > 10:
        best_adj = maximize_edges(best_n, best_adj)
        # Verify
        if is_c4_free(best_n, best_adj):
            alpha = max_independent_set_size(best_n, best_adj)
            if alpha < 11:
                pass
            else:
                # Revert if alpha increased
                best_adj = best_adj[:]  # Keep previous
    
    # Convert to edge list
    edge_list = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edge_list.append([i, j])

    return {'n': best_n, 'edges': edge_list}
