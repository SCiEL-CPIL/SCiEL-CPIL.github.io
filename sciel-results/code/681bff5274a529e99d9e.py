import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    n_target = 42
    
    # Helper to compute binomial coefficient n choose k
    def nCk(n, k):
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)
        res = 1
        for i in range(k):
            res = res * (n - i) // (i + 1)
        return res

    # Precompute binomials up to n_target
    # We need binom(x, 3) for x up to n_target
    binom3 = [nCk(i, 3) for i in range(n_target + 1)]

    def count_violations(adj, n):
        """
        Counts the number of K5s and independent 5-sets.
        adj: list of bitmasks
        """
        # Count K5s
        # A K5 is a set of 5 vertices all connected.
        # We can count by iterating over all 5-subsets? No, too slow.
        # Better: Sum over all edges (u,v) of number of common neighbors C.
        # Each K5 contributes to 10 edges. For each edge in K5, the other 3 vertices are in C.
        # So sum_{u<v} binom(common_neighbors(u,v), 3) / 10 gives number of K5s?
        # Let's verify.
        # In a K5, pick an edge (u,v). There are 3 other vertices in the K5. They are all in N(u) and N(v).
        # So for that edge, binom(3,3)=1 is counted.
        # There are 10 edges in a K5. So each K5 is counted 10 times.
        # Thus, K5_count = (1/10) * sum_{u<v} binom(popcount(adj[u] & adj[v]), 3)
        
        # Similarly for Independent Sets (K5 in complement):
        # I5_count = (1/10) * sum_{u<v} binom(popcount((~adj[u]) & (~adj[v]) & mask), 3)
        
        mask = (1 << n) - 1
        k5_sum = 0
        i5_sum = 0
        
        for u in range(n):
            for v in range(u + 1, n):
                common = (adj[u] & adj[v]).bit_count()
                k5_sum += binom3[common]
                
                # Non-neighbors: 
                # adj[u] contains u? No, usually adj[i] doesn't have bit i set.
                # Let's ensure adj[i] doesn't have bit i.
                # common_non = ((~adj[u]) & (~adj[v]) & mask).bit_count()
                # But ~adj[u] in Python is infinite bits. We mask it.
                # Also, u and v are not neighbors of themselves.
                # So ~adj[u] has bit u set? No, adj[u] has bit u clear, so ~adj[u] has bit u set.
                # We want common non-neighbors excluding u and v.
                # The mask handles the range.
                # We must exclude u and v from the count of non-neighbors?
                # In the K5 count, common neighbors of u and v cannot be u or v.
                # adj[u] doesn't have bit u. adj[v] doesn't have bit v.
                # So adj[u] & adj[v] doesn't have bits u or v. Correct.
                
                # For independent set:
                # We want vertices w != u, v such that w is not connected to u and not connected to v.
                # This is equivalent to w is a neighbor of u in complement and neighbor of v in complement.
                # adj_comp[u] = mask ^ adj[u] ^ (1<<u)  (since u is not connected to u in G, it is connected to u in comp? No, loopless.)
                # Standard: adj_comp[u] = (mask ^ (1<<u)) ^ adj[u]
                # Let's just use the property:
                # non_common = (mask ^ (1<<u) ^ (1<<v) ^ adj[u] ^ adj[v]).bit_count()
                # Wait.
                # Set of vertices not connected to u: V \ ({u} U N(u))
                # Set of vertices not connected to v: V \ ({v} U N(v))
                # Intersection: V \ ({u, v} U N(u) U N(v))
                # Size = n - 2 - |N(u) U N(v)|
                # |N(u) U N(v)| = |N(u)| + |N(v)| - |N(u) & N(v)|
                # So non_common = n - 2 - (deg(u) + deg(v) - common)
                
                # Let's stick to bitmasks for safety.
                # neighbors_u = adj[u]
                # neighbors_v = adj[v]
                # non_neighbors_u = mask ^ (1<<u) ^ neighbors_u
                # non_neighbors_v = mask ^ (1<<v) ^ neighbors_v
                # common_non = (non_neighbors_u & non_neighbors_v).bit_count()
                
                # Optimization:
                # non_common = (mask ^ (1<<u) ^ (1<<v) ^ adj[u] ^ adj[v]).bit_count() is WRONG.
                # XOR is not union.
                
                # Let's use the formula:
                # deg_u = adj[u].bit_count()
                # deg_v = adj[v].bit_count()
                # non_common = n - 2 - (deg_u + deg_v - common)
                
                deg_u = adj[u].bit_count()
                deg_v = adj[v].bit_count()
                non_common = n - 2 - (deg_u + deg_v - common)
                i5_sum += binom3[non_common]
                
        k5_count = k5_sum // 10
        i5_count = i5_sum // 10
        return k5_count + i5_count, k5_count, i5_count

    def get_delta(adj, u, v, n, is_edge):
        """
        Calculate change in total violations if we flip edge (u,v).
        is_edge: True if (u,v) is currently an edge.
        """
        common = (adj[u] & adj[v]).bit_count()
        deg_u = adj[u].bit_count()
        deg_v = adj[v].bit_count()
        non_common = n - 2 - (deg_u + deg_v - common)
        
        if is_edge:
            # Removing edge
            # Destroys K5s: binom(common, 3)
            # Creates I5s: binom(non_common, 3)
            delta = binom3[non_common] - binom3[common]
        else:
            # Adding edge
            # Creates K5s: binom(common, 3)
            # Destroys I5s: binom(non_common, 3)
            delta = binom3[common] - binom3[non_common]
            
        return delta

    def local_search(adj, n, max_iters=50000):
        """
        Greedy local search with random restarts inside.
        Returns adj if valid, else None.
        """
        current_viol, _, _ = count_violations(adj, n)
        if current_viol == 0:
            return adj
            
        best_adj = adj[:]
        best_viol = current_viol
        
        # Simulated Annealing parameters
        temp = 10.0
        cooling = 0.99
        
        for i in range(max_iters):
            if time.time() > deadline:
                return None
                
            # Pick random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            is_edge = bool(adj[u] & (1 << v))
            
            delta = get_delta(adj, u, v, n, is_edge)
            
            # Accept if better, or with probability exp(-delta/temp)
            if delta < 0:
                accept = True
            else:
                if temp > 0:
                    accept = random.random() < math.exp(-delta / temp)
                else:
                    accept = False
                    
            if accept:
                # Flip edge
                if is_edge:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                current_viol += delta
                
                if current_viol < best_viol:
                    best_viol = current_viol
                    best_adj = adj[:]
                    
                if current_viol == 0:
                    return adj
                    
            temp *= cooling
            
        # If we didn't reach 0, return the best found
        if best_viol == 0:
            return best_adj
        return None

    # Try to find a 42-vertex graph
    # Start with multiple random initializations or specific seeds
    found = None
    
    # Attempt 1: Random starts
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        # Random graph
        adj = [0] * n_target
        for i in range(n_target):
            for j in range(i + 1, n_target):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        res = local_search(adj, n_target, max_iters=20000)
        if res:
            found = res
            break

    if found:
        edges = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                if found[i] & (1 << j):
                    edges.append((i, j))
        return n_target, edges

    # Fallback: Return the seed (37 vertices)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    seed_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    return q, seed_edges

