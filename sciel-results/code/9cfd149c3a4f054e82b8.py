import random
import time
import sys

def construct():
    """
    Construct a K4-free graph with no independent set of size 6.
    Strategy: Iterated Local Search (ILS) on a fixed vertex set.
    1. Generate random regular graphs (Configuration Model) for various n.
    2. Use bitset-based fast violation counting (K4s and Ind-6s).
    3. Perform local search: flip edges to reduce violation count.
    4. If stuck, perturb and restart.
    """
    deadline = time.time() + 115.0
    rng = random.Random(42)
    
    # We want to maximize n.
    # Known lower bound is 35. We will try to find 36 or higher.
    # If we can't find 36 quickly, we fall back to 35.
    
    # Precompute powers of 2 for fast bit manipulation if needed, 
    # but Python ints are arbitrary precision, so bit ops are fast.
    
    def count_k4s(n, adj):
        """
        Count the number of K4s in the graph using bitsets.
        adj[i] is a bitmask of neighbors of i.
        """
        count = 0
        for i in range(n):
            # To count each K4 exactly once, we can iterate over edges (i, j) with i < j
            # and count common neighbors k, l with k, l > j? 
            # Actually, simpler:
            # For each pair (i, j) with i < j and edge (i, j) exists:
            #   Common neighbors C = adj[i] & adj[j]
            #   For each pair (k, l) in C with i < j < k < l? No, that's complex.
            
            # Let's use the standard triangle counting extension.
            # A K4 is a set of 4 vertices {a,b,c,d} all connected.
            # We can count by iterating vertices a, and for each a,
            # looking at the subgraph induced by N(a).
            # Any triangle in N(a) forms a K4 with a.
            
            # To avoid overcounting, let's only count K4s where 'a' is the smallest index.
            # So for a given a, we look at neighbors b, c, d such that a < b < c < d.
            # This is O(n * (deg(a))^3) which might be slow if degrees are high.
            
            # Alternative: Iterate over all pairs (i, j) with i < j.
            # If (i, j) is an edge:
            #   C = adj[i] & adj[j] & mask_greater_than_j
            #   Any edge (k, l) within C forms a K4 {i, j, k, l}.
            #   So count edges in the subgraph induced by C.
            
            # Let's optimize:
            # Precompute for each i, the mask of vertices > i.
            # But n is up to 40, so we can just use shifts.
            
            for i in range(n):
                # Neighbors of i greater than i
                # We want to form triangles (i, j, k) with i < j < k.
                # Then for each such triangle, count common neighbors l > k?
                # No, K4 is {i, j, k, l}.
                
                # Let's stick to:
                # For i in 0..n-1:
                #   For j in i+1..n-1 where (i,j) is edge:
                #     C = adj[i] & adj[j] & mask(j+1..n-1)
                #     Count edges in subgraph induced by C.
                #     Each edge (k, l) in C corresponds to K4 {i, j, k, l}.
                
                # This counts each K4 exactly once?
                # Let K4 be {a, b, c, d} with a < b < c < d.
                # It is counted when i=a, j=b.
                # C includes c, d.
                # Edge (c, d) is in C.
                # Is it counted elsewhere?
                # If i=a, j=c: C includes b, d. But b < c, so b is not in mask(c+1..). So b not in C.
                # If i=b, j=c: i > a, but we only count if we iterate i from 0.
                # Wait, the loop `for i in range(n)` iterates all i.
                # The condition `mask(j+1..n-1)` ensures k, l > j.
                # So for K4 {a, b, c, d}:
                # Pairs (i, j) from the set:
                # (a, b): C has {c, d}. Edge (c, d) exists. Count 1.
                # (a, c): C has {d} (b is < c, excluded). No edge in {d}. Count 0.
                # (a, d): C has {}. Count 0.
                # (b, c): i=b. C has {d} (a is < b, excluded). Count 0.
                # (b, d): C has {}. Count 0.
                # (c, d): C has {}. Count 0.
                # So yes, each K4 is counted exactly once.
                
                for j in range(i + 1, n):
                    if (adj[j] >> i) & 1:
                        # Edge (i, j) exists
                        # Common neighbors greater than j
                        # mask = ~((1 << (j + 1)) - 1)
                        C = adj[i] & adj[j] & (~((1 << (j + 1)) - 1))
                        # Count edges in subgraph induced by C
                        # Iterate through bits in C
                        c_mask = C
                        while c_mask:
                            low = c_mask & -c_mask
                            k = low.bit_length() - 1
                            # Check if k is connected to any other in C
                            # Neighbors of k in C:
                            neighbors_in_c = adj[k] & C & (~((1 << (k + 1)) - 1))
                            # Count bits in neighbors_in_c
                            count += bin(neighbors_in_c).count('1')
                            c_mask ^= low
        return count

    def count_ind6s(n, adj):
        """
        Count the number of independent sets of size 6.
        Equivalent to counting K6s in the complement graph.
        """
        # Build complement adjacencies
        comp_adj = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp_adj[i] = all_mask & ~(adj[i] | (1 << i))
            
        return count_k4s_in_comp(n, comp_adj) # Reuse logic? No, count_k4s counts K4s. We need K6s.
        # Wait, independent set of size 6 is K6 in complement.
        # My count_k4s function counts K4s. I need a function to count K6s.
        # Let's generalize count_cliques(n, adj, size).
        
    def count_cliques(n, adj, size):
        """
        Count cliques of given size using the method:
        For each vertex v, count cliques of size-1 in N(v) with all vertices > v.
        This is recursive and might be slow for size=6.
        """
        # For size=6, this is heavy.
        # Alternative: Use Bron-Kerbosch to find ALL maximal independent sets?
        # Or just check if max independent set >= 6?
        # The task requires "no independent set of 6".
        # So we just need to know if alpha(G) >= 6.
        # For local search, we need a gradient.
        # Counting the number of independent 6-sets is a good gradient.
        
        # Let's implement a specialized counter for K6 in complement.
        # Or, simpler: Just find IF there is an ind-6.
        # For local search, we can use "number of edges in Ind-6s" or just "0 if valid, 1 if invalid"?
        # That's a binary signal, too weak for local search.
        
        # Let's count the number of independent 6-sets.
        # This is equivalent to counting K6s in the complement.
        
        # Efficient way:
        # Iterate over all 6-combinations? C(40,6) is ~3.8M. Too slow for each step.
        
        # Let's use the recursive clique counting on the complement.
        # Count K6 in G_comp.
        
        # Function to count K_k in a graph given by adj masks.
        def count_kk(n, adj, k):
            if k == 2:
                # Count edges
                c = 0
                for i in range(n):
                    # Count neighbors > i
                    c += bin(adj[i] & (~((1 << (i + 1)) - 1))).count('1')
                return c
            
            if k == 1:
                return n
                
            total = 0
            for i in range(n):
                # Consider cliques where i is the smallest vertex
                # Neighbors of i greater than i
                N_i = adj[i] & (~((1 << (i + 1)) - 1))
                # Build subgraph on N_i
                # We need to count K_{k-1} in this subgraph.
                # The vertices in N_i are not contiguous, so we can't just call count_kk directly on a smaller n.
                # We need a version that works on a subset of vertices.
                
                # Let's map N_i to a contiguous range 0..m-1.
                verts = []
                mask = N_i
                while mask:
                    low = mask & -mask
                    v = low.bit_length() - 1
                    verts.append(v)
                    mask ^= low
                
                m = len(verts)
                if m < k - 1:
                    continue
                
                # Build adj masks for this subgraph
                sub_adj = [0] * m
                idx_map = {v: idx for idx, v in enumerate(verts)}
                for idx, v in enumerate(verts):
                    # Neighbors of v in the subgraph
                    # These are neighbors of v in original graph that are also in verts
                    # And > v? No, in the subgraph, we just need the internal edges.
                    # The "smallest vertex" logic in the recursive call will handle the ordering.
                    # So sub_adj[idx] should contain bits for other vertices in verts that are connected to v.
                    neighbors_in_sub = adj[v] & N_i # N_i is mask of verts
                    # Convert this mask to subgraph indices
                    sub_mask = 0
                    temp = neighbors_in_sub
                    while temp:
                        low = temp & -temp
                        u = low.bit_length() - 1
                        if u in idx_map:
                            sub_mask |= (1 << idx_map[u])
                        temp ^= low
                    sub_adj[idx] = sub_mask
                
                total += count_kk(m, sub_adj, k - 1)
            return total

        # This recursive approach is O(n * (deg)^(k-1)) roughly.
        # For n=40, k=6, deg~20. 40 * 20^5 = 128M. Too slow in Python.
        
        # We need a faster check for "Independent Set of 6".
        # Since we are doing local search, maybe we can just check validity?
        # If we use a binary penalty (0 if valid, large number if invalid), local search might get stuck.
        # However, if we start from a random regular graph, it's likely invalid.
        
        # Alternative: Use a SAT solver? No, stdlib only.
        
        # Let's try to optimize the K6 count in complement.
        # Or, use a heuristic: Count the number of independent 5-sets that can be extended?
        
        # Actually, for the purpose of this task, maybe we can just use `max_independent_set_size < 6` as the goal,
        # and use a simpler penalty: number of K4s + (1 if alpha >= 6 else 0).
        # But the gradient for alpha is 0 almost everywhere.
        
        # Let's try a different penalty: 
        # Penalty = (# K4s) + 100 * (# of edges missing from Ind-6s) ? No.
        
        # How about: Penalty = (# K4s) + 10 * (number of independent 5-sets that are NOT extendable to 6)?
        # No, we want to MINIMIZE independent 6-sets.
        
        # Let's implement a fast check for "Is there an independent set of 6?"
        # Bron-Kerbosch with pivot is fast for n=40.
        # We can use it in the local search loop.
        # If we find an Ind-6, we add a penalty.
        # To get a gradient, we can count how many Ind-6s exist?
        # Or just pick one random Ind-6 and try to break it.
        
        # Strategy:
        # 1. Count K4s (fast enough with bitsets? Let's check. n=40, edges~400.
        #    Loop i, j. 800 pairs. Inner loop counts edges in C.
        #    Average C size ~ 10. Edges in C ~ 25.
        #    800 * 25 = 20k ops. Very fast.
        # 2. Check for Ind-6.
        #    If found, penalty += 1000.
        #    If not, penalty = K4_count.
        
        # This binary penalty for Ind-6 is weak.
        # Better: Count the number of independent 6-sets using a limited Bron-Kerbosch?
        # Or, count the number of "near" independent 6-sets?
        
        # Let's try: Penalty = K4_count + 100 * (1 if alpha >= 6 else 0).
        # And in local search, if alpha >= 6, we specifically target edges within the found Ind-6.
        
    def find_ind6(n, adj):
        """Find one independent set of size 6, or None."""
        comp_adj = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp_adj[i] = all_mask & ~(adj[i] | (1 << i))
        
        found = [None]
        
        def bk(R, P, X):
            if found[0] is not None:
                return
            if len(R) == 6:
                found[0] = tuple(R)
                return
            if not P:
                return
            if len(R) + bin(P).count("1") < 6:
                return
            # Pivot
            if X:
                u_bit = X & -X
                u = u_bit.bit_length() - 1
                candidates = P & ~comp_adj[u]
            else:
                candidates = P
            c = candidates
            while c and found[0] is None:
                low = c & -c
                v = low.bit_length() - 1
                R.append(v)
                bk(R, P & comp_adj[v], X & comp_adj[v])
                R.pop()
                P = P & ~low
                X = X | low
                c = c & ~low
        
        bk([], (1 << n) - 1, 0)
        return found[0]

    def penalty(n, adj):
        k4_count = count_k4s(n, adj)
        ind6 = find_ind6(n, adj)
        if ind6:
            return k4_count + 1000
        return k4_count

    def generate_random_regular(n, d, rng):
        """Generate a random d-regular graph using configuration model."""
        if d * n % 2 != 0:
            return None
        if d >= n:
            return None
            
        stubs = []
        for i in range(n):
            for _ in range(d):
                stubs.append(i)
        
        rng.shuffle(stubs)
        
        adj = [0] * n
        edges = set()
        
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                # Self loop, retry or swap
                # Simple fix: find another stub
                # For simplicity, if self-loop, skip and try to fix later?
                # Or just accept and remove? No, must be simple graph.
                # Let's try to swap with a random other stub.
                j = rng.randint(0, len(stubs)//2 - 1) * 2
                if j == i: continue
                if stubs[j] != u and stubs[j+1] != u:
                    # Swap v with stubs[j]
                    stubs[i+1], stubs[j] = stubs[j], stubs[i+1]
                    v = stubs[i+1]
                    if u == v: continue
            if (u, v) in edges or (v, u) in edges:
                # Multi-edge, skip or fix
                continue
                
            edges.add((u, v))
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # Check if it's actually d-regular? 
        # Configuration model with swaps usually yields simple graphs if n is large enough.
        # For n=36, d=18, it should be fine.
        
        # Verify degrees
        for i in range(n):
            if bin(adj[i]).count('1') != d:
                return None
                
        return adj

    def local_search(n, adj, rng, max_steps=5000):
        """
        Local search: flip edges to reduce penalty.
        """
        current_pen = penalty(n, adj)
        if current_pen == 0:
            return adj
        
        for step in range(max_steps):
            if time.time() > deadline - 10:
                break
                
            # Pick a random edge to flip
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
                
            # Flip edge (i, j)
            if (adj[j] >> i) & 1:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
            else:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
            new_pen = penalty(n, adj)
            
            if new_pen < current_pen:
                current_pen = new_pen
                if current_pen == 0:
                    return adj
            else:
                # Revert
                if (adj[j] >> i) & 1:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        return adj

    def perturb(n, adj, rng, num_flips=10):
        for _ in range(num_flips):
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j: continue
            if (adj[j] >> i) & 1:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
            else:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        return adj

    best_n = 0
    best_adj = None
    
    # Try to find a valid graph for n=36, 35, etc.
    # Start with higher n, fall back to lower.
    target_ns = [36, 35, 34, 33, 32, 31, 30]
    
    for n in target_ns:
        if time.time() > deadline - 20:
            break
            
        found = False
        attempts = 0
        max_attempts = 50 # Limit attempts per n to avoid getting stuck
        
        while attempts < max_attempts and not found and time.time() < deadline - 10:
            attempts += 1
            
            # Try different degrees
            # For n=36, avg degree for R(4,6) should be around n/2 = 18.
            # Let's try d=17, 18, 19.
            for d in [17, 18, 19, 16, 20]:
                if time.time() > deadline - 5:
                    break
                    
                adj = generate_random_regular(n, d, rng)
                if adj is None:
                    continue
                
                # Check initial penalty
                pen = penalty(n, adj)
                
                if pen == 0:
                    found = True
                    break
                
                # Local search
                adj = local_search(n, adj, rng, max_steps=2000)
                pen = penalty(n, adj)
                
                if pen == 0:
                    found = True
                    break
                
                # If stuck, perturb and try again?
                # Or just break and try next d.
                
            if found:
                break
            
        if found:
            best_n = n
            best_adj = adj
            break # Found a valid graph for this n, stop searching higher n? 
            # Wait, we iterate n from high to low. So if we find 36, we stop.
            # If we don't find 36, we try 35.
            
    # If we didn't find anything (unlikely), return a small valid graph
    if best_adj is None:
        # Fallback: Paley(17) or similar
        # Let's just construct a small one manually or use a known valid small graph.
        # K4-free, alpha < 6.
        # n=10?
        # Let's just return n=1 if all else fails, but we should do better.
        # Let's try n=17 Paley again as a last resort.
        q = 17
        squares = {x * x % q for x in range(1, q)}
        n0 = q
        A = [0] * n0
        for i in range(n0):
            for j in range(n0):
                if i != j and (j - i) % q in squares:
                    A[i] |= (1 << j)
        best_n = n0
        best_adj = A
        
    # Convert to edges
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[j] >> i) & 1:
                edges.append([i, j])
                
    return {'n': best_n, 'edges': edges}
