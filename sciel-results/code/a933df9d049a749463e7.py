import time
import random

def construct():
    start_time = time.time()
    time_limit = 115  # Safety margin under 120s

    # --- Fast Bitmask-based Validators ---

    def find_k4(adj, n):
        """Find a K4 in the graph. Returns tuple of 4 vertices or None."""
        # Iterate over edges (u, v)
        for u in range(n):
            # Only consider v > u to avoid duplicates and simplify logic
            # We need to find w, x such that u, v, w, x form a clique.
            # This means w, x must be in N(u) ∩ N(v) and w ~ x.
            common = adj[u] & adj[v]
            # We want w, x > v? No, just distinct from u, v.
            # To ensure we find *a* K4, we can just check if the induced subgraph on common has an edge.
            # But we must be careful not to pick u or v.
            # common contains neighbors of both u and v.
            # If common contains an edge (w, x), then {u, v, w, x} is a K4.
            
            # Optimization: We can restrict common to vertices > v to reduce checks?
            # Not strictly necessary for correctness if we just check existence of edge in common.
            # But checking all pairs in common is O(d^2).
            
            # Let's check if common has any edge.
            # If common is small, this is fast.
            # We need to ensure w != u and w != v.
            # Since u and v are not adjacent to themselves, u not in adj[u] and v not in adj[v].
            # So u not in common and v not in common.
            
            c = common
            while c:
                lsb = c & -c
                w = lsb.bit_length() - 1
                c &= c - 1
                # Check if w has any neighbor in (common & ~(1<<w))
                # Actually, if w is in common, it is neighbor to u and v.
                # We just need another vertex x in common such that w ~ x.
                # So we check if adj[w] & (common & ~(1<<w)) is non-zero.
                # Note: common already excludes w? No, common is a set of vertices.
                # adj[w] contains neighbors of w.
                # If any neighbor of w is in common (and not w itself), we have a K4.
                
                # Optimization: Only check x > w to avoid duplicate checks?
                # Doesn't matter for boolean existence.
                
                if adj[w] & (common & ~(1 << w)):
                    # Find x
                    x_mask = adj[w] & (common & ~(1 << w))
                    x_lsb = x_mask & -x_mask
                    x = x_lsb.bit_length() - 1
                    return (u, v, w, x)
        return None

    def find_i6(adj, n):
        """Find an independent set of size 6. Returns list of 6 vertices or None."""
        # Backtracking search in the complement graph for K6.
        # Independent set in G = Clique in G_bar.
        # G_bar edge (u, v) exists if not (adj[u] & (1 << v)).
        
        # Heuristic: Order vertices by degree in G_bar (descending).
        # High degree in G_bar means low degree in G.
        # Low degree in G means high degree in G_bar.
        # Vertices with high degree in G_bar are more likely to be in a clique.
        
        degs = [0] * n
        for i in range(n):
            degs[i] = adj[i].bit_count()
        
        # Degree in complement = (n - 1) - degs[i]
        # Sort indices by comp_degree descending
        order = sorted(range(n), key=lambda x: (n - 1 - degs[x]), reverse=True)
        
        # Map vertex -> position in order?
        # We will use a list of candidate vertices.
        
        def backtrack(candidates, current_set):
            if len(current_set) == 6:
                return current_set[:]
            
            # Pruning: if not enough candidates left
            if len(candidates) < 6 - len(current_set):
                return None
            
            # Choose a vertex to branch on.
            # Heuristic: Pick vertex with highest degree in the induced subgraph on candidates?
            # Or just pick the first one?
            # Let's pick the first one in the sorted order (which is already in candidates if we filter correctly).
            # However, candidates is a list.
            
            # To speed up, we can filter candidates based on the last added vertex.
            # But here we branch on inclusion/exclusion.
            
            # Let's pick the vertex from candidates that has the highest degree within candidates in the COMPLEMENT graph.
            # i.e., lowest degree in G within candidates.
            
            best_v = -1
            best_score = -1
            for v in candidates:
                # Count neighbors of v in candidates in G
                # If few neighbors in G, many non-neighbors (edges in G_bar).
                cnt = 0
                for u in candidates:
                    if u == v: continue
                    if adj[v] & (1 << u):
                        cnt += 1
                # We want to maximize (len(candidates) - 1 - cnt)
                # Or minimize cnt.
                score = len(candidates) - 1 - cnt
                if score > best_score:
                    best_score = score
                    best_v = v
            
            if best_v == -1:
                return None
                
            # Branch 1: Include best_v
            # New candidates must be non-neighbors of best_v in G (i.e., neighbors in G_bar)
            # And must be from candidates (excluding best_v)
            new_cands = []
            for u in candidates:
                if u == best_v: continue
                if not (adj[best_v] & (1 << u)):
                    new_cands.append(u)
            
            res = backtrack(new_cands, current_set + [best_v])
            if res:
                return res
                
            # Branch 2: Exclude best_v
            new_cands2 = [u for u in candidates if u != best_v]
            res = backtrack(new_cands2, current_set)
            if res:
                return res
                
            return None
            
        return backtrack(order, [])

    def is_valid(adj, n):
        """Check if graph is K4-free and I6-free."""
        if find_k4(adj, n):
            return False
        if find_i6(adj, n):
            return False
        return True

    def to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    # --- Spectral / Algebraic Construction Helpers ---

    def generate_circulant(n, S):
        """Generate a circulant graph C_n(S)."""
        adj = [0] * n
        for i in range(n):
            for s in S:
                j = (i + s) % n
                if j != i:
                    adj[i] |= (1 << j)
        # Symmetry is handled by S containing -s or by checking both
        # If S contains s, then edge i->i+s exists.
        # For undirected, we need i+s ~ i.
        # If we add edge (i, i+s), we must also add (i+s, i).
        # The loop above adds adj[i] |= 1<<j.
        # If j = i+s, then adj[i] has bit j.
        # Does adj[j] have bit i?
        # When loop is at i' = j, s' = -s mod n.
        # If -s is in S, then adj[j] |= 1<<i.
        # So S must be symmetric: if s in S, then -s mod n in S.
        
        # Let's enforce symmetry in the caller or here.
        # For now, assume S is symmetric.
        return adj

    def random_circulant(n):
        """Generate a random symmetric connection set for a circulant graph."""
        S = set()
        # Possible steps: 1 to n//2
        for s in range(1, n // 2 + 1):
            if random.random() < 0.5:
                S.add(s)
                S.add((n - s) % n)
        
        # Ensure self-loops are not added (s != 0)
        # s=0 is not in range(1, ...)
        
        # Avoid trivial cases?
        return S

    def local_search(adj, n, time_budget):
        """
        Improve a graph using local search.
        Returns adj if valid, else None.
        """
        start = time.time()
        
        # Initial check
        if is_valid(adj, n):
            return adj
            
        while time.time() - start < time_budget:
            # Check for K4
            k4 = find_k4(adj, n)
            if k4:
                u, v, w, x = k4
                # Remove a random edge from the K4
                edges_to_remove = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                ru, rv = random.choice(edges_to_remove)
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                continue
                
            # Check for I6
            i6 = find_i6(adj, n)
            if i6:
                # Add a random edge within the independent set
                # Pick two distinct vertices
                u, v = random.sample(i6, 2)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                continue
                
            # If we get here, no K4 and no I6 found in this iteration?
            # Wait, the loop continues only if a violation was found and fixed.
            # If no violation was found, we should have returned at the start of the loop?
            # No, the check is at the top.
            # If is_valid returns True, we return.
            # So if we are in the loop, is_valid was False.
            # So either k4 or i6 must have been found.
            # If neither is found, it's a bug or race condition?
            # Actually, find_k4 and find_i6 are deterministic.
            # If is_valid is False, one of them must return non-None.
            
            # However, if we fix a K4, we might create an I6.
            # If we fix an I6, we might create a K4.
            # The loop handles this.
            
            # Just in case, check again at the end of the iteration?
            # No, the top of the loop does that.
            
        # Time expired
        if is_valid(adj, n):
            return adj
        return None

    # --- Main Logic ---

    # Fallback: Paley 17
    p = 17
    squares = {x * x % p for x in range(1, p)}
    best_n = p
    best_edges = [[i, j] for i in range(p) for j in range(i + 1, p) if (j - i) % p in squares]

    # Target n=35 primarily, as it is the known record.
    # Try to construct n=35 using spectral/circulant initializations + local search.
    
    # Time allocation
    # We have ~115s.
    # Let's try a few random circulant graphs for n=35.
    # Each local search gets a slice of time.
    
    n_target = 35
    attempts = 5
    time_per_attempt = (time_limit - 5) / attempts  # 5s margin for final checks/returns
    
    for i in range(attempts):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Generate random circulant
        S = random_circulant(n_target)
        adj = generate_circulant(n_target, S)
        
        # Local search
        improved_adj = local_search(adj, n_target, time_per_attempt)
        
        if improved_adj is not None:
            best_n = n_target
            best_edges = to_edges(improved_adj, n_target)
            # If we found n=35, we can stop?
            # The task says maximize n. 35 is the known max.
            # Trying 36 is possible but unlikely to succeed quickly.
            # Given the time constraint, securing 35 is the main goal.
            # If we have time left, we could try 36, but it's risky.
            # Let's stick with 35 if found, as it's a strong result (score ~1.47+).
            break

    # If we didn't find 35, try a generic local search on a random graph for n=35?
    # Or try smaller n?
    # If we fail 35, we should try to get the best possible n < 35.
    # But the seed is 17.
    # Let's try one more strategy: Random graph + local search for n=35.
    
    if best_n < 35 and time.time() - start_time < time_limit - 10:
        # Try random graph initialization
        adj = [0] * n_target
        for i in range(n_target):
            for j in range(i + 1, n_target):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        improved_adj = local_search(adj, n_target, time_limit - (time.time() - start_time) - 5)
        if improved_adj is not None:
            best_n = n_target
            best_edges = to_edges(improved_adj, n_target)
            break # Found 35

    # If still < 35, try n=34, 33... down to 18?
    # This might be too slow if we spend too much time failing 35.
    # But if we failed 35 quickly, we have time.
    
    if best_n < 35:
        # Try to find a valid graph for smaller n
        # We can reuse the local search with random initialization
        # Let's try n=25, 30, 34 etc.
        # To be safe, we'll iterate down from 34 to 18.
        # But we must be careful with time.
        
        # Remaining time
        rem_time = time_limit - (time.time() - start_time) - 2
        
        # Try a few intermediate sizes
        for n_try in [34, 30, 25]:
            if rem_time < 2:
                break
                
            # Allocate 1/3 of remaining time to each?
            t_slice = rem_time / 3
            if t_slice < 1:
                t_slice = 1
                
            adj = [0] * n_try
            for i in range(n_try):
                for j in range(i + 1, n_try):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            improved_adj = local_search(adj, n_try, t_slice)
            if improved_adj is not None:
                best_n = n_try
                best_edges = to_edges(improved_adj, n_try)
                break # Found a valid graph for this n, stop trying smaller?
                # Actually, we want max n. So if we found 34, we stop.
                # If we failed 34, we try 30.
                # So we should NOT break if we found it?
                # Wait, if we found 34, that's better than 30. So break is correct.
                # If we didn't find 34, we continue to 30.
                
            rem_time = time_limit - (time.time() - start_time) - 2

    return {'n': best_n, 'edges': best_edges}
