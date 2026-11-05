import random
import time

def construct():
    deadline = time.time() + 115
    rng = random.Random(42)
    
    # Helper to check if a circulant graph is valid
    def valid_circulant(n, jumps):
        # Build adjacency bitmasks
        A = [0] * n
        for j in jumps:
            jj = j % n
            for i in range(n):
                A[i] |= (1 << ((i + jj) % n))
                A[i] |= (1 << ((i - jj) % n))
        
        # Remove self loops (though (i+j)%n == i implies j%n==0, which we avoid)
        for i in range(n):
            A[i] &= ~(1 << i)

        # Check K4
        # A K4 in circulant graph corresponds to 4 vertices a, b, c, d
        # such that all pairwise differences are in J (or -J).
        # Efficient check: iterate triples a, b, c that form a triangle, 
        # and see if there is a d connected to all.
        # Even better: just use the generic bitset check but optimized.
        
        # Check for K4
        for a in range(n):
            # Neighbors of a
            nb_a = A[a]
            # Iterate pairs of neighbors b, c
            # To find K4, we need a, b, c, d all mutually connected.
            # This is equivalent to finding a triangle in the neighborhood of a,
            # where the third vertex d is also a neighbor of a.
            # Actually, simpler: 
            # For each a, consider the subgraph induced by N(a).
            # If N(a) contains a K3, then a + that K3 is a K4.
            # So we just need to check if the subgraph induced by N(a) has a triangle.
            
            # Let's extract the subgraph on N(a)
            # Vertices in N(a): v such that (v-a)%n in J or (a-v)%n in J
            # Edges between u, v in N(a) exist if (u-v)%n in J
            
            # Optimization: 
            # A triangle in N(a) exists if there are u, v in N(a) such that 
            # u~v and they have a common neighbor w in N(a).
            
            # Let's just do the O(n^4) check but with bitsets it's fast enough for n<=40?
            # n=40, n^4 = 2.5M operations. In Python, 2.5M loops is ~0.1s.
            # But we do this for each step. 
            # Let's try to be smarter.
            
            # Check K4 by iterating a, b, c, d?
            # No, let's use the property: 
            # G has K4 iff there exists a such that G[N(a)] has a triangle.
            
            # Let's find all triangles in G[N(a)].
            # Vertices of N(a):
            # We can represent N(a) as a bitmask.
            # For each pair (u, v) in N(a) with u < v and u~v:
            #   Check if (N(u) & N(v) & N(a) & ~(1<<u) & ~(1<<v)) is non-empty.
            
            # Let's precompute neighbors for speed? No, A is already bitsets.
            
            # Iterate u in N(a)
            mask_u = nb_a
            while mask_u:
                lu = mask_u & -mask_u
                u = lu.bit_length() - 1
                mask_u ^= lu
                
                # Iterate v in N(a) with v > u (or just v != u)
                # Actually, we need u~v.
                # v must be in N(a) AND N(u).
                candidates_v = nb_a & A[u]
                
                # We need to find a triangle u-v-w in N(a).
                # So for each v in candidates_v:
                #   w must be in N(a) & N(u) & N(v).
                #   We can just check if (N(a) & N(u) & N(v)) has any bit other than u, v?
                #   Actually, if u, v, w are all in N(a) and form a triangle, 
                #   then w is in N(u) & N(v) & N(a).
                
                # Let's iterate v
                mask_v = candidates_v
                while mask_v:
                    lv = mask_v & -mask_v
                    v = lv.bit_length() - 1
                    mask_v ^= lv
                    
                    # Check for w
                    # w must be in N(a) & N(u) & N(v)
                    # And w != u, w != v
                    common = nb_a & A[u] & A[v]
                    # Remove u and v
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    if common:
                        return False, "K4"
                        
        # Check Independent Set of size 6
        # This is harder to do structurally fast.
        # But n is small. 
        # We can use a simple backtracking or just check all 6-subsets? 
        # C(40,6) is ~3.8M. Too slow for every step.
        
        # Better: Use the same bitset backtracking as the original, 
        # but only when necessary? 
        # Or, since we are doing local search, we can just run the full check.
        # The original program's `max_independent_set_size` is decent.
        
        # Let's implement a fast alpha check for n<=40.
        # If alpha >= 6, invalid.
        
        # Heuristic: if the graph is dense, alpha is small.
        # Circulant graphs are regular.
        # Let's just run a quick check.
        
        # Re-use logic from original but inline for speed.
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(A[i] | (1 << i))
            
        # Find if there is an independent set of size 6
        found = [False]
        
        def bk(R, P, X):
            if found[0]:
                return
            if len(R) == 6:
                found[0] = True
                return
            if not P:
                return
            if len(R) + bin(P).count("1") < 6:
                return
            
            # Choose u
            if X:
                u_bit = X & -X
                u = u_bit.bit_length() - 1
                candidates = P & ~comp[u]
            else:
                # Pick vertex with max degree in P
                max_deg = -1
                u = -1
                p_temp = P
                while p_temp:
                    v_bit = p_temp & -p_temp
                    v = v_bit.bit_length() - 1
                    deg = bin(comp[v] & P).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        u = v
                    p_temp ^= v_bit
                candidates = P
                
            c = candidates
            while c and not found[0]:
                low = c & -c
                v = low.bit_length() - 1
                R.append(v)
                bk(R, P & comp[v], X & comp[v])
                R.pop()
                P = P & ~low
                X = X | low
                c = c & ~low
        
        bk([], all_mask, 0)
        if found[0]:
            return False, "Ind6"
            
        return True, None

    def to_edges(n, jumps):
        edges = []
        for j in jumps:
            jj = j % n
            for i in range(n):
                u = i
                v = (i + jj) % n
                if u < v:
                    edges.append([u, v])
        # Deduplicate
        edges_set = set(tuple(e) for e in edges)
        edges = [list(e) for e in edges_set]
        return edges

    best_n = 17
    # Start with Paley(17) jumps? 
    # Paley(17) is a circulant graph?
    # Yes, Paley(q) for q=4k+1 is a circulant graph on Z_q where edges connect x, y if x-y is a quadratic residue.
    # Quadratic residues mod 17: 1, 4, 9, 16, 8, 2, 15, 13.
    # Sorted: 1, 2, 4, 8, 9, 13, 15, 16.
    # Note: 16 is -1, 15 is -2, 13 is -4, 9 is -8.
    # So jumps = [1, 2, 4, 8] is sufficient for undirected circulant.
    
    start_jumps = [1, 2, 4, 8]
    valid, _ = valid_circulant(best_n, start_jumps)
    if not valid:
        # Fallback to random
        best_n = 20
        start_jumps = [1, 2, 3, 5, 7]
        # Ensure valid
        while not valid_circulant(best_n, start_jumps)[0]:
             start_jumps = [rng.randint(1, best_n//2) for _ in range(rng.randint(2, 5))]
             if len(start_jumps) == 0: start_jumps = [1]

    best_jumps = list(start_jumps)
    
    # Search loop
    # We will try to increase n, and for each n, optimize jumps.
    
    current_n = best_n
    current_jumps = list(best_jumps)
    
    while time.time() < deadline - 5:
        # Try to increase n
        if current_n < 40:
            next_n = current_n + 1
            # Try to find valid jumps for next_n
            # Start with current jumps, maybe adjusted
            candidate_jumps = list(current_jumps)
            
            # Try a few random modifications or direct check
            found_valid = False
            for attempt in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Check current candidate
                valid, _ = valid_circulant(next_n, candidate_jumps)
                if valid:
                    current_n = next_n
                    current_jumps = candidate_jumps
                    best_n = current_n
                    best_jumps = list(current_jumps)
                    found_valid = True
                    break
                
                # Modify candidate
                # Add a random jump
                if attempt % 3 == 0:
                    j = rng.randint(1, next_n // 2)
                    if j not in candidate_jumps:
                        candidate_jumps.append(j)
                # Remove a random jump
                elif attempt % 3 == 1 and len(candidate_jumps) > 1:
                    candidate_jumps.pop(rng.randrange(len(candidate_jumps)))
                # Swap
                else:
                    if len(candidate_jumps) >= 2:
                        i, j = rng.sample(range(len(candidate_jumps)), 2)
                        candidate_jumps[i], candidate_jumps[j] = candidate_jumps[j], candidate_jumps[i]
                    else:
                        j = rng.randint(1, next_n // 2)
                        candidate_jumps[0] = j

            if not found_valid:
                # Failed to grow, try to optimize current n
                # Or break if we can't grow anymore?
                # Let's try to optimize current n to see if it helps for future?
                # Actually, if we can't grow, we are stuck at this n for circulant.
                # We should return the best n found.
                # But maybe we can try different n? No, we want max n.
                # So if we can't grow from current_n, we are done?
                # Not necessarily, maybe a different set of jumps for current_n 
                # allows growth? But we already optimized current_n?
                # We didn't explicitly optimize current_n, we just started from Paley(17).
                
                # Let's do a local search on current_n to find a "better" structure
                # that might allow growth?
                # Actually, the set of jumps doesn't change n.
                # So if we can't find valid jumps for n+1, we might be stuck.
                
                # Let's try to optimize current_n just to be sure we have a good base,
                # though it doesn't change n.
                # Actually, let's just break if we fail to grow a few times?
                # No, let's try a few random restarts of the jump search for next_n.
                
                # If we fail 50 times, maybe we should try a completely random start for next_n?
                # Or just accept current_n.
                
                # Let's try one more time with random jumps for next_n
                for _ in range(10):
                    if time.time() > deadline - 5:
                        break
                    cand = [rng.randint(1, next_n // 2) for _ in range(rng.randint(3, 6))]
                    if valid_circulant(next_n, cand)[0]:
                        current_n = next_n
                        current_jumps = cand
                        best_n = current_n
                        best_jumps = list(current_jumps)
                        found_valid = True
                        break
                
                if not found_valid:
                    # Cannot grow
                    break

        # If we can grow, we loop. 
        # If we reached 40, break.
        if current_n >= 40:
            break

    # Final verification
    valid, _ = valid_circulant(best_n, best_jumps)
    if not valid:
        # Fallback to a known small valid graph if something went wrong
        # e.g. Paley(17)
        best_n = 17
        best_jumps = [1, 2, 4, 8]

    edges = to_edges(best_n, best_jumps)
    return {'n': best_n, 'edges': edges}
