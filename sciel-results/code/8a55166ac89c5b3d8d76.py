import time
import random
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 145.0

    def time_left():
        return DEADLINE - time.time()

    # ---------- Deterministic Fallbacks ----------
    # Fallback 1: n=30 (10 disjoint triangles)
    def fallback_30():
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

    # ---------- Bitset Utilities ----------
    def bit_count(x):
        return bin(x).count('1')

    # ---------- C4-Free Graph Generation ----------
    def generate_c4free(n, seed):
        rng = random.Random(seed)
        adj = [0] * n
        
        # Strategy: Random greedy edge addition
        # We try to add edges in a random order of pairs
        # If adding an edge creates a C4, we skip it.
        # To get denser graphs, we can try multiple random orders and keep the best.
        
        best_adj = [0] * n
        best_edges = 0
        
        # Run a few internal trials to find a dense C4-free subgraph
        for _ in range(5):
            adj_try = [0] * n
            # Generate a list of all pairs and shuffle
            pairs = list(combinations(range(n), 2))
            rng.shuffle(pairs)
            
            for u, v in pairs:
                # Check C4 condition: u and v can have at most 1 common neighbor
                # If they are not adjacent, common neighbors must be <= 1?
                # Wait, if u and v are NOT adjacent, and they have 2 common neighbors x, y,
                # then u-x-v-y-u is a C4? No, x-u-y-v-x?
                # A C4 is a cycle of length 4.
                # If u and v are not connected, they cannot form a C4 directly as a diagonal?
                # No, if u and v have two common neighbors x, y, then the subgraph induced by {u,v,x,y}
                # has edges (u,x), (u,y), (v,x), (v,y).
                # This is a K_{2,2}. Is it a C4?
                # Yes, u-x-v-y-u is a cycle of length 4 if (u,v) is NOT an edge and (x,y) is NOT an edge.
                # If (x,y) is an edge, it's a triangle K3 plus a vertex? No.
                # The condition "No two vertices may have two common neighbours" is exactly
                # the condition that the graph is C4-free (contains no induced C4 or any C4).
                # Actually, if u and v have two common neighbors x, y, then u-x-v-y-u is a C4
                # UNLESS x and y are connected?
                # If x and y are connected, then u-x-y-v-u is a C4? No, u-x, x-y, y-v, v-u.
                # If u and v are not connected, we don't have the edge v-u.
                # The definition of C4 in a simple graph usually refers to the cycle $v_1-v_2-v_3-v_4-v_1$.
                # If u and v have common neighbors x, y, then we have edges (u,x), (v,x), (u,y), (v,y).
                # This forms a cycle $x-u-y-v-x$ (length 4) if (x,y) is not an edge and (u,v) is not an edge?
                # Wait. The cycle is $x \to u \to y \to v \to x$.
                # This cycle exists if edges (x,u), (u,y), (y,v), (v,x) exist.
                # They do.
                # So, if ANY two vertices (adjacent or not) have $\ge 2$ common neighbors, there is a C4.
                # Thus, the condition `bit_count(adj[u] & adj[v]) <= 1` must hold for ALL pairs (u,v).
                
                common = adj_try[u] & adj_try[v]
                if bit_count(common) <= 1:
                    # Safe to add edge?
                    # Adding edge (u,v) changes common neighbors for other pairs?
                    # Yes. If we add (u,v), then for any neighbor w of u, the common neighbors of v and w increase by 1 (u becomes common).
                    # We must check if adding (u,v) violates the condition for any pair involving u or v.
                    # Actually, the condition is local to the pair.
                    # If we maintain the invariant that ALL pairs have <= 1 common neighbor,
                    # and we add (u,v), we must ensure that for any x, the number of common neighbors of (u,x) and (v,x) doesn't exceed 1?
                    # No, we just need to ensure that the new edge doesn't create a new C4.
                    # A new C4 involving (u,v) would be of the form u-w-v-x-u?
                    # This requires w and x to be distinct neighbors of u and v?
                    # If u and v are connected, they cannot have 2 common neighbors.
                    # So we check: if u and v are connected, common neighbors must be <= 1.
                    # If u and v are NOT connected, they must have <= 1 common neighbors.
                    # So the check `bit_count(adj_try[u] & adj_try[v]) <= 1` is necessary and sufficient
                    # for the pair (u,v) to be "C4-safe" with respect to their current neighborhoods.
                    # BUT, adding (u,v) makes u and v adjacent.
                    # Does it create a C4 with other vertices?
                    # A C4 is a set of 4 vertices with 4 edges.
                    # If we add (u,v), the only new C4s that can form must include the edge (u,v).
                    # Such a C4 would be $u-v-w-x-u$.
                    # This requires:
                    # 1. (u,v) edge (new)
                    # 2. (v,w) edge
                    # 3. (w,x) edge
                    # 4. (x,u) edge
                    # And $w \neq x$.
                    # This implies that w is a common neighbor of u and v? No.
                    # w is neighbor of v. x is neighbor of u.
                    # And w is neighbor of x.
                    # This is complex.
                    
                    # Simpler check:
                    # After adding (u,v), we must verify that for all z,
                    # bit_count(adj_try[u] & adj_try[z]) <= 1
                    # and bit_count(adj_try[v] & adj_try[z]) <= 1
                    # is NOT sufficient.
                    
                    # The standard condition for C4-free is:
                    # For any two vertices x, y, |N(x) \cap N(y)| <= 1.
                    # If we add edge (u,v):
                    # For any z not in {u,v}, N(z) is unchanged.
                    # So we only need to check pairs involving u or v.
                    # Pair (u, v): new common neighbors?
                    # N(u) and N(v) don't change for the purpose of intersection?
                    # Wait, N(u) gains v. N(v) gains u.
                    # Intersection N(u) \cap N(v) is unchanged by adding (u,v) because u not in N(u) and v not in N(v).
                    # So |N(u) \cap N(v)| is the same as before.
                    # We already checked it is <= 1.
                    
                    # Now check pair (u, z) for z != v.
                    # N(u) gains v. N(z) is same.
                    # New intersection N(u) \cap N(z) = (N_old(u) \cap N(z)) \cup ({v} \cap N(z)).
                    # If v is in N(z), then the size increases by 1.
                    # So if v \in N(z) and |N_old(u) \cap N(z)| == 1, then new size is 2. BAD.
                    # So we must check:
                    # For all z, if (adj_try[z] >> v) & 1, then bit_count(adj_try[u] & adj_try[z]) must be 0 (before adding).
                    # Similarly for (v, z).
                    
                    # This check is O(n) per edge. Total O(n^3). For n=40, 64000 ops. Very fast.
                    
                    can_add = True
                    # Check all z != u, v
                    for z in range(n):
                        if z == u or z == v:
                            continue
                        # If z is connected to v, then u and z will have v as a common neighbor?
                        # No, v is a neighbor of z. Is v a neighbor of u? Yes, we are adding (u,v).
                        # So v becomes a common neighbor of u and z.
                        # If u and z already had 1 common neighbor, they now have 2.
                        if (adj_try[z] >> v) & 1:
                            if bit_count(adj_try[u] & adj_try[z]) > 0:
                                can_add = False
                                break
                        if (adj_try[z] >> u) & 1:
                            if bit_count(adj_try[v] & adj_try[z]) > 0:
                                can_add = False
                                break
                            
                    if can_add:
                        adj_try[u] |= (1 << v)
                        adj_try[v] |= (1 << u)
            
            edge_count = sum(bit_count(b) for b in adj_try) // 2
            if edge_count > best_edges:
                best_edges = edge_count
                best_adj = adj_try[:]
        
        return best_adj

    # ---------- Exact Maximum Independent Set ----------
    def max_independent_set(adj, n):
        # Bitset based branch and bound
        # We use a simple recursion with bit masks
        # candidates_mask: vertices that can still be added
        # current_set: count of vertices in current independent set
        
        best_alpha = 0
        
        # Optimization: Precompute neighborhood masks
        # adj[i] is the mask of neighbors of i
        
        def solve(candidates, current_size):
            nonlocal best_alpha
            
            if candidates == 0:
                if current_size > best_alpha:
                    best_alpha = current_size
                return
            
            # Upper bound: current_size + bit_count(candidates)
            if current_size + bit_count(candidates) <= best_alpha:
                return
            
            # Heuristic: Pick vertex with max degree in induced subgraph (candidates)
            # Or min degree? Max degree usually prunes better for independence number.
            # Actually, for MIS, picking a vertex with high degree in the candidate set
            # removes many neighbors, reducing the branching factor.
            
            v = -1
            max_deg = -1
            temp = candidates
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                deg = bit_count(adj[idx] & candidates)
                if deg > max_deg:
                    max_deg = deg
                    v = idx
                temp ^= lsb
            
            if v == -1:
                return

            # Branch 1: Include v
            # Remove v and its neighbors from candidates
            neighbors = adj[v] & candidates
            new_cand_include = candidates & ~(neighbors | (1 << v))
            solve(new_cand_include, current_size + 1)
            
            # Branch 2: Exclude v
            new_cand_exclude = candidates & ~(1 << v)
            solve(new_cand_exclude, current_size)

        solve((1 << n) - 1, 0)
        return best_alpha

    # ---------- Main Search ----------
    best_result = fallback_30()
    
    # Target n values, descending
    target_ns = [43, 42, 41, 40, 39, 38]
    
    for n in target_ns:
        if time.time() >= DEADLINE - 5.0:
            break
            
        # Try to find a valid graph for this n
        # We run multiple random seeds to generate different C4-free graphs
        max_trials = 200 if n <= 40 else 100
        
        found = False
        for trial in range(max_trials):
            if time.time() >= DEADLINE - 5.0:
                break
            
            adj = generate_c4free(n, trial)
            
            # Quick check: if density is very low, alpha is likely high
            # But we must be exact.
            # For n=39, alpha < 11 is the target.
            
            # To save time, we can do a greedy MIS first?
            # No, exact is safer.
            
            alpha = max_independent_set(adj, n)
            
            if alpha < 11:
                # Convert to edge list
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                
                return {'n': n, 'edges': edges}
                
    return best_result
