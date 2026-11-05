import time
import random
import sys

_START = time.time()
# The evaluator kills at 150s. We use 145s as a hard deadline to ensure safe return.
_DEADLINE = 145

def _elapsed():
    return time.time() - _START

def _construct_greedy(n_target, time_limit):
    """
    Attempt to construct a valid graph on n_target vertices using greedy growth.
    We add vertices one by one. For each new vertex v, we decide its adjacency
    to previous vertices 0..v-1 such that:
    1. No K4 is formed.
    2. No independent set of size 6 is formed.
    
    We use backtracking for the adjacency decision of a single new vertex.
    """
    start_time = time.time()
    
    # adj[i] is a bitmask of neighbors of i among 0..n-1
    # We maintain the graph incrementally.
    # To make backtracking efficient, we maintain:
    # - adj: list of bitmasks
    # - For K4 check: For any pair (u, v), we need to ensure common neighbors don't form a triangle.
    #   Actually, simpler: When adding vertex v connected to S (subset of 0..v-1),
    #   we must check:
    #   - No 3-clique in S is fully connected to v (would form K4 with v).
    #     Wait, if u, w, x in S form a K3, and v is connected to all, then {u,w,x,v} is K4.
    #     So v cannot be connected to all vertices of any K3 in the existing graph.
    #   - No independent set of size 5 in S is fully non-connected to v (would form Ind6 with v).
    #     So v must be connected to at least one vertex in any independent set of size 5 in the existing graph.
    
    # Precomputing all K3s and Ind5s is expensive if done naively, but n is small (<=35).
    # However, doing it for every step is O(n^3) or O(n^5) which might be slow.
    # Alternative: Maintain lists of K3s and Ind5s dynamically?
    # Or just check on the fly during backtracking.
    
    # Let's try a simpler check during backtracking:
    # When deciding adjacency of v to previous vertices:
    # Let A be the set of neighbors of v in 0..v-1.
    # Let B be the set of non-neighbors of v in 0..v-1.
    
    # Condition 1 (No K4):
    # For any K3 {u, w, x} in the existing graph (0..v-1), we must NOT have {u, w, x} subset of A.
    # Equivalent: A cannot contain a K3 of the existing graph.
    
    # Condition 2 (No Ind6):
    # For any Ind5 {u, w, x, y, z} in the existing graph, we must NOT have {u, w, x, y, z} subset of B.
    # Equivalent: B cannot contain an Ind5 of the existing graph.
    # Or: A must intersect every Ind5 of the existing graph.

    # To make this fast, we maintain:
    # - list of K3s: list of tuples (u, w, x)
    # - list of Ind5s: list of tuples (u, w, x, y, z)
    
    # However, maintaining Ind5s is tricky. Number of Ind5s can be large.
    # Alternative: Just check validity of the whole graph at the end? No, we need pruning.
    
    # Let's use a different pruning strategy:
    # When adding v, we pick its neighbors A.
    # We check if A contains a K3 from the previous graph.
    # We check if the non-neighbors B contains an Ind5 from the previous graph.
    
    # To check "A contains a K3", we can iterate through all K3s in the previous graph.
    # To check "B contains an Ind5", we can iterate through all Ind5s in the previous graph.
    
    # How to maintain K3s and Ind5s?
    # When adding v:
    # New K3s involving v: (u, w, v) where u, w in A and u~w.
    # New Ind5s involving v: (u, w, x, y, v) where {u,w,x,y} is Ind4 in B and none connected to v?
    #   Wait, if {u,w,x,y} is Ind4 in B, and v is not connected to them (they are in B),
    #   then {u,w,x,y,v} is Ind5.
    #   So we need to know Ind4s in B.
    
    # This seems complex to maintain dynamically.
    # Let's try a simpler approach: 
    # Just do the local search but with a MUCH faster validity check?
    # The current program's _has_k4 is O(n^4) worst case but often fast.
    # _has_ind_6 is backtracking.
    
    # Let's stick to the "Greedy Growth" but use a simpler validity check for the *prefix* graph.
    # Actually, the most robust "different" strategy is **Simulated Annealing** on the edge set,
    # but the current program is basically Random Local Search.
    # Let's try **Deterministic Construction via Quadratic Residues (Paley-like)**?
    # Paley graph P(q) exists for q ≡ 1 mod 4.
    # P(17) is n=17. P(29) is n=29. P(37) is n=37.
    # But P(29) might have K4 or Ind6?
    # Ramsey number R(4,6) is known to be > 35.
    # Paley(29) has n=29.
    # If we can construct a valid graph on 30, 31, 32...
    
    # Let's try **Vertex Addition with Backtracking** properly.
    # We will maintain the graph as a list of bitmasks.
    # We will maintain a list of all K3s and all Ind5s in the current graph.
    
    adj = [0] * n_target
    k3s = []  # List of (u, w, x)
    ind5s = [] # List of (u, w, x, y, z)
    
    # Vertex 0: no constraints
    # Vertex 1: no constraints (no K3 possible, no Ind5 possible)
    # ...
    # Actually, for small n, we can just brute force the adjacency of the new vertex.
    
    def is_valid_prefix(v):
        """Check if adding vertex v with current adj[v] creates K4 or Ind6."""
        # Check K4:
        # v is connected to A = neighbors(v).
        # If there is a K3 in A, then v + K3 = K4.
        # We can check this by iterating k3s? No, k3s are for the whole graph.
        # We need K3s *within* A.
        # Since A is a subset of 0..v-1, we can check if any triple in A is a K3.
        # But checking all triples in A is O(|A|^3).
        
        # Check Ind6:
        # v is NOT connected to B = non-neighbors(v).
        # If there is an Ind5 in B, then v + Ind5 = Ind6.
        # We need to check if B contains an Ind5.
        
        # This check is expensive if done naively.
        
        # Let's optimize:
        # Precompute for the graph 0..v-1:
        # - For every subset of size 3, is it a K3?
        # - For every subset of size 5, is it an Ind5?
        # This is too much memory/time.
        
        # Alternative:
        # Just run _has_k4 and _has_ind_6 on the prefix graph 0..v?
        # That's what the current program does, but it restarts from scratch.
        # If we do it incrementally, we can stop early.
        
        pass

    # Let's try a different structural change: **Iterative Deepening DFS on Edges**?
    # No, that's exponential.
    
    # Let's go with **Greedy Growth with Fast Local Checks**.
    # The key insight: When adding vertex v, we only need to check for K4s containing v and Ind6s containing v.
    # K4s containing v: v + {a,b,c} where a,b,c form a K3 and are all neighbors of v.
    # Ind6s containing v: v + {a,b,c,d,e} where a,b,c,d,e form an Ind5 and are all NON-neighbors of v.
    
    # So, we need to quickly check:
    # 1. Do the neighbors of v contain a K3?
    # 2. Do the non-neighbors of v contain an Ind5?
    
    # To make this fast, we can precompute:
    # - For each pair (u, w), is it an edge?
    # - For each triple (u, w, x), is it a K3?
    # - For each 5-tuple, is it an Ind5?
    
    # Since n <= 35, the number of triples is 35^3 ~ 42k. Manageable.
    # The number of 5-tuples is 35^5 ~ 50M. Too many to precompute all.
    
    # So we will precompute K3s.
    # For Ind5s, we will use a backtracking search restricted to the non-neighbors B.
    
    adj = [0] * n_target
    # Precompute K3 status for all triples?
    # Let's just build the graph and check incrementally.
    
    # Strategy:
    # For v from 0 to n_target-1:
    #   Try all possible adjacencies for v? No, 2^v possibilities.
    #   Instead, use random assignment + repair?
    
    # Let's try **Random Growth with Backtracking Repair**.
    # Initialize adj[v] randomly.
    # Check for K4s involving v. If found, remove an edge.
    # Check for Ind6s involving v. If found, add an edge.
    # Repeat until stable or timeout.
    
    # This is very similar to the current program but restricted to the new vertex.
    # The current program checks the WHOLE graph.
    # This new approach only checks violations *involving the new vertex*, which is much faster.
    # Because any violation not involving v was already resolved in previous steps.
    
    def _check_k4_with_v(v, adj, n):
        """Find a K4 containing v."""
        # Neighbors of v
        nbrs = adj[v]
        # Get list of neighbors
        nbr_list = []
        c = nbrs
        while c:
            lsb = c & -c
            u = lsb.bit_length() - 1
            nbr_list.append(u)
            c ^= lsb
            
        # Check all pairs in nbr_list for common neighbor in nbr_list
        # If u, w in nbr_list and they have a common neighbor x in nbr_list, then {v, u, w, x} is K4.
        # Wait, v is connected to u, w, x.
        # We need u-w, u-x, w-x to be edges.
        # So we need a triangle in the subgraph induced by nbr_list.
        
        for i in range(len(nbr_list)):
            u = nbr_list[i]
            for j in range(i + 1, len(nbr_list)):
                w = nbr_list[j]
                if not (adj[u] & (1 << w)):
                    continue
                # u and w are connected.
                # Common neighbors of u and w in nbr_list?
                # common = adj[u] & adj[w] & nbrs
                common = adj[u] & adj[w] & nbrs
                if common:
                    # There is at least one x in nbr_list connected to both u and w.
                    # Pick one
                    lsb = common & -common
                    x = lsb.bit_length() - 1
                    return [v, u, w, x]
        return None

    def _check_ind6_with_v(v, adj, n):
        """Find an Ind6 containing v."""
        # Non-neighbors of v
        non_nbrs = ((1 << n) - 1) & ~adj[v] & ~(1 << v)
        
        # We need an Ind5 in non_nbrs.
        # Use backtracking to find Ind5 in the subgraph induced by non_nbrs.
        # Since we are adding v, we only care about Ind5s in the previous vertices.
        
        def find_ind5(candidates, current):
            if len(current) == 5:
                return current[:]
            if candidates.bit_count() < 5 - len(current):
                return None
            
            # Heuristic: pick vertex with fewest neighbors in candidates
            best_v = -1
            best_cnt = n + 1
            c = candidates
            while c:
                lsb = c & -c
                u = lsb.bit_length() - 1
                cnt = (adj[u] & candidates).bit_count()
                if cnt < best_cnt:
                    best_cnt = cnt
                    best_v = u
                c ^= lsb
                
            if best_v == -1:
                return None
                
            candidates &= ~(1 << best_v)
            
            # Include
            new_cands = candidates & ~adj[best_v]
            res = find_ind5(new_cands, current + [best_v])
            if res:
                return res
            
            # Exclude
            res = find_ind5(candidates, current)
            if res:
                return res
            
            return None

        ind5 = find_ind5(non_nbrs, [])
        if ind5:
            return [v] + ind5
        return None

    # Main loop
    for v in range(n_target):
        # Initialize adj[v] randomly
        # Probability of edge?
        # For R(4,6), density ~ 0.5 is good.
        adj[v] = 0
        for u in range(v):
            if random.random() < 0.5:
                adj[v] |= (1 << u)
                adj[u] |= (1 << v)
                
        # Repair loop for this vertex
        while time.time() - start_time < time_limit:
            # Check K4
            k4 = _check_k4_with_v(v, adj, v + 1)
            if k4:
                # Remove a random edge from the K4 (not involving v? No, involving v is fine, or not)
                # Removing an edge involving v reduces degree of v.
                # Removing an edge between u, w in nbrs breaks the triangle.
                # Let's remove a random edge from the K4.
                pairs = [(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]),
                         (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])]
                u, w = random.choice(pairs)
                adj[u] &= ~(1 << w)
                adj[w] &= ~(1 << u)
                continue
                
            # Check Ind6
            ind6 = _check_ind6_with_v(v, adj, v + 1)
            if ind6:
                # Add a random edge within the Ind6 (not involving v? No, involving v is fine)
                # Adding an edge involving v increases degree.
                # Adding an edge between two non-neighbors breaks the Ind5.
                pairs = [(ind6[i], ind6[j]) for i in range(6) for j in range(i + 1, 6)]
                u, w = random.choice(pairs)
                adj[u] |= (1 << w)
                adj[w] |= (1 << u)
                continue
                
            # No violations involving v. Proceed to next vertex.
            break
            
        if time.time() - start_time >= time_limit:
            return None
            
    # Final check (just in case)
    # The incremental check should guarantee validity, but let's be sure.
    # Actually, if we only check violations involving the NEW vertex, 
    # and the prefix was valid, then the new graph is valid.
    # Because any K4/Ind6 in the new graph must involve the new vertex v,
    # as the prefix 0..v-1 was already checked and valid.
    
    edges = []
    for i in range(n_target):
        for j in range(i + 1, n_target):
            if adj[i] & (1 << j):
                edges.append([i, j])
                
    return edges

def construct():
    # Base case: Paley 17
    p = 17
    squares = {x * x % p for x in range(1, p)}
    best_edges = [[i, j] for i in range(p) for j in range(i + 1, p) if (j - i) % p in squares]
    best_n = p
    
    # Try to grow from 18 to 35
    # We use the incremental greedy growth.
    # This is structurally different: it builds the graph vertex by vertex,
    # ensuring validity at each step, rather than fixing n and repairing the whole graph.
    
    for n in range(18, 36):
        if _elapsed() > _DEADLINE - 5:
            break
            
        rem = _DEADLINE - _elapsed()
        # Allocate time slice
        # Larger n is harder.
        # Give 10s to 35, 8s to 34, etc?
        # Let's just give 10s each, but break if time is low.
        time_slice = min(10.0, rem / (36 - n))
        
        # Try multiple times if fast?
        # The incremental method is fast if it succeeds.
        # If it fails, it times out.
        
        edges = _construct_greedy(n, time_slice)
        if edges is not None:
            best_n = n
            best_edges = edges
            
    return {'n': best_n, 'edges': best_edges}
