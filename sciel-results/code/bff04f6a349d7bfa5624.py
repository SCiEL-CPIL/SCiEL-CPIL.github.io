import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    N = 128
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Build conflict graph: edge (u,v) exists if there exists w != u,v
    # such that gcd(u,v) == gcd(u,w) == gcd(v,w)
    # conflict[u] is a bitmask of all v that conflict with u
    conflict = [0]*(N+1)
    for u in range(1, N+1):
        for v in range(u+1, N+1):
            g_uv = gcd_table[u][v]
            # Check if there exists w such that gcd(u,w)==g_uv and gcd(v,w)==g_uv
            # This means w must be a multiple of g_uv, and w/g_uv must be coprime to u/g_uv and v/g_uv
            u_reduced = u // g_uv
            v_reduced = v // g_uv
            # w = g_uv * k where gcd(k, u_reduced) == 1 and gcd(k, v_reduced) == 1
            # and w != u, w != v, w in [1, 128]
            found = False
            max_k = N // g_uv
            for k in range(1, max_k+1):
                w = g_uv * k
                if w == u or w == v:
                    continue
                if gcd(k, u_reduced) == 1 and gcd(k, v_reduced) == 1:
                    found = True
                    break
            if found:
                conflict[u] |= (1 << v)
                conflict[v] |= (1 << u)

    # Precompute for each x, the set of (a,b) pairs where gcd(a,x)==gcd(b,x)==gcd(a,b)
    # conflict_mask[x] is a bitmask over elements: for each pair (a,b) with a<b,
    # if adding x to {a,b} creates a bad triple, mark both a and b in conflict_mask[x]
    # Actually, let's think differently:
    # For a given set S and candidate x, x conflicts with S if there exist a,b in S (a≠b, a≠x, b≠x)
    # such that gcd(a,x) == gcd(b,x) == gcd(a,b).
    #
    # Key observation: gcd(a,x) == gcd(b,x) == gcd(a,b) = g means:
    #   g | a, g | b, g | x, and a/g, b/g, x/g are pairwise coprime... no that's not right.
    #   Actually: gcd(a,x)=g means g|a and g|x and gcd(a/g, x/g)=1
    #   Similarly gcd(b,x)=g means g|b and g|x and gcd(b/g, x/g)=1
    #   And gcd(a,b)=g means g|a and g|b and gcd(a/g, b/g)=1
    #
    # So for a fixed x and fixed g = gcd(a,x) = gcd(b,x) = gcd(a,b):
    #   g must divide x, and a/g and b/g must be coprime to x/g and to each other.
    #
    # Alternative approach: For each x, precompute for each possible gcd value g that divides x,
    # which elements a satisfy gcd(a,x) = g. Then two elements a,b in S conflict with x if
    # gcd(a,x) == gcd(b,x) == gcd(a,b).
    #
    # Simpler bitmask approach: For each x, define conflict[x] as a bitmask where bit i is set
    # if there exists some y (y≠x) such that... no, this doesn't work directly because the
    # conflict depends on the pair (a,b) both being in S.
    #
    # Let me use a different approach: For each x, precompute a list of "witness pairs" - 
    # pairs (a,b) such that gcd(a,x)==gcd(b,x)==gcd(a,b). Then x can be added to S iff
    # no witness pair has both elements in S.
    #
    # Even better: For each x, for each a ≠ x, compute g_ax = gcd(a,x). Then x conflicts with
    # the pair (a,b) in S iff gcd(b,x) == g_ax and gcd(a,b) == g_ax.
    #
    # For efficient checking: For each x, group elements by their gcd with x.
    # gcd_with_x[a] = gcd(a,x) for all a.
    # Then x can be added to S iff for every g, the set {a in S : gcd(a,x)=g} has the property
    # that no two elements a,b in it have gcd(a,b)=g.
    #
    # Let me precompute: for each x, for each a, gcd(a,x). Store as gcd_with[x][a].
    # Then for a candidate x and set S (as bitmask), check:
    #   For each a in S: g = gcd_with[x][a]
    #     For each b in S with b > a: if gcd_with[x][b] == g and gcd_table[a][b] == g: conflict
    #
    # This is still O(n²) per check. Let me think of a better way.
    #
    # Alternative: Precompute for each x a "conflict bitmask" approach:
    # For each x, define a graph on {1..128}\{x} where edge (a,b) exists if 
    # gcd(a,x)==gcd(b,x)==gcd(a,b). Then x can be added to S iff S∩({1..128}\{x}) is an
    # independent set in this graph.
    #
    # For n=128, checking if a set is an independent set in a graph can be done by:
    # For each vertex v in S, check if any neighbor of v is also in S.
    # Using bitmasks: neighbor_mask[v] has bits set for all neighbors of v.
    # Then S is independent iff for all v in S: (S & neighbor_mask[v]) == 0 (excluding v itself).
    #
    # But building this graph for each x is expensive. Let me just use the direct approach
    # but with bitmasks to speed up the inner loop.
    
    def can_add_mask(S_mask, x):
        """Check if x can be added to independent set S_mask."""
        # x can be added iff no neighbor of x is in S_mask
        return (S_mask & conflict[x]) == 0

    def mask_to_list(S_mask):
        result = []
        temp = S_mask
        while temp:
            lowbit = temp & (-temp)
            result.append(lowbit.bit_length() - 1)
            temp ^= lowbit
        return result

    # Greedy coloring for upper bound (Tomita algorithm)
    # Returns a list of colors (0-indexed) for vertices in the given set
    def greedy_coloring(vertices, adj):
        """Greedy coloring. vertices is a list of vertex indices.
        adj is a list of adjacency bitmasks.
        Returns (num_colors, color_list) where color_list[i] is color of vertices[i]."""
        n = len(vertices)
        if n == 0:
            return 0, []
        colors = [-1] * n
        # Order by degree (descending) for better coloring
        vertex_order = sorted(range(n), key=lambda i: bin(adj[vertices[i]] & vertices_mask_of_set).bit_count(), reverse=True)
        # Actually, let's use a simpler approach
        num_colors = 0
        for i in range(n):
            v = vertices[i]
            # Find first available color
            used_colors = set()
            for j in range(n):
                if j != i and colors[j] >= 0 and (adj[v] >> vertices[j]) & 1:
                    used_colors.add(colors[j])
            c = 0
            while c in used_colors:
                c += 1
            colors[i] = c
            if c >= num_colors:
                num_colors = c + 1
        return num_colors, colors

    # Better: use the standard Tomita coloring with bitsets
    def color_bound(vertex_set_mask):
        """Compute an upper bound on the size of MIS using greedy coloring.
        vertex_set_mask is a bitmask of candidate vertices.
        Returns the number of colors (upper bound on clique size, which is lower bound... 
        wait, for MIS we need upper bound on independent set size = clique number of complement.
        Actually, for branch and bound on MIS, we use:
        upper_bound = |P| where P is the candidate set, refined by coloring.
        The coloring gives an upper bound: if we can color P with k colors, then 
        the maximum independent set in P has size at most... no.
        
        Let me reconsider. For Maximum Independent Set:
        - We branch on a vertex v: either include v (remove v and its neighbors from P) 
          or exclude v (remove v from P).
        - Upper bound on the size of any independent set in P is |P|.
        - A tighter bound: if we color P such that vertices of the same color are 
          pairwise non-adjacent (i.e., form independent sets), then the maximum 
          independent set can contain at most one vertex from each color class... 
          NO, that's wrong. If vertices of the same color are non-adjacent, they 
          CAN all be in the independent set.
        
        I'm confusing this with Maximum Clique. For Maximum Clique:
        - Upper bound via coloring: if P can be colored with k colors (adjacent vertices 
          get different colors), then max clique in P has size at most k.
        
        For Maximum Independent Set, we can convert to Maximum Clique on the complement graph,
        or we can directly use: the upper bound is |P|, and we can refine it.
        
        Actually, the standard approach for MIS branch-and-bound:
        - Use the fact that if we partition P into independent sets S_1, ..., S_k,
          then the maximum independent set in P has size at most sum of |S_i|... 
          that's just |P| again.
        
        Let me use a different approach: convert to maximum clique on complement.
        Or use a simple bound: upper_bound = |P|, and prune if |current| + |P| <= best.
        
        For a tighter bound, I'll use the following: 
        In the complement graph, find a clique cover of P. The size of the largest 
        independent set in P equals the size of the largest clique in the complement 
        restricted to P. So I can use maximum clique algorithms on the complement.
        
        But for simplicity, let me just use |P| as the bound and rely on good 
        branching. With N=128, this might be too slow.
        
        Alternative: use a heuristic bound. For each vertex v in P, at most one of 
        {v} and its non-neighbors can be in the independent set... this gets complex.
        
        Let me try a different strategy: use the greedy approach to get a good initial 
        solution, then use local search (add/remove/swap) with the conflict graph.
        """
        return bin(vertex_set_mask).count('1')

    # Simple greedy independent set
    def greedy_is(vertices_list, adj):
        """Greedy: pick vertices in order, add if no neighbor already picked."""
        result = set()
        for v in vertices_list:
            if not (result & adj[v]):
                result.add(v)
        return result

    # Convert set to mask
    def set_to_mask(s):
        m = 0
        for v in s:
            m |= (1 << v)
        return m

    # Branch and bound for Maximum Independent Set
    # Using a simple recursive approach with bitmasks
    best_size = 0
    best_mask = 0
    nodes_visited = [0]
    
    def branch_and_bound(P_mask, current_mask, current_size):
        """
        P_mask: bitmask of candidate vertices (can still be added)
        current_mask: bitmask of vertices already in the independent set
        current_size: size of current independent set
        """
        nonlocal best_size, best_mask
        if time.time() > deadline:
            return
        
        nodes_visited[0] += 1
        
        # Upper bound: current_size + |P|
        P_count = bin(P_mask).count('1')
        if current_size + P_count <= best_size:
            return
        
        if P_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in P
        temp = P_mask
        best_v = 0
        best_deg = -1
        while temp:
            lowbit = temp & (-temp)
            v = lowbit.bit_length() - 1
            deg = bin(conflict[v] & P_mask).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lowbit
        
        v = best_v
        v_bit = 1 << v
        
        # Branch 1: Include v
        # Remove v and all its neighbors from P
        new_P = P_mask & ~conflict[v] & ~v_bit
        branch_and_bound(new_P, current_mask | v_bit, current_size + 1)
        
        # Branch 2: Exclude v
        branch_and_bound(P_mask & ~v_bit, current_mask, current_size)

    # Run branch and bound
    all_vertices_mask = 0
    for i in range(1, N+1):
        all_vertices_mask |= (1 << i)
    
    branch_and_bound(all_vertices_mask, 0, 0)
    
    # If B&B didn't find anything or timed out, fall back to greedy + local search
    if best_size == 0 or time.time() > deadline - 5:
        # Greedy construction
        S_mask = 0
        vertices = list(range(1, N+1))
        random.shuffle(vertices)
        for x in vertices:
            if can_add_mask(S_mask, x):
                S_mask |= (1 << x)
        sz = bin(S_mask).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = S_mask
        
        # Local search: try to add remaining vertices
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            candidates = [x for x in range(1, N+1) if not (best_mask & (1 << x))]
            random.shuffle(candidates)
            for x in candidates:
                if can_add_mask(best_mask, x):
                    best_mask |= (1 << x)
                    best_size += 1
                    improved = True
                    break
        
        # 1-1 swaps
        for _ in range(50):
            if time.time() >= deadline - 2:
                break
            candidates = [x for x in range(1, N+1) if not (best_mask & (1 << x))]
            if not candidates:
                break
            temp = best_mask
            bits_in_S = []
            while temp:
                lowbit = temp & (-temp)
                bits_in_S.append(lowbit.bit_length() - 1)
                temp ^= lowbit
            if not bits_in_S:
                break
            u = random.choice(bits_in_S)
            v = random.choice(candidates)
            S_temp = best_mask & ~(1 << u)
            if can_add_mask(S_temp, v):
                best_mask = S_temp | (1 << v)

    return mask_to_list(best_mask)
