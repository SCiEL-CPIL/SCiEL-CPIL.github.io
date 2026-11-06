"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    N = 256
    # Precompute which sums are perfect squares
    # a+b ranges from 2 to 512
    max_sum = 2 * N
    is_square = [False] * (max_sum + 1)
    for k in range(1, int(math.isqrt(max_sum)) + 1):
        is_square[k * k] = True
    
    # Build conflict graph: edge between i and j (i<=j) if i+j is a perfect square
    # We work with indices 0..255 representing values 1..256
    # conflict[i][j] = True if (i+1)+(j+1) is a perfect square and i <= j
    # For MIS, we need undirected edges; diagonal edges (i==i) mean vertex i cannot be in any independent set? 
    # Wait: the constraint is "a+b is never a perfect square for a<=b in the set, INCLUDING a=b".
    # So if 2*a is a perfect square, then a cannot be in the set at all (since a+a=2a is a square).
    # If a+b is a perfect square for a<b, then a and b cannot both be in the set.
    
    # So:
    # 1. Vertices where 2*(i+1) is a square are forbidden (they conflict with themselves).
    # 2. For remaining vertices, edges exist where (i+1)+(j+1) is a square.
    
    # Let's identify forbidden vertices first
    forbidden = [False] * N
    for i in range(N):
        val = i + 1
        if is_square[2 * val]:
            forbidden[i] = True
    
    # Build adjacency list for the conflict graph among allowed vertices
    allowed = [i for i in range(N) if not forbidden[i]]
    allowed_set = set(allowed)
    
    # Map allowed vertices to indices 0..m-1
    m = len(allowed)
    idx_map = {v: i for i, v in enumerate(allowed)}
    
    # Build adjacency list
    adj = [[] for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            a = allowed[i] + 1
            b = allowed[j] + 1
            if is_square[a + b]:
                adj[i].append(j)
                adj[j].append(i)
    
    # Now solve Maximum Independent Set on this graph
    # Use a branch-and-bound approach
    
    best_set = []  # best independent set found so far (as indices into allowed[])
    best_size = 0
    
    # Initial greedy solution: min-degree ordering
    def greedy_min_degree():
        nonlocal best_set, best_size
        in_set = [False] * m
        remaining = list(range(m))
        # Compute degrees
        deg = [len(adj[i]) for i in range(m)]
        while remaining:
            # Pick vertex with minimum degree among remaining
            min_idx = min(remaining, key=lambda x: deg[x])
            in_set[min_idx] = True
            # Remove min_idx and its neighbors from remaining
            to_remove = {min_idx} | set(adj[min_idx])
            remaining = [v for v in remaining if v not in to_remove]
            for v in to_remove:
                for u in adj[v]:
                    if u in set(remaining):
                        deg[u] -= 1
        sol = [i for i in range(m) if in_set[i]]
        if len(sol) > best_size:
            best_size = len(sol)
            best_set = sol[:]
        return sol
    
    greedy_min_degree()
    
    # Local search: try to add vertices that don't conflict
    def local_search_improve(sol):
        nonlocal best_set, best_size
        sol_set = set(sol)
        improved = True
        while improved:
            improved = False
            for v in range(m):
                if v in sol_set:
                    continue
                # Check if v can be added
                can_add = True
                for u in adj[v]:
                    if u in sol_set:
                        can_add = False
                        break
                if can_add:
                    sol_set.add(v)
                    improved = True
        if len(sol_set) > best_size:
            best_size = len(sol_set)
            best_set = list(sol_set)
        return sol_set
    
    local_search_improve(set(best_set))
    
    # Branch and bound for MIS
    # We'll use a recursive approach with bitsets for speed
    
    # Convert adj to bitsets
    adj_bits = [0] * m
    for i in range(m):
        bits = 0
        for j in adj[i]:
            bits |= (1 << j)
        adj_bits[i] = bits
    
    # Use a greedy coloring upper bound for the subgraph
    def greedy_color_upper_bound(vertices):
        """Return an upper bound on MIS size using greedy coloring."""
        if not vertices:
            return 0
        color_class = []
        remaining = list(vertices)
        colors = 0
        while remaining:
            colors += 1
            # Greedy: pick vertices that don't conflict with each other in this color class
            color_class = []
            used = set()
            for v in remaining:
                if v in used:
                    continue
                # Check if v conflicts with any in color_class
                ok = True
                for u in color_class:
                    if u in adj_bits[v] >> u & 1 or (adj_bits[v] >> u) & 1:
                        ok = False
                        break
                if ok:
                    color_class.append(v)
                    used.add(v)
            remaining = [v for v in remaining if v not in used]
        return colors
    
    # Better: use a proper greedy coloring to get upper bound
    def greedy_coloring(vertices):
        """Assign colors to vertices such that adjacent vertices have different colors.
        Return number of colors used (upper bound on clique, so lower bound on MIS... 
        wait, we need upper bound on MIS).
        Actually, for MIS, we can use the fact that the complement graph's clique number = MIS of original.
        A coloring of the original graph gives an upper bound on the size of any independent set? No.
        Let me think again.
        
        For Maximum Independent Set: 
        - The graph's independence number α(G) ≤ χ(G) (chromatic number)? No, that's not right either.
        - Actually, α(G) * χ(G) ≥ n, so α(G) ≥ n/χ(G). That's a lower bound.
        - For upper bound: α(G) ≤ n - τ(G) where τ is minimum vertex cover. And τ(G) ≥ maximum matching.
        
        Let me use a simpler approach: for the branch and bound, at each node with candidate set S,
        the upper bound is |S| (trivial) or we can use a greedy coloring of the subgraph induced by S
        to get a better bound. Actually, a proper coloring of the subgraph G[S] with k colors means
        S can be partitioned into k independent sets, so the maximum independent set in G[S] has size
        at most... no, that doesn't directly give an upper bound.
        
        Wait, I'm confusing myself. Let me use a standard approach:
        - In the conflict graph, we want the largest set of vertices with no edges between them.
        - At a branch node, we have a set of candidates C.
        - Upper bound: |C| (trivial).
        - Better upper bound: use a greedy coloring of the subgraph induced by C. If we can color C with k colors,
          then... hmm, actually the number of colors is an upper bound on the clique number of the complement,
          which is the independence number of the original. So yes, greedy coloring of G[C] gives an upper
          bound on α(G[C]).
        
        Let me implement a simple greedy coloring for upper bound.
        """
        if not vertices:
            return 0
        color_of = {}
        max_color = 0
        for v in vertices:
            # Find the smallest color not used by neighbors of v
            used_colors = set()
            for u in adj[v]:
                if u in color_of:
                    used_colors.add(color_of[u])
            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            max_color = max(max_color, c + 1)
        return max_color
    
    # Branch and bound
    # State: current set (list of chosen vertices), candidates (set of vertices that can still be added)
    # We'll use a recursive DFS with pruning
    
    def branch_and_bound(chosen, candidates):
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline:
            return
        
        # Upper bound: len(chosen) + greedy_coloring(candidates)
        ub = len(chosen) + greedy_coloring(list(candidates))
        if ub <= best_size:
            return
        
        if not candidates:
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_set = chosen[:]
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in the candidate subgraph
        cand_list = list(candidates)
        # Compute degree within candidates
        cand_set = set(candidates)
        best_v = max(cand_list, key=lambda v: sum(1 for u in adj[v] if u in cand_set))
        
        # Branch 1: include best_v
        new_chosen = chosen + [best_v]
        # New candidates: candidates minus best_v and its neighbors
        neighbors = set(adj[best_v])
        new_candidates = cand_set - {best_v} - neighbors
        branch_and_bound(new_chosen, new_candidates)
        
        # Branch 2: exclude best_v
        new_candidates2 = cand_set - {best_v}
        branch_and_bound(chosen, new_candidates2)
    
    # Run branch and bound from the current best solution's perspective
    # Actually, let's just run it from scratch with all allowed vertices as candidates
    # But that might be too slow. Let's use the greedy solution as a starting point.
    
    # For now, try B&B with a time limit
    initial_candidates = set(range(m))
    branch_and_bound([], initial_candidates)
    
    # Check time
    if time.time() > deadline - 5:
        # Convert best_set back to original values
        result = [allowed[i] + 1 for i in best_set]
        if not result:
            # Fallback: return a known valid set
            result = list(range(1, 257, 3))
        return result
    
    # Try multiple random restarts with local search
    random.seed(42)
    for restart in range(100):
        if time.time() > deadline - 5:
            break
        
        # Random greedy: pick vertices in random order, add if no conflict
        random_order = list(range(m))
        random.shuffle(random_order)
        sol_set = set()
        for v in random_order:
            can_add = True
            for u in adj[v]:
                if u in sol_set:
                    can_add = False
                    break
            if can_add:
                sol_set.add(v)
        
        # Local search
        sol_set = local_search_improve(sol_set)
        
        if time.time() > deadline - 5:
            break
    
    # Final result
    result = [allowed[i] + 1 for i in best_set]
    if not result:
        result = list(range(1, 257, 3))
    return result
# EVOLVE-BLOCK-END
