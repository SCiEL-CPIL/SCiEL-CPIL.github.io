"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import sys
    from itertools import combinations

    deadline = time.time() + 145  # safety margin before 150s kill
    best_total = float('inf')
    best_result = None

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split_graph(n, k, p, seed):
        """
        Solve the clique partition problem for a split graph.
        
        Structure:
        - K = {0, ..., k-1} is a clique (the "kernel" / "clique part")
        - I = {k, ..., n-1} is an independent set (the "independent part")
        - Cross edges: each (u, v) with u in K, v in I exists with prob p
        
        Edges to cover:
        1. Internal edges of K: C(k, 2) edges
        2. Cross edges: for each v in I, edges (u, v) for u in N_K(v)
        
        Strategy:
        - For internal K edges: we can use cliques within K. The simplest is 
          to cover all internal K edges with a small number of cliques.
          Note: any clique within K is valid. We need to partition E(K) into 
          cliques (as edge sets). Since K is a clique, we can use:
            - One big clique K itself covers ALL internal K edges at once!
          So: clique K covers all C(k,2) internal edges.
        
        - For cross edges: For each v in I, the edges incident to v are 
          (u, v) for u in N_K(v). These edges must be covered by cliques that 
          contain v. A clique containing v can contain v and a subset S of K 
          such that S ∪ {v} forms a clique. Since I is independent, no two 
          vertices from I can be in the same clique. So each clique containing 
          v must be of the form S ∪ {v} where S ⊆ N_K(v).
          
          The edges covered by clique S ∪ {v} are:
          - All edges within S (internal K edges)
          - All edges (u, v) for u in S (cross edges)
          
          But we already covered all internal K edges with the clique K. So we 
          must be careful: each edge must lie in EXACTLY one clique.
          
          Revised strategy:
          - We cannot use the full clique K if we also use cliques that contain 
            subsets of K and cross vertices, because internal K edges would be 
            double-covered.
          
          Alternative:
          - Partition the problem by assigning each cross edge to a clique.
          - For each v in I, we need to cover the star of edges from v to N_K(v).
          - We can form cliques of the form {v} ∪ S where S ⊆ N_K(v), and 
            these cliques cover:
              * Cross edges (u, v) for u in S
              * Internal edges within S
          - But internal edges within S must not be covered elsewhere.
          
          Better approach:
          - Use cliques that contain at most one vertex from I (since I is independent).
          - Type A: cliques entirely within K.
          - Type B: cliques containing exactly one vertex v from I and some subset of K.
          
          For Type B cliques {v} ∪ S: they cover cross edges (u,v) for u in S, 
          AND internal K edges within S.
          
          So if we use Type B cliques, the internal K edges within their S sets 
          are covered by those cliques, not by Type A cliques.
          
          We need to partition ALL edges into cliques. Let's think of it as:
          - Each edge is either an internal K edge or a cross edge.
          - Cross edges (u, v) must be in a clique containing both u and v. 
            Since v is in I, the clique is {v} ∪ S where u ∈ S ⊆ N_K(v).
          - Internal K edge (u, w) must be in a clique. It can be in:
            * A Type A clique (subset of K)
            * A Type B clique {v} ∪ S where u, w ∈ S (and v is some vertex in I 
              such that u, w ∈ N_K(v))
          
          To minimize the number of cliques, we want to maximize the size of 
          each clique. 
          
          Key insight: If we use one big clique K for all internal edges, then 
          for cross edges, we can form cliques {v} ∪ S where S ⊆ N_K(v), but 
          these would double-cover internal edges within S. So we can't do that.
          
          Therefore, we must either:
          1. Use Type A cliques to cover some internal K edges, and Type B 
             cliques to cover the rest (including cross edges).
          2. Or, design the Type B cliques so that their S sets are disjoint, 
             and cover the remaining internal K edges with Type A cliques.
          
          Let's try approach: 
          - For each v in I, partition N_K(v) into subsets S_{v,1}, S_{v,2}, ...
          - For each subset S_{v,j}, create clique {v} ∪ S_{v,j}.
          - These cliques cover all cross edges incident to v.
          - They also cover internal K edges within each S_{v,j}.
          - The remaining internal K edges (those not within any S_{v,j}) must 
            be covered by Type A cliques (subsets of K).
          
          To minimize total cliques:
          - We want to minimize the number of subsets per v, and minimize the 
            number of Type A cliques.
          - The internal K edges covered by Type B cliques are the union of 
            E(S_{v,j}) for all v, j.
          - The remaining internal K edges form a graph G_rem on K. We need to 
            partition E(G_rem) into cliques (Type A).
          
          This is still complex. Let's try a simpler greedy:
          
          Simplified greedy:
          1. Start with no cliques.
          2. For each v in I (in some order), cover the edges incident to v.
             - The edges are (u, v) for u in N_K(v).
             - We can form cliques {v} ∪ S where S ⊆ N_K(v).
             - To cover all cross edges of v with few cliques, we want large S.
             - But S must be such that the internal edges within S are not 
               already covered.
             
          Given the complexity and small sizes (k up to 30, n up to 90), let's 
          try a direct approach:
          
          Approach: 
          - We'll use a greedy algorithm that processes vertices in I one by one.
          - Maintain a set of covered edges.
          - For each v in I:
            - Let U = N_K(v) \ {already fully handled}... no, edges are what matter.
            - The uncovered cross edges incident to v are (u, v) where (u,v) is 
              not yet covered.
            - We want to form cliques {v} ∪ S where S ⊆ K, such that:
              * All (u, v) for u in S are uncovered cross edges.
              * No internal edge (a, b) with a,b in S is already covered.
            - To maximize S, we want the largest subset of N_K(v) such that:
              * All cross edges (u,v) for u in S are uncovered.
              * The subgraph induced by S on K has no covered internal edges.
            - This is like finding a clique in a modified graph.
            
          Given time constraints, let's use a very simple but valid approach:
          
          Simple valid approach:
          - For each cross edge (u, v), create a clique [u, v]. This covers 
            exactly that edge. But this uses too many cliques.
          
          Better simple approach:
          - For each v in I, let N_v = N_K(v).
          - Partition N_v into singletons: for each u in N_v, create clique [u, v].
          - This covers all cross edges. Total cross cliques = sum_{v in I} |N_v|.
          - For internal K edges: create one clique K. This covers all C(k,2) 
            internal edges.
          - But wait: do the cross cliques [u, v] cover any internal edges? No, 
            they only have two vertices, one from K and one from I. So they only 
            cover the cross edge (u, v).
          - The internal clique K covers all internal edges.
          - Total cliques = 1 + sum_{v in I} |N_v|.
          
          This is valid! Let's compute it.
          
          Can we do better? Yes, by grouping multiple u's with the same v.
          - For each v in I, partition N_v into subsets S_1, S_2, ...
          - For each S_j, create clique S_j ∪ {v}.
          - This covers cross edges (u, v) for u in S_j, AND internal edges 
            within S_j.
          - The internal edges within S_j are now covered, so we must exclude 
            them from the internal clique K.
          
          Revised:
          - Let's collect all internal K edges that are "covered" by Type B 
            cliques.
          - For each v in I, partition N_v into subsets. For each subset S, 
            the internal edges within S are covered by clique S ∪ {v}.
          - After processing all v, the remaining internal K edges (not covered 
            by any Type B clique) must be covered by Type A cliques.
          - To minimize Type A cliques, we want to cover the remaining internal 
            edges with as few cliques as possible.
          
          For small k, we can use a greedy: 
          - The remaining internal edges form a graph G_rem on K.
          - We need to partition E(G_rem) into cliques.
          - Greedy: repeatedly find a maximal clique in G_rem, remove its edges, 
            repeat.
          
          Let's implement this.
          
          Steps for solve_split_graph:
          1. Generate the graph (edges).
          2. Identify K = {0, ..., k-1}, I = {k, ..., n-1}.
          3. For each v in I, compute N_v = {u in K : (u, v) is an edge}.
          4. Greedily partition N_v for each v to form Type B cliques.
             - To minimize the number of Type B cliques, we want each S to be 
               as large as possible.
             - Constraint: the internal edges within S must not conflict with 
               other Type B cliques' S sets? Actually, they CAN overlap in 
               vertices, but the internal edges within overlapping regions 
               would be double-covered. So we need the S sets (across all v) 
               to have disjoint internal edge sets.
             - This is equivalent to: the family of subsets {S} (one per Type 
               B clique) must be such that no internal edge (a,b) is contained 
               in more than one S.
             - So we need to choose S_{v,1}, S_{v,2}, ... for each v such that 
               the edge-disjointness condition holds.
             
          This is getting complex. Given the time budget, let's use the simple 
          valid approach first, and if we have time, try to improve.
          
          Simple valid approach:
          - Type B: for each cross edge (u, v), clique [u, v].
          - Type A: clique K for all internal edges.
          - Total = 1 + |E_cross|.
          
          Let's compute |E_cross| for the given instances.
          """
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        adj = [[False] * n for _ in range(n)]
        for (u, v) in edges:
            adj[u][v] = adj[v][u] = True
        
        K_vertices = list(range(k))
        I_vertices = list(range(k, n))
        
        # Compute N_v for each v in I
        N_v = {}
        for v in I_vertices:
            N_v[v] = [u for u in K_vertices if adj[u][v]]
        
        # Simple approach: one clique per cross edge, one clique for K
        cliques = []
        
        # Internal clique K
        cliques.append(list(K_vertices))
        
        # Cross edges: each as a 2-clique
        for v in I_vertices:
            for u in N_v[v]:
                cliques.append([u, v])
        
        return cliques

    def try_improved_solve(n, k, p, seed):
        """
        Try to improve by grouping cross edges per v into larger cliques.
        """
        edges = instance(n, k, p, seed)
        
        adj = [[False] * n for _ in range(n)]
        for (u, v) in edges:
            adj[u][v] = adj[v][u] = True
        
        K_vertices = list(range(k))
        I_vertices = list(range(k, n))
        
        N_v = {}
        for v in I_vertices:
            N_v[v] = [u for u in K_vertices if adj[u][v]]
        
        # Greedy: for each v, partition N_v into subsets S such that 
        # the internal edges within S are disjoint across all cliques.
        
        # Track which internal K edges are already covered
        covered_internal = set()
        
        def internal_edge_key(u, w):
            return (min(u, w), max(u, w))
        
        cliques = []
        
        # Process v in I in some order
        # Sort by |N_v| descending to handle larger ones first?
        I_sorted = sorted(I_vertices, key=lambda v: len(N_v[v]), reverse=True)
        
        for v in I_sorted:
            remaining = list(N_v[v])
            
            while remaining:
                # Find largest subset S of remaining such that 
                # all internal edges within S are not in covered_internal
                # Greedy: start with all remaining, remove conflicts
                S = list(remaining)
                
                # Check for conflicts: internal edges within S that are covered
                conflict_found = True
                while conflict_found and len(S) > 1:
                    conflict_found = False
                    for i in range(len(S)):
                        for j in range(i+1, len(S)):
                            key = internal_edge_key(S[i], S[j])
                            if key in covered_internal:
                                # Remove the vertex involved in the most conflicts
                                # For simplicity, remove S[j]
                                S.pop(j)
                                conflict_found = True
                                break
                        if conflict_found:
                            break
                
                if len(S) >= 1:
                    # Create clique S ∪ {v}
                    clique = S + [v]
                    cliques.append(clique)
                    
                    # Mark internal edges within S as covered
                    for i in range(len(S)):
                        for j in range(i+1, len(S)):
                            key = internal_edge_key(S[i], S[j])
                            covered_internal.add(key)
                    
                    # Remove S from remaining
                    S_set = set(S)
                    remaining = [x for x in remaining if x not in S_set]
        
        # Now cover remaining internal K edges
        # Find all internal K edges not in covered_internal
        remaining_internal_edges = []
        for i in range(k):
            for j in range(i+1, k):
                key = (i, j)
                if key not in covered_internal:
                    remaining_internal_edges.append((i, j))
        
        if remaining_internal_edges:
            # Greedy cover remaining internal edges with cliques
            # Build graph of remaining internal edges
            rem_adj = {u: set() for u in range(k)}
            for (u, v) in remaining_internal_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
            
            # Repeatedly find maximal clique
            uncovered = set(remaining_internal_edges)
            
            while uncovered:
                # Find a maximal clique in the remaining graph
                # Start with a random edge, try to extend
                first_edge = next(iter(uncovered))
                u, v = first_edge
                
                # Try to build clique containing u, v
                clique = [u, v]
                
                # Try to add vertices
                changed = True
                while changed:
                    changed = False
                    for w in range(k):
                        if w in clique:
                            continue
                        # Check if w is connected to all in clique
                        connected = all(w in rem_adj[x] for x in clique)
                        if connected:
                            # Check that all edges from w to clique are uncovered
                            all_uncovered = all(
                                (min(w, x), max(w, x)) in uncovered 
                                for x in clique
                            )
                            if all_uncovered:
                                clique.append(w)
                                changed = True
                
                # Create clique
                cliques.append(clique)
                
                # Remove edges covered by this clique
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        key = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                        if key in uncovered:
                            uncovered.discard(key)
        
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    # First pass: simple approach
    results_simple = []
    for spec in specs:
        if time.time() > deadline:
            break
        results_simple.append(solve_split_graph(*spec))
    
    if len(results_simple) == 3:
        total_simple = sum(len(r) for r in results_simple)
        best_total = total_simple
        best_result = results_simple
    
    # Try improved approach if time permits
    if time.time() < deadline - 5:
        results_improved = []
        for spec in specs:
            if time.time() > deadline:
                break
            results_improved.append(try_improved_solve(*spec))
        
        if len(results_improved) == 3:
            total_improved = sum(len(r) for r in results_improved)
            if total_improved < best_total:
                best_total = total_improved
                best_result = results_improved
    
    # Validate and return
    if best_result is None:
        # Fallback: should not happen
        return [[[]] for _ in range(3)]
    
    # Convert to ensure proper types
    final = []
    for cliques in best_result:
        final_cliques = []
        for clique in cliques:
            final_cliques.append([int(v) for v in clique])
        final.append(final_cliques)
    
    return final

# EVOLVE-BLOCK-END
