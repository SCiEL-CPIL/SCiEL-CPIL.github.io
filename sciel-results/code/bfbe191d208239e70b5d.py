import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # Pre-compute neighbor sets for each v in I
        N = [set(neighbors[v]) for v in range(k, n)]
        
        # All K-edges
        k_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        k_edges_set = set(k_edges)
        
        # For each K-edge, which vertices in I contain both endpoints?
        edge_to_vs = {}
        for e in k_edges:
            u, w = e
            vs = [v for v in range(k, n) if u in N[v-k] and w in N[v-k]]
            if vs:
                edge_to_vs[e] = vs
        
        # Strategy:
        # 1. Assign K-edges to vertices in I to form size-3 cliques {u, w, v}.
        #    Constraint: For a fixed v, the assigned edges must form a matching
        #    (no shared vertices) because if (u,w) and (u,z) are both assigned to v,
        #    then the cliques are {u,w,v} and {u,z,v}. They share edge (u,v)? No.
        #    They share vertex u and v.
        #    Edge (u,v) is in {u,w,v}. Is it in {u,z,v}? Yes.
        #    So (u,v) would be in two cliques. Invalid.
        #    Therefore, for each v, the neighbors of v used in size-3 cliques must be disjoint.
        #    i.e., the edges assigned to v must form a matching in the graph induced by N(v).
        #    Since K is a clique, any subset of N(v) is a clique.
        #    So we need a matching in the complete graph on N(v)? No, just that the edges don't share endpoints.
        #    So for each v, we pick a set of disjoint pairs from N(v).
        
        # 2. For each v, the remaining neighbors (not used in size-3 cliques) form size-2 cliques {u, v}.
        
        # 3. K-edges not assigned to any v must be covered by cliques within K.
        #    Since K is a clique, any subset is a clique. We want to cover the remaining K-edges
        #    with minimum number of cliques.
        #    Actually, since K is a clique, we can just cover the remaining edges with one big clique?
        #    No, we only need to cover the *remaining* edges.
        #    The remaining edges form a subgraph of K. Since K is a clique, the remaining edges
        #    form a graph where every pair of vertices that has an edge remaining is connected.
        #    Wait, K is a complete graph. So the remaining edges form a graph G_rem on subset of K.
        #    We need to partition the edges of G_rem into cliques.
        #    Since G_rem is a subgraph of a clique, its cliques are just any subsets of its vertices that are edges in G_rem?
        #    No, a clique in the partition must be a set of vertices such that all pairs are edges in G_rem.
        #    Since G_rem is a subgraph of K (complete), a set S is a clique in G_rem iff all pairs in S are edges in G_rem.
        #    This is equivalent to S being a clique in the graph G_rem.
        #    To minimize the number of cliques to cover the edges of G_rem, we can use a greedy approach:
        #    Find a maximal clique in G_rem, remove its edges, repeat.
        
        # Optimization:
        # The "value" of assigning an edge (u,w) to v is 1 (it saves 1 clique compared to covering it in K,
        # and it combines two K-I edges into one 3-clique instead of two 2-cliques? 
        # Let's count:
        # If (u,w) is NOT assigned to v:
        #   - (u,v) is a clique {u,v} (1 clique)
        #   - (w,v) is a clique {w,v} (1 clique)
        #   - (u,w) is covered in K (part of some K-clique)
        # If (u,w) IS assigned to v:
        #   - {u,w,v} is a clique (1 clique)
        #   - This covers (u,v), (w,v), and (u,w).
        # So we save 1 clique for the K-I edges (2 become 1) and we "consume" the K-edge (u,w) 
        # which would have been covered by a K-clique.
        # Does consuming (u,w) reduce the number of K-cliques?
        # If we cover all K-edges with one big clique K, it costs 1.
        # If we remove some edges, we might still need 1 clique if the remaining graph is connected/clique-like.
        # But if we remove enough edges, we might need more cliques? No, removing edges makes it easier to cover?
        # Actually, covering edges with cliques: if we have a complete graph, 1 clique covers all.
        # If we remove edges, the remaining graph is sparser. We might need more cliques?
        # Example: Path of length 2 (3 vertices, 2 edges). 1 clique covers 1 edge, 1 clique covers the other. Total 2.
        # Complete graph K3: 1 clique covers all 3 edges.
        # So removing edges can INCREASE the number of cliques needed to cover the rest?
        # No, that's not right.
        # Let's look at the objective: Minimize Total Cliques = C_I + C_K.
        # C_I: cliques involving I.
        # C_K: cliques within K.
        
        # Let M be the set of K-edges assigned to I.
        # For each v, let M_v be the edges assigned to v. M_v must be a matching in N(v).
        # Number of 3-cliques = |M|.
        # Number of 2-cliques for I = sum_v (|N(v)| - 2|M_v|).
        # Total I-cliques = |M| + sum_v (|N(v)| - 2|M_v|) = sum_v |N(v)| - |M|.
        # So each assigned edge saves 1 clique in the I-part.
        
        # Now for K-part.
        # The edges remaining in K are E_K \ M.
        # We need to cover E_K \ M with cliques.
        # Let f(G) be the minimum number of cliques to cover edges of G.
        # We want to minimize: (sum_v |N(v)| - |M|) + f(E_K \ M).
        
        # Note that f(E_K) = 1 (since K is a clique).
        # If M is small, E_K \ M is close to E_K, so f(E_K \ M) is likely 1 or 2.
        # If M is large, E_K \ M is sparse, f might be larger.
        
        # However, saving 1 clique per edge in M is significant.
        # Max possible |M| is bounded by matching constraints.
        
        # Let's try to maximize |M| subject to M_v being a matching for each v.
        # This is a matching problem in a hypergraph? Or we can use greedy.
        
        # Greedy:
        # Sort edges by "scarcity" (fewer options for v).
        # For each edge, try to assign it to a v that can take it (i.e., u,w not already used in M_v).
        
        # We will run this greedy multiple times with random tie-breaking to find a good M.
        
        best_cliques = None
        best_count = float('inf')
        
        # Pre-compute list of edges for each v that can be in a matching?
        # Just use the graph of N(v). Since N(v) is a clique, any matching is valid.
        # So we just need to pick disjoint pairs from N(v).
        
        # To speed up, we can pre-compute for each edge e, the list of valid vs.
        # We already have edge_to_vs.
        
        # Let's do 50 restarts.
        
        for _ in range(50):
            if time.time() > deadline:
                break
            
            # Randomize edge order
            edges = list(k_edges)
            random.shuffle(edges)
            
            # Sort by scarcity (number of valid vs)
            # This might be expensive to do every time, but k is small.
            # edge_to_vs is static.
            edges.sort(key=lambda e: len(edge_to_vs.get(e, [])))
            
            # State: used vertices in K for each v
            used = [set() for _ in range(k, n)]
            assigned_edges = [] # list of (u, w, v)
            
            for e in edges:
                u, w = e
                candidates = edge_to_vs.get(e, [])
                # Filter candidates where u and w are not used
                valid_vs = [v for v in candidates if u not in used[v-k] and w not in used[v-k]]
                if not valid_vs:
                    continue
                
                # Pick one randomly
                v = random.choice(valid_vs)
                used[v-k].add(u)
                used[v-k].add(w)
                assigned_edges.append((u, w, v))
            
            # Build cliques
            cliques = []
            
            # 3-cliques
            for u, w, v in assigned_edges:
                cliques.append([u, w, v])
            
            # 2-cliques for I
            # For each v, neighbors not in used[v]
            for v in range(k, n):
                N_v = N[v-k]
                used_v = used[v-k]
                for u in N_v:
                    if u not in used_v:
                        cliques.append([u, v])
            
            # K-cliques for remaining edges
            # Remaining edges are those in k_edges but NOT in assigned_edges
            assigned_set = set((u, w) for u, w, v in assigned_edges)
            remaining_edges = [e for e in k_edges if e not in assigned_set]
            
            if remaining_edges:
                rem_set = set(remaining_edges)
                # Greedy clique cover for remaining edges
                # Since it's a subgraph of a clique, we can just find maximal cliques in the remaining graph.
                # Actually, since it's a subgraph of K, any set of vertices is a clique in K,
                # but we only care about edges in rem_set.
                # A clique in the partition must be a set S such that all pairs in S are in rem_set.
                # So we need to find cliques in the graph (V_K, rem_set).
                
                # Simple greedy:
                # While rem_set not empty:
                #   Pick an edge (u,w)
                #   Extend to maximal clique containing u,w in (V_K, rem_set)
                #   Add clique, remove its edges from rem_set
                
                while rem_set:
                    e = next(iter(rem_set))
                    u, w = e
                    clique = {u, w}
                    # Try to add vertices
                    changed = True
                    while changed:
                        changed = False
                        for x in range(k):
                            if x not in clique:
                                # Check if x is connected to all in clique in rem_set
                                if all((min(x, y), max(x, y)) in rem_set for y in clique):
                                    clique.add(x)
                                    changed = True
                                    break # restart loop to allow more additions? 
                                    # Actually, if we add x, we should check again.
                                    # But breaking and restarting the outer while is fine.
                    
                    if len(clique) >= 2:
                        cliques.append(sorted(clique))
                        for a, b in combinations(clique, 2):
                            rem_set.discard((min(a, b), max(a, b)))
                    else:
                        # Should not happen
                        cliques.append([u, w])
                        rem_set.discard(e)
            
            # Convert to int lists
            clean_cliques = [[int(x) for x in c] for c in cliques]
            count = len(clean_cliques)
            
            if count < best_count:
                best_count = count
                best_cliques = clean_cliques
        
        if best_cliques is None:
            # Fallback
            cliques = []
            for v in range(k, n):
                for u in neighbors[v]:
                    cliques.append([u, v])
            for u in range(k):
                for w in range(u+1, k):
                    cliques.append([u, w])
            best_cliques = [[int(x) for x in c] for c in cliques]
            
        return best_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
    return results
