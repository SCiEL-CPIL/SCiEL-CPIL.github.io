import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        # K is 0..k-1, I is k..n-1
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # Strategy:
        # 1. For each v in I, we must cover the edges (u, v) for u in N(v).
        #    We can form cliques of the form {v} U S where S is a subset of N(v).
        #    Since K is a clique, any subset S of K is a clique.
        #    The clique {v} U S covers:
        #      - Cross edges (u, v) for u in S.
        #      - K-edges (u, w) for u, w in S.
        #    We want to minimize the total number of such cliques plus the cliques needed for remaining K-edges.
        #    
        # 2. If we use singletons S={u}, we use |N(v)| cliques and cover 0 K-edges.
        #    If we use pairs S={u, w}, we use |N(v)|/2 cliques (approx) and cover 1 K-edge per clique.
        #    If we use larger sets, we cover more K-edges per clique, but the "efficiency" of covering K-edges
        #    decreases because a clique of size s covers s*(s-1)/2 K-edges but only saves (s-1) cross-cliques
        #    compared to singletons.
        #    
        #    Let's compare:
        #    Singletons: cost 1 clique, covers 0 K-edges, covers 1 cross-edge.
        #    Pair: cost 1 clique, covers 1 K-edge, covers 2 cross-edges.
        #       Saving: 1 clique compared to 2 singletons. Cost: 1 K-edge removed from K.
        #    Triple: cost 1 clique, covers 3 K-edges, covers 3 cross-edges.
        #       Saving: 2 cliques compared to 3 singletons. Cost: 3 K-edges removed.
        #       Ratio of savings to K-edges removed:
        #       Pair: 1/1 = 1.
        #       Triple: 2/3 = 0.67.
        #       Quad: 3/6 = 0.5.
        #    
        #    So, using pairs is more efficient in terms of "cliques saved per K-edge consumed" than larger sets.
        #    However, we must also consider the cost of covering the remaining K-edges.
        #    Removing edges from K makes the remaining K-graph sparser.
        #    Covering a sparse graph with cliques might require more cliques if the graph is not a union of disjoint cliques.
        #    But K_k is dense. Removing a small number of edges (a matching) leaves a graph that is still very dense.
        #    
        #    Let's look at the structure of K_k \ M where M is a matching.
        #    If M is a matching of size m, the remaining graph has k-2m vertices not in M and m pairs.
        #    Actually, it's simpler: the vertices are partitioned into m pairs (non-adjacent) and k-2m singletons.
        #    Any clique can contain at most one vertex from each pair.
        #    So the maximum clique size is m + (k-2m) = k-m? No.
        #    If we pick one from each pair and all singletons, we get a set of size m + (k-2m) = k-m.
        #    Are they all connected?
        #    Yes, because any two vertices from different pairs or singletons are connected in K_k \ M.
        #    So there is a clique of size k-m.
        #    Does this clique cover all edges?
        #    No, it covers edges between the selected vertices.
        #    What about edges between a selected vertex u (from pair {u, u'}) and an unselected vertex u'?
        #    (u, u') is NOT an edge in K_k \ M. So that edge is gone.
        #    But what about edges between u and some other vertex w?
        #    If w is selected, (u, w) is covered.
        #    If w is not selected (i.e., w is u''), is (u, w) covered?
        #    No, because w is not in the clique.
        #    So we need more cliques.
        #    
        #    Actually, the graph K_k \ M can be covered by:
        #    1. One large clique C consisting of one vertex from each pair in M and all vertices not in M.
        #       Size of C = m + (k-2m) = k-m.
        #       This covers all edges incident to vertices in C, except edges between C and the "other" vertices of the pairs.
        #       Wait, the "other" vertices are not in C.
        #       Let P = { {u_i, v_i} for i=1..m } be the pairs in M.
        #       Let S be the set of singletons.
        #       Let C = { u_i for i=1..m } U S.
        #       Edges covered by C: all edges between any two vertices in C.
        #       Remaining edges:
        #       - Edges between v_i and u_j (i != j). (v_i, u_j) is an edge in K_k \ M.
        #       - Edges between v_i and s (s in S). (v_i, s) is an edge.
        #       - Edges between v_i and v_j (i != j). (v_i, v_j) is an edge.
        #       So the remaining graph is induced by V' = { v_i for i=1..m }.
        #       Wait, are there edges between v_i and v_j? Yes.
        #       Are there edges between v_i and u_j? Yes.
        #       So the remaining edges are incident to V'.
        #       Can we cover them with one more clique?
        #       The subgraph induced by V' is a clique K_m.
        #       But we also have edges from V' to C.
        #       A clique containing v_i cannot contain u_i (no edge).
        #       But it can contain u_j (j != i) and s.
        #       So we can form a clique C' = { v_i for i=1..m } U { u_j for j=1..m, j != i? } No.
        #       Let's just use a greedy algorithm for the remaining edges.
        #       Since k is small (10, 20, 30), and the number of remaining edges is small if M is large,
        #       greedy should be fast and good.
        
        # Implementation:
        # 1. For each v in I, find a maximal matching in the graph of available K-edges induced by N(v).
        #    Since K is a clique, the available K-edges induced by N(v) is a complete graph on N(v) minus the edges already removed.
        #    The removed edges form a matching globally.
        #    So the graph is a complete graph minus a matching.
        #    A maximal matching in such a graph is maximum.
        #    We can just greedily pair up vertices in N(v) that are still connected.
        
        available_k_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        cliques = []
        
        # Sort I vertices by degree in N(v) descending? Or ascending?
        # If we process high-degree vertices first, they can "consume" more K-edges.
        # This might leave fewer K-edges for low-degree vertices, forcing them to use more singletons?
        # Or it might be better to leave K-edges for vertices that can use them efficiently?
        # Let's try sorting by degree descending.
        vs = sorted(range(k, n), key=lambda v: len(neighbors[v]), reverse=True)
        
        for v in vs:
            N_v = neighbors[v]
            if not N_v:
                continue
            
            # Find maximal matching in N_v using available_k_edges
            matched = set()
            pairs = []
            N_v_list = list(N_v)
            
            # To make matching maximal, we can just iterate and pair up.
            # Since the graph is dense, greedy works well.
            i = 0
            while i < len(N_v_list):
                if N_v_list[i] in matched:
                    i += 1
                    continue
                u = N_v_list[i]
                found_pair = False
                for j in range(i + 1, len(N_v_list)):
                    w = N_v_list[j]
                    if w in matched:
                        continue
                    edge = (min(u, w), max(u, w))
                    if edge in available_k_edges:
                        pairs.append((u, w))
                        matched.add(u)
                        matched.add(w)
                        available_k_edges.remove(edge)
                        found_pair = True
                        break
                if not found_pair:
                    # u cannot be paired with anyone else
                    # It will be a singleton
                    pass
                i += 1
            
            for u, w in pairs:
                cliques.append([u, w, v])
            
            for u in N_v:
                if u not in matched:
                    cliques.append([u, v])
        
        # Cover remaining K-edges
        remaining_edges = list(available_k_edges)
        
        # Greedy clique cover for remaining edges
        # Since k is small, we can use a simple greedy:
        # 1. Pick an edge (u, v).
        # 2. Find the largest clique containing (u, v) in the remaining graph.
        # 3. Add it, remove its edges.
        
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: just add edges as 2-cliques
                for e in remaining_edges:
                    cliques.append(list(e))
                break
            
            u, v = remaining_edges[0]
            # Find all x such that (u, x) and (v, x) are in remaining_edges
            # This forms a candidate clique {u, v} U {x...}
            candidates = set()
            # Optimization: only check edges incident to u or v
            # But for k=30, O(E) is fine.
            for e in remaining_edges:
                a, b = e
                if a == u:
                    candidates.add(b)
                elif b == u:
                    candidates.add(a)
                if a == v:
                    candidates.add(b)
                elif b == v:
                    candidates.add(a)
            
            # Filter candidates to ensure they form a clique with u and v
            # i.e., (x, y) must be in remaining_edges for all x, y in candidates
            # And (u, x), (v, x) must be in remaining_edges (already true by construction of candidates?
            # No, we added b if a==u. So (u, b) is in remaining_edges.
            # But we need (v, b) to be in remaining_edges too.
            
            valid_candidates = []
            for c in candidates:
                if c == u or c == v:
                    continue
                edge_uf = (min(u, c), max(u, c))
                edge_vf = (min(v, c), max(v, c))
                if edge_uf in remaining_edges and edge_vf in remaining_edges:
                    valid_candidates.append(c)
            
            # Now we need to select a subset of valid_candidates that forms a clique among themselves.
            # Since the graph is dense, we can just take all of them?
            # No, (c1, c2) might not be an edge.
            # But in K_k \ M, the only non-edges are the matching edges.
            # So if (c1, c2) is not an edge, then (c1, c2) is in M.
            # Since M is a matching, c1 and c2 are matched to each other.
            # But if c1 and c2 are matched, they are not connected.
            # So we can include at most one from each pair in M.
            # But we don't have M explicitly. We have remaining_edges.
            # So we just need to check if (c1, c2) is in remaining_edges.
            
            # Greedy: sort valid_candidates and add if it connects to all already added.
            # Or just take the first one?
            # Let's try to maximize the clique size.
            # Since k is small, we can try a simple greedy:
            # Start with {u, v}.
            # Try to add each c in valid_candidates if it connects to all current members.
            
            clique = {u, v}
            for c in valid_candidates:
                # Check if c connects to all in clique
                can_add = True
                for x in clique:
                    if x == c:
                        continue
                    edge = (min(x, c), max(x, c))
                    if edge not in remaining_edges:
                        can_add = False
                        break
                if can_add:
                    clique.add(c)
            
            clique_list = list(clique)
            cliques.append(clique_list)
            
            # Remove edges covered by this clique
            edges_to_remove = set()
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    edges_to_remove.add((min(a, b), max(a, b)))
            
            remaining_edges = [e for e in remaining_edges if e not in edges_to_remove]
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Should not happen, but just in case
            pass
        cliques = solve_instance(*spec)
        clean_cliques = []
        for c in cliques:
            # Ensure all are ints
            clean_cliques.append([int(x) for x in c])
        results.append(clean_cliques)
        
    return results
