import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 20
    
    def time_ok():
        return time.time() < deadline

    # The constraint: For any three vertices u, v, w, 
    # |N(u) ∩ N(v) ∩ N(w)| <= 2.
    # This is equivalent to saying that no three vertices have 3 common neighbors.
    
    # Strategy: 
    # 1. Start with an empty graph.
    # 2. Use a deterministic ordering of edges, prioritized by potential to maintain density.
    # 3. Greedily add edges if they don't violate the constraint.
    # 4. Use a "look-ahead" or simple backtracking if we hit a dead end, 
    #    but given n=20, a smart greedy might suffice or we can use a 
    #    "repair" phase that is more structured than random.
    
    # However, the inspirations suggest "Paley-type" or "deterministic exhaustive" works better.
    # Let's try a hybrid: 
    # 1. Generate a candidate graph based on a known dense K3,3-free structure (e.g., based on finite fields or just a strong greedy).
    # 2. Refine it using a deterministic local search that tries all possible single-edge swaps 
    #    in a specific order, not random.
    
    # Actually, the most robust "different" strategy is **Simulated Annealing** with a 
    # carefully designed cooling schedule and a different neighborhood definition 
    # (e.g., 2-edge swaps instead of single edge add/remove).
    # But the prompt asks for a STRUCTURAL change. 
    # Let's try **Integer Linear Programming (ILP)**? No, scipy is available but no dedicated ILP solver like pulp/cvxpy is guaranteed.
    # SciPy has `linprog` but it's for LP, not IP.
    
    # Let's stick to a **Deterministic Greedy with Backtracking** that is exhaustive enough for n=20.
    # We will order edges by a heuristic: edges between high-degree potential vertices first.
    # But "potential" is unknown.
    
    # Alternative Structural Change: **Partition-based Construction**.
    # We know R(3,3)=6. This means any graph on 6 vertices contains a triangle or an independent set of size 3.
    # This doesn't directly give K3,3.
    # K3,3 is bipartite.
    
    # Let's look at the "Inspirations":
    # Insp 1: "Paley-type graph". Paley graphs are strongly regular. 
    # Paley graph of order q (q ≡ 1 mod 4) is K_{3,3}-free?
    # For q=17 (prime), Paley(17) has 17 vertices. We need 20.
    # We can take Paley(17) and add 3 vertices? Or use a different algebraic construction.
    
    # Let's try a **Deterministic Local Search** that is exhaustive over a subset of moves.
    # The seed uses random shuffles. I will use a **systematic sweep**.
    
    def build_adjacency(edge_list):
        adj = [0] * n
        for u, v in edge_list:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def edge_count(adj):
        total = 0
        for i in range(n):
            total += adj[i].bit_count()
        return total // 2

    def is_k33_free(adj):
        """Check if graph is K3,3-free."""
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                aij = ai & adj[j]
                for k in range(j + 1, n):
                    if (aij & adj[k]).bit_count() >= 3:
                        return False
        return True
        
    def find_violations(adj):
        """Return list of (i, j, k) triples with >= 3 common neighbors."""
        viols = []
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                aij = ai & adj[j]
                for k in range(j + 1, n):
                    if (aij & adj[k]).bit_count() >= 3:
                        viols.append((i, j, k))
        return viols

    # Phase 1: Deterministic Greedy Construction
    # Order edges by a fixed heuristic: e.g., sum of indices, or just lexicographic.
    # To make it "dense", we should try to add edges that don't create violations.
    # A simple greedy: iterate all edges (u,v) in a fixed order. Add if valid.
    # This is too weak.
    
    # Better Greedy: 
    # 1. Start with empty graph.
    # 2. While there are edges to add:
    #    - Find the edge (u,v) that, if added, results in the fewest new violations 
    #      (ideally 0) and maximizes some secondary metric.
    #    - If no edge can be added without violation, we are stuck? 
    #      No, we can remove an edge to allow others.
    
    # Let's use a **Repair-based Deterministic Search**:
    # 1. Start with a random dense graph (seeded for reproducibility in this run, but we want deterministic).
    #    Actually, let's start with the **Complement of a Sparse Graph**.
    #    If G is K3,3-free, is G^c K3,3-free? Not necessarily.
    
    # Let's try the **Paley Graph** approach explicitly.
    # Paley graph P(q) for prime q ≡ 1 mod 4.
    # Vertices 0..q-1. Edge (u,v) if (v-u) is a quadratic residue mod q.
    # For q=17, P(17) has 16 edges? No, degree (q-1)/2 = 8. Total edges 17*8/2 = 68.
    # We need 20 vertices.
    # We can embed P(17) into 20 vertices by adding 3 isolated vertices? No, we want max edges.
    # We can add edges from the 3 new vertices to the 17 old ones.
    
    def is_quadratic_residue(a, p):
        if a == 0:
            return True
        return pow(a, (p-1)//2, p) == 1

    def create_paley_subgraph(p, offset=0):
        """Create Paley graph on p vertices, mapped to offset..offset+p-1"""
        adj = [0] * n
        for i in range(p):
            u = i + offset
            for j in range(i+1, p):
                v = j + offset
                if is_quadratic_residue((v - u) % p, p) or is_quadratic_residue((u - v) % p, p):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    # Try p=17.
    # Add 3 vertices: 17, 18, 19.
    # Connect them to the Paley graph in a way that preserves K3,3-freeness.
    # This is a specific structural construction.
    
    best_adj = [0] * n
    best_count = -1
    
    # Candidate 1: Paley(17) + 3 vertices connected greedily
    if time_ok():
        adj_paley = create_paley_subgraph(17, 0)
        # Greedily add edges for vertices 17, 18, 19
        for new_v in [17, 18, 19]:
            # Try connecting to all 17 vertices, check validity
            # We want to maximize edges, so try all, then remove conflicts.
            # Simpler: iterate all possible edges for new_v, add if valid.
            for u in range(17):
                # Try adding (new_v, u)
                adj_paley[new_v] |= (1 << u)
                adj_paley[u] |= (1 << new_v)
                if not is_k33_free(adj_paley):
                    adj_paley[new_v] &= ~(1 << u)
                    adj_paley[u] &= ~(1 << new_v)
            # Also try connecting new_v to other new vertices
        for u in [17, 18]:
            for v in [u+1, 19]:
                if v <= 19 and v > u:
                    adj_paley[u] |= (1 << v)
                    adj_paley[v] |= (1 << u)
                    if not is_k33_free(adj_paley):
                        adj_paley[u] &= ~(1 << v)
                        adj_paley[v] &= ~(1 << u)
                        
        count = edge_count(adj_paley)
        if count > best_count:
            best_count = count
            best_adj = adj_paley[:]

    # Candidate 2: Two disjoint Paley(13)? No, 13*2=26 > 20.
    # Candidate 3: Complete Bipartite K10,10 is NOT K3,3-free (it contains K3,3).
    # Candidate 4: Randomized Deterministic Search (Simulated Annealing)
    # Since the "Paley" construction might not be optimal for n=20, 
    # we use the Paley result as a strong seed for a **Deterministic Local Search**.
    
    # The structural difference here: 
    # The seed program uses RANDOM local search.
    # This program uses a SYSTEMATIC local search (sweeping all moves) 
    # starting from a STRUCTURAL seed (Paley).
    
    def systematic_local_search(adj):
        adj = adj[:]
        improved = True
        # Limit iterations to avoid infinite loops if we oscillate
        max_sweeps = 50
        sweep_count = 0
        while improved and sweep_count < max_sweeps and time_ok():
            improved = False
            sweep_count += 1
            
            # Generate all possible single-edge additions and removals?
            # No, let's do **Edge Swaps** (remove one, add one) and **Additions**.
            
            # 1. Try all possible additions
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        # Try adding
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        if is_k33_free(adj):
                            improved = True
                        else:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
            
            # 2. Try all possible swaps (remove (a,b), add (c,d))
            # This is O(E * Non-E) which is ~100*100 = 10k checks. Fast enough.
            edges = []
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        edges.append((i, j))
                    else:
                        non_edges.append((i, j))
                        
            for a, b in edges:
                for c, d in non_edges:
                    if (a == c and b == d) or (a == d and b == c):
                        continue
                    # Remove (a,b), Add (c,d)
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    adj[c] |= (1 << d)
                    adj[d] |= (1 << c)
                    
                    if is_k33_free(adj):
                        improved = True
                        # Do not break immediately, continue sweeping to find more improvements
                        # But we must be careful not to revert a good swap in the same sweep.
                        # Since we are modifying adj in place, subsequent checks in this sweep
                        # use the new state. This is a "greedy sweep".
                    else:
                        # Revert
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                        adj[c] &= ~(1 << d)
                        adj[d] &= ~(1 << c)
                        
        return adj

    if time_ok():
        refined = systematic_local_search(best_adj)
        refined_count = edge_count(refined)
        if refined_count > best_count:
            best_count = refined_count
            best_adj = refined

    # If the Paley seed was weak, try a second structural seed: 
    # "Turán-type" or "Random Dense" but processed by the SAME systematic search.
    # But to be materially different, we stick to the Structural Seed + Deterministic Refinement.
    
    # One more structural idea: **Complement of a Triangle-Free Graph**?
    # No, K3,3 is not related directly to triangles.
    
    # Let's ensure we have a fallback if Paley is bad.
    # Seed with a random graph of density 0.5, then run systematic search.
    if time_ok():
        random.seed(42)
        adj_rand = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_rand[i] |= (1 << j)
                    adj_rand[j] |= (1 << i)
        
        # First, repair it to be K3,3-free using deterministic deletion
        # This is similar to the seed's phase 1, but deterministic.
        # We remove the edge that is part of the most violations.
        
        def deterministic_repair(adj):
            adj = adj[:]
            max_iter = 1000
            for _ in range(max_iter):
                if not time_ok():
                    break
                viols = find_violations(adj)
                if not viols:
                    break
                
                # Score edges
                edge_score = {}
                for (i, j, k) in viols:
                    cn = adj[i] & adj[j] & adj[k]
                    if cn.bit_count() < 3:
                        continue
                    # For each common neighbor w, edges (w,i), (w,j), (w,k) are candidates
                    for w in range(n):
                        if cn & (1 << w):
                            if w != i:
                                key = (min(w, i), max(w, i))
                                edge_score[key] = edge_score.get(key, 0) + 1
                            if w != j:
                                key = (min(w, j), max(w, j))
                                edge_score[key] = edge_score.get(key, 0) + 1
                            if w != k:
                                key = (min(w, k), max(w, k))
                                edge_score[key] = edge_score.get(key, 0) + 1
                
                if not edge_score:
                    break
                    
                best_edge = max(edge_score, key=edge_score.get)
                u, v = best_edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
            return adj

        repaired_rand = deterministic_repair(adj_rand)
        # Then refine with systematic search
        if time_ok():
            refined_rand = systematic_local_search(repaired_rand)
            count_rand = edge_count(refined_rand)
            if count_rand > best_count:
                best_count = count_rand
                best_adj = refined_rand

    # Convert to edge list
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edge_list.append([i, j])
    
    return edge_list
