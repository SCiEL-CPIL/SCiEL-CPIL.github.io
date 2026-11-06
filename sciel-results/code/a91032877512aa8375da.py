import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Precompute edge representation for quick lookup
        # We represent an edge as (min, max)
        
        # Fallback: each edge is a clique
        best_cliques = [list(e) for e in edges]
        best_count = len(best_cliques)
        
        end_time = start_time + time_budget
        
        # Helper to check if a set of vertices forms a clique using only remaining edges
        def is_clique_in_remaining(vertices, remaining_edges):
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    u, v = vertices[i], vertices[j]
                    e = (min(u, v), max(u, v))
                    if e not in remaining_edges:
                        return False
            return True
        
        # Helper to get the set of edges covered by a clique
        def clique_edges(vertices):
            res = set()
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    u, v = vertices[i], vertices[j]
                    res.add((min(u, v), max(u, v)))
            return res

        # Greedy algorithm with multiple restarts
        # Strategy:
        # 1. Pick a seed edge.
        # 2. Grow the clique greedily by adding the vertex that covers the most "critical" edges.
        # 3. To improve, we can try different growth heuristics or randomizations.
        
        # Heuristic for adding vertex w to clique C:
        # Score = number of edges (w, x) where x in remaining_edges and x not in C.
        # Or: Score = number of edges (w, x) where x is a common neighbor of C in remaining graph.
        
        # Let's use a simple but effective heuristic:
        # Add w that maximizes the number of remaining edges incident to w that are also incident to some vertex in C?
        # Actually, adding w covers edges (w, c) for c in C.
        # It also "consumes" w, so it prevents w from being used in other cliques.
        # A good proxy for "value" is the degree of w in the remaining graph, restricted to vertices that can join the clique.
        
        # Better heuristic:
        # When choosing next vertex w for clique C:
        # Candidates are vertices w such that (w, c) in remaining_edges for all c in C.
        # Among candidates, pick w that maximizes the number of edges (w, x) in remaining_edges 
        # where x is also a candidate (or just x in remaining graph).
        # This tends to find larger cliques.
        
        attempts = 0
        # Use a fixed number of restarts or time-based
        # For n=32, we can afford many restarts. For n=64, fewer.
        
        while time.time() < end_time:
            attempts += 1
            attempt_rng = random.Random(seed + attempts * 12345 + 67890)
            
            remaining_edges = set(edges)
            cliques = []
            
            while remaining_edges:
                # Pick a seed edge
                # To diversify, pick randomly from remaining edges
                edge_list = list(remaining_edges)
                attempt_rng.shuffle(edge_list)
                
                best_seed_clique = None
                max_cleared = 0
                best_seed_clique_edges = None
                
                # Try a few seed edges to find a good starting clique
                # The "quality" of a seed edge is how large a clique we can form from it
                for seed_edge in edge_list[:15]:
                    u, v = seed_edge
                    current_clique = [u, v]
                    
                    # Candidates: vertices w such that (w, c) in remaining_edges for all c in current_clique
                    # Start with intersection of neighbors of u and v in the remaining graph
                    # Note: adj[u] contains all neighbors in original graph.
                    # We need to filter by remaining_edges.
                    
                    # Efficient way to get candidates:
                    # For each candidate w, check if (w, c) in remaining for all c in clique.
                    # Since clique is small, this is fast.
                    
                    # Initialize candidates as intersection of adj[u] and adj[v]
                    # But we must also ensure edges are in remaining.
                    
                    cand_set = set(adj[u])
                    cand_set.intersection_update(adj[v])
                    cand_set.discard(u)
                    cand_set.discard(v)
                    
                    # Filter candidates: must have edges to all in current_clique in remaining_edges
                    # Since current_clique is [u, v], we need (w, u) and (w, v) in remaining.
                    # We know (w, u) and (w, v) are in original graph (from adj).
                    # Check remaining.
                    
                    valid_cands = []
                    for w in cand_set:
                        if (min(w, u), max(w, u)) in remaining_edges and (min(w, v), max(w, v)) in remaining_edges:
                            valid_cands.append(w)
                    
                    cand_set = set(valid_cands)
                    
                    while cand_set:
                        # Pick best candidate
                        # Heuristic: Maximize degree in remaining graph among candidates?
                        # Or maximize number of edges to other candidates?
                        
                        # Let's try: Maximize number of edges (w, x) in remaining_edges where x in cand_set.
                        # This encourages forming dense subgraphs (cliques).
                        
                        best_w = None
                        max_score = -1
                        
                        # For small n, we can afford O(|cand_set|^2)
                        # Precompute degrees in remaining graph for all vertices?
                        # Or just compute on the fly.
                        
                        for w in cand_set:
                            score = 0
                            for x in cand_set:
                                if w == x:
                                    continue
                                e = (min(w, x), max(w, x))
                                if e in remaining_edges:
                                    score += 1
                            # Also add degree to non-candidates?
                            # No, just internal density is a good proxy for clique potential.
                            
                            # Tie-breaker: random
                            if score > max_score:
                                max_score = score
                                best_w = w
                            elif score == max_score:
                                # Random tie-break
                                if attempt_rng.random() < 0.5:
                                    best_w = w
                            
                        if best_w is None:
                            break
                            
                        current_clique.append(best_w)
                        cand_set.remove(best_w)
                        
                        # Update candidates:
                        # New candidate x must be in old cand_set \ {best_w}
                        # AND have edge to best_w in remaining_edges
                        # AND have edges to all other clique members (already satisfied for old cands, 
                        # but we must check edge to best_w)
                        
                        new_cand_set = set()
                        for w in cand_set:
                            e_bw = (min(w, best_w), max(w, best_w))
                            if e_bw in remaining_edges:
                                # Check edges to other clique members?
                                # They were valid for [u, v, ..., prev_best].
                                # Now we added best_w.
                                # We checked (w, best_w).
                                # Do we need to check (w, c) for c in clique?
                                # Yes, but they were checked when w was added to cand_set initially or in previous steps.
                                # Wait, if w was in cand_set, it had edges to all previous clique members.
                                # It does NOT necessarily have edge to best_w (we just checked that).
                                # So it is valid for the new clique.
                                new_cand_set.add(w)
                        
                        cand_set = new_cand_set

                    # Evaluate this clique
                    if len(current_clique) >= 2:
                        c_edges = clique_edges(current_clique)
                        # Verify all edges are in remaining
                        valid = True
                        for e in c_edges:
                            if e not in remaining_edges:
                                valid = False
                                break
                        if valid:
                            if len(c_edges) > max_cleared:
                                max_cleared = len(c_edges)
                                best_seed_clique = current_clique
                                best_seed_clique_edges = c_edges

                if not best_seed_clique:
                    # Fallback: take the first edge
                    e = edge_list[0]
                    best_seed_clique = list(e)
                    best_seed_clique_edges = {e}
                
                cliques.append(best_seed_clique)
                for e in best_seed_clique_edges:
                    remaining_edges.discard(e)
            
            # Check if this solution is better
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = [list(c) for c in cliques]
                
            # If we hit a very good solution, maybe stop?
            # Lower bound: sum of degrees / 2 / max_clique_size?
            # Or just if best_count is close to theoretical min?
            # For now, keep searching until time runs out.
            
            if time.time() > end_time - 2:
                break

        return best_cliques

    # Solve both instances
    t1 = time.time()
    # Allocate more time to the larger, harder instance (n=64)
    # Total time ~145s.
    # n=32 is smaller. Let's give 30s to n=32 and 110s to n=64.
    elapsed1 = t1 - start_time
    budget1 = min(30, DEADLINE - elapsed1 - 5)
    if budget1 < 1:
        budget1 = 1
    cliques1 = solve_instance(32, 0.6, 601, budget1)
    
    t2 = time.time()
    elapsed2 = t2 - start_time
    budget2 = min(110, DEADLINE - elapsed2)
    if budget2 < 1:
        budget2 = 1
    cliques2 = solve_instance(64, 0.55, 602, budget2)
    
    return [cliques1, cliques2]
