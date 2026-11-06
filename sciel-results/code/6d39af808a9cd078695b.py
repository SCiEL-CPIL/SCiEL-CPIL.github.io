import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 145

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)

        # Build adjacency as sets for fast intersection
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Fallback: each edge is a clique
        best_cliques = [list(e) for e in edges]
        best_count = len(best_cliques)

        end_time = start_time + time_budget

        def enumerate_cliques(start, current, rem_edges):
            """
            Enumerate all maximal cliques in the subgraph induced by rem_edges,
            starting from 'current' and only considering vertices >= start.
            Yields cliques as lists of vertices.
            """
            # Find candidates: vertices w > start such that w is connected to all in current
            # and all edges (w, c) for c in current are in rem_edges
            candidates = []
            if not current:
                # If current is empty, we don't really enumerate from scratch this way in this context
                # But usually we start with an edge.
                return

            # Candidates must be connected to all vertices in current
            cand_set = set()
            for c in current:
                cand_set = cand_set.union(adj[c])
            
            # Filter: must be connected to ALL in current
            # Actually, intersection is better
            if len(current) > 0:
                cand_set = set(adj[current[0]])
                for c in current[1:]:
                    cand_set.intersection_update(adj[c])
            else:
                cand_set = set(range(n))

            # Filter by index > start and edge existence in rem_edges
            for w in cand_set:
                if w <= start:
                    continue
                # Check if all edges (w, c) for c in current are in rem_edges
                valid = True
                for c in current:
                    e = (min(w, c), max(w, c))
                    if e not in rem_edges:
                        valid = False
                        break
                if valid:
                    candidates.append(w)

            if not candidates:
                # Maximal clique found
                if len(current) >= 2:
                    yield list(current)
                return

            # Bron-Kerbosch style: pick a pivot to reduce branching?
            # For simplicity and correctness, let's just iterate.
            # But to avoid duplicates, we restrict to w > start.
            # This standard recursive approach with 'start' ensures we don't revisit.
            
            for w in candidates:
                current.append(w)
                # Recurse with start = w (since we only look at vertices > w next? 
                # No, standard BK uses P and X. 
                # This simplified version: "grow clique by adding w, then look for vertices > w"
                # This enumerates each clique exactly once if we enforce increasing order.
                yield from enumerate_cliques(w, current, rem_edges)
                current.pop()

        def find_maximal_cliques_covering_most(rem_edges):
            """
            Find a clique that covers the most edges in rem_edges.
            We sample some seed edges, enumerate maximal cliques containing them,
            and pick the one with the most edges.
            """
            if not rem_edges:
                return []

            # Heuristic: pick a few random edges from rem_edges to seed the search
            # This avoids enumerating ALL maximal cliques which is too slow for n=64.
            edge_list = list(rem_edges)
            if len(edge_list) > 100:
                # Sample up to 100 random edges
                sample_size = min(100, len(edge_list))
                sampled_edges = random.sample(edge_list, sample_size)
            else:
                sampled_edges = edge_list

            best_clique = []
            best_edge_count = 0

            for seed_edge in sampled_edges:
                u, v = seed_edge
                # Enumerate maximal cliques containing u and v
                # We call enumerate_cliques with current=[u,v] and start=min(u,v)?
                # No, the recursion in enumerate_cliques relies on 'start' to be the last added vertex.
                # Let's adjust: we want cliques containing {u, v}.
                # Let's just run a local Bron-Kerbosch on the subgraph induced by 
                # common neighbors of u and v, plus u and v.
                
                # Simpler approach: 
                # The clique must contain u and v.
                # Any other vertex w must be connected to u and v.
                # So candidates are in adj[u] & adj[v].
                # Let's use a modified enumeration that starts with [u, v].
                
                # We need to be careful with the 'start' parameter logic in my enumerate_cliques.
                # It assumes we are building a clique in increasing order.
                # If we start with [u, v] where u < v, the next vertex must be > v.
                # This misses cliques where a vertex w < u is in the clique.
                
                # So, let's sort the seed edge.
                if u > v:
                    u, v = v, u
                
                # Now we look for vertices w > v that connect to u and v.
                # This finds cliques of the form {u, v, w1, w2, ...} with w1 < w2 ...
                # It misses cliques that include vertices < u.
                
                # To be safe, let's just find the "greedy best" from this seed.
                # Or better: just do a local greedy expansion from u, v.
                
                current = [u, v]
                # Candidates: connected to u and v, and edges in rem_edges
                cand_set = adj[u].intersection(adj[v])
                cand_set.discard(u)
                cand_set.discard(v)
                
                # Filter for edges in rem_edges
                cand_set = {w for w in cand_set if (min(u,w),max(u,w)) in rem_edges and (min(v,w),max(v,w)) in rem_edges}
                
                while cand_set:
                    # Pick w with highest degree in rem_edges among candidates
                    best_w = None
                    max_deg = -1
                    for w in cand_set:
                        # Count edges from w to other candidates? Or just total remaining degree?
                        # Let's use total remaining degree as a proxy for "valuable"
                        deg = sum(1 for nb in adj[w] if (min(w,nb),max(w,nb)) in rem_edges)
                        if deg > max_deg:
                            max_deg = deg
                            best_w = w
                            
                    if best_w is None:
                        break
                            
                    current.append(best_w)
                    cand_set.discard(best_w)
                    
                    # Update candidates
                    new_cand = set()
                    for w in cand_set:
                        if w in adj[best_w] and (min(w,best_w),max(w,best_w)) in rem_edges:
                            # Check if w is still connected to all in current?
                            # Since we only add w if it connects to all, and we filter cand_set,
                            # we just need to ensure w connects to best_w.
                            # But w must also connect to all OTHERS in current.
                            # Since w was in cand_set, it was connected to all previous members.
                            # So just checking best_w is sufficient if we maintain the invariant.
                            new_cand.add(w)
                    cand_set = new_cand

                # Evaluate this clique
                clique_edges = set()
                for i in range(len(current)):
                    for j in range(i + 1, len(current)):
                        clique_edges.add((min(current[i], current[j]), max(current[i], current[j])))
                        
                # Verify all edges are in rem_edges
                if all(e in rem_edges for e in clique_edges):
                    if len(clique_edges) > best_edge_count:
                        best_edge_count = len(clique_edges)
                        best_clique = current

            return best_clique

        attempts = 0
        while time.time() < end_time:
            attempts += 1
            attempt_rng = random.Random(seed + attempts * 12345 + 67890)

            remaining_edges = set(edges)
            cliques = []

            while remaining_edges:
                if time.time() > end_time - 1:
                    break

                # Find the clique that covers the most remaining edges
                best_c = find_maximal_cliques_covering_most(remaining_edges)

                if not best_c:
                    # Fallback: pick any edge
                    e = next(iter(remaining_edges))
                    best_c = [e[0], e[1]]

                # Add to partition
                cliques.append(best_c)
                
                # Remove edges covered by this clique
                for i in range(len(best_c)):
                    for j in range(i + 1, len(best_c)):
                        e = (min(best_c[i], best_c[j]), max(best_c[i], best_c[j]))
                        remaining_edges.discard(e)

            # Check if this solution is better
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques

            if time.time() > end_time - 2:
                break

        return best_cliques

    # Solve both instances
    t1 = time.time()
    # Allocate more time to the larger, harder instance (n=64)
    budget1 = min(30, DEADLINE - (t1 - start_time) - 5)
    cliques1 = solve_instance(32, 0.6, 601, budget1)

    t2 = time.time()
    budget2 = min(110, DEADLINE - (t2 - start_time))
    cliques2 = solve_instance(64, 0.55, 602, budget2)

    return [cliques1, cliques2]
