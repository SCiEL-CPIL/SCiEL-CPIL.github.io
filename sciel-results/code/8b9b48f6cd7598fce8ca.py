import random
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_instance(edges):
        if not edges:
            return []
        
        # Map edges to indices in line graph
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        n_edges = len(edges)
        
        # Build line graph adjacency
        # adj[i] is a set of j such that edge i and edge j share a vertex
        adj = [set() for _ in range(n_edges)]
        for i in range(n_edges):
            u, v = edges[i]
            for j in range(i + 1, n_edges):
                x, y = edges[j]
                if u in (x, y) or v in (x, y):
                    adj[i].add(j)
                    adj[j].add(i)
        
        best_matching = set()
        best_size = 0
        
        def try_greedy_and_improve(seed):
            rng = random.Random(seed)
            # Greedy phase
            matching = set()
            matched = [False] * n_edges
            
            # Random order of edges in line graph
            # We iterate over all pairs (i, j) in line graph? No, that's O(E^2).
            # We can just iterate i from 0 to n_edges-1 and pick a random neighbor.
            # Better: create list of all line graph edges, shuffle, and pick.
            lg_edges = []
            for i in range(n_edges):
                for j in adj[i]:
                    if i < j:
                        lg_edges.append((i, j))
            
            rng.shuffle(lg_edges)
            
            for i, j in lg_edges:
                if not matched[i] and not matched[j]:
                    matching.add((i, j))
                    matched[i] = True
                    matched[j] = True
            
            # Local search phase: Find augmenting paths of length 3
            # Path a-b-c-d in L(G) where (b,c) in M, (a,b) not in M, (c,d) not in M
            # a, d are unmatched.
            improved = True
            while improved:
                improved = False
                # List of current matched edges
                m_edges = list(matching)
                for b, c in m_edges:
                    # Check neighbors of b
                    for a in adj[b]:
                        if a == c: continue
                        if matched[a]: continue
                        # a is unmatched
                        # Check if (a,b) is in matching? No, we checked matched[a]
                        # Now check neighbors of c
                        for d in adj[c]:
                            if d == b: continue
                            if d == a: continue # a and d must be distinct? 
                            # If a == d, then path is a-b-c-a? Cycle. 
                            # In a simple path a-b-c-d, a!=d.
                            if matched[d]: continue
                            # d is unmatched
                            # Can we swap?
                            # Remove (b,c), add (a,b) and (c,d)
                            # Check if (a,b) and (c,d) are valid edges in L(G). Yes, by construction.
                            # Check if they conflict with other matching edges?
                            # a is unmatched, d is unmatched.
                            # b is matched to c. c is matched to b.
                            # So removing (b,c) frees b and c.
                            # Adding (a,b) matches a and b.
                            # Adding (c,d) matches c and d.
                            # No conflicts.
                            matching.remove((b, c))
                            matching.add((a, b))
                            matching.add((c, d))
                            matched[b] = True # still matched
                            matched[c] = True # still matched
                            matched[a] = True
                            matched[d] = True
                            improved = True
                            break
                        if improved:
                            break
                    if improved:
                        break
            
            return matching

        # Run multiple times
        num_trials = 20
        for i in range(num_trials):
            if time.time() > deadline:
                break
            m = try_greedy_and_improve(i)
            if len(m) > best_size:
                best_size = len(m)
                best_matching = m
        
        # Construct paths from matching
        # matched edges in line graph correspond to pairs of adjacent edges in G
        # We need to form paths.
        # A path in G of length k corresponds to a path of length k-1 in L(G).
        # The matching covers some edges in L(G).
        # The remaining edges in L(G) are not in the matching.
        # Actually, the matching partitions the edges of G into pairs and singles.
        # We can just reconstruct the paths by traversing.
        
        # Build adjacency of G
        g_adj = defaultdict(list)
        for u, v in edges:
            g_adj[u].append(v)
            g_adj[v].append(u)
        
        # Mark edges used in matching
        used_edges = set()
        for i, j in best_matching:
            used_edges.add((i, j))
            used_edges.add((j, i))
            
        # We need to group edges into paths.
        # Start with all edges as potential path segments.
        # If edge i is matched with edge j, they form a segment of length 2.
        # We can represent the decomposition as a list of paths.
        
        # Simpler: 
        # 1. Create a list of all edges.
        # 2. For each matched pair (i, j), merge them into a path of 3 vertices.
        # 3. The remaining edges are paths of 2 vertices.
        
        paths = []
        visited_edges = [False] * n_edges
        
        for i in range(n_edges):
            if visited_edges[i]:
                continue
            
            # Check if i is part of a matched pair
            # Find j such that (i,j) in best_matching
            partner = -1
            for a, b in best_matching:
                if a == i:
                    partner = b
                    break
                elif b == i:
                    partner = a
                    break
            
            if partner == -1:
                # Single edge
                u, v = edges[i]
                paths.append([u, v])
                visited_edges[i] = True
            else:
                # Pair (i, partner)
                u1, v1 = edges[i]
                u2, v2 = edges[partner]
                
                # Determine order
                # They share a vertex.
                if v1 == u2:
                    path = [u1, v1, v2]
                elif v1 == v2:
                    path = [u1, v1, u2]
                elif u1 == u2:
                    path = [v1, u1, v2]
                elif u1 == v2:
                    path = [v1, u1, u2]
                else:
                    # Should not happen if line graph is correct
                    path = [u1, v1]
                
                paths.append(path)
                visited_edges[i] = True
                visited_edges[partner] = True
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Fallback: return empty or simple if time runs out
            # But we must return valid result.
            # If time runs out, we can return the seed result (all edges separate)
            # or whatever we have.
            pass
        
        edges = instance(*spec)
        paths = solve_instance(edges)
        results.append(paths)
        
    return results
