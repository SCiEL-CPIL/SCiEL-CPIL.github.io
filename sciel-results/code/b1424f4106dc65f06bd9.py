import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed, time_limit):
        edges = random_graph(n, p, seed)
        
        # Build adjacency sets
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # rem_adj tracks remaining edges
        rem_adj = [set(s) for s in adj]
        remaining_edges = len(edges)
        
        best_partition = None
        best_score = float('inf')
        
        # We will run multiple randomized greedy passes.
        # Each pass picks vertices in a randomized order (or based on dynamic degree) 
        # and forms a maximal clique in the remaining neighborhood.
        
        # Optimization: Pre-compute bitmasks for adjacency to speed up clique finding?
        # n is up to 64, so int bitmasks work well.
        
        # Let's stick to sets for clarity but optimize the inner loop.
        
        num_restarts = 20 # We will check time frequently
        
        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Copy state
            curr_rem_adj = [set(s) for s in rem_adj]
            curr_edges = remaining_edges
            partition = []
            
            # Create a list of vertices to process
            # Strategy: Pick vertex with max degree in remaining graph, 
            # but add randomness to break ties or slightly perturb order.
            
            # To speed up, we can maintain a list of active vertices.
            active = list(range(n))
            
            while curr_edges > 0:
                if time.time() > deadline - 5.0:
                    break
                
                # Find vertex with max degree
                # Instead of scanning all n every time (n is small, so it's fine),
                # we can just scan.
                max_deg = -1
                best_v = -1
                # Randomize tie-breaking by adding a small random noise or just shuffling candidates
                candidates = []
                for v in active:
                    d = len(curr_rem_adj[v])
                    if d > 0:
                        if d > max_deg:
                            max_deg = d
                            candidates = [v]
                        elif d == max_deg:
                            candidates.append(v)
                
                if not candidates:
                    break
                
                v = random.choice(candidates)
                
                # Find a large clique in the neighborhood of v within the remaining graph.
                # Greedy: start with v, then iteratively add vertex with max degree in common neighborhood.
                
                clique = [v]
                # Candidates for next vertex: neighbors of v in curr_rem_adj
                cand_set = set(curr_rem_adj[v])
                
                while cand_set:
                    # Pick vertex u in cand_set with max degree within cand_set
                    max_deg_u = -1
                    best_u = -1
                    # To add randomness, we can shuffle if degrees are close, or just pick max
                    # Let's pick max degree in cand_set
                    for u in cand_set:
                        # Count neighbors of u within cand_set
                        deg_u = 0
                        # Optimization: use set intersection size?
                        # len(curr_rem_adj[u] & cand_set)
                        deg_u = len(curr_rem_adj[u] & cand_set)
                        if deg_u > max_deg_u:
                            max_deg_u = deg_u
                            best_u = u
                            
                    if best_u == -1:
                        break
                    
                    clique.append(best_u)
                    cand_set.remove(best_u)
                    
                    # Update cand_set: must be neighbor of best_u
                    cand_set = cand_set & curr_rem_adj[best_u]
                
                # Remove edges covered by this clique
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        u, w = clique[i], clique[j]
                        if w in curr_rem_adj[u]:
                            curr_rem_adj[u].remove(w)
                            curr_rem_adj[w].remove(u)
                            curr_edges -= 1
                            
                partition.append(clique)
                
                # Update active list? 
                # If a vertex has no remaining edges, it's effectively done.
                # We don't strictly need to remove from 'active' if we check len(curr_rem_adj[v]) > 0.
                
            # Fallback for remaining edges
            for u in range(n):
                for v in list(curr_rem_adj[u]):
                    if u < v:
                        partition.append([u, v])
                        curr_rem_adj[u].remove(v)
                        curr_rem_adj[w].remove(u) # Bug in original seed? No, seed was correct. Let's be careful.
                        curr_rem_adj[v].remove(u)
                        curr_edges -= 1
                        
            score = len(partition)
            if score < best_score:
                best_score = score
                best_partition = partition
                
        # If no partition found (e.g. timeout), return a valid but potentially suboptimal one
        if best_partition is None:
            # Just do a single greedy pass quickly
            curr_rem_adj = [set(s) for s in rem_adj]
            curr_edges = remaining_edges
            partition = []
            active = list(range(n))
            while curr_edges > 0:
                max_deg = -1
                candidates = []
                for v in active:
                    d = len(curr_rem_adj[v])
                    if d > 0:
                        if d > max_deg:
                            max_deg = d
                            candidates = [v]
                        elif d == max_deg:
                            candidates.append(v)
                if not candidates:
                    break
                v = candidates[0]
                clique = [v]
                cand_set = set(curr_rem_adj[v])
                while cand_set:
                    max_deg_u = -1
                    best_u = -1
                    for u in cand_set:
                        deg_u = len(curr_rem_adj[u] & cand_set)
                        if deg_u > max_deg_u:
                            max_deg_u = deg_u
                            best_u = u
                    if best_u == -1:
                        break
                    clique.append(best_u)
                    cand_set.remove(best_u)
                    cand_set = cand_set & curr_rem_adj[best_u]
                
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        u, w = clique[i], clique[j]
                        if w in curr_rem_adj[u]:
                            curr_rem_adj[u].remove(w)
                            curr_rem_adj[w].remove(u)
                            curr_edges -= 1
                partition.append(clique)
                
            for u in range(n):
                for v in list(curr_rem_adj[u]):
                    if u < v:
                        partition.append([u, v])
                        curr_rem_adj[u].remove(v)
                        curr_rem_adj[v].remove(u)
            best_partition = partition
            
        return best_partition

    result1 = solve_instance(32, 0.6, 601, deadline)
    result2 = solve_instance(64, 0.55, 602, deadline)
    
    return [result1, result2]
