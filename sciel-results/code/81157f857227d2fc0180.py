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
        
        # List of independent vertices
        i_vertices = list(range(k, n))
        
        # Sort by degree in K descending to try to pack larger cliques first
        # This might help cover more K-edges early on
        i_vertices.sort(key=lambda v: len(neighbors[v]), reverse=True)
        
        cliques = []
        used_k_edges = set()
        
        # Precompute all edges in K
        all_k_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_k_edges.add((u, w))
        
        # Process each independent vertex
        for v in i_vertices:
            Nv = neighbors[v]
            if not Nv:
                # If v has no neighbors in K, we cannot form a clique with K vertices.
                # But v must be in a clique of size >= 2.
                # This case shouldn't happen with p=0.7 and reasonable n,k, but handle it.
                # We can't put it with another I vertex (independent).
                # So this instance would be invalid or require a dummy?
                # The problem states "Every listed set must be a clique...".
                # If v is isolated in the bipartite part, it's isolated in G? No, it's in I.
                # If it has no neighbors in K, it has no edges at all?
                # Then it's an isolated vertex. A clique of size 1 is not allowed.
                # However, with p=0.7, probability of 0 neighbors is (0.3)^k.
                # For k=30, (0.3)^30 is extremely small.
                # We assume Nv is not empty.
                pass
            
            # Find the largest subset S_v of Nv such that all edges in S_v are not in used_k_edges
            # This is a Maximum Clique problem on the graph H_v where vertices are Nv
            # and edges are those in all_k_edges \ used_k_edges.
            
            # Greedy approximation for Max Clique:
            # Since k is small, we can try a simple greedy or even brute force if Nv is small.
            # Nv size is at most k.
            
            # Let's try a simple greedy:
            # Sort Nv by some heuristic?
            # Just pick vertices one by one if they connect to all currently picked.
            
            # Better: Since we want to maximize |S_v| to cover edges, 
            # and the graph is dense (subset of K), let's try to find a large clique.
            
            # Simple Greedy:
            # Start with empty set.
            # Iterate through Nv.
            # If adding u to current set S keeps it a clique in the available graph, add it.
            
            # To make it better, sort Nv by "availability" degree?
            # Or just random order?
            # Let's try sorting by index to be deterministic.
            
            S_v = []
            # We need to check if (u, w) is available for all u, w in S_v U {u_new}
            
            # Optimization: Precompute available adjacency for Nv?
            # avail_adj[u] = set of w in Nv such that (min(u,w), max(u,w)) not in used_k_edges
            
            # For efficiency, let's just check on the fly.
            
            # Heuristic: Try to find a clique of size at least 2 or 3.
            # If we can't find one, S_v will be size 1 (or 0).
            
            # Let's try to build S_v greedily.
            # Sort Nv by degree in the available graph?
            
            # Calculate available neighbors for each u in Nv
            # This might be slow if done naively for each v.
            # But k is small (<=30).
            
            # Let's just iterate through Nv in a fixed order.
            # To improve packing, maybe pick vertices that have many available edges first?
            
            # Calculate "score" for each u in Nv: number of other vertices in Nv 
            # such that edge (u, w) is available.
            
            scores = {}
            for u in Nv:
                cnt = 0
                for w in Nv:
                    if u == w: continue
                    edge = (min(u, w), max(u, w))
                    if edge not in used_k_edges:
                        cnt += 1
                scores[u] = cnt
            
            # Sort Nv by score descending
            sorted_Nv = sorted(Nv, key=lambda u: scores[u], reverse=True)
            
            S_v = []
            for u in sorted_Nv:
                # Check if u can be added to S_v
                can_add = True
                for w in S_v:
                    edge = (min(u, w), max(u, w))
                    if edge in used_k_edges:
                        can_add = False
                        break
                if can_add:
                    S_v.append(u)
            
            if len(S_v) >= 2:
                cliques.append([v] + S_v)
                # Mark edges in S_v as used
                for i in range(len(S_v)):
                    for j in range(i + 1, len(S_v)):
                        u, w = S_v[i], S_v[j]
                        used_k_edges.add((min(u, w), max(u, w)))
            elif len(S_v) == 1:
                # Clique must have at least 2 vertices.
                # {v, u} is a clique if (v,u) is an edge. It is.
                # Does it cover any K-edge? No.
                cliques.append([v, S_v[0]])
                # No K-edges marked used
            else:
                # S_v is empty. v has no available neighbors?
                # This means all neighbors of v have their edges to other Nv vertices used?
                # Or Nv is empty.
                # If Nv is not empty, we must pick at least one.
                # If we picked one, len(S_v) would be 1.
                # So this case implies Nv is empty.
                # As discussed, unlikely.
                pass
        
        # Cover remaining K-edges
        for edge in all_k_edges:
            if edge not in used_k_edges:
                cliques.append(list(edge))
        
        return [[int(x) for x in c] for c in cliques]

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
    return results
