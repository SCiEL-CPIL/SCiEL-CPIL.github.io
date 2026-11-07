import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        # Core clique edges
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, k, p, seed):
        rng = random.Random(seed)
        
        # Build adjacency sets
        # core_adj[u] is set of core neighbors (all core vertices)
        # cross_adj[i] is set of core neighbors for independent vertex i
        cross_adj = [set() for _ in range(n)]
        for i in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_adj[i].add(u)
        
        # Full adjacency for core vertices is complete, but we only care about edges remaining
        # We will maintain a set of remaining core edges
        remaining_core_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                remaining_core_edges.add((u, v))
        
        cliques = []
        
        # Process independent vertices
        # Sort by size of neighborhood in core, descending, to pack larger cliques first
        i_vertices = list(range(k, n))
        i_vertices.sort(key=lambda x: len(cross_adj[x]), reverse=True)
        
        for i in i_vertices:
            if time.time() > deadline:
                break
            
            N_i = cross_adj[i]
            if not N_i:
                continue
            
            # We want to find a clique S in N_i such that:
            # 1. All edges in S are in remaining_core_edges (i.e., S is a clique in the graph of remaining core edges)
            # 2. S is as large as possible to minimize the number of cliques needed for core edges
            
            # Greedy approach to find a large clique in the intersection of N_i and the remaining core edge graph
            candidates = set(N_i)
            S = []
            
            while candidates:
                # Find vertex v in candidates with max degree in the subgraph induced by candidates
                # where edges are from remaining_core_edges
                best_v = None
                best_deg = -1
                for v in candidates:
                    # Count neighbors in candidates that have an edge to v in remaining_core_edges
                    deg = 0
                    for w in candidates:
                        if v == w: continue
                        if (min(v, w), max(v, w)) in remaining_core_edges:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                
                if best_v is None:
                    break
                
                # Check if best_v can connect to all existing members of S
                # Since we start with S empty, this is initially true.
                # As we add to S, we must ensure new vertex connects to all in S.
                # The greedy max-degree heuristic doesn't strictly guarantee this unless we prune candidates.
                
                # Let's prune candidates to only those that can connect to best_v
                # And also ensure they can connect to all in S?
                # Actually, standard greedy max clique:
                # 1. Pick max degree vertex v.
                # 2. Add v to S.
                # 3. Remove v from candidates.
                # 4. Remove from candidates any vertex that is NOT a neighbor of v in the current graph.
                
                S.append(best_v)
                candidates.remove(best_v)
                
                # Prune candidates: keep only those connected to best_v via remaining_core_edges
                # AND connected to all other members of S? 
                # If we just keep neighbors of best_v, we might pick a vertex later that doesn't connect to an earlier member of S.
                # So we must intersect with neighbors of ALL members of S.
                
                # For efficiency, let's maintain a set of valid candidates that form a clique with S.
                # Initially, valid = candidates (all connect to empty S).
                # When we add best_v, new_valid = valid ∩ neighbors(best_v).
                
                # Let's re-implement with this pruning.
                # Restarting the inner loop logic is simpler but slower.
                # Given small k (10, 20, 30), O(k^3) or similar is fine.
                
            # The above greedy was slightly flawed in description. Let's do a robust greedy:
            # S = []
            # candidates = set(N_i)
            # while candidates:
            #   find v in candidates with max number of neighbors in (candidates ∩ valid_for_S)
            #   if no such v or v doesn't connect to all in S: break? No, if v doesn't connect to S, it shouldn't be in candidates.
            
            # Let's redo the clique finding properly.
            S = []
            candidates = set(N_i)
            
            while candidates:
                # Filter candidates to those that can connect to all in S
                valid_candidates = []
                for v in candidates:
                    ok = True
                    for u in S:
                        if (min(u, v), max(u, v)) not in remaining_core_edges:
                            ok = False
                            break
                    if ok:
                        valid_candidates.append(v)
                            
                if not valid_candidates:
                    break
                    
                # Among valid candidates, pick one with max degree in the subgraph induced by valid_candidates
                best_v = None
                best_deg = -1
                for v in valid_candidates:
                    deg = 0
                    for w in valid_candidates:
                        if v == w: continue
                        if (min(v, w), max(v, w)) in remaining_core_edges:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                        
                S.append(best_v)
                candidates.remove(best_v)
                
            if S:
                clique = [i] + sorted(S)
                cliques.append(clique)
                
                # Remove edges in S from remaining_core_edges
                for idx_u in range(len(S)):
                    for idx_v in range(idx_u + 1, len(S)):
                        u = S[idx_u]
                        v = S[idx_v]
                        edge = (min(u, v), max(u, v))
                        remaining_core_edges.discard(edge)
            
            # If S is empty, it means i has neighbors but none could be added?
            # This happens if S is empty initially, which means valid_candidates was empty?
            # No, if S is empty, valid_candidates = candidates (all connect to empty set).
            # So if candidates is not empty, valid_candidates is not empty, and best_v is found.
            # So S will have at least 1 vertex if N_i is not empty.
            # Thus clique size is at least 2.

        # Cover remaining core edges
        # The remaining graph is a subgraph of a complete graph (since we only removed edges).
        # We need to cover these edges with cliques.
        # Since the graph is a subgraph of a clique, we can just find cliques in this remaining graph.
        
        remaining_edges = set(remaining_core_edges)
        
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: cover each edge separately
                for e in remaining_edges:
                    cliques.append(list(e))
                break
                
            # Find a max clique in the remaining graph
            # Build adjacency from remaining_edges
            rem_adj = defaultdict(set)
            for u, v in remaining_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
                
            candidates = set(rem_adj.keys())
            if not candidates:
                break
                
            S = []
            while candidates:
                # Find vertex with max degree in candidates
                best_v = None
                best_deg = -1
                for v in candidates:
                    deg = len(rem_adj[v] & candidates)
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                        
                if best_v is None:
                    break
                    
                S.append(best_v)
                candidates.remove(best_v)
                
                # Prune candidates to neighbors of best_v
                candidates = candidates & rem_adj[best_v]
                
            if len(S) < 2:
                # Should not happen if remaining_edges is not empty, but just in case
                # Pick an edge
                if remaining_edges:
                    u, v = next(iter(remaining_edges))
                    S = [u, v]
                else:
                    break
            else:
                S = sorted(S)
                
            cliques.append(S)
            
            # Remove edges in S from remaining_edges
            for idx_u in range(len(S)):
                for idx_v in range(idx_u + 1, len(S)):
                    u = S[idx_u]
                    v = S[idx_v]
                    edge = (min(u, v), max(u, v))
                    remaining_edges.discard(edge)
                    
        return cliques

    results = []
    specs = [(30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)]
    
    for spec in specs:
        if time.time() > deadline:
            # This shouldn't happen if previous instances were fast, but just in case
            # We need to return valid results. If we run out of time, we can't compute.
            # But the problem says "Any broken rule makes the whole result invalid".
            # So we must ensure we finish.
            # Given the sizes, the greedy is very fast (O(n*k^2) roughly).
            pass
        
        cliques = solve_instance(*spec)
        
        # Validate and clean types
        clean_cliques = []
        for c in cliques:
            # Ensure list of ints
            clean_cliques.append([int(v) for v in c])
        results.append(clean_cliques)
        
    return results
