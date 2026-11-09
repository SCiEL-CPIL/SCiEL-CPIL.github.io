import time
import sys
from itertools import combinations
import random

def construct():
    from random import Random
    
    # Keep the instance generator as is
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def dsatur_coloring(vertices, adj):
        """
        Color a graph using DSATUR heuristic.
        vertices: list of vertex labels
        adj: dict mapping vertex to set of neighbors
        Returns: dict mapping vertex to color
        """
        if not vertices:
            return {}
        
        uncolored = set(vertices)
        colors = {}
        # degree in the subgraph
        degree = {v: len(adj[v]) for v in vertices}
        # saturation degree: number of distinct colors in neighbors
        sat = {v: 0 for v in vertices}
        neighbor_colors = {v: set() for v in vertices}
        
        while uncolored:
            # Select vertex with highest saturation, then highest degree
            # v* = argmax (sat[v], degree[v])
            best_v = None
            best_key = (-1, -1)
            for v in uncolored:
                key = (sat[v], degree[v])
                if key > best_key:
                    best_key = key
                    best_v = v
            
            # Find smallest available color
            used_colors = neighbor_colors[best_v]
            c = 0
            while c in used_colors:
                c += 1
            
            colors[best_v] = c
            uncolored.remove(best_v)
            
            # Update neighbors
            for u in adj[best_v]:
                if u in uncolored:
                    if c not in neighbor_colors[u]:
                        neighbor_colors[u].add(c)
                        sat[u] += 1
        
        return colors

    def partition_remaining_edges(rem_edges, k):
        """
        Partition the remaining edges in C into cliques.
        rem_edges: set of tuples (u, v)
        k: size of C
        """
        if not rem_edges:
            return []
            
        cliques = []
        rem = set(rem_edges)
        
        # Build adjacency for remaining graph
        # We will update this as we remove edges
        
        while rem:
            # Find a maximal clique
            # Heuristic: pick a vertex with highest degree in rem
            # Then expand to maximal clique
            
            # Calculate degrees in rem
            deg = {i: 0 for i in range(k)}
            for u, v in rem:
                deg[u] += 1
                deg[v] += 1
            
            # Pick vertex with max degree
            if not any(deg.values()):
                break
                
            start_v = max(deg, key=lambda x: deg[x])
            
            # Build clique containing start_v
            # Candidates must be connected to all current clique members
            clique = {start_v}
            candidates = set()
            for u, v in rem:
                if u == start_v:
                    candidates.add(v)
                elif v == start_v:
                    candidates.add(u)
            
            # Iterate to add candidates
            changed = True
            while changed:
                changed = False
                for w in list(candidates):
                    # Check if w is connected to all in clique
                    can_add = True
                    for x in clique:
                        e = (min(w, x), max(w, x))
                        if e not in rem:
                            can_add = False
                            break
                    if can_add:
                        clique.add(w)
                        changed = True
                        candidates.discard(w)
                        # Update candidates: remove those not connected to w
                        # Actually, we can just filter candidates in next iteration
                        # Or update candidates to be intersection of neighbors
                        new_candidates = set()
                        for u, v in rem:
                            if u == w: new_candidates.add(v)
                            elif v == w: new_candidates.add(u)
                        candidates = candidates & new_candidates
            
            # Add clique
            cliques.append(sorted(list(clique)))
            
            # Remove edges of this clique from rem
            for u in clique:
                for v in clique:
                    if u < v:
                        rem.discard((u, v))
                        
        return cliques

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        C = list(range(k))
        I = list(range(k, n))
        
        # Internal edges of C
        internal_edges = set((u, v) for u in range(k) for v in range(u+1, k))
        
        best_partition = None
        best_count = float('inf')
        
        # Try different orderings of I
        # 1. Degree descending
        # 2. Degree ascending
        # 3. Random orders
        
        orderings = []
        orderings.append(sorted(I, key=lambda v: len(adj[v] & set(C)), reverse=True))
        orderings.append(sorted(I, key=lambda v: len(adj[v] & set(C)), reverse=False))
        
        rng = Random(seed + 1000)
        for _ in range(10):
            order = I[:]
            rng.shuffle(order)
            orderings.append(order)
            
        for order in orderings:
            used_edges = set()
            partition = []
            
            for v in order:
                N_v = adj[v] & set(C)
                if not N_v:
                    continue
                    
                # Build conflict graph H on N_v
                h_vertices = list(N_v)
                h_adj = {u: set() for u in h_vertices}
                
                for u in h_vertices:
                    for w in h_vertices:
                        if u == w: continue
                        e = (min(u,w), max(u,w))
                        if e in used_edges:
                            h_adj[u].add(w)
                            
                # Color H using DSATUR
                colors = dsatur_coloring(h_vertices, h_adj)
                
                # Group by color
                color_classes = {}
                for u, c in colors.items():
                    if c not in color_classes:
                        color_classes[c] = []
                    color_classes[c].append(u)
                    
                for c in color_classes:
                    S = color_classes[c]
                    if not S:
                        continue
                    clique = sorted(S + [v])
                    partition.append(clique)
                    
                    # Mark edges in S as used
                    for u in S:
                        for w in S:
                            if u < w:
                                used_edges.add((u, w))
                                
            # Partition remaining edges
            remaining_edges = internal_edges - used_edges
            rem_cliques = partition_remaining_edges(remaining_edges, k)
            partition.extend(rem_cliques)
            
            count = len(partition)
            if count < best_count:
                best_count = count
                best_partition = partition
                
        return best_partition

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    for spec in specs:
        res = solve_instance(*spec)
        # Ensure types are correct
        # res is list of lists of ints
        results.append(res)
        
        if time.time() > deadline:
            # If we run out of time, we might have incomplete results
            # But the problem says "return the best result found"
            # We should probably return what we have if it's valid, 
            # but the contract says "Any broken rule makes the whole result invalid".
            # So we must ensure we return a valid list of 3 lists.
            # If we haven't finished, we might return partial or empty?
            # Better to optimize speed.
            pass

    return results
