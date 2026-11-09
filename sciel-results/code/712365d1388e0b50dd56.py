import time
import sys
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def dsatur_coloring(h_vertices, h_adj, rng):
        """
        Randomized DSATUR coloring.
        Returns a dictionary mapping vertex to color.
        """
        if not h_vertices:
            return {}
        
        colors = {}
        sat_degree = {u: 0 for u in h_vertices}
        color_set = {u: set() for u in h_vertices}
        uncolored = set(h_vertices)
        
        while uncolored:
            # Calculate saturation and degree for all uncolored
            # To allow random tie-breaking, we collect candidates
            # Score: (saturation, degree)
            candidates = []
            for u in uncolored:
                sat = len(color_set[u])
                deg = len(h_adj[u])
                # We want max sat, then max deg
                # We will pick randomly from the top few to add noise
                candidates.append((sat, deg, u))
            
            # Sort by score descending
            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            
            # Pick randomly from the top 3 (or 1 if deterministic mode, but we want random)
            # To ensure diversity, pick from top min(3, len(candidates))
            top_k = min(3, len(candidates))
            best = rng.choice(candidates[:top_k])[2]
            
            # Find smallest available color
            used_colors = color_set[best]
            c = 0
            while c in used_colors:
                c += 1
            colors[best] = c
            color_set[best].add(c)
            uncolored.discard(best)
            
            # Update neighbors
            for w in h_adj[best]:
                if w in uncolored:
                    if c not in color_set[w]:
                        color_set[w].add(c)
        
        return colors

    def best_coloring(h_vertices, h_adj, rng, iterations=50):
        """
        Run randomized DSATUR multiple times and return the coloring with min colors.
        """
        if not h_vertices:
            return {}
        
        best_colors = None
        best_count = float('inf')
        
        # Deterministic DSATUR first (top 1)
        # We can simulate deterministic by setting top_k=1 in a modified dsatur, 
        # or just run the standard one. 
        # For simplicity, we run the randomized one, but maybe force first run to be deterministic?
        # The prompt says "noise-based tie-breaking".
        
        for _ in range(iterations):
            colors = dsatur_coloring(h_vertices, h_adj, rng)
            if colors:
                count = max(colors.values()) + 1
                if count < best_count:
                    best_count = count
                    best_colors = colors
            else:
                best_colors = {}
                best_count = 0
                break
        
        return best_colors

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
        
        # Used edges in C
        used_edges = set()
        
        partition = []
        
        # Group independent vertices by their neighborhood in C
        neighborhood_groups = defaultdict(list)
        for v in I:
            N_v = frozenset(adj[v] & set(C))
            if N_v:
                neighborhood_groups[N_v].append(v)
        
        # Sort groups by size or complexity? 
        # Processing larger groups first might be better as they constrain more?
        # Or smaller first? 
        # Current code processed I_sorted. 
        # Let's sort groups by the size of the neighborhood (ascending) to leave more options for larger ones?
        # Actually, if we process a group, we mark edges as used.
        # If we process a group with small N first, we use few edges.
        # If we process a group with large N first, we use many edges.
        # Let's stick to the previous order or sort by len(N_v) descending?
        # Let's try sorting by len(N_v) descending (hardest first) to see if it helps.
        # Or keep it simple: process in order of appearance or sorted by v.
        # The previous code sorted I by degree. 
        # Let's sort the groups by the number of vertices in the neighborhood (descending).
        
        sorted_groups = sorted(neighborhood_groups.items(), key=lambda item: len(item[0]), reverse=True)
        
        rng = Random(seed + 1000) # Local RNG for heuristics
        
        for N_v_frozen, group_vertices in sorted_groups:
            N_v = set(N_v_frozen)
            n_v = len(N_v)
            
            if n_v == 1:
                u = next(iter(N_v))
                for v in group_vertices:
                    partition.append(sorted([v, u]))
                continue
                
            # Build graph H on N_v using used_edges
            # H has an edge (u,w) if (u,w) is in internal_edges but NOT in used_edges
            # Wait, the logic in the current code is:
            # "Build graph H on N_v using used_edges"
            # h_adj[u].add(w) if e in used_edges.
            # Then it colors H.
            # If u and w are adjacent in H, they have the SAME color?
            # No, in coloring, adjacent vertices get DIFFERENT colors.
            # So if (u,w) is in H (i.e., edge (u,w) is ALREADY used), u and w MUST have different colors?
            # Let's re-read the logic.
            
            # The clique formed is {v} U S, where S is a color class.
            # For {v} U S to be a clique, all vertices in S must be mutually adjacent in the original graph G.
            # Since G[C] is a complete graph, any subset of C is a clique.
            # However, the edges (u,w) for u,w in S must be covered by THIS clique or a previous one.
            # If (u,w) was already covered by a previous clique, we cannot cover it again.
            # So, we cannot put u and w in the same clique {v} U S if (u,w) is already used.
            # Therefore, u and w must have DIFFERENT colors if (u,w) is in used_edges.
            # This means H should have an edge (u,w) if (u,w) is in used_edges.
            # And we color H such that adjacent vertices (in H) have different colors.
            # This matches the current code logic.
            
            h_vertices = list(N_v)
            h_adj = {u: set() for u in h_vertices}
            
            for i in range(len(h_vertices)):
                u = h_vertices[i]
                for j in range(i+1, len(h_vertices)):
                    w = h_vertices[j]
                    e = (min(u, w), max(u, w))
                    if e in used_edges:
                        h_adj[u].add(w)
                        h_adj[w].add(u)
                        
            # Get best coloring
            colors = best_coloring(h_vertices, h_adj, rng, iterations=100)
            
            if not colors:
                continue
                
            color_count = max(colors.values()) + 1
            
            # Group vertices by color
            color_classes = defaultdict(list)
            for u, c in colors.items():
                color_classes[c].append(u)
                
            # For each color class, create one clique containing all group vertices + color class vertices
            for c in range(color_count):
                S = color_classes.get(c, [])
                if len(S) < 1:
                    continue
                    
                clique = S + group_vertices
                partition.append(sorted(clique))
                
                # Mark edges in S as used
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        used_edges.add((min(S[i], S[j]), max(S[i], S[j])))
                            
        # Remaining edges in C
        remaining_edges = internal_edges - used_edges
        
        # Partition remaining_edges into cliques
        rem_set = set(remaining_edges)
        
        def greedy_clique_partition(rem_set, strategy='max_common'):
            """
            Greedily partition edges into cliques.
            """
            if not rem_set:
                return []
            
            local_partition = []
            rem = set(rem_set)
            
            # Precompute adjacencies for rem? 
            # Since rem changes, we might need to recompute or update.
            # For k<=30, recomputing is fast.
            
            while rem:
                if not rem:
                    break
                
                # Pick best edge
                best_edge = None
                best_score = -1
                
                # Strategy 1: Max common neighbors
                if strategy == 'max_common':
                    # Build adj bitmasks for current rem
                    rem_adj = [0] * k
                    for (u, v) in rem:
                        rem_adj[u] |= (1 << v)
                        rem_adj[v] |= (1 << u)
                    
                    for e in rem:
                        u, v = e
                        common = bin(rem_adj[u] & rem_adj[v]).count('1')
                        if common > best_score:
                            best_score = common
                            best_edge = e
                            
                elif strategy == 'max_degree_sum':
                    # Build degrees
                    deg = [0] * k
                    for (u, v) in rem:
                        deg[u] += 1
                        deg[v] += 1
                    for e in rem:
                        u, v = e
                        score = deg[u] + deg[v]
                        if score > best_score:
                            best_score = score
                            best_edge = e
                            
                else: # random
                    best_edge = rng.choice(list(rem))
                
                if best_edge is None:
                    best_edge = next(iter(rem))
                    
                u, v = best_edge
                
                # Greedy maximal clique growth
                # Start with {u, v}
                current = {u, v}
                
                # Find candidates: vertices connected to all in current
                # We need to check if (w, x) is in rem for all x in current
                # Since rem is a set of edges, we check membership.
                
                changed = True
                while changed:
                    changed = False
                    # Try to add any vertex w not in current
                    # w must be connected to all in current via edges in rem
                    # Note: (w, u) and (w, v) must be in rem.
                    # And (w, x) for all x in current must be in rem.
                    
                    # To be efficient, iterate through vertices in rem that are adjacent to u
                    # Actually, just iterate through all vertices 0..k-1
                    # k is small (<=30)
                    
                    for w in range(k):
                        if w in current:
                            continue
                        
                        can_add = True
                        for x in current:
                            e = (min(w, x), max(w, x))
                            if e not in rem:
                                can_add = False
                                break
                                
                        if can_add:
                            current.add(w)
                            changed = True
                            # We can break early if we want a specific clique size? 
                            # No, we want MAXIMAL clique to remove max edges.
                            # So keep adding.
                            
                # Add clique to partition
                clique_list = sorted(list(current))
                if len(clique_list) >= 2:
                    local_partition.append(clique_list)
                    
                # Remove edges of this clique from rem
                for i in range(len(clique_list)):
                    for j in range(i+1, len(clique_list)):
                        a, b = clique_list[i], clique_list[j]
                        rem.discard((min(a, b), max(a, b)))
                        
            return local_partition

        # Try multiple strategies and pick best
        best_rem_partition = None
        best_rem_count = float('inf')
        
        strategies = ['max_common', 'max_degree_sum', 'random']
        
        for strat in strategies:
            # For random, try a few times
            if strat == 'random':
                for _ in range(5):
                    part = greedy_clique_partition(rem_set, strat)
                    if len(part) < best_rem_count:
                        best_rem_count = len(part)
                        best_rem_partition = part
            else:
                part = greedy_clique_partition(rem_set, strat)
                if len(part) < best_rem_count:
                    best_rem_count = len(part)
                    best_rem_partition = part
                    
        if best_rem_partition:
            partition.extend(best_rem_partition)
            
        return partition

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        
        res = solve_instance(*spec)
        cleaned = []
        for clique in res:
            cleaned.append([int(x) for x in clique])
        results.append(cleaned)
        
    return results

if __name__ == "__main__":
    construct()
