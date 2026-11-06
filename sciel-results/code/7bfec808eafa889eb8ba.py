import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    N = 256
    # Build conflict graph: edge between i,j (1-indexed values) if (i*j) % (i+j) == 0
    # Use 0-indexed: vertex i represents value i+1
    n = N
    # conflicts[i] is a set of j such that (i+1)*(j+1) % ((i+1)+(j+1)) == 0
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # Degree for ordering
    degrees = [len(conflicts[i]) for i in range(n)]
    
    # Tomita's maximum clique algorithm on the complement graph
    # Independent set in G = Clique in complement of G
    # complement: edge between i,j if i,j NOT in conflicts
    # We want max clique in complement graph
    
    # For efficiency, represent complement adjacency as bitsets or sets
    # complement_neighbors[i] = set of j != i such that j not in conflicts[i]
    comp_neighbors = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and j not in conflicts[i]:
                comp_neighbors[i].add(j)
    
    # Tomita's MCS (Maximum Clique Search) with coloring
    best_clique = []
    best_size = 0
    
    def color_sort(candidates):
        """Color the candidates and return them sorted by color (ascending),
        along with the color of each vertex (used as upper bound)."""
        # Greedy coloring: assign colors greedily
        uncolored = list(candidates)
        colors = {}
        color_classes = []
        while uncolored:
            color_class = set()
            i = 0
            while i < len(uncolored):
                v = uncolored[i]
                # v can be in this color class if it's not adjacent to any in color_class
                if all(v not in comp_neighbors[u] for u in color_class):
                    color_class.add(v)
                    uncolored.pop(i)
                else:
                    i += 1
            colors_list = color_class
            for v in colors_list:
                colors[v] = len(color_classes) + 1
            color_classes.append(colors_list)
        
        # Sort by color ascending
        sorted_cands = sorted(candidates, key=lambda v: colors[v])
        return sorted_cands, colors
    
    def expand(clique, candidates, colors):
        """Recursively search for maximum clique."""
        nonlocal best_clique, best_size
        nonlocal start_time, deadline
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = list(clique)
            return
        
        # Upper bound: max color
        max_color = max(colors.values()) if colors else 0
        if len(clique) + max_color <= best_size:
            return
        
        # Process vertices in reverse color order (highest color first)
        while candidates:
            # Pick vertex with highest color
            v = candidates[-1]
            candidates.pop()
            
            # If even with this vertex we can't beat best
            if len(clique) + colors[v] <= best_size:
                break
            
            # Add v to clique, recurse
            new_clique = clique + [v]
            # New candidates: intersection of current candidates with neighbors of v
            new_candidates = [u for u in candidates if u in comp_neighbors[v]]
            if new_candidates:
                new_sorted, new_colors = color_sort(new_candidates)
                expand(new_clique, new_sorted, new_colors)
    
    # Initial: all vertices
    initial_candidates = list(range(n))
    initial_sorted, initial_colors = color_sort(initial_candidates)
    expand([], initial_sorted, initial_colors)
    
    # Convert clique (0-indexed) to values (1-indexed)
    result = [v + 1 for v in best_clique]
    result.sort()
    
    # Verify
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                # Should not happen, but fallback
                return list(range(1, 257, 2))
    
    return result if result else list(range(1, 257, 2))
