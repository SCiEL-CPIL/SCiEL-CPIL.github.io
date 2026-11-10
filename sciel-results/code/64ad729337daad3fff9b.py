import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 256
    
    # Precompute conflicts
    # conflicts[i] is a set of j such that i and j conflict
    conflicts = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            x = i // g
            y = j // g
            if g % (x + y) == 0:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # Build non-conflict adjacency (for maximum clique)
    # Non-conflict means (a*b)%(a+b) != 0, i.e., NOT in conflicts
    # We want maximum independent set in conflict graph = maximum clique in non-conflict graph
    non_conflict_adj = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if i != j and j not in conflicts[i]:
                non_conflict_adj[i].add(j)
    
    # Maximum clique algorithm (Tomita with coloring)
    # We work with 0-indexed internally for the algorithm
    # Map: node i (1..N) -> index i-1 (0..N-1)
    
    best_clique = []
    best_size = 0
    
    def max_clique_tomita():
        nonlocal best_clique, best_size
        
        # Convert to 0-indexed adjacency lists
        n = N
        adj = [set() for _ in range(n)]
        for i in range(1, N + 1):
            for j in non_conflict_adj[i]:
                adj[i - 1].add(j - 1)
        
        # Tomita's algorithm with coloring
        # P = candidate set, X = excluded set
        # Color P to get upper bound on clique size
        
        def color_sort(P):
            """Color the set P and return (sorted_P, colors) where colors[i] is the color of P[i]"""
            P_list = sorted(P)
            colors = []
            used_colors = set()
            # Greedy coloring: process in order, assign smallest available color
            for v in P_list:
                neighbor_colors = set()
                for u in P_list:
                    if u != v and u in adj[v]:
                        if u in colors_map:
                            neighbor_colors.add(colors_map[u])
                c = 1
                while c in neighbor_colors:
                    c += 1
                colors_map[v] = c
                colors.append(c)
                if c > len(used_colors):
                    used_colors.add(c)
            return P_list, colors
        
        # Simpler approach: use the standard Tomita coloring
        def color_sort_standard(P):
            """
            Returns (sorted_vertices, color_bounds) where color_bounds[i] 
            is the maximum color assigned to vertices up to position i.
            Vertices are sorted so that color_bounds is non-decreasing.
            """
            P_list = list(P)
            # Greedy coloring
            color_of = {}
            for v in P_list:
                # Get colors of neighbors in P_list
                neighbor_colors = set()
                for u in P_list:
                    if u != v and u in adj[v] and u in color_of:
                        neighbor_colors.add(color_of[u])
                c = 1
                while c in neighbor_colors:
                    c += 1
                color_of[v] = c
            
            # Sort by color (ascending), break ties arbitrarily
            sorted_P = sorted(P_list, key=lambda v: color_of[v])
            color_bounds = []
            max_c = 0
            for v in sorted_P:
                max_c = max(max_c, color_of[v])
                color_bounds.append(max_c)
            
            return sorted_P, color_bounds
        
        def expand(R, P):
            nonlocal best_clique, best_size
            
            if time.time() > deadline:
                return
            
            if not P:
                if len(R) > best_size:
                    best_size = len(R)
                    best_clique = list(R)
                return
            
            if len(R) + len(P) <= best_size:
                return
            
            # Color sort P
            sorted_P, color_bounds = color_sort_standard(P)
            
            # Process in reverse order (highest color first)
            for idx in range(len(sorted_P) - 1, -1, -1):
                v = sorted_P[idx]
                
                if time.time() > deadline:
                    return
                
                # Upper bound: len(R) + color_bounds[idx]
                if len(R) + color_bounds[idx] <= best_size:
                    return
                
                R.append(v)
                # New P = P ∩ N(v), only vertices before idx in sorted_P
                new_P = set()
                for j in range(idx):
                    u = sorted_P[j]
                    if u in adj[v]:
                        new_P.add(u)
                
                expand(R, new_P)
                R.pop()
        
        # Initial call: R = empty, P = all vertices
        all_vertices = set(range(N))
        expand([], all_vertices)
    
    # Run maximum clique
    max_clique_tomita()
    
    if best_size > 0:
        # Convert back to 1-indexed
        S = set(v + 1 for v in best_clique)
    else:
        # Fallback: use the odd-based approach
        S = set(range(1, N + 1, 2))
        evens = [i for i in range(2, N + 1, 2) if i not in S]
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for e in list(evens):
                if time.time() > deadline:
                    break
                if not conflicts[e].intersection(S):
                    S.add(e)
                    evens.remove(e)
                    changed = True
    
    # Local search polish: try to add more elements
    def try_add_greedy(S, remaining, conflicts, deadline):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for e in list(remaining):
                if time.time() > deadline:
                    break
                if not conflicts[e].intersection(S):
                    S.add(e)
                    remaining.remove(e)
                    changed = True
    
    remaining = [i for i in range(1, N + 1) if i not in S]
    try_add_greedy(S, remaining, conflicts, deadline)
    
    # 1-for-k swap local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = list(S)
        for s_elem in S_list:
            if time.time() > deadline:
                break
            S_test = S - {s_elem}
            candidates = []
            for e in remaining:
                if not conflicts[e].intersection(S_test):
                    candidates.append(e)
            if candidates:
                S_test2 = S_test.copy()
                remaining_test = list(remaining)
                candidates.sort(key=lambda e: len(conflicts[e] & S_test))
                added = 0
                for e in candidates:
                    if time.time() > deadline:
                        break
                    if not conflicts[e].intersection(S_test2):
                        S_test2.add(e)
                        if e in remaining_test:
                            remaining_test.remove(e)
                        added += 1
                if added > 1:
                    S = S_test2
                    remaining = remaining_test
                    improved = True
                    break
    
    # Final greedy pass
    if time.time() < deadline:
        try_add_greedy(S, remaining, conflicts, deadline)
    
    # Verify result is valid
    S_list = list(S)
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            a, b = S_list[i], S_list[j]
            if (a * b) % (a + b) == 0:
                # Invalid - should not happen, but safety check
                # Remove the conflict
                S.remove(b)
                break
    
    return list(S)
# EVOLVE-BLOCK-END
