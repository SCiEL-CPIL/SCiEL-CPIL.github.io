import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 17
    MIN_DIST = 6
    
    def time_left():
        return deadline - time.time()
    
    # Generate all weight-8 and weight-9 words
    candidates = []
    for w in [8, 9]:
        for combo in combinations(range(N), w):
            c = 0
            for p in combo:
                c |= (1 << p)
            candidates.append(c)
    
    n = len(candidates)
    
    # Build conflict graph: edge between i,j if dist(candidates[i], candidates[j]) < MIN_DIST
    # Use bitset representation for fast neighbor queries
    # neighbors[i] is a set of indices j where dist < MIN_DIST
    neighbors = [set() for _ in range(n)]
    
    for i in range(n):
        ci = candidates[i]
        for j in range(i + 1, n):
            if (ci ^ candidates[j]).bit_count() < MIN_DIST:
                neighbors[i].add(j)
                neighbors[j].add(i)
    
    # Convert to sorted lists for faster iteration
    neighbor_lists = [sorted(s) for s in neighbors]
    
    # Greedy coloring for upper bound: assign colors to remaining vertices
    # such that no two adjacent vertices have the same color.
    # Number of colors is an upper bound on the size of any independent set.
    def color_bound(vertices):
        """Assign greedy colors to vertices, return (coloring, num_colors)."""
        colors = {}
        color_count = 0
        # Process vertices in order (could use degeneracy ordering for better bounds)
        for v in vertices:
            # Find smallest color not used by neighbors of v that are already colored
            used = set()
            for u in neighbor_lists[v]:
                if u in colors:
                    used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
            color_count = max(color_count, c + 1)
        return colors, color_count
    
    # Branch and bound for Maximum Independent Set
    best_set = set()
    best_size = 0
    nodes_explored = 0
    max_nodes = 500000  # Limit to avoid timeout
    
    def bnb(available, current_set):
        nonlocal best_set, best_size, nodes_explored
        nodes_explored += 1
        
        if nodes_explored > max_nodes:
            return
        
        if time_left() < 5.0:
            return
        
        if not available:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        
        # Upper bound: |current_set| + |available|
        if len(current_set) + len(available) <= best_size:
            return
        
        # Better upper bound using coloring
        avail_list = list(available)
        colors, num_colors = color_bound(avail_list)
        if len(current_set) + num_colors <= best_size:
            return
        
        # Select vertex with maximum degree in the induced subgraph
        v = max(avail_list, key=lambda x: len([y for y in neighbor_lists[x] if y in available]))
        
        # Branch 1: include v
        new_available = {u for u in available if u != v and u not in neighbor_lists[v]}
        current_set.add(v)
        bnb(new_available, current_set)
        current_set.remove(v)
        
        # Branch 2: exclude v
        new_available = available - {v}
        bnb(new_available, current_set)
    
    bnb(set(range(n)), set())
    
    # Convert best_set to codewords
    code = [candidates[i] for i in best_set]
    code_set = set(code)
    
    # Try to add more words (including weights other than 8,9)
    if time_left() > 10.0:
        code_matrix = np.array([[(c >> i) & 1 for i in range(N)] for c in code], dtype=np.uint8)
        
        def can_add(cand):
            if len(code_matrix) == 0:
                return True
            cand_bits = np.array([(cand >> i) & 1 for i in range(N)], dtype=np.uint8)
            diffs = code_matrix ^ cand_bits
            dists = diffs.sum(axis=1)
            return bool(np.min(dists) >= MIN_DIST)
        
        # Try adding weight 7,10 words
        for w in [7, 10]:
            if time_left() < 5.0:
                break
            for combo in combinations(range(N), w):
                if time_left() < 5.0:
                    break
                c = 0
                for p in combo:
                    c |= (1 << p)
                if c not in code_set:
                    if can_add(c):
                        code.append(c)
                        code_set.add(c)
                        new_row = np.array([(c >> i) & 1 for i in range(N)], dtype=np.uint8)
                        code_matrix = np.vstack([code_matrix, new_row])
        
        # Random search for additional words
        rng = random.Random(42)
        for _ in range(5000):
            if time_left() < 5.0:
                break
            w = rng.randint(6, 12)
            positions = rng.sample(range(N), w)
            candidate = 0
            for p in positions:
                candidate |= (1 << p)
            if candidate not in code_set:
                if can_add(candidate):
                    code.append(candidate)
                    code_set.add(candidate)
                    new_row = np.array([(candidate >> i) & 1 for i in range(N)], dtype=np.uint8)
                    code_matrix = np.vstack([code_matrix, new_row])
    
    return code

def greedy_construct(deadline):
    code = [0]
    code_set = {0}
    rng = random.Random(42)
    while time.time() < deadline:
        w = rng.randint(3, 14)
        positions = rng.sample(range(17), w)
        c = 0
        for p in positions:
            c |= (1 << p)
        if c not in code_set:
            ok = True
            for w2 in code:
                if (c ^ w2).bit_count() < 6:
                    ok = False
                    break
            if ok:
                code.append(c)
                code_set.add(c)
    return code
