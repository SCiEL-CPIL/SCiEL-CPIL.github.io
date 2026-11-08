"""Construction of a kissing configuration in R^11 using integer vectors."""
import time
import numpy as np
from itertools import combinations
from functools import lru_cache

D = 11

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 105  # safety margin from 150s limit

    def generate_weight_k(k):
        """Generate all integer vectors with exactly k nonzero entries, each ±1."""
        vectors = []
        for coords in combinations(range(D), k):
            for signs in _signs_iter(k):
                v = np.zeros(D, dtype=np.int64)
                for idx, s in zip(coords, signs):
                    v[idx] = s
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    def _signs_iter(k):
        """Generate all 2^k sign combinations."""
        for mask in range(1 << k):
            yield tuple(1 if (mask >> i) & 1 else -1 for i in range(k))

    def branch_and_bound_mis(n, adj, time_limit):
        """
        Branch and Bound Maximum Independent Set.
        Uses degree-based upper bound for pruning.
        Returns list of selected vertex indices.
        """
        best_set = set()
        best_size = 0
        deadline = time.time() + time_limit
        
        def upper_bound(vertices):
            """
            Upper bound on MIS size using greedy coloring of the induced subgraph.
            This is a standard technique: color the complement graph greedily,
            each color class is an independent set in original graph.
            """
            if not vertices:
                return 0
            
            # Greedy coloring of the induced subgraph on vertices
            # Colors represent independent sets in the original graph
            uncolored = set(vertices)
            colors = []
            
            while uncolored:
                # Find a maximal independent set in the induced subgraph
                # using greedy approach on remaining vertices
                color_class = set()
                available = list(uncolored)
                # Sort by degree within available for better coloring
                available.sort(key=lambda v: sum(1 for u in available if u != v and u in adj[v]))
                
                for v in available:
                    if v not in uncolored:
                        continue
                    # Check if v can be added to current color class
                    # (i.e., v is not adjacent to any vertex in color_class)
                    can_add = True
                    for u in color_class:
                        if u in adj[v]:
                            can_add = False
                            break
                    if can_add:
                        color_class.add(v)
                        uncolored.discard(v)
                
                if color_class:
                    colors.append(len(color_class))
            
            return sum(colors)
        
        def search(vertices, current_set):
            nonlocal best_set, best_size, deadline
            
            if time.time() > deadline:
                return False
            
            if not vertices:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                return True
            
            # Upper bound: if current + upper_bound <= best, prune
            ub = upper_bound(vertices)
            if len(current_set) + ub <= best_size:
                return True
            
            # Choose vertex with minimum degree in induced subgraph
            v = min(vertices, key=lambda x: sum(1 for y in vertices if y != x and y in adj[x]))
            
            # Branch 1: include v
            new_vertices = {u for u in vertices if u != v and u not in adj[v]}
            search(new_vertices, current_set | {v})
            
            if time.time() > deadline:
                return False
            
            # Branch 2: exclude v
            vertices.discard(v)
            search(vertices, current_set)
            vertices.add(v)  # restore
            
            return True
        
        search(set(range(n)), set())
        return list(best_set)

    def greedy_mis(n, adj):
        """
        Minimum Degree Greedy for Maximum Independent Set (fallback).
        """
        selected = []
        remaining = set(range(n))
        deg = [len(adj[i]) for i in range(n)]
        
        while remaining:
            min_deg = min(deg[i] for i in remaining)
            candidates = [i for i in remaining if deg[i] == min_deg]
            v = candidates[0]
            selected.append(v)
            remaining.discard(v)
            for u in list(adj[v]):
                if u in remaining:
                    remaining.discard(u)
                    for w in adj[u]:
                        if w in remaining:
                            deg[w] -= 1
        return selected

    # Baseline: all weight-2 vectors (220)
    w2 = generate_weight_k(2)
    best_result = w2
    best_count = len(w2)

    if time.time() > deadline:
        return best_result

    # Try weight-3 vectors
    w3 = generate_weight_k(3)
    n3 = len(w3)
    
    # For weight-3 vectors, norm² = 3. Need distance² >= 3.
    # distance² = 3+3-2*dot = 6-2*dot >= 3 => dot <= 1.5 => dot <= 1
    # Conflict if dot > 1, i.e., dot >= 2
    
    dots = w3 @ w3.T
    conflict_mask = (dots >= 2)
    np.fill_diagonal(conflict_mask, False)
    
    adj = [set() for _ in range(n3)]
    conflict_indices = np.argwhere(conflict_mask)
    for i, j in conflict_indices:
        adj[i].add(j)
        adj[j].add(i)
    
    # Use branch and bound for weight-3 (n3 = 1320, manageable)
    time_remaining = deadline - time.time()
    if time_remaining > 20:
        selected = branch_and_bound_mis(n3, adj, min(time_remaining - 5, 40))
    else:
        selected = greedy_mis(n3, adj)
    
    result_w3 = w3[selected]
    
    if len(selected) > best_count:
        best_result = result_w3
        best_count = len(selected)
    
    if time.time() > deadline:
        return best_result

    # Try weight-4 vectors
    w4 = generate_weight_k(4)
    n4 = len(w4)
    
    # For weight-4 vectors, norm² = 4. Need distance² >= 4.
    # distance² = 4+4-2*dot = 8-2*dot >= 4 => dot <= 2
    # Conflict if dot > 2, i.e., dot >= 3
    
    if n4 < 50000:  # safety check
        dots4 = w4 @ w4.T
        conflict_mask4 = (dots4 >= 3)
        np.fill_diagonal(conflict_mask4, False)
        
        adj4 = [set() for _ in range(n4)]
        conflict_indices4 = np.argwhere(conflict_mask4)
        for i, j in conflict_indices4:
            adj4[i].add(j)
            adj4[j].add(i)
        
        time_remaining = deadline - time.time()
        if time_remaining > 15:
            selected4 = branch_and_bound_mis(n4, adj4, min(time_remaining - 5, 30))
        else:
            selected4 = greedy_mis(n4, adj4)
        
        result_w4 = w4[selected4]
        
        if len(selected4) > best_count:
            best_result = result_w4
            best_count = len(selected4)
    
    if time.time() > deadline:
        return best_result

    # Try weight-5 vectors if time permits
    time_remaining = deadline - time.time()
    if time_remaining > 25:
        w5 = generate_weight_k(5)
        n5 = len(w5)
        
        # For weight-5 vectors, norm² = 5. Need distance² >= 5.
        # distance² = 5+5-2*dot = 10-2*dot >= 5 => dot <= 2.5 => dot <= 2
        # Conflict if dot > 2, i.e., dot >= 3
        
        if n5 < 100000:
            dots5 = w5 @ w5.T
            conflict_mask5 = (dots5 >= 3)
            np.fill_diagonal(conflict_mask5, False)
            
            adj5 = [set() for _ in range(n5)]
            conflict_indices5 = np.argwhere(conflict_mask5)
            for i, j in conflict_indices5:
                adj5[i].add(j)
                adj5[j].add(i)
            
            time_remaining = deadline - time.time()
            if time_remaining > 10:
                selected5 = branch_and_bound_mis(n5, adj5, min(time_remaining - 5, 20))
            else:
                selected5 = greedy_mis(n5, adj5)
            
            result_w5 = w5[selected5]
            
            if len(selected5) > best_count:
                best_result = result_w5
                best_count = len(selected5)
    
    return best_result
# EVOLVE-BLOCK-END
