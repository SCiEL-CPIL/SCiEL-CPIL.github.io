import numpy as np
import itertools
import time
import sys
from collections import deque

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Generate candidates: weight-3 and weight-4 vectors
    # Apply sign convention: first non-zero entry is positive to reduce symmetry
    candidates = []
    norms_sq = []
    
    # Weight 3 vectors
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            # Sign convention: first non-zero is positive
            if signs[0] == -1:
                continue
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(3)
    
    # Weight 4 vectors
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            # Sign convention: first non-zero is positive
            if signs[0] == -1:
                continue
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(4)
    
    candidates = np.array(candidates, dtype=np.int64)
    norms_sq = np.array(norms_sq, dtype=np.int64)
    n = len(candidates)
    
    if n == 0:
        v = [0] * D
        v[0] = 2
        return np.array([v], dtype=np.int64)
    
    # Compute dot products
    dots = candidates @ candidates.T
    
    # Conflict: dist_sq[i,j] < 4
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflicts = dist_sq < 4
    np.fill_diagonal(conflicts, False)
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for i in range(n):
        adj[i] = set(np.where(conflicts[i])[0].tolist())
    
    # Greedy coloring bound for branch-and-bound
    def greedy_color_bound(subset):
        """Compute a greedy coloring upper bound on clique size in the conflict graph
        restricted to subset. For MIS, we use the complement: color the non-conflict graph."""
        # Actually, for MIS in G, we can use: |MIS(G[S])| <= max clique in complement of G[S]
        # Simpler: use greedy coloring on the conflict graph to get an upper bound on independent set
        # An independent set in G is a clique in complement(G)
        # Greedy coloring of complement(G) gives upper bound on clique size = upper bound on MIS
        
        # Alternative simpler approach: use degree-based bound
        # For each vertex v in subset, the number of non-neighbors in subset is an upper bound contribution
        # Use a simple bound: for each v, count non-neighbors in subset
        # Better: use a greedy independent set as lower bound, and n - max_degree as upper bound
        
        # Let's use a proper greedy coloring on the conflict graph
        # The chromatic number of the conflict graph restricted to S gives an upper bound on the
        # size of a clique in the complement, which is the MIS size.
        
        # Actually, simpler: use the fact that in any independent set, no two vertices are adjacent.
        # A simple upper bound: for each vertex v, if we pick v, we can pick at most
        # (number of non-neighbors of v in S) + 1 vertices total.
        
        # Let's use a better approach: sequential coloring bound
        # Order vertices by degree (descending), color greedily
        vertices = list(subset)
        if not vertices:
            return 0
        
        # Sort by degree in the induced subgraph (descending)
        degs = {}
        for v in vertices:
            degs[v] = len(adj[v] & set(vertices))
        
        vertices_sorted = sorted(vertices, key=lambda v: -degs[v])
        
        # Greedy coloring
        color = {}
        num_colors = 0
        for v in vertices_sorted:
            # Find available colors
            used_colors = set()
            for u in vertices_sorted:
                if u == v:
                    break
                if u in color and v in adj[u]:
                    used_colors.add(color[u])
            
            c = 0
            while c in used_colors:
                c += 1
            color[v] = c
            num_colors = max(num_colors, c + 1)
        
        return num_colors
    
    # Branch and bound for Maximum Independent Set
    def branch_and_bound():
        """Find a large independent set using branch and bound with coloring bound."""
        best_set = set()
        best_size = 0
        best_set_list = []
        
        # Initial greedy solution
        greedy_set = greedy_is()
        best_set = set(greedy_set)
        best_size = len(greedy_set)
        best_set_list = list(greedy_set)
        
        # Recursive branch and bound
        def bb(remaining, current_set, current_size):
            nonlocal best_set, best_size, best_set_list
            
            if time.time() > deadline - 5.0:
                return
            
            if not remaining:
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
                    best_set_list = list(current_set)
                return
            
            # Upper bound using greedy coloring
            ub = greedy_color_bound(remaining)
            if current_size + ub <= best_size:
                return
            
            # Pick a vertex to branch on
            # Choose vertex with highest degree in remaining
            rem_list = list(remaining)
            if not rem_list:
                return
            
            # Sort by degree in remaining
            rem_set = set(rem_list)
            degs = {v: len(adj[v] & rem_set) for v in rem_list}
            v = max(rem_list, key=lambda x: degs[x])
            
            # Branch 1: Include v
            new_remaining = rem_set - adj[v] - {v}
            bb(new_remaining, current_set | {v}, current_size + 1)
            
            if time.time() > deadline - 5.0:
                return
            
            # Branch 2: Exclude v
            new_remaining2 = rem_set - {v}
            bb(new_remaining2, current_set, current_size)
        
        # Start with all vertices
        bb(set(range(n)), set(), 0)
        
        return best_set_list
    
    def greedy_is():
        """Greedy independent set: repeatedly pick vertex with fewest conflicts with remaining."""
        remaining = set(range(n))
        selected = []
        
        while remaining:
            if time.time() > deadline - 10.0:
                break
            
            # Find vertex with minimum degree in remaining
            best_v = None
            min_deg = float('inf')
            
            # Sample if large
            rem_list = list(remaining)
            if len(rem_list) > 100:
                import random
                sample = random.sample(rem_list, 100)
            else:
                sample = rem_list
            
            for v in sample:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            
            if best_v is None:
                break
            
            selected.append(best_v)
            remaining.discard(best_v)
            for nb in adj[best_v]:
                if nb in remaining:
                    remaining.discard(nb)
        
        return selected
    
    # Run multiple greedy trials to get a good initial solution
    import random
    random.seed(42)
    np.random.seed(42)
    
    best_selected = []
    
    # Multiple greedy trials with different random seeds
    for trial in range(50):
        if time.time() > deadline - 30.0:
            break
        
        # Shuffle and do greedy
        order = list(range(n))
        random.shuffle(order)
        
        remaining = set(order)
        selected = []
        
        while remaining:
            if time.time() > deadline - 10.0:
                break
            
            # Pick first available in order that has min conflicts
            best_v = None
            min_conf = float('inf')
            
            for v in order:
                if v not in remaining:
                    continue
                conf = len(adj[v] & remaining)
                if conf < min_conf:
                    min_conf = conf
                    best_v = v
            
            if best_v is None:
                break
            
            selected.append(best_v)
            remaining.discard(best_v)
            for nb in adj[best_v]:
                if nb in remaining:
                    remaining.discard(nb)
        
        if len(selected) > len(best_selected):
            best_selected = selected
    
    # Try branch and bound if time permits
    if time.time() < deadline - 40.0:
        bb_result = branch_and_bound()
        if len(bb_result) > len(best_selected):
            best_selected = bb_result
    
    # Local search: try to improve by swapping
    if best_selected and time.time() < deadline - 10.0:
        current_set = set(best_selected)
        current_size = len(best_selected)
        
        for ls_iter in range(20):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            
            # Try removing one vertex and adding two
            for v in list(current_set):
                if time.time() > deadline - 5.0:
                    break
                
                test_set = current_set - {v}
                
                # Find vertices not in test_set that can be added
                candidates_to_add = []
                for u in range(n):
                    if u not in test_set:
                        if not (adj[u] & test_set):
                            candidates_to_add.append(u)
                
                # Try adding up to 2
                if len(candidates_to_add) >= 2:
                    # Pick two that don't conflict with each other
                    for i in range(len(candidates_to_add)):
                        if time.time() > deadline - 5.0:
                            break
                        for j in range(i+1, len(candidates_to_add)):
                            a, b = candidates_to_add[i], candidates_to_add[j]
                            if b not in adj[a]:
                                new_set = test_set | {a, b}
                                if len(new_set) > current_size:
                                    current_set = new_set
                                    current_size = len(new_set)
                                    best_selected = list(current_set)
                                    improved = True
                                    break
                        if improved:
                            break
                if improved:
                    break
            
            if not improved:
                # Try removing 2 and adding 3
                removed = list(current_set)
                if len(removed) >= 2:
                    for i in range(min(10, len(removed))):
                        if time.time() > deadline - 5.0:
                            break
                        for j in range(i+1, min(10, len(removed))):
                            v1, v2 = removed[i], removed[j]
                            test_set = current_set - {v1, v2}
                            
                            candidates_to_add = []
                            for u in range(n):
                                if u not in test_set:
                                    if not (adj[u] & test_set):
                                        candidates_to_add.append(u)
                            
                            if len(candidates_to_add) >= 3:
                                # Try to find 3 mutually non-conflicting
                                found = []
                                for a in candidates_to_add:
                                    if len(found) >= 3:
                                        break
                                    if not any(a in adj[f] for f in found):
                                        found.append(a)
                                
                                if len(found) >= 3:
                                    new_set = test_set | set(found[:3])
                                    if len(new_set) > current_size:
                                        current_set = new_set
                                        current_size = len(new_set)
                                        best_selected = list(current_set)
                                        improved = True
                                        break
                        if improved:
                            break
                if improved:
                    break
            
            if not improved:
                break
    
    if not best_selected:
        v = [0] * D
        v[0] = 2
        return np.array([v], dtype=np.int64)
    
    return candidates[best_selected].astype(np.int64)
