import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    d = 7
    n = 2 ** d
    deadline = time.time() + 110.0

    edges = []
    edge_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_idx[(x, y)] = idx
                edge_idx[(y, x)] = idx

    m = len(edges)  # 448

    # Build conflict graph: two edges conflict if they share a face
    # Each edge belongs to (d-1) = 6 faces, each face has 4 edges
    # So each edge conflicts with at most 6*3 = 18 others (some overlap)
    conflict = [set() for _ in range(m)]
    
    # Build face-to-edge mapping
    face_edges = []
    for b in range(d):
        for c in range(b + 1, d):
            for a in range(n):
                if (a & (1 << b)) or (a & (1 << c)):
                    continue
                v0 = a
                v1 = a ^ (1 << b)
                v2 = a ^ (1 << c)
                v3 = a ^ (1 << b) ^ (1 << c)
                e0 = edge_idx[(min(v0, v1), max(v0, v1))]
                e1 = edge_idx[(min(v0, v2), max(v0, v2))]
                e2 = edge_idx[(min(v1, v3), max(v1, v3))]
                e3 = edge_idx[(min(v2, v3), max(v2, v3))]
                face_edges.append((e0, e1, e2, e3))
    
    # Build conflict sets
    for face in face_edges:
        e0, e1, e2, e3 = face
        for i in range(4):
            for j in range(i+1, 4):
                conflict[face[i]].add(face[j])
                conflict[face[j]].add(face[i])
    
    # Convert to sorted lists and bitmasks for fast operations
    conflict_list = [sorted(c) for c in conflict]
    
    # For bit-parallel: represent sets of vertices as Python ints (bitmasks)
    # 448 bits fits in a Python int
    def to_mask(lst):
        mask = 0
        for v in lst:
            mask |= (1 << v)
        return mask
    
    conflict_masks = [to_mask(c) for c in conflict_list]
    
    # Greedy coloring of the conflict graph for upper bound (Tomita algorithm)
    # Color the graph so that vertices with same color are independent
    # This gives an upper bound on clique size in the complement (= independent set in original)
    
    def greedy_color(order):
        """Assign colors greedily to vertices in given order."""
        colors = {}
        color_classes = []
        for v in order:
            # Find first color where v has no neighbor in that color class
            placed = False
            for ci, cls in enumerate(color_classes):
                if not any(u in conflict_list[v] for u in cls):
                    cls.append(v)
                    colors[v] = ci
                    placed = True
                    break
            if not placed:
                color_classes.append([v])
                colors[v] = len(color_classes) - 1
        return colors, len(color_classes)
    
    # Maximum independent set via branch and bound on the conflict graph
    # We want max set of vertices with no conflicts = max independent set
    # Equivalent to max clique in complement graph
    
    best_iset = []
    best_size = 0
    node_count = 0
    
    def branch_and_bound(candidates, current_set):
        """
        candidates: list of vertex indices that can still be added
        current_set: list of vertices in current independent set
        """
        nonlocal best_iset, best_size, node_count
        
        if time.time() >= deadline - 2.0:
            return
        
        node_count += 1
        
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_iset = current_set[:]
            return
        
        # Upper bound: current size + len(candidates)
        if len(current_set) + len(candidates) <= best_size:
            return
        
        # Greedy coloring upper bound on independent set in subgraph induced by candidates
        # In the conflict graph, we want independent set, so we color the conflict graph
        # on candidates. Max independent set <= sum over colors of min(color_class_size, ...)
        # Actually simpler: use greedy coloring where same color = no conflicts
        # Then max independent set <= number of color classes (each color class is independent)
        # Wait, that's not right. Let me think again.
        
        # For maximum independent set in graph G:
        # Color G greedily. The chromatic number chi(G) >= alpha(G) is wrong.
        # Actually: alpha(G) <= chi(G) is false in general.
        # Correct bound: If we color G with k colors, then alpha(G) can be up to n (if one color class is all vertices).
        # Better: use the fact that in each color class (independent set in G), we can pick at most all of them.
        # So a valid upper bound is: for each color class, the max independent set within it is the whole class.
        # Total upper bound = sum of sizes of color classes = n. That's trivial.
        
        # Better approach: use greedy coloring to get an upper bound via:
        # alpha(G) <= max over colorings of (sum over colors of alpha(G[color_class]))
        # But within a color class, there are no edges, so alpha = size of class.
        # So this gives alpha <= n, which is useless.
        
        # Alternative: use the complement. Max independent set in G = max clique in complement of G.
        # For max clique, greedy coloring of the complement gives an upper bound.
        # Let's build the complement conflict structure.
        
        # Actually, let's use a simpler but effective bound:
        # Pick a vertex v from candidates. Either include v (remove its neighbors from candidates)
        # or exclude v (remove v from candidates).
        # The bound: if we include v, we can add at most (len(candidates) - 1 - deg_in_candidates(v)) more.
        # If we exclude v, we can add at most len(candidates) - 1 more.
        
        # Let's use a randomized pivot selection for branching
        rng = random.Random(node_count)
        
        # Try to find a good branching vertex
        # Use degree heuristic: branch on vertex with max degree in candidate subgraph
        if len(candidates) > 100:
            # Sample to speed up
            sample = rng.sample(candidates, min(50, len(candidates)))
            best_v = sample[0]
            best_deg = 0
            for v in sample:
                deg = sum(1 for u in candidates if u in conflict_list[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
        else:
            best_v = candidates[0]
            best_deg = 0
            for v in candidates:
                deg = sum(1 for u in candidates if u in conflict_list[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
        
        # Branch 1: include best_v
        # Remove best_v and all its neighbors from candidates
        neighbors = set(conflict_list[best_v])
        new_candidates_include = [v for v in candidates if v != best_v and v not in neighbors]
        
        # Upper bound for include branch: len(current_set) + 1 + len(new_candidates_include)
        if len(current_set) + 1 + len(new_candidates_include) > best_size:
            branch_and_bound(new_candidates_include, current_set + [best_v])
        
        # Branch 2: exclude best_v
        new_candidates_exclude = [v for v in candidates if v != best_v]
        
        # Upper bound for exclude branch: len(current_set) + len(new_candidates_exclude)
        if len(current_set) + len(new_candidates_exclude) > best_size:
            branch_and_bound(new_candidates_exclude, current_set)
    
    # Initialize with a good greedy solution
    def greedy_independent_set(seed):
        rng = random.Random(seed)
        remaining = set(range(m))
        result = []
        while remaining:
            # Pick random vertex from remaining
            v = rng.choice(list(remaining))
            result.append(v)
            remaining.discard(v)
            # Remove all neighbors of v
            for u in conflict_list[v]:
                remaining.discard(u)
        return result
    
    # Get initial solution from multiple greedy runs
    for seed in range(20):
        if time.time() >= deadline - 5.0:
            break
        sol = greedy_independent_set(seed)
        if len(sol) > best_size:
            best_size = len(sol)
            best_iset = sol
    
    # Run branch and bound
    initial_candidates = list(range(m))
    branch_and_bound(initial_candidates, [])
    
    # Additional local search: try to improve best_iset
    def local_search_improve(iset, max_rounds=50):
        nonlocal best_size, best_iset
        iset = iset[:]
        iset_set = set(iset)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 3.0:
            improved = False
            rounds += 1
            # Try swapping: remove one edge, add up to two non-conflicting edges
            for ri in list(iset):
                iset_set.discard(ri)
                # Find all edges that can be added
                addable = []
                for ei in range(m):
                    if ei in iset_set:
                        continue
                    if any(ei in conflict_list[u] for u in iset_set):
                        continue
                    addable.append(ei)
                
                # Try adding pairs
                best_pair = None
                best_pair_gain = 0
                if len(addable) > 1:
                    # Try random pairs
                    rng = random.Random(rounds * 1000 + ri)
                    tries = min(20, len(addable) * (len(addable)-1) // 2)
                    for _ in range(tries):
                        i, j = rng.sample(addable, 2)
                        # Check if i and j conflict
                        if j in conflict_list[i]:
                            continue
                        # Check both don't conflict with iset_set
                        if any(i in conflict_list[u] or j in conflict_list[u] for u in iset_set):
                            continue
                        gain = 2 - 1  # remove 1, add 2
                        if gain > best_pair_gain:
                            best_pair_gain = gain
                            best_pair = (i, j)
                
                if best_pair:
                    i, j = best_pair
                    iset_set.add(i)
                    iset_set.add(j)
                    iset.append(i)
                    iset.append(j)
                    improved = True
                    if len(iset) > best_size:
                        best_size = len(iset)
                        best_iset = iset[:]
                    break
                else:
                    # Try adding single edge
                    if addable:
                        rng = random.Random(rounds * 1000 + ri)
                        ai = rng.choice(addable)
                        iset_set.add(ai)
                        iset.append(ai)
                        improved = True
                        if len(iset) > best_size:
                            best_size = len(iset)
                            best_iset = iset[:]
                        break
                # Restore ri if we didn't improve
                if not improved:
                    iset_set.add(ri)
        
        return iset
    
    if best_iset and time.time() < deadline - 5.0:
        improved_iset = local_search_improve(best_iset)
        if len(improved_iset) > best_size:
            best_size = len(improved_iset)
            best_iset = improved_iset
    
    out = []
    for ei in best_iset:
        x, y = edges[ei]
        out.append([x, y])
    
    return out
# EVOLVE-BLOCK-END
