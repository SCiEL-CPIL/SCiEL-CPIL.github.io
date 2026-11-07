import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    trees = instance_trees(20, 611)
    n = 20
    
    # K20 edge set
    K20_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            K20_edges.add((i, j))
    
    # Process trees in reverse order (largest first)
    order = list(range(n - 1, -1, -1))
    
    used_edges = set()
    result = [None] * n
    
    def try_embed(tree_idx, tree_edges, k, used_edges_set, host_map, assigned_count, unassigned):
        """
        Try to embed tree with k vertices into K20 using dynamic vertex ordering.
        Returns: list of k host vertices or None
        """
        if time.time() > deadline:
            return None
        
        if assigned_count == k:
            return list(host_map)
        
        # Pick the unassigned vertex with the most assigned neighbors (most constrained)
        best_v = -1
        best_constraint = -1
        for v in unassigned:
            constraint = 0
            for (a, b) in tree_edges:
                if a == v and b not in unassigned:
                    constraint += 1
                elif b == v and a not in unassigned:
                    constraint += 1
            if constraint > best_constraint:
                best_constraint = constraint
                best_v = v
        
        v = best_v
        
        # Find neighbors of v in the tree that are already assigned
        assigned_neighbors = []
        for (a, b) in tree_edges:
            if a == v and b not in unassigned:
                assigned_neighbors.append((b, host_map[b]))
            elif b == v and a not in unassigned:
                assigned_neighbors.append((a, host_map[a]))
        
        # Candidate host vertices for v
        candidates = list(range(n))
        
        # Filter: if v has assigned neighbors, host must be adjacent to their hosts
        if assigned_neighbors:
            filtered = []
            for h in candidates:
                ok = True
                for (nb_tree, nb_host) in assigned_neighbors:
                    edge = (min(h, nb_host), max(h, nb_host))
                    if edge not in K20_edges:
                        ok = False
                        break
                    if edge in used_edges_set:
                        ok = False
                        break
                if ok:
                    filtered.append(h)
            candidates = filtered
        
        # Sort candidates by how many unused edges they have (prefer vertices with more options)
        if assigned_neighbors:
            def count_free_edges(h):
                cnt = 0
                for nb in range(n):
                    if nb == h:
                        continue
                    e = (min(h, nb), max(h, nb))
                    if e in K20_edges and e not in used_edges_set:
                        cnt += 1
                return cnt
            candidates.sort(key=count_free_edges, reverse=True)
        
        new_unassigned = [u for u in unassigned if u != v]
        
        # Try candidates
        for h in candidates:
            # Check that host h is not already used by another assigned tree vertex
            h_used = False
            for i in range(k):
                if i != v and host_map[i] == h:
                    h_used = True
                    break
            if h_used:
                continue
            
            # Check all edges from v to assigned neighbors
            valid = True
            new_edges = []
            for (a, b) in tree_edges:
                if a == v and b not in unassigned:
                    nb_host = host_map[b]
                    edge = (min(h, nb_host), max(h, nb_host))
                    if edge in used_edges_set:
                        valid = False
                        break
                    new_edges.append(edge)
                elif b == v and a not in unassigned:
                    nb_host = host_map[a]
                    edge = (min(h, nb_host), max(h, nb_host))
                    if edge in used_edges_set:
                        valid = False
                        break
                    new_edges.append(edge)
            
            if not valid:
                continue
            
            # Assign
            host_map[v] = h
            used_edges_set.update(new_edges)
            
            res = try_embed(tree_idx, tree_edges, k, used_edges_set, host_map, assigned_count + 1, new_unassigned)
            if res is not None:
                return res
            
            # Undo
            host_map[v] = -1
            for e in new_edges:
                used_edges_set.discard(e)
        
        return None
    
    # For each tree in order, try to embed
    for tree_idx in order:
        if time.time() > deadline:
            break
        
        k = tree_idx + 1
        tree_edges = trees[tree_idx]
        
        if k == 1:
            # Single vertex, any host works
            result[tree_idx] = [0]
            continue
        
        if k == 2:
            # Edge (0,1), need any unused edge
            found = False
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) not in used_edges:
                        result[tree_idx] = [i, j]
                        used_edges.add((i, j))
                        found = True
                        break
                if found:
                    break
            if not found:
                result[tree_idx] = None
            continue
        
        # Backtracking for k >= 3
        host_map = [-1] * k
        unassigned = list(range(k))
        res = try_embed(tree_idx, tree_edges, k, used_edges, host_map, 0, unassigned)
        
        if res is not None:
            result[tree_idx] = res
            # Add all edges of this tree to used_edges
            for (a, b) in tree_edges:
                ha, hb = res[a], res[b]
                edge = (min(ha, hb), max(ha, hb))
                used_edges.add(edge)
        else:
            result[tree_idx] = None
    
    # Verify and count
    total_edges = 0
    for i in range(n):
        if result[i] is not None:
            k = i + 1
            if len(result[i]) != k:
                result[i] = None
                continue
            if len(set(result[i])) != k:
                result[i] = None
                continue
            if any(v < 0 or v >= n for v in result[i]):
                result[i] = None
                continue
            total_edges += len(trees[i])
    
    return result
