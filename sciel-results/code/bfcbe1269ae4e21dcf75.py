import random
import time

def construct():
    start_time = time.time()
    deadline = 40.0  # safety margin before 60s limit
    
    n = 15
    
    def is_valid_fast(edges_list):
        """Fast validity check: no 3-edge path s-u-v-t with d(s)==d(t)"""
        adj = [set() for _ in range(n)]
        degrees = [0] * n
        
        for u, v in edges_list:
            adj[u].add(v)
            adj[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
        
        # Group vertices by degree
        deg_to_verts = {}
        for i in range(n):
            deg_to_verts.setdefault(degrees[i], []).append(i)
        
        # For each pair of vertices with same degree, check if a 3-edge path exists
        for deg, verts in deg_to_verts.items():
            if len(verts) < 2:
                continue
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    s, t = verts[i], verts[j]
                    # Check if there's a path s-u-v-t
                    for u in adj[s]:
                        if u == t:
                            continue
                        for v in adj[u]:
                            if v == s or v == t or v == u:
                                continue
                            if t in adj[v]:
                                return False
        return True
    
    def get_missing_edges(edges_set):
        missing = []
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in edges_set:
                    missing.append((i, j))
        return missing
    
    # Strategy: Start with structured graphs based on degree partitions
    # Key idea: vertices with same degree should not be connected by 3-edge paths
    
    def try_add_edges(edges_set):
        """Greedy: try to add all possible edges that maintain validity"""
        changed = True
        while changed:
            changed = False
            missing = get_missing_edges(edges_set)
            random.shuffle(missing)
            for e in missing:
                new_edges = edges_set | {e}
                if is_valid_fast(list(new_edges)):
                    edges_set.add(e)
                    changed = True
        return edges_set
    
    def local_search_with_swaps(edges_set, max_steps=2000):
        """Local search: try single-edge additions, then swaps"""
        edges = set(edges_set)
        best = len(edges)
        
        for step in range(max_steps):
            if time.time() - start_time > deadline:
                break
            
            # Try adding edges
            missing = get_missing_edges(edges)
            random.shuffle(missing)
            added = False
            for e in missing:
                new_edges = edges | {e}
                if is_valid_fast(list(new_edges)):
                    edges.add(e)
                    added = True
                    break
            
            if not added:
                # Try swaps: remove one edge, add two
                edge_list = list(edges)
                random.shuffle(edge_list)
                swapped = False
                for e_remove in edge_list[:20]:  # limit for speed
                    edges.remove(e_remove)
                    # Try to add edges
                    missing = get_missing_edges(edges)
                    random.shuffle(missing)
                    added_count = 0
                    for e_add in missing:
                        new_edges = edges | {e_add}
                        if is_valid_fast(list(new_edges)):
                            edges.add(e_add)
                            added_count += 1
                    if added_count > 1:
                        swapped = True
                        break
                    else:
                        edges.add(e_remove)  # restore
            
            if len(edges) > best:
                best = len(edges)
        
        return edges, best
    
    best_edges = set()
    best_score = 0
    
    # Multiple structured initializations
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Strategy 1: Random starts with varying densities
    for density in [0.3, 0.5, 0.7]:
        if time.time() - start_time > deadline:
            break
        num_initial = int(len(all_edges) * density)
        initial = set(random.sample(all_edges, num_initial))
        
        # Greedy add
        initial = try_add_edges(initial)
        
        # Local search
        edges, score = local_search_with_swaps(initial)
        if score > best_score:
            best_score = score
            best_edges = edges
    
    # Strategy 2: Structured degree-based initialization
    # Try graphs where vertices are partitioned by target degree
    # and connections respect the 3-path constraint
    
    # Try a "layered" construction: 
    # Group vertices, connect within and between groups carefully
    for trial in range(30):
        if time.time() - start_time > deadline:
            break
        
        # Random degree assignment pattern
        # Assign target degrees, then try to realize them
        target_degs = [random.randint(1, 8) for _ in range(n)]
        random.shuffle(target_degs)
        
        # Build graph trying to achieve these degrees while maintaining validity
        edges = set()
        # Sort by target degree descending, add edges greedily
        order = sorted(range(n), key=lambda x: -target_degs[x])
        
        for u in order:
            for v in order:
                if u >= v:
                    continue
                if (u, v) in edges:
                    continue
                # Check if adding would violate degree constraints or validity
                new_edges = edges | {(u, v)}
                if is_valid_fast(list(new_edges)):
                    edges.add((u, v))
        
        edges = try_add_edges(edges)
        edges, score = local_search_with_swaps(edges, max_steps=1000)
        if score > best_score:
            best_score = score
            best_edges = edges
    
    # Final attempt: if we have a good solution, try more aggressive optimization
    if best_score > 0 and time.time() - start_time < deadline - 5:
        edges, score = local_search_with_swaps(best_edges, max_steps=3000)
        if score > best_score:
            best_score = score
            best_edges = edges
    
    if not best_edges:
        # Fallback: star graph (always valid since all paths from leaf to leaf go through center)
        # Actually star: 0 connected to all. Path s-0-v-t? No, s-0-x-t where x is neighbor of 0.
        # d(s)=1, d(t)=1 for leaves. Path s-0-x-t: s and t both degree 1. VIOLATION!
        # So star is NOT valid. Use empty graph.
        return []
    
    result = [[u, v] for u, v in sorted(best_edges)]
    return result
