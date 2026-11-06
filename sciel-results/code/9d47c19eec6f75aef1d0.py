import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(n, edges, time_limit):
        # Build adjacency list
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Initialize paths: each edge is a path
        # Path representation: list of vertices
        paths = [[u, v] for u, v in edges]
        
        # Index paths by start and end vertices to speed up merging
        # start_index: vertex -> list of path indices starting at vertex
        # end_index: vertex -> list of path indices ending at vertex
        # Note: A path [u, v] starts at u and ends at v.
        # We can also reverse it: starts at v, ends at u.
        
        def rebuild_indices(paths):
            start_idx = defaultdict(list)
            end_idx = defaultdict(list)
            for i, p in enumerate(paths):
                if not p: continue
                start_idx[p[0]].append(i)
                end_idx[p[-1]].append(i)
            return start_idx, end_idx

        start_idx, end_idx = rebuild_indices(paths)
        
        # To allow reversing paths, we treat a path [u, ..., v] as having two "ends": (u, v) and (v, u).
        # For merging, we need:
        # Path A ends at u.
        # Path B starts at v.
        # Edge (u, v) exists and is a path itself (or we are merging A, B, and the edge path).
        
        # Strategy:
        # Iterate while time permits.
        # Find a triple (P1, P_edge, P2) that can be merged.
        # P1 ends at u, P_edge is [u, v] or [v, u], P2 starts at v (or u if reversed).
        
        # Optimization: Only consider P_edge as a single edge path for now.
        # This is a valid subset of merges.
        
        # We need to track which paths are single edges.
        # Actually, just checking len(p) == 1 is O(1).
        
        # To avoid O(N^3) search, we can iterate over single edge paths.
        # For each single edge path E = [u, v]:
        #   Check neighbors of u in start_idx? No, P1 must END at u.
        #   So look in end_idx[u].
        #   Check neighbors of v in end_idx? No, P2 must START at v.
        #   So look in start_idx[v].
        
        # If we find P1 in end_idx[u] and P2 in start_idx[v]:
        #   Check if P1 + E + P2 is simple.
        #   If yes, merge.
        
        # We should also consider E = [v, u] orientation.
        
        merged = True
        while merged and time.time() < time_limit:
            merged = False
            # Shuffle single edge paths to avoid bias
            single_edges = [i for i, p in enumerate(paths) if len(p) == 2]
            random.shuffle(single_edges)
            
            for i_e in single_edges:
                if time.time() > time_limit: break
                p_e = paths[i_e]
                if len(p_e) != 2: continue # might have changed? No, we rebuild indices after merge
                
                u, v = p_e[0], p_e[1]
                
                # Case 1: P1 ends at u, P2 starts at v
                # P1 from end_idx[u], P2 from start_idx[v]
                if u in end_idx and v in start_idx:
                    p1_candidates = end_idx[u]
                    p2_candidates = start_idx[v]
                    
                    # Filter out i_e itself if it's in candidates (it shouldn't be, as it's single edge, 
                    # but end_idx[u] contains paths ending at u. If p_e=[u,v], it ends at v. 
                    # So i_e is NOT in end_idx[u]. 
                    # i_e is in end_idx[v] and start_idx[u].)
                    
                    for i1 in p1_candidates:
                        if i1 == i_e: continue
                        p1 = paths[i1]
                        # Check simplicity: p1[-1] is u.
                        # New path: p1 + [v] + p2[1:] ? 
                        # p1 = [..., u]. p_e = [u, v]. p2 = [v, ...].
                        # Combined: p1 + p2[1:]
                        # We need to ensure no vertex in p2[1:] is in p1.
                        # And no vertex in p1 is in p2[1:].
                        
                        p2_candidates_valid = []
                        for i2 in p2_candidates:
                            if i2 == i_e or i2 == i1: continue
                            p2 = paths[i2]
                            # p2 starts at v.
                            # Check if any vertex in p2[1:] is in p1
                            # And if any vertex in p1 is in p2[1:]
                            # This is O(L1 + L2).
                            
                            # Quick check: sets
                            set_p1 = set(p1)
                            set_p2_rest = set(p2[1:])
                            if not set_p1.isdisjoint(set_p2_rest):
                                continue
                            
                            # Also need to ensure p1 and p2 don't share the connecting edge? 
                            # They are edge-disjoint by construction (partition).
                            # The only new edge is (u,v).
                            # (u,v) is in p_e.
                            # We remove p1, p2, p_e.
                            # So we just need vertex simplicity.
                            
                            p2_candidates_valid.append(i2)
                        
                        if p2_candidates_valid:
                            # Pick one (random or first)
                            i2 = random.choice(p2_candidates_valid)
                            p2 = paths[i2]
                            
                            # Perform merge
                            new_path = p1 + p2[1:]
                            
                            # Remove i1, i2, i_e from paths
                            # Be careful with indices.
                            # Mark as None or filter.
                            # Since we rebuild indices anyway, we can just filter.
                            
                            # To avoid index shifting issues, we collect indices to remove.
                            # But we are in a loop.
                            # Let's just do the removal and break.
                            
                            # Remove i1, i2, i_e
                            # Add new_path
                            
                            # We must ensure we don't remove the same index twice.
                            # i1, i2, i_e are distinct.
                            
                            # Create new list
                            new_paths = []
                            for idx, p in enumerate(paths):
                                if idx == i1 or idx == i2 or idx == i_e:
                                    continue
                                new_paths.append(p)
                            new_paths.append(new_path)
                            
                            paths = new_paths
                            start_idx, end_idx = rebuild_indices(paths)
                            merged = True
                            break # Restart loop with new indices
                            
                # Case 2: P1 ends at v, P2 starts at u
                # p_e is [u, v]. We can use it as [v, u].
                if v in end_idx and u in start_idx:
                    p1_candidates = end_idx[v]
                    p2_candidates = start_idx[u]
                    
                    for i1 in p1_candidates:
                        if i1 == i_e: continue
                        p1 = paths[i1]
                        # p1 ends at v.
                        # p_e used as [v, u].
                        # p2 starts at u.
                        
                        p2_candidates_valid = []
                        for i2 in p2_candidates:
                            if i2 == i_e or i2 == i1: continue
                            p2 = paths[i2]
                            
                            set_p1 = set(p1)
                            set_p2_rest = set(p2[1:])
                            if not set_p1.isdisjoint(set_p2_rest):
                                continue
                            
                            p2_candidates_valid.append(i2)
                        
                        if p2_candidates_valid:
                            i2 = random.choice(p2_candidates_valid)
                            p2 = paths[i2]
                            
                            new_path = p1 + p2[1:]
                            
                            new_paths = []
                            for idx, p in enumerate(paths):
                                if idx == i1 or idx == i2 or idx == i_e:
                                    continue
                                new_paths.append(p)
                            new_paths.append(new_path)
                            
                            paths = new_paths
                            start_idx, end_idx = rebuild_indices(paths)
                            merged = True
                            break
                            
                if merged:
                    break

        return paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for n, p, seed in specs:
        if time.time() > deadline - 10:
            # Fallback: return what we have or trivial solution
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
            
        edges = instance(n, p, seed)
        # Allocate time per graph
        # Total time 110s. 3 graphs. ~35s each.
        time_limit = time.time() + 35.0
        paths = decompose_graph(n, edges, time_limit)
        results.append(paths)
        
    return results

