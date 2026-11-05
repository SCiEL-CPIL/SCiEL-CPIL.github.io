"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import networkx as nx
from collections import defaultdict

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        graph = nx.Graph()
        for u, v in edges:
            graph.add_edge(u, v)
        
        # Initialize paths as single edges
        # Each path is represented as (vertices_list, vertex_set)
        # But we need to return just vertex lists at the end.
        # Let's keep a list of paths (lists of vertices) and a parallel list of sets for fast intersection.
        
        paths = [[u, v] for u, v in edges]
        path_sets = [set(p) for p in paths]
        
        # Index paths by endpoints
        # endpoint_map: vertex -> list of path indices where vertex is an endpoint
        endpoint_map = defaultdict(list)
        
        def update_endpoint_map(path_idx, old_path, new_path=None):
            if old_path:
                endpoints = {old_path[0], old_path[-1]}
                for ep in endpoints:
                    if path_idx in endpoint_map[ep]:
                        endpoint_map[ep].remove(path_idx)
            if new_path:
                endpoints = {new_path[0], new_path[-1]}
                for ep in endpoints:
                    endpoint_map[ep].append(path_idx)
        
        # Initialize endpoint map
        for i, p in enumerate(paths):
            endpoints = {p[0], p[-1]}
            for ep in endpoints:
                endpoint_map[ep].append(i)
                
        changed = True
        while changed:
            changed = False
            n_paths = len(paths)
            if n_paths < 2:
                break
            
            # Build auxiliary graph for matching
            merge_graph = nx.Graph()
            merge_graph.add_nodes_from(range(n_paths))
            
            # Find potential merges
            # We iterate over vertices that are endpoints of at least two paths
            candidate_pairs = set()
            
            for v, p_indices in endpoint_map.items():
                if len(p_indices) < 2:
                    continue
                # Check all pairs in p_indices
                # Note: A path has two endpoints. It might appear in p_indices for two different vertices.
                # We just need to ensure we check unique pairs of path indices.
                for i in range(len(p_indices)):
                    for j in range(i + 1, len(p_indices)):
                        idx1 = p_indices[i]
                        idx2 = p_indices[j]
                        if idx1 == idx2:
                            continue
                        # Avoid duplicate pair checks
                        pair = (min(idx1, idx2), max(idx1, idx2))
                        if pair in candidate_pairs:
                            continue
                        candidate_pairs.add(pair)
                        
                        p1 = paths[idx1]
                        p2 = paths[idx2]
                        s1 = path_sets[idx1]
                        s2 = path_sets[idx2]
                        
                        # Check if they share exactly one vertex
                        # We know they share v. Check if they share any other.
                        # Intersection size must be 1.
                        # Since v is in both, if intersection is {v}, then size is 1.
                        # We can check if s1 & s2 has any element other than v.
                        # Or simply check len(s1 & s2) == 1.
                        # Given small path sizes, set intersection is fast.
                        common = s1 & s2
                        if len(common) == 1:
                            # Verify v is the common vertex
                            # It must be, because v is in s1 and s2, and common has size 1.
                            # Also verify v is endpoint of both (guaranteed by endpoint_map)
                            merge_graph.add_edge(idx1, idx2)
            
            if merge_graph.number_of_edges() == 0:
                break
                
            # Find maximum matching
            matching = nx.max_weight_matching(merge_graph, maxcardinality=True)
            
            if not matching:
                break
                
            # Apply merges
            new_paths = []
            new_path_sets = []
            used = set()
            
            # We need to map old indices to new indices or just rebuild
            # Since we are removing pairs, we can just build new lists.
            # However, endpoint_map needs to be updated.
            # It's easier to rebuild endpoint_map from scratch if changes are significant,
            # or update it carefully.
            # Given the small size, rebuilding might be safer and fast enough.
            # But let's try to update incrementally or just rebuild at the end of the loop.
            
            # Let's collect the new paths first
            merged_paths_data = []
            for i, j in matching:
                if i in used or j in used:
                    continue
                used.add(i)
                used.add(j)
                
                p1 = paths[i]
                p2 = paths[j]
                s1 = path_sets[i]
                s2 = path_sets[j]
                
                # Determine orientation
                # Find common vertex v
                v = next(iter(s1 & s2))
                
                # p1 endpoints: p1[0], p1[-1]
                # p2 endpoints: p2[0], p2[-1]
                
                if p1[0] == v:
                    p1 = p1[::-1]
                    # Now p1 ends at v
                if p2[0] == v:
                    p2 = p2[::-1]
                    # Now p2 ends at v
                    
                # Now p1 ends at v, p2 ends at v?
                # If p1 was reversed, it ends at v. If not, it started at v?
                # Let's trace:
                # If p1[0] == v, we reversed p1. So new p1[-1] == v.
                # If p1[-1] == v, we didn't reverse. So new p1[-1] == v.
                # So in both cases, p1 now ends at v.
                # Same for p2.
                
                # So we have p1 ending at v and p2 ending at v.
                # To join them, we reverse p2 and append to p1?
                # p1: ... -> v
                # p2: ... -> v (reversed: v -> ... )
                # So p1 + p2[1:] works if p2 starts with v.
                # Since we ensured p2 ends at v, reversing p2 makes it start at v.
                
                p2_rev = p2[::-1]
                merged_path = p1 + p2_rev[1:]
                merged_set = s1 | s2
                
                merged_paths_data.append((merged_path, merged_set))
                
            # Add unmerged paths
            for k in range(n_paths):
                if k not in used:
                    merged_paths_data.append((paths[k], path_sets[k]))
                    
            paths = [d[0] for d in merged_paths_data]
            path_sets = [d[1] for d in merged_paths_data]
            
            # Rebuild endpoint map
            endpoint_map.clear()
            for i, p in enumerate(paths):
                endpoints = {p[0], p[-1]}
                for ep in endpoints:
                    endpoint_map[ep].append(i)
                    
            changed = True

        results.append(paths)
        
    return results
# EVOLVE-BLOCK-END
