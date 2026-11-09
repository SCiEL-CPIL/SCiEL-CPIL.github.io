"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges, max_time=40.0):
        """
        Find a minimum-ish path cover using local search on a multigraph.
        
        Representation: We maintain for each vertex a list of incident edge-ids.
        Each edge-id corresponds to an original edge. We track the current
        path/cycle structure implicitly via a union-find-like structure on
        directed half-edges.
        
        Strategy: Start with each edge as its own component. Use local moves:
        1. Merge two paths if they share a vertex (connect at that vertex).
        2. Break a cycle into a path (reduces cycle count but doesn't reduce
           path count directly - actually cycles are fine, we just need to
           convert them).
        3. Reconnect: if two paths P1 and P2 have endpoints at the same vertex,
           we can merge them into one path.
        
        Actually, let's use a cleaner formulation:
        - Build a directed multigraph where each undirected edge becomes two
          directed half-edges.
        - A path cover corresponds to pairing up half-edges at each vertex
          such that the result is a set of vertex-disjoint paths and cycles.
        - Minimizing the number of paths is equivalent to maximizing the number
          of pairs at each vertex (since each pair reduces the component count).
        
        Better approach: Use the standard reduction.
        - For each vertex v with degree d(v), we need to pair up d(v) incident
          edge-ends. Each pairing corresponds to a "turn" at v.
        - The number of paths in the cover = (number of vertices with odd 
          contribution) / 2 + number_of_cycles... 
        
        Let me use a simpler, proven approach:
        1. Find a maximum matching in an auxiliary graph to pair edge-ends.
        2. The remaining unpaired edge-ends define the path endpoints.
        
        Even simpler and effective: Iterative improvement.
        Start with all edges separate. Repeatedly find two paths that can be
        merged (share a vertex) and merge them. This is essentially finding
        a maximum set of vertex-disjoint connections.
        
        Let me implement a practical local search:
        - Maintain a list of paths.
        - Build a vertex-to-paths map.
        - Find merges: if two paths share a vertex, they can be merged into one.
        - Use a priority queue or random sampling to find merges.
        """
        rng = Random(42)
        adj = defaultdict(list)  # vertex -> list of (neighbor, edge_id)
        edge_map = {}  # edge_id -> (u, v)
        for eid, (u, v) in enumerate(edges):
            adj[u].append((v, eid))
            adj[v].append((u, eid))
            edge_map[eid] = (u, v)
        
        # State: list of paths, each path is a list of vertices
        # Start: each edge is a path of length 1
        paths = [[u, v] for (u, v) in edge_map.values()]
        
        # Build index: vertex -> set of path indices containing it
        # and position of vertex in path
        v_to_paths = defaultdict(list)  # v -> [(path_idx, pos)]
        for pi, path in enumerate(paths):
            for pos, v in enumerate(path):
                v_to_paths[v].append((pi, pos))
        
        def try_merge(pi1, pi2):
            """Try to merge paths pi1 and pi2 if they share a vertex."""
            path1 = paths[pi1]
            path2 = paths[pi2]
            # Find common vertex
            set1 = set(path1)
            common = None
            for v in path2:
                if v in set1:
                    common = v
                    break
            if common is None:
                return False
            
            # Find positions
            pos1 = path1.index(common)
            pos2 = path2.index(common)
            
            # We can merge in different ways depending on positions
            # The merged path must be simple.
            # Option: if common is at an endpoint of both, easy merge.
            # If common is in the middle, we need to reverse one part.
            
            # Let's try all valid merge options
            candidates = []
            
            # Option 1: reverse path2 if needed, concatenate at common
            # This is complex. Let's use a simpler heuristic:
            # Only merge if common is at an endpoint of at least one path.
            
            p1_endpoints = (pos1 == 0, pos1 == len(path1) - 1)
            p2_endpoints = (pos2 == 0, pos2 == len(path2) - 1)
            
            if p1_endpoints[0] and p2_endpoints[0]:
                # Both at start
                new_path = path1 + path2[1:]
                if len(set(new_path)) == len(new_path):
                    candidates.append(new_path)
            elif p1_endpoints[1] and p2_endpoints[1]:
                # Both at end
                new_path = path1 + path2[1:]
                if len(set(new_path)) == len(new_path):
                    candidates.append(new_path)
            elif p1_endpoints[0] and p2_endpoints[1]:
                # path1 starts at common, path2 ends at common
                new_path = path1 + path2[:-1]
                if len(set(new_path)) == len(new_path):
                    candidates.append(new_path)
            elif p1_endpoints[1] and p2_endpoints[0]:
                # path1 ends at common, path2 starts at common
                new_path = path1 + path2[:-1]
                if len(set(new_path)) == len(new_path):
                    candidates.append(new_path)
            
            if not candidates:
                return False
            
            # Pick the first valid candidate
            new_path = candidates[0]
            
            # Update state
            # Remove old paths
            for pi in (pi1, pi2):
                for v in paths[pi]:
                    v_to_paths[v] = [(p, pos) for (p, pos) in v_to_paths[v] if p != pi]
            paths[pi1] = new_path
            paths[pi2] = None  # mark as deleted
            for pos, v in enumerate(new_path):
                v_to_paths[v].append((pi1, pos))
            
            return True
        
        # Local search: repeatedly find and apply merges
        improved = True
        while improved and time.time() < start_time + max_time:
            improved = False
            # Build list of path indices that are still active
            active_paths = [i for i, p in enumerate(paths) if p is not None]
            
            # For each vertex, get the active paths containing it
            # Try to find a pair to merge
            rng.shuffle(active_paths)
            
            # Build vertex -> active paths map
            v_active = defaultdict(list)
            for pi in active_paths:
                for v in paths[pi]:
                    v_active[v].append(pi)
            
            # Try merges at each vertex
            for v in list(v_active.keys()):
                if time.time() >= start_time + max_time:
                    break
                plist = v_active[v]
                if len(plist) < 2:
                    continue
                # Try pairs
                for i in range(len(plist)):
                    for j in range(i + 1, len(plist)):
                        pi1, pi2 = plist[i], plist[j]
                        if paths[pi1] is None or paths[pi2] is None:
                            continue
                        if try_merge(pi1, pi2):
                            improved = True
                            # Rebuild v_active for this vertex
                            v_active[v] = [p for p in v_active[v] if paths[p] is not None]
                            break
                    if improved:
                        break
            
            # Also try random pairs of paths that share a vertex
            if not improved:
                for _ in range(100):
                    if time.time() >= start_time + max_time:
                        break
                    pi1 = rng.choice(active_paths)
                    pi2 = rng.choice(active_paths)
                    if pi1 == pi2 or paths[pi1] is None or paths[pi2] is None:
                        continue
                    set1 = set(paths[pi1])
                    if not set1.intersection(set(paths[pi2])):
                        continue
                    if try_merge(pi1, pi2):
                        improved = True
                        break
        
        # Clean up: remove None paths
        final_paths = [p for p in paths if p is not None]
        
        # Verify edge coverage and edge-disjointness
        # Each edge should appear exactly once
        edge_count = defaultdict(int)
        for path in final_paths:
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                edge_count[e] += 1
        
        # Check all edges covered exactly once
        for e in edges:
            if edge_count[tuple(e)] != 1:
                # Fallback: return individual edges
                return [[u, v] for (u, v) in edges]
        
        return final_paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose_graph(edges)
        results.append(paths)
        if time.time() > deadline:
            # Emergency: fill remaining with individual edges
            for spec2 in ((25, 0.3, 2), (30, 0.25, 3))[len(results)-1:]:
                if len(results) == 1 and spec2 == (25, 0.3, 2):
                    results.append([[u, v] for (u, v) in instance(*spec2)])
                elif len(results) == 2 and spec2 == (30, 0.25, 3):
                    results.append([[u, v] for (u, v) in instance(*spec2)])
            break

    return results
# EVOLVE-BLOCK-END
