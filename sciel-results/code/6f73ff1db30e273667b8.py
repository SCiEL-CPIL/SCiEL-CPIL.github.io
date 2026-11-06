import time
from collections import defaultdict
from random import Random
import numpy as np
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def maximum_matching_decompose(edges):
        """
        Model path decomposition as maximum matching on bipartite graph.
        Left nodes: directed edge-stubs (u->v) for each undirected edge {u,v}
        Right nodes: directed edge-stubs (v->u) for each undirected edge {u,v}
        An edge in the bipartite graph connects (u->v) on left to (w->x) on right
        if v == w (they share a vertex, allowing path continuation).
        
        The maximum matching M gives us the minimum number of paths = |E| - |M|.
        We recover paths by following the matched edges.
        """
        if not edges:
            return []
        
        E = len(edges)
        # Create directed edge stubs
        # For each undirected edge (u,v), create two directed stubs: u->v and v->u
        # Left side: all directed stubs where we're "leaving" a vertex
        # Right side: all directed stubs where we're "arriving" at a vertex
        
        # Index edges
        edge_list = list(edges)
        edge_to_idx = {e: i for i, e in enumerate(edge_list)}
        
        # Build adjacency for the bipartite graph
        # Left node i corresponds to directed edge from edge_list[i][0] to edge_list[i][1]
        # Right node j corresponds to directed edge from edge_list[j][1] to edge_list[j][0]
        # Actually, let me think more carefully.
        
        # For a path decomposition, we want to chain edges together.
        # If edge e1 = (a,b) and edge e2 = (b,c) are in the same path, 
        # we "connect" the b-end of e1 to the b-end of e2.
        
        # Bipartite graph:
        # Left node: (edge_idx, end) where end=0 means the first vertex, end=1 means the second
        # Right node: (edge_idx, end) where end=0 means the first vertex, end=1 means the second
        # Edge in bipartite graph: left (i, end_i) connected to right (j, end_j) 
        #   if the vertex at that end is the same, i.e., edge_list[i][end_i] == edge_list[j][end_j]
        #   AND i != j (can't connect an edge to itself)
        
        # This creates a 2E x 2E bipartite graph.
        
        # For efficiency, let's build the cost matrix.
        # We want to maximize the number of matches, so we minimize -1 for valid connections.
        
        # Actually, let's use a simpler formulation:
        # Left nodes: each directed half-edge (u, v, edge_idx) meaning "arriving at v from u via edge_idx"
        # Right nodes: each directed half-edge (u, v, edge_idx) meaning "leaving u toward v via edge_idx"
        # Connection: left (u, v, i) connects to right (v, w, j) if i != j
        # This means after traversing edge i to reach v, we can continue with edge j from v.
        
        # Number of left nodes = 2E (each edge has two endpoints)
        # Number of right nodes = 2E
        
        # Build the cost matrix for Hungarian algorithm
        # Cost = 0 if connection is valid, INF if not
        # We want to maximize matches, so valid connections have cost 0, invalid have large cost.
        
        # For 2E up to ~60 (30 vertices, ~150 edges max), 2E = 300, so 300x300 matrix.
        # That's manageable.
        
        # Let's index:
        # Left node (i, s) where i in [0, E), s in {0, 1}
        # This represents: we just traversed edge i, arriving at vertex edge_list[i][s]
        # Wait, that's not right either.
        
        # Let me redefine:
        # Left node (i, s): we are at vertex edge_list[i][1-s] and want to traverse edge i in direction s
        #   Actually, let's say left node (i, s) means: "the s-th endpoint of edge i is the START of this edge in the path"
        #   So if s=0, the edge goes from edge_list[i][0] to edge_list[i][1]
        #   If s=1, the edge goes from edge_list[i][1] to edge_list[i][0]
        
        # Right node (j, t): "the t-th endpoint of edge j is the END of this edge in the path"
        #   So if t=0, the edge arrives at edge_list[j][0]
        #   If t=1, the edge arrives at edge_list[j][1]
        
        # Connection: left (i, s) can connect to right (j, t) if:
        #   The end vertex of edge i (when oriented with s as start) equals the start vertex of edge j (when oriented with t as end)
        #   End vertex of edge i with start=s: edge_list[i][1-s]
        #   Start vertex of edge j with end=t: edge_list[j][1-t]
        #   So: edge_list[i][1-s] == edge_list[j][1-t] AND i != j
        
        # This is getting complicated. Let me use a cleaner formulation.
        
        # Cleaner: 
        # Each undirected edge e=(u,v) can be traversed in two directions: u->v or v->u.
        # Left node: (e, dir) where dir=0 means u->v, dir=1 means v->u. This is a "tail" at the start vertex.
        # Right node: (e, dir) where dir=0 means u->v, dir=1 means v->u. This is a "head" at the end vertex.
        # 
        # For left (e1, d1): start vertex = e1[d1], end vertex = e1[1-d1]
        # For right (e2, d2): start vertex = e2[d2], end vertex = e2[1-d2]
        # 
        # We can chain left (e1, d1) to right (e2, d2) if end_vertex(e1,d1) == start_vertex(e2,d2) and e1 != e2
        # i.e., e1[1-d1] == e2[d2] and e1 != e2
        
        # Total left nodes = 2E, total right nodes = 2E
        # Build 2E x 2E cost matrix
        
        n_left = 2 * E
        n_right = 2 * E
        
        # Index: left node index = e * 2 + d, right node index = e * 2 + d
        INF = 1e9
        cost = np.full((n_left, n_right), INF, dtype=float)
        
        for e1 in range(E):
            for d1 in range(2):
                left_idx = e1 * 2 + d1
                end_v1 = edge_list[e1][1 - d1]  # end vertex of edge e1 when traversed in direction d1
                
                for e2 in range(E):
                    if e1 == e2:
                        continue
                    for d2 in range(2):
                        right_idx = e2 * 2 + d2
                        start_v2 = edge_list[e2][d2]  # start vertex of edge e2 when traversed in direction d2
                        
                        if end_v1 == start_v2:
                            cost[left_idx][right_idx] = 0.0
        
        # Solve assignment problem to maximize number of zero-cost assignments
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Count how many assignments have cost 0 (valid connections)
        valid_matches = 0
        match_map = {}  # left_idx -> right_idx
        for r, c in zip(row_ind, col_ind):
            if cost[r][c] < INF / 2:
                valid_matches += 1
                match_map[r] = c
        
        # Number of paths = E - valid_matches
        # Recover the paths
        
        # Build the directed edges for each match
        # For each left node (e1, d1) matched to right node (e2, d2):
        # This means edge e1 is followed by edge e2 in a path
        # Edge e1 is traversed from e1[d1] to e1[1-d1]
        # Edge e2 is traversed from e2[d2] to e2[1-d2]
        # And e1[1-d1] == e2[d2]
        
        # Each edge appears exactly once as a left node and once as a right node.
        # But in the matching, each left node is matched to at most one right node,
        # and each right node is matched to at most one left node.
        
        # To recover paths:
        # - A path starts at a left node that is NOT matched (i.e., no edge follows it)
        #   Wait, that's wrong. A path starts at a left node where the edge is the FIRST edge.
        #   Actually, let me think again.
        
        # The matching tells us which edge follows which edge.
        # If left (e1, d1) is matched to right (e2, d2), then in some path, e1 is immediately followed by e2.
        # 
        # To find paths:
        # 1. For each edge e, determine its direction in the path:
        #    - If left (e, d) is unmatched (no successor), then e is the LAST edge in its path, traversed in direction d.
        #    - If right (e, d) is unmatched (no predecessor), then e is the FIRST edge in its path, traversed in direction d.
        #    - If both are matched, e is in the middle.
        #
        # Actually, each edge has exactly one direction in the path. The direction is determined by:
        # - If the edge is a start: the right node (e, d) is unmatched, so d is the direction.
        # - If the edge is an end: the left node (e, d) is unmatched, so d is the direction.
        # - If the edge is in the middle: both left and right are matched. The direction must be consistent.
        #   Left (e, d1) matched means e is traversed with start at e[d1].
        #   Right (e, d2) matched means e is traversed with end at e[d2].
        #   For consistency, we need d1 = d2 (same direction).
        
        # Let's determine the direction for each edge:
        # For edge e, possible directions are 0 and 1.
        # - If left (e, 0) is in match_map, then e could be traversed in direction 0.
        # - If left (e, 1) is in match_map, then e could be traversed in direction 1.
        # - If right (e, 0) is NOT in the values of match_map, then e is a start with direction 0.
        # - If right (e, 1) is NOT in the values of match_map, then e is a start with direction 1.
        
        # Let me build:
        # successor[e, d] = (e2, d2) if left (e,d) is matched to right (e2,d2), else None
        # predecessor[e, d] = (e1, d1) if right (e,d) is matched from left (e1,d1), else None
        
        successor = {}  # (e, d) -> (e2, d2)
        predecessor = {}  # (e, d) -> (e1, d1)
        
        for left_idx, right_idx in match_map.items():
            e1 = left_idx // 2
            d1 = left_idx % 2
            e2 = right_idx // 2
            d2 = right_idx % 2
            successor[(e1, d1)] = (e2, d2)
            predecessor[(e2, d2)] = (e1, d1)
        
        # For each edge e, determine its direction:
        # Check if (e, 0) has a predecessor and (e, 1) has a predecessor
        # The valid direction is the one that is consistent.
        
        edge_direction = {}  # e -> d
        for e in range(E):
            # Try direction 0
            pred0 = predecessor.get((e, 0), None)
            succ0 = successor.get((e, 0), None)
            # Try direction 1
            pred1 = predecessor.get((e, 1), None)
            succ1 = successor.get((e, 1), None)
            
            # The edge must have a consistent direction.
            # If (e,0) has a predecessor, the edge is entered at vertex e[1] (since direction 0 means start at e[0], end at e[1])
            # Wait, I need to be more careful.
            
            # If edge e is traversed in direction d:
            #   Start vertex: e[d]
            #   End vertex: e[1-d]
            #   Predecessor: some edge that ends at e[d]
            #   Successor: some edge that starts at e[1-d]
            
            # So if (e, d) has a predecessor, that's fine.
            # If (e, d) has a successor, that's fine.
            # The direction is valid if it's consistent with all connections.
            
            # Since each edge can only be in one direction in the path decomposition,
            # and the matching is a valid assignment, the direction should be uniquely determined.
            
            # If (e, 0) is in successor keys, then e is traversed in direction 0.
            # If (e, 1) is in successor keys, then e is traversed in direction 1.
            # Similarly for predecessor.
            
            in_succ_0 = (e, 0) in successor
            in_succ_1 = (e, 1) in successor
            in_pred_0 = (e, 0) in predecessor
            in_pred_1 = (e, 1) in predecessor
            
            if in_succ_0 or in_pred_0:
                edge_direction[e] = 0
            elif in_succ_1 or in_pred_1:
                edge_direction[e] = 1
            else:
                # Isolated edge, can go either way
                edge_direction[e] = 0
        
        # Now find all paths:
        # A path starts at an edge (e, d) where predecessor is None.
        paths = []
        visited_edges = set()
        
        for e in range(E):
            d = edge_direction[e]
            if (e, d) not in predecessor and e not in visited_edges:
                # Start a new path
                path = []
                current_e = e
                current_d = d
                while current_e is not None and current_e not in visited_edges:
                    visited_edges.add(current_e)
                    # Add the two vertices of this edge
                    u = edge_list[current_e][current_d]
                    v = edge_list[current_e][1 - current_d]
                    if not path:
                        path.append(u)
                    path.append(v)
                    # Get successor
                    next_info = successor.get((current_e, current_d), None)
                    if next_info:
                        current_e, current_d = next_info
                    else:
                        current_e = None
                
                if path:
                    paths.append(path)
        
        # Check if we covered all edges
        if len(visited_edges) != E:
            # Some edges weren't covered, which shouldn't happen
            # Fallback: add them as single-edge paths
            for e in range(E):
                if e not in visited_edges:
                    paths.append(list(edge_list[e]))
        
        return paths

    def greedy_decompose(edges, rng):
        """Greedy path decomposition as fallback."""
        if not edges:
            return []
            
        current_adj = defaultdict(list)
        for u, v in edges:
            current_adj[u].append(v)
            current_adj[v].append(u)
            
        remaining_edges = set(edges)
        paths = []
        
        while remaining_edges:
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
                
            if not active_vertices:
                break
                
            start = rng.choice(list(active_vertices))
            path = [start]
            current = start
            visited = {start}
            
            while True:
                avail = [nb for nb in current_adj[current] 
                         if tuple(sorted((current, nb))) in remaining_edges 
                         and nb not in visited]
                
                if not avail:
                    break
                
                avail.sort(key=lambda x: len([e for e in remaining_edges if x in e]), reverse=True)
                
                if len(avail) > 1 and rng.random() < 0.2:
                    next_v = rng.choice(avail[:3])
                else:
                    next_v = avail[0]
                
                edge = tuple(sorted((current, next_v)))
                remaining_edges.remove(edge)
                current_adj[current].remove(next_v)
                current_adj[next_v].remove(current)
                
                path.append(next_v)
                visited.add(next_v)
                current = next_v
                
            paths.append(path)
            
        return paths

    def decompose_graph(edges):
        if not edges:
            return []
        
        # Try maximum matching approach
        try:
            paths = maximum_matching_decompose(edges)
            # Validate
            if _validate_paths(paths, edges):
                return paths
        except Exception:
            pass
        
        # Fallback to greedy
        rng = Random(42)
        best_paths = None
        best_count = float('inf')
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            paths = greedy_decompose(edges, rng)
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = paths
        
        if best_paths is None:
            return [list(e) for e in edges]
        
        return best_paths

    def _validate_paths(paths, edges):
        """Validate that paths are simple, use graph edges, are edge-disjoint, and cover all edges."""
        edge_set = set(edges)
        used_edges = set()
        
        for path in paths:
            if len(path) < 1:
                return False
            # Check simplicity
            if len(set(path)) != len(path):
                return False
            # Check edges exist and are unused
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        # Check all edges are covered
        if used_edges != edge_set:
            return False
        
        return True

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for spec in specs:
        edges = instance(*spec)
        paths = decompose_graph(edges)
        results.append(paths)
        
    return results
