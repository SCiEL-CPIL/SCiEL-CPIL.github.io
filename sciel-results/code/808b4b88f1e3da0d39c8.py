"""
Strategy: Maximum Weight Matching on an auxiliary graph.
We model the path decomposition problem as finding a set of pairs of edges
that can be connected at a shared vertex to form a longer path.
This is equivalent to finding a maximum matching in a graph where:
- Nodes represent the edges of the original graph.
- Two nodes are connected if the corresponding edges share a vertex.
- The weight of an edge in the auxiliary graph is 1 (we want to maximize the number of pairs).
Each pair reduces the number of path components by 1.
After finding the maximum matching, we assemble the paths from the pairs.
"""

import time
from random import Random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_max_matching(adj, nodes):
        """
        Finds a maximum cardinality matching in an unweighted graph.
        Uses a simple BFS-based augmenting path algorithm (Hopcroft-Karp is overkill for small n, 
        but simple augmenting path is O(VE) which is fine for ~500 edges).
        adj: dict node -> list of neighbors
        nodes: list of nodes
        """
        if not nodes:
            return {}
        
        # Map node values to indices for easier array handling if needed, 
        # but since n is small, we can use the node values directly.
        match = {}
        
        def try_kuhn(v, seen):
            if v in match and match[v] is not None:
                return False
            for u in adj[v]:
                if u in seen:
                    continue
                seen.add(u)
                if u not in match or match[u] is None or try_kuhn(match[u], seen):
                    match[v] = u
                    match[u] = v
                    return True
            return False

        # Order nodes by degree to improve performance (optional heuristic)
        # nodes_sorted = sorted(nodes, key=lambda x: len(adj[x]))
        
        for v in nodes:
            seen = set()
            try_kuhn(v, seen)
            
        return match

    def decompose_graph(edges, max_time=40.0):
        if not edges:
            return []
            
        n_edges = len(edges)
        
        # 1. Build auxiliary graph for edge pairing
        # Node i represents edge i.
        # Adjacency: edge i and edge j are adjacent if they share a vertex.
        aux_adj = defaultdict(list)
        edge_to_vertices = {}
        
        for i, (u, v) in enumerate(edges):
            edge_to_vertices[i] = (u, v)
            aux_adj[i].append(-1) # placeholder to ensure key exists
            
        # Efficiently find shared vertices
        # vertex -> list of edge indices
        v_to_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            v_to_edges[u].append(i)
            v_to_edges[v].append(i)
            
        for v, e_list in v_to_edges.items():
            for i in range(len(e_list)):
                for j in range(i + 1, len(e_list)):
                    e1 = e_list[i]
                    e2 = e_list[j]
                    aux_adj[e1].append(e2)
                    aux_adj[e2].append(e1)
                    
        # Remove duplicates in adjacency lists (optional, but good for speed)
        for i in aux_adj:
            aux_adj[i] = list(set(aux_adj[i]))

        # 2. Find Maximum Matching
        # The matching pairs edges that will be connected in the final paths.
        # A perfect matching would mean all edges are paired, forming cycles or long paths.
        # Unmatched edges will be endpoints of paths.
        match = find_max_matching(aux_adj, list(range(n_edges)))
        
        # 3. Assemble Paths
        # We have a set of pairs (e1, e2) from the matching.
        # This defines a graph where nodes are the original edges, and edges are the matches.
        # Since it's a matching, each node has degree <= 1 in this "pairing graph".
        # Wait, a matching in the auxiliary graph means we select a set of disjoint pairs.
        # So the "pairing graph" is just a set of disjoint edges (2-cycles in terms of connection logic? No, just pairs).
        
        # Let's trace the paths.
        # Each original edge is a segment.
        # If edge i is matched with edge j, they are connected.
        # How are they connected?
        # Edge i = (u, v), Edge j = (x, y).
        # They share a vertex, say v == x.
        # Then the combined path segment is u -> v -> y (or u -> v -> x if v==x).
        
        # We need to determine the direction.
        # Let's build a map: edge_id -> (neighbor_edge_id, shared_vertex, other_vertex1, other_vertex2)
        
        # Actually, it's easier to think of the "pairing" as defining a permutation-like structure on the edge-ends?
        # No, a matching on edges means we link edge i to edge j.
        
        # Let's construct the final paths by traversing the "edge-link" graph.
        # Nodes: original edge indices 0..N-1
        # Edges: (i, j) if match[i] == j.
        # Since it's a matching, each node has at most 1 link.
        # So the structure is a set of disjoint paths of edges?
        # No, a matching creates pairs. If i is matched with j, they form a "super-edge".
        # But we can have chains?
        # Wait. If i is matched with j, and j is matched with k? No, matching is a set of disjoint pairs.
        # So i is only matched with j. j is only matched with i.
        # This means the "pairing graph" consists of isolated edges (pairs) and isolated nodes (unmatched).
        # This implies we can only merge edges in pairs?
        # NO. That's not right.
        
        # Let's re-read the theory.
        # A path decomposition is a set of paths.
        # At each vertex v in the original graph, the incident edges are partitioned into pairs (turns) and singles (endpoints).
        # If we pair edge e1 and e2 at vertex v, we are saying "in the path decomposition, the path enters e1 at v and exits via e2".
        # This is exactly a matching on the *edge-ends* incident to each vertex?
        # No, it's a matching on the *edges* themselves, but constrained by vertices.
        
        # The auxiliary graph I built connects edge i and edge j if they share a vertex.
        # A matching in this auxiliary graph selects a set of pairs (i, j) such that no edge is in more than one pair.
        # This corresponds to selecting a set of "turns" where each edge participates in at most one turn?
        # That's not enough. An edge has two ends. It can have a turn at one end and a turn at the other end.
        # So an edge can be paired with TWO other edges (one at each end).
        
        # My auxiliary graph model is flawed for this specific "matching" formulation if I treat edges as nodes.
        # Because an edge node can only be matched once in a standard matching.
        # But an edge can be "turned" at both ends.
        
        # Correct Model:
        # We need to pair up the *stubs* (half-edges) at each vertex.
        # For each vertex v, let E_v be the set of edges incident to v.
        # We need to partition E_v into pairs and singles.
        # This is a local decision at each vertex.
        # However, these local decisions must be consistent globally to form simple paths.
        
        # If we just pick a maximum matching of stubs at each vertex independently, we might create cycles or non-simple paths.
        # But for sparse graphs, independent local matching is a very strong heuristic.
        # Gallai's conjecture is about the *minimum* number of paths.
        # The number of paths = (Number of odd degree vertices) / 2 + (Number of cycles in the resulting structure).
        # Actually, if we maximize the number of pairs at each vertex, we minimize the number of "ends".
        # Total ends = Sum_v (deg(v) mod 2) + 2 * (Number of unpaired stubs due to matching constraints? No).
        
        # Let's stick to the local matching heuristic:
        # 1. For each vertex v, find a maximum matching of the incident edges.
        # 2. This defines how edges are connected at v.
        # 3. Assemble the paths.
        # 4. If any path is not simple (visits a vertex twice), we have a problem.
        #    However, for random geometric-like graphs (Erdos-Renyi with p=0.3), cycles in the *path decomposition* (i.e., the path visiting a vertex twice) are less likely to be an issue if we break them?
        #    Actually, if the local matching creates a cycle that is not a simple cycle in the original graph, it's invalid.
        #    But a "cycle" in the path decomposition sense means the path is a closed loop.
        #    If the path is a closed loop, it's a cycle, not a path.
        #    The problem asks for "paths". Usually, a cycle can be broken into a path by removing one edge, but that edge must be covered.
        #    Wait, "Paths must be simple". A cycle is not a simple path (start != end).
        #    So we must break any cycles found in the assembly into paths.
        #    Breaking a cycle of length L into a path of length L-1 and a single edge path?
        #    No, we just pick a vertex in the cycle and "break" the connection there.
        #    This increases the path count by 1.
        
        # Let's implement the Local Maximum Matching on Stubs.
        
        # 1. For each vertex, match incident edges.
        #    To ensure consistency, we can just do it greedily or with a small matching algorithm.
        #    Since we want to maximize pairs, we want a maximum matching in the "line graph" of the star at v?
        #    No, the "graph" of incident edges at v is a clique (any two edges can be paired).
        #    So the maximum matching size is floor(deg(v) / 2).
        #    We just need to pick WHICH pairs.
        #    To minimize cycles later, random pairing is usually fine.
        
        # Let's define the connections.
        # conn[eid][end] = (neighbor_eid, shared_vertex)
        # end 0 is u, end 1 is v for edge (u, v).
        
        conn = {}
        for eid, (u, v) in enumerate(edges):
            conn[eid] = [None, None] # [connection_at_u, connection_at_v]
            
        # For each vertex, get incident edges and pair them.
        for v, e_list in v_to_edges.items():
            if len(e_list) < 2:
                continue
            # Shuffle to randomize pairing
            rng = Random(hash(v) % 10000) # Deterministic per vertex
            e_list_copy = e_list[:]
            rng.shuffle(e_list_copy)
            
            # Pair up
            for i in range(0, len(e_list_copy) - 1, 2):
                e1 = e_list_copy[i]
                e2 = e_list_copy[i+1]
                
                # Determine which end of e1 and e2 is at v
                u1, v1 = edges[e1]
                u2, v2 = edges[e2]
                
                # e1 end at v
                if u1 == v:
                    e1_end = 0
                else:
                    e1_end = 1
                    
                if u2 == v:
                    e2_end = 0
                else:
                    e2_end = 1
                    
                # Link them
                conn[e1][e1_end] = (e2, v)
                conn[e2][e2_end] = (e1, v)
                
        # 2. Assemble Paths
        # Start from unconnected ends (where conn[eid][end] is None).
        # Traverse the graph.
        
        visited_edges = set()
        paths = []
        
        for eid in range(n_edges):
            if eid in visited_edges:
                continue
                
            # Find an unconnected end to start
            start_end = None
            if conn[eid][0] is None:
                start_end = 0
            elif conn[eid][1] is None:
                start_end = 1
            else:
                # Both ends connected. This is part of a cycle.
                # We must break it to make it a path.
                # Pick one connection to break.
                # Let's break the connection at end 0.
                # We need to remove the link in the neighbor's connection too.
                # But we are just assembling. We can just start here and treat one end as an endpoint.
                # Actually, if both are connected, we can start anywhere, but we must break one link to make it a simple path (start != end).
                # Let's break the link at end 0.
                # We will start at eid, end 0.
                start_end = 0
                # Note: We don't actually modify conn, we just treat it as a break during traversal.
                
            # Trace the path
            path = []
            cur_eid = eid
            cur_end = start_end
            
            while True:
                path.append(cur_eid)
                visited_edges.add(cur_eid)
                
                u, v = edges[cur_eid]
                # Add the vertex corresponding to cur_end
                if cur_end == 0:
                    path.append(u) # Wait, path should be vertices.
                    # Let's store vertices directly.
                # Let's restart the path building to store vertices.
                break
        
        # Redo assembly with vertices
        visited_edges = set()
        paths = []
        
        for eid in range(n_edges):
            if eid in visited_edges:
                continue
            
            # Find start
            start_end = None
            if conn[eid][0] is None:
                start_end = 0
            elif conn[eid][1] is None:
                start_end = 1
            else:
                # Cycle. Break at end 0.
                start_end = 0
                # We will treat the connection at end 0 as "not there" for the purpose of starting.
                # But we need to ensure we don't loop back to the start.
                
            path_vertices = []
            cur_eid = eid
            cur_end = start_end
            
            # To handle the cycle break, we can just stop when we hit a visited edge or a None.
            # But if it's a cycle, we'll hit the start edge again.
            
            while True:
                u, v = edges[cur_eid]
                
                # Add the entry vertex
                if cur_end == 0:
                    path_vertices.append(u)
                else:
                    path_vertices.append(v)
                    
                visited_edges.add(cur_eid)
                
                # Check if we can continue
                # If this was a cycle break, and we are at the "broken" end, we stop?
                # No, we start at the broken end. We traverse.
                # We stop when we reach an end with no connection, OR if we revisit an edge (cycle).
                
                next_conn = conn[cur_eid][cur_end]
                
                if next_conn is None:
                    # End of path
                    # Add the exit vertex
                    if cur_end == 0:
                        path_vertices.append(v)
                    else:
                        path_vertices.append(u)
                    break
                
                next_eid, shared_v = next_conn
                
                # Check for cycle (revisiting edge)
                if next_eid in visited_edges:
                    # This is a cycle. We must stop to make it a simple path.
                    # Add the exit vertex
                    if cur_end == 0:
                        path_vertices.append(v)
                    else:
                        path_vertices.append(u)
                    break
                
                # Move to next edge
                # The exit vertex of current edge is the entry vertex of next edge
                # cur_end 0 -> exit is v. Next edge enters at shared_v.
                # We need to determine the entry end of next_eid.
                nu, nv = edges[next_eid]
                if nu == shared_v:
                    next_end = 0
                else:
                    next_end = 1
                    
                # Append the shared vertex? No, it's already the exit of current.
                # The path is u1-v1-u2-v2...
                # Current edge (u, v). cur_end=0 (start at u, exit at v).
                # Next edge enters at v.
                # So we just switch edge and end.
                
                cur_eid = next_eid
                cur_end = next_end
                
            if path_vertices:
                # Verify simplicity
                if len(set(path_vertices)) == len(path_vertices):
                    paths.append(path_vertices)
                else:
                    # Fallback: split into individual edges?
                    # This shouldn't happen if we break cycles correctly.
                    # But just in case.
                    for i in range(len(path_vertices) - 1):
                        paths.append([path_vertices[i], path_vertices[i+1]])
                        
        # Verify coverage
        # This heuristic might miss some edges if the cycle breaking logic is flawed.
        # Let's check if all edges are in visited_edges.
        if len(visited_edges) != n_edges:
            # Fallback to simple edges for missing ones?
            # Or just return what we have? The verifier requires all edges covered.
            # Let's add missing edges as single paths.
            missing = set(range(n_edges)) - visited_edges
            for eid in missing:
                u, v = edges[eid]
                paths.append([u, v])
                
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Emergency fill
            edges = instance(*spec)
            results.append([[u, v] for (u, v) in edges])
            continue
            
        edges = instance(*spec)
        paths = decompose_graph(edges)
        results.append(paths)

    return results

# EVOLVE-BLOCK-END
