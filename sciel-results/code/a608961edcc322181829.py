"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque

def construct():
    """
    Decompose three fixed graphs into minimum number of edge-disjoint paths covering all edges.
    Uses reduction to min-cost perfect matching on odd-degree vertices + Eulerian circuit.
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def min_path_decomposition(n, edge_list):
        if not edge_list:
            return []
        
        # Build adjacency list and degree count
        adj = defaultdict(list)
        deg = [0] * n
        for u, v in edge_list:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
            
        # Identify odd degree vertices
        odd_vertices = [i for i in range(n) if deg[i] % 2 == 1]
        
        # Helper to find Eulerian circuit in a multigraph represented by adjacency list
        # We need to handle edge consumption.
        def find_eulerian_circuit(start, adj_list, n):
            # Hierholzer's algorithm
            # adj_list is a dict of lists of (neighbor, edge_id)
            # We modify adj_list in place to remove edges
            stack = [start]
            circuit = []
            while stack:
                v = stack[-1]
                if adj_list[v]:
                    u, eid = adj_list[v].pop()
                    stack.append(u)
                else:
                    circuit.append(stack.pop())
            return circuit[::-1]
        
        # If no odd vertices, graph is Eulerian
        if len(odd_vertices) == 0:
            # Build edge-indexed adjacency
            # We need to distinguish parallel edges if any, but simple graph here.
            # However, we might add edges in augmentation, so use edge IDs.
            edge_ids = {}
            eid_counter = 0
            adj_list = defaultdict(list)
            for u, v in edge_list:
                edge_ids[(u, v)] = eid_counter
                edge_ids[(v, u)] = eid_counter
                adj_list[u].append((v, eid_counter))
                adj_list[v].append((u, eid_counter))
                eid_counter += 1
                
            # Find start node with non-zero degree
            start = next((i for i in range(n) if deg[i] > 0), 0)
            circuit = find_eulerian_circuit(start, adj_list, n)
            
            # Circuit is a list of vertices. It's a closed loop: v0...vk=v0
            # We need to convert to a simple path.
            # If the circuit is simple (all vertices unique except start=end), we can just return it as a path (dropping the last vertex).
            # If not, we have a problem.
            # For now, assume it's simple or try to split.
            # A circuit v0...vk is a path v0...v_{k-1} if we drop vk.
            # But this leaves the edge (v_{k-1}, v0) uncovered? No, the path uses edges (v0,v1)...(v_{k-2}, v_{k-1}).
            # The edge (v_{k-1}, v0) is NOT used in the path v0...v_{k-1}.
            # So we need to cover that edge separately.
            # Path 1: v0...v_{k-1}
            # Path 2: v_{k-1}...v0 (which is just the edge)
            
            # Check if circuit is simple
            vertices_in_circuit = circuit[:-1] # exclude last which is same as first
            if len(set(vertices_in_circuit)) == len(vertices_in_circuit):
                # It is a simple cycle
                # Split into two paths
                path1 = vertices_in_circuit
                path2 = [vertices_in_circuit[-1], vertices_in_circuit[0]]
                return [path1, path2]
            else:
                # Not simple. Fallback to trivial
                return [[list(e)] for e in edge_list]

        else:
            # Odd vertices exist
            # 1. Compute distances between all odd vertices
            # Use Floyd-Warshall or Dijkstra. N is small, Floyd-Warshall is easy.
            INF = 10**9
            dist = [[INF]*n for _ in range(n)]
            for i in range(n):
                dist[i][i] = 0
            for u, v in edge_list:
                dist[u][v] = 1
                dist[v][u] = 1
                
            for k in range(n):
                for i in range(n):
                    if dist[i][k] == INF: continue
                    for j in range(n):
                        if dist[k][j] == INF: continue
                        if dist[i][j] > dist[i][k] + dist[k][j]:
                            dist[i][j] = dist[i][k] + dist[k][j]
                        
            # 2. Min Cost Perfect Matching on odd vertices
            m = len(odd_vertices)
            # Build cost matrix
            cost = [[0]*m for _ in range(m)]
            for i in range(m):
                for j in range(m):
                    if i == j:
                        cost[i][j] = 0
                    else:
                        cost[i][j] = dist[odd_vertices[i]][odd_vertices[j]]
                        
            # Use linear_sum_assignment
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost)
            
            # Pairs of odd vertices to connect
            added_edges = []
            for i, j in zip(row_ind, col_ind):
                u = odd_vertices[i]
                v = odd_vertices[j]
                if u != v:
                    # Add edge (u,v) to augmentation
                    added_edges.append((u, v))
                    
            # 3. Build augmented graph
            aug_edges = list(edge_list) + added_edges
            adj_list = defaultdict(list)
            deg_aug = [0] * n
            eid_counter = 0
            edge_map = {} # eid -> (u,v)
            for u, v in aug_edges:
                edge_map[eid_counter] = (u, v)
                adj_list[u].append((v, eid_counter))
                adj_list[v].append((u, eid_counter))
                deg_aug[u] += 1
                deg_aug[v] += 1
                eid_counter += 1
                
            # 4. Find Eulerian circuit in augmented graph
            start = next((i for i in range(n) if deg_aug[i] > 0), 0)
            circuit = find_eulerian_circuit(start, adj_list, n)
            
            # 5. Extract paths by removing added edges
            # The circuit is a sequence of vertices.
            # We need to identify which edges in the circuit are "added".
            # We can track edge IDs.
            # But find_eulerian_circuit didn't return edge IDs.
            # Let's modify find_eulerian_circuit to return edges or track them.
            
            # Re-implement circuit finding to track edges
            def find_eulerian_circuit_with_edges(start, adj_list, n):
                stack = [(start, -1)] # (node, incoming_edge_id)
                circuit_nodes = []
                circuit_edges = []
                while stack:
                    v, in_e = stack[-1]
                    if adj_list[v]:
                        u, eid = adj_list[v].pop()
                        stack.append((u, eid))
                    else:
                        node, e = stack.pop()
                        circuit_nodes.append(node)
                        if e != -1:
                            circuit_edges.append(e)
                return circuit_nodes[::-1], circuit_edges[::-1]
                
            # Re-run with edge tracking
            # Need to reset adj_list because it was modified?
            # Yes, we need a fresh adj_list.
            adj_list2 = defaultdict(list)
            for u, v in aug_edges:
                adj_list2[u].append((v, edge_map[(u,v)] if (u,v) in edge_map else -1)) # This logic is messy
                # Let's just rebuild properly
                pass
            
            # Simpler: Just check if the resulting paths are simple.
            # If we split the circuit at added edges, we get paths.
            # The added edges are not in the original graph, so they act as "breaks".
            # We need to know which edges in the circuit are added.
            # Let's identify added edges by their endpoints.
            # Added edges are exactly the pairs in `added_edges`.
            # In the circuit, if we traverse an edge (u,v) that is in `added_edges`, it's a break.
            
            # Let's reconstruct the circuit with edge info.
            adj_list3 = defaultdict(list)
            eid = 0
            edge_info = {}
            for u, v in aug_edges:
                edge_info[eid] = (u, v)
                adj_list3[u].append((v, eid))
                adj_list3[v].append((u, eid))
                eid += 1
                
            start = next((i for i in range(n) if deg_aug[i] > 0), 0)
            stack = [start]
            circuit = []
            while stack:
                v = stack[-1]
                if adj_list3[v]:
                    u, eid = adj_list3[v].pop()
                    stack.append(u)
                else:
                    circuit.append(stack.pop())
            circuit = circuit[::-1]
            
            # Now circuit is a list of vertices.
            # We need to split it based on added edges.
            # An edge (circuit[i], circuit[i+1]) is an added edge if it's in `added_edges` (as a set of tuples).
            added_set = set(tuple(sorted(e)) for e in added_edges)
            
            paths = []
            current_path = [circuit[0]]
            for i in range(len(circuit) - 1):
                u = circuit[i]
                v = circuit[i+1]
                e = tuple(sorted((u, v)))
                current_path.append(v)
                if e in added_set:
                    # Break here
                    if len(current_path) > 1:
                        paths.append(current_path)
                    current_path = [v]
                    
            # Close the loop? The circuit is closed, so circuit[0] == circuit[-1].
            # The last edge is (circuit[-2], circuit[-1]=circuit[0]).
            # If it's an added edge, we break.
            # If not, we continue?
            # Actually, the loop closes.
            # If the last edge is NOT added, the path continues to the start.
            # But we started a path at circuit[0].
            # If the last edge is not added, then the last segment connects back to circuit[0].
            # This forms a cycle.
            # If the last edge IS added, the path ends at circuit[0].
            
            # Let's handle the closing edge explicitly.
            # The loop above processes edges (0,1), (1,2), ..., (k-2, k-1).
            # The edge (k-1, 0) is not processed.
            u = circuit[-1]
            v = circuit[0]
            e = tuple(sorted((u, v)))
            
            if e in added_set:
                # The path ends at v (which is circuit[0])
                current_path.append(v)
                if len(current_path) > 1:
                    paths.append(current_path)
            else:
                # The path connects back to start.
                # This implies the last segment is a cycle.
                # We need to break this cycle.
                # We can just add the edge to the current path?
                # current_path currently ends at circuit[-1].
                # If we add v (circuit[0]), we close the cycle.
                # But we need simple paths.
                # If current_path + [v] is a simple cycle, we split it into two paths.
                # Or we can just treat it as a path that goes from start to end?
                # Actually, if the closing edge is NOT added, it means the "break" didn't happen at the end.
                # This means the entire circuit was one segment?
                # No, `added_edges` must be present in the circuit.
                # If the closing edge is not added, then the path `current_path` does not close.
                # Wait, if no break happened at the end, then `current_path` is just the vertices from the last break to the end.
                # But the circuit is closed.
                # So the last edge connects `circuit[-1]` to `circuit[0]`.
                # If this edge is NOT added, then the path continues.
                # This means the "start" of the current segment is not `circuit[0]`?
                # This logic is getting messy.
                
                # Fallback: If any path is not simple, return trivial.
                pass
            
            # Check simplicity of generated paths
            for p in paths:
                if len(set(p)) != len(p):
                    return [[list(e)] for e in edge_list]
                    
            # If we didn't break at the end, we might have missed the last edge or formed a cycle.
            # To be safe, if we don't have exactly len(added_edges)+1 paths (or similar), fallback.
            # Actually, number of paths should be len(added_edges) if we split at each added edge?
            # No, if we have K added edges, we split the circuit into K paths?
            # Yes, K cuts in a cycle produce K paths.
            
            if len(paths) != len(added_edges):
                 # Something went wrong with the splitting logic
                 return [[list(e)] for e in edge_list]
                 
            return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        n, p, seed = spec
        edge_list = instance(n, p, seed)
        paths = min_path_decomposition(n, edge_list)
        results.append(paths)
        if time.time() > deadline:
            while len(results) < 3:
                spec2 = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results)]
                n2, p2, seed2 = spec2
                edge_list2 = instance(n2, p2, seed2)
                results.append([[list(e)] for e in edge_list2])
            break
    
    return results
# EVOLVE-BLOCK-END
