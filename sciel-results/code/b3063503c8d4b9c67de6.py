import time
import random
import sys
from collections import defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    graphs = [instance(*spec) for spec in specs]
    
    def find_all_cycles(adj, nodes):
        """
        Find all simple cycles in the graph.
        Returns a list of cycles, where each cycle is a list of nodes.
        To avoid duplicates, we only consider cycles where the smallest node is the first.
        """
        cycles = []
        n = len(nodes)
        
        # For each node, try to find cycles starting and ending at it,
        # but only if it's the smallest node in the cycle.
        for start in nodes:
            # DFS to find cycles starting from 'start'
            # We only allow visiting nodes > start, except for closing the cycle back to start.
            # This ensures 'start' is the smallest node in the cycle.
            
            stack = [(start, [start])]
            # To optimize, we can use a recursive DFS or iterative.
            # Let's use iterative to avoid recursion limits, though N is small.
            
            # Actually, a standard DFS for simple cycles:
            # path = [start]
            # visited = {start}
            # We want to find paths that start at start and end at a neighbor of start,
            # where all intermediate nodes are > start.
            
            def dfs(current, path, visited):
                for neighbor in adj[current]:
                    if neighbor == start:
                        if len(path) >= 3: # Cycle of length at least 3
                            cycles.append(list(path))
                    elif neighbor > start and neighbor not in visited:
                        visited.add(neighbor)
                        path.append(neighbor)
                        dfs(neighbor, path, visited)
                        path.pop()
                        visited.remove(neighbor)
            
            visited = {start}
            path = [start]
            dfs(start, path, visited)
            
        return cycles

    def decompose_graph_ilp(edges, time_limit):
        if not edges:
            return []
            
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        nodes = sorted(adj.keys())
        
        # Find all cycles
        cycles = find_all_cycles(adj, nodes)
        
        # If no cycles, all edges are single edges
        if not cycles:
            return [list(e) for e in edges]
            
        # ILP Formulation
        # Variables: x_i for each cycle i
        # Maximize sum(len(cycle_i) * x_i)
        # Subject to: for each edge e, sum(x_i for i where e in cycle_i) <= 1
        
        m = len(cycles)
        if m == 0:
            return [list(e) for e in edges]
            
        # Objective coefficients: length of cycle
        c = [len(cyc) for cyc in cycles]
        
        # Constraints: for each edge, sum of cycles containing it <= 1
        # We need to map edges to constraints.
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        num_edges = len(edges)
        
        # A matrix A where A[j, i] = 1 if edge j is in cycle i
        # This might be large, but m is usually small.
        # Let's build the constraint matrix.
        A_rows = []
        for i, cyc in enumerate(cycles):
            # For each edge in the cycle, add 1 to the corresponding row
            # We are building columns for the matrix A in scipy.milp?
            # scipy.milp expects A in shape (num_constraints, num_variables)
            # So we need to build rows for each edge.
            pass
            
        # Let's build A directly
        A = [[0] * m for _ in range(num_edges)]
        for i, cyc in enumerate(cycles):
            for j in range(len(cyc) - 1):
                u, v = cyc[j], cyc[j+1]
                e = tuple(sorted((u, v)))
                if e in edge_to_idx:
                    A[edge_to_idx[e]][i] = 1
            u, v = cyc[-1], cyc[0]
            e = tuple(sorted((u, v)))
            if e in edge_to_idx:
                A[edge_to_idx[e]][i] = 1
                
        # Constraints: A @ x <= 1
        # Bounds: 0 <= x_i <= 1
        # Integrality: x_i is integer
        
        # Check time
        if time.time() > time_limit:
            # Fallback to greedy
            return decompose_graph_greedy(edges)
            
        try:
            res = milp(
                c=c,
                constraints=[LinearConstraint(A, ub=1)],
                integrality=[1] * m,
                bounds=Bounds(lb=0, ub=1)
            )
            
            if res.status != 0: # 0 is Optimal
                # Fallback
                return decompose_graph_greedy(edges)
                
            selected_cycles = []
            for i in range(m):
                if res.x[i] > 0.5:
                    selected_cycles.append(cycles[i])
                    
            # Determine remaining edges
            used_edges = set()
            for cyc in selected_cycles:
                for j in range(len(cyc) - 1):
                    u, v = cyc[j], cyc[j+1]
                    used_edges.add(tuple(sorted((u, v))))
                u, v = cyc[-1], cyc[0]
                used_edges.add(tuple(sorted((u, v))))
                
            remaining_edges = [e for e in edges if e not in used_edges]
            
            pieces = [list(cyc) for cyc in selected_cycles]
            pieces.extend([list(e) for e in remaining_edges])
            
            return pieces
            
        except Exception:
            return decompose_graph_greedy(edges)
            
    def decompose_graph_greedy(edges, time_limit):
        # Simple greedy fallback: find longest cycles repeatedly
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining_edges = set(edges)
        pieces = []
        
        while remaining_edges:
            if time.time() > time_limit:
                pieces.extend([list(e) for e in remaining_edges])
                break
                
            # Find a cycle
            cycle = None
            # Try to find a long cycle
            # Heuristic: pick a vertex with high degree, do DFS
            candidates = sorted(adj.keys(), key=lambda x: len(adj[x]), reverse=True)
            
            for start in candidates:
                if len(adj[start]) < 2:
                    continue
                # DFS
                path = [start]
                visited = {start}
                
                def dfs(current):
                    nonlocal cycle
                    if cycle:
                        return
                    for neighbor in adj[current]:
                        if neighbor == start:
                            if len(path) >= 3:
                                cycle = list(path)
                                return
                        elif neighbor not in visited:
                            visited.add(neighbor)
                            path.append(neighbor)
                            dfs(neighbor)
                            path.pop()
                            visited.remove(neighbor)
                            if cycle:
                                return
                            
                dfs(start)
                if cycle:
                    break
                    
            if cycle:
                pieces.append(cycle)
                # Remove edges
                for i in range(len(cycle) - 1):
                    u, v = cycle[i], cycle[i+1]
                    e = tuple(sorted((u, v)))
                    remaining_edges.discard(e)
                    adj[u].discard(v)
                    adj[v].discard(u)
                u, v = cycle[-1], cycle[0]
                e = tuple(sorted((u, v)))
                remaining_edges.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)
            else:
                # No cycle, take an edge
                edge = next(iter(remaining_edges))
                pieces.append(list(edge))
                remaining_edges.discard(edge)
                u, v = edge
                adj[u].discard(v)
                adj[v].discard(u)
                
        return pieces

    results = []
    for i, edges in enumerate(graphs):
        # Allocate time
        time_per_graph = (deadline - time.time()) / (len(graphs) - i)
        time_limit = time.time() + max(1.0, time_per_graph * 0.9)
        
        # Try ILP
        pieces = decompose_graph_ilp(edges, time_limit)
        results.append(pieces)
        
        if time.time() > deadline - 2.0:
            for j in range(i + 1, len(graphs)):
                results.append([list(e) for e in graphs[j]])
            break
            
    return results

if __name__ == "__main__":
    res = construct()
    # Quick sanity check
    for i, spec in enumerate([(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]):
        n, p, seed = spec
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        edge_set = set(edges)
        
        used_edges = set()
        valid = True
        for piece in res[i]:
            if len(piece) == 2:
                u, v = piece
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            elif len(piece) >= 3:
                if len(set(piece)) != len(piece):
                    valid = False
                    break
                for k in range(len(piece) - 1):
                    u, v = piece[k], piece[k+1]
                    e = tuple(sorted((u, v)))
                    if e not in edge_set or e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
                u, v = piece[-1], piece[0]
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            else:
                valid = False
                break
                
        if valid and used_edges != edge_set:
            valid = False
            
        print(f"Graph {i}: Valid={valid}, Pieces={len(res[i])}")
