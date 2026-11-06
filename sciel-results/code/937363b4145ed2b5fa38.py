import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    graphs = [instance(*spec) for spec in specs]

    def decompose_graph(edges, time_limit):
        if not edges:
            return []
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Enumerate all simple cycles up to max_len
        max_len = 20
        cycles = []
        
        # To avoid duplicates, we only start DFS from the smallest vertex in the cycle
        # and ensure the path is in increasing order? No, that misses cycles.
        # Standard way: 
        # For each start vertex s:
        #   DFS from s, only visiting vertices > s? No, that misses cycles where s is not min.
        # Correct way: 
        # For each start vertex s:
        #   DFS from s.
        #   Only close cycle if s is the smallest vertex in the cycle.
        
        def find_cycles(start, max_len):
            # Returns list of cycles (as lists of vertices) starting/ending at start
            # where start is the smallest vertex in the cycle
            results = []
            
            def dfs(current, path, visited):
                if time.time() > time_limit:
                    return
                if len(path) > max_len:
                    return
                
                # Check if we can close cycle
                if len(path) >= 3 and current in adj[start] and start < min(path[1:]):
                    # start is min of path[1:] implies start is min of whole cycle
                    # Also need to ensure we don't add duplicate cycles.
                    # Since we enforce start < all other nodes, each cycle is found exactly once
                    # when started from its minimum vertex.
                    # But we must ensure we don't add the same cycle in different orders?
                    # A simple cycle is defined by the set of edges.
                    # The path [s, v1, ..., vk, s] is unique for a given start and direction?
                    # No, [s, v1, v2, s] and [s, v2, v1, s] are the same cycle.
                    # To avoid this, we can enforce an order on the second step?
                    # Or just deduplicate later using frozenset of edges.
                    
                    # Let's just collect and deduplicate.
                    cycle_edges = frozenset()
                    for i in range(len(path)-1):
                        u, v = path[i], path[i+1]
                        cycle_edges.add(tuple(sorted((u, v))))
                    u, v = path[-1], path[0]
                    cycle_edges.add(tuple(sorted((u, v))))
                    
                    results.append((list(path), cycle_edges))
                    
                # Extend
                for neighbor in adj[current]:
                    if neighbor == start:
                        continue
                    if neighbor in visited:
                        continue
                    # Pruning: if neighbor < start, we can't use it because start must be min
                    if neighbor < start:
                        continue
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)
            
            visited = {start}
            path = [start]
            dfs(start, path, visited)
            return results

        all_cycles = []
        # Sort vertices to process smaller ones first?
        vertices = sorted(adj.keys())
        for s in vertices:
            if time.time() > time_limit:
                break
            found = find_cycles(s, max_len)
            all_cycles.extend(found)
            
        # Deduplicate cycles based on edge set
        unique_cycles = {}
        for path, edge_set in all_cycles:
            # Use frozenset of edges as key
            if edge_set not in unique_cycles:
                unique_cycles[edge_set] = (path, len(edge_set) - 1) # path, weight
                
        if not unique_cycles:
            return [list(e) for e in edges]
            
        # Prepare for MWIS
        # Nodes: indices in unique_cycles
        cycle_list = list(unique_cycles.values())
        edge_sets = list(unique_cycles.keys())
        weights = [w for _, w in cycle_list]
        n_cycles = len(cycle_list)
        
        # Build conflict graph
        # conflict[i] is a set of j such that i and j share an edge
        conflict = [set() for _ in range(n_cycles)]
        
        # Map edge to list of cycle indices
        edge_to_cycles = defaultdict(list)
        for i, e_set in enumerate(edge_sets):
            for e in e_set:
                edge_to_cycles[e].append(i)
                
        for e, indices in edge_to_cycles.items():
            for i in indices:
                for j in indices:
                    if i != j:
                        conflict[i].add(j)
                        
        # Solve MWIS
        # Greedy initialization
        selected = set()
        available = set(range(n_cycles))
        
        # Sort by weight descending
        sorted_indices = sorted(available, key=lambda x: weights[x], reverse=True)
        
        for i in sorted_indices:
            if i not in available:
                continue
            # Check if i conflicts with any selected
            if not conflict[i].intersection(selected):
                selected.add(i)
                # Remove i and all its conflicts from available
                to_remove = {i} | conflict[i]
                available -= to_remove
                
        # Local Search Improvement
        # We try to swap a selected cycle with an unselected one if it improves weight
        # And doesn't violate constraints.
        
        # To make this efficient, we can just iterate a few times.
        improved = True
        iterations = 0
        max_iterations = 50
        
        while improved and iterations < max_iterations and time.time() < time_limit:
            improved = False
            iterations += 1
            # Create a copy of selected to iterate
            selected_list = list(selected)
            for i in selected_list:
                # Try to remove i and add some j
                # We need to find a j in available (or previously removed due to i) 
                # that doesn't conflict with (selected - {i})
                
                # Actually, simpler:
                # Just try to add any unselected cycle if it fits? No, we want to improve.
                
                # Let's try swapping i with j.
                # j must be such that:
                # 1. j is not in selected
                # 2. j does not conflict with (selected - {i})
                # 3. weights[j] > weights[i]
                
                # To find such j, we can look at all j not in selected.
                # This is O(N^2) per swap, which might be slow if N is large.
                # But N is small (hundreds).
                
                best_j = -1
                best_gain = 0
                for j in range(n_cycles):
                    if j in selected:
                        continue
                    if weights[j] <= weights[i]:
                        continue
                    # Check conflict with selected - {i}
                    if not conflict[j].intersection(selected - {i}):
                        gain = weights[j] - weights[i]
                        if gain > best_gain:
                            best_gain = gain
                            best_j = j
                
                if best_j != -1:
                    selected.remove(i)
                    selected.add(best_j)
                    improved = True
                    # Update available? 
                    # We don't strictly need to maintain 'available' for this simple swap logic
                    # as long as we check 'selected' directly.
                    # But we must ensure we don't pick a j that conflicts with the NEW selected set.
                    # The check `not conflict[j].intersection(selected - {i})` ensures j is compatible
                    # with the rest of the set. Since we add j, the new set is (selected - {i}) U {j}.
                    # This is valid.
        
        # Construct final pieces
        pieces = []
        used_edges = set()
        
        for i in selected:
            path, w = cycle_list[i]
            pieces.append(path)
            used_edges.update(edge_sets[i])
            
        # Add remaining edges as singletons
        for e in edges:
            if e not in used_edges:
                pieces.append(list(e))
                
        return pieces

    results = []
    for i, edges in enumerate(graphs):
        time_per_graph = (deadline - time.time()) / (len(graphs) - i)
        time_limit = time.time() + max(1.0, time_per_graph * 0.9)
        
        best_pieces = decompose_graph(edges, time_limit)
        results.append(best_pieces)
        
        if time.time() > deadline - 2.0:
            for j in range(i + 1, len(graphs)):
                results.append([list(e) for e in graphs[j]])
            break
            
    return results

if __name__ == "__main__":
    res = construct()
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
