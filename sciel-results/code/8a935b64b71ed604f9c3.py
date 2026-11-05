import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin out of 90s
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    def is_valid_path(path):
        if not path or len(path) < 1:
            return False
        seen = set(path)
        if len(seen) != len(path):
            return False
        for i in range(len(path) - 1):
            if path[i + 1] not in adj[path[i]]:
                return False
        return True
    
    def greedy_path(start, strategy='min_degree'):
        path = [start]
        seen = {start}
        while True:
            candidates = adj[path[-1]] - seen
            if not candidates:
                break
            if strategy == 'min_degree':
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            elif strategy == 'max_degree':
                v = max(candidates, key=lambda w: (len(adj[w] - seen), -w))
            elif strategy == 'random':
                v = random.choice(list(candidates))
            else:
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.append(v)
            seen.add(v)
        return path
    
    def try_extend(path, adj, seen):
        """Try to extend path from both ends."""
        changed = True
        while changed:
            changed = False
            # Try extending from end
            if path:
                last = path[-1]
                candidates = adj[last] - seen
                if candidates:
                    v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                    path.append(v)
                    seen.add(v)
                    changed = True
            # Try extending from start
            if path:
                first = path[0]
                candidates = adj[first] - seen
                if candidates:
                    v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                    path.insert(0, v)
                    seen.add(v)
                    changed = True
        return path
    
    def local_improve(path, adj, max_iters=500):
        """Try to improve path by swapping segments or reordering."""
        best = list(path)
        best_len = len(best)
        seen = set(best)
        
        for _ in range(max_iters):
            improved = False
            # Try 2-opt style: reverse a segment
            for i in range(len(best) - 1):
                for j in range(i + 1, len(best)):
                    # Check if reversing best[i..j] is valid
                    valid = True
                    if i > 0 and best[j] not in adj[best[i-1]]:
                        valid = False
                    if valid and j < len(best) - 1 and best[i] not in adj[best[j+1]]:
                        valid = False
                    if valid:
                        new_path = best[:i] + best[i:j+1][::-1] + best[j+1:]
                        if len(new_path) > best_len:
                            best = new_path
                            best_len = len(best)
                            seen = set(best)
                            improved = True
                            break
                if improved:
                    break
            
            # Try to extend from both ends
            extended = try_extend(best, adj, set(best))
            if len(extended) > best_len:
                best = extended
                best_len = len(best)
                seen = set(best)
                improved = True
            
            if not improved:
                break
        
        return best
    
    def max_weight_matching_path(adj, n):
        """
        Formulate longest path as maximum weight matching on a transformed graph.
        
        We create a bipartite graph where left nodes represent "in" edges and 
        right nodes represent "out" edges. Each original vertex v becomes two nodes:
        - Left node v_in (represents entering v)
        - Right node v_out (represents leaving v)
        
        For each edge (u,v) in original graph, we create:
        - Edge from u_out to v_in with weight 1
        - Edge from v_out to u_in with weight 1
        
        A matching in this bipartite graph corresponds to a set of vertex-disjoint paths/cycles.
        We want to maximize the number of vertices covered, which equals the number of matched edges.
        
        However, a matching gives us paths/cycles, not necessarily a single path.
        We'll extract the longest path from the resulting structure.
        """
        # Build bipartite graph for matching
        # Left side: 0..n-1 represent "out" of vertex i
        # Right side: 0..n-1 represent "in" to vertex i
        
        # For maximum weight matching, we'll use a simple approach:
        # Since the graph is small (100 vertices), we can use a greedy or
        # augmenting path approach.
        
        # Actually, let's use a simpler formulation:
        # We want to find a maximum set of edges such that each vertex has degree <= 2
        # in the selected subgraph, and the subgraph consists of paths and cycles.
        
        # This is equivalent to finding a maximum 2-matching.
        # A 2-matching is a collection of vertex-disjoint paths and cycles.
        
        # For a 2-matching, we can use the following approach:
        # - Start with empty set of edges
        # - Repeatedly add edges that don't violate degree constraint
        # - Use augmenting paths to improve
        
        # Let's implement a simple 2-matching algorithm:
        
        degree = [0] * n  # degree of each vertex in current 2-matching
        selected_edges = set()  # set of (u,v) with u < v
        
        # First, greedily add edges
        all_edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    all_edges.append((u, v))
        
        # Sort by some heuristic - prefer edges that connect high-degree vertices
        all_edges.sort(key=lambda e: (len(adj[e[0]]) + len(adj[e[1]]), e[0], e[1]))
        
        # Greedy 2-matching
        for u, v in all_edges:
            if degree[u] < 2 and degree[v] < 2:
                selected_edges.add((u, v))
                degree[u] += 1
                degree[v] += 1
        
        # Now extract paths from the 2-matching
        # Build adjacency for the 2-matching
        match_adj = [set() for _ in range(n)]
        for u, v in selected_edges:
            match_adj[u].add(v)
            match_adj[v].add(u)
        
        # Find all paths and cycles
        visited = set()
        paths = []
        
        for start in range(n):
            if start in visited or not match_adj[start]:
                continue
            
            # Follow the path
            path = [start]
            visited.add(start)
            current = start
            
            while True:
                # Find next unvisited neighbor
                nexts = [v for v in match_adj[current] if v not in visited]
                if not nexts:
                    break
                current = nexts[0]
                path.append(current)
                visited.add(current)
            
            if len(path) > 1:
                paths.append(path)
        
        # Return the longest path
        if not paths:
            return [0] if n > 0 else []
        
        longest = max(paths, key=len)
        return longest
    
    def augmenting_path_improve(path, adj, n, max_iters=100):
        """
        Try to improve a path by finding augmenting structures.
        This is a local search that tries to insert unvisited vertices.
        """
        best = list(path)
        best_seen = set(best)
        best_len = len(best)
        rng_local = random.Random(123)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            unvisited = set(range(n)) - best_seen
            
            if not unvisited:
                break
            
            # Try to insert a random unvisited vertex
            v = rng_local.choice(list(unvisited))
            
            # Find all valid positions to insert v
            valid_positions = []
            for i in range(len(best) + 1):
                if i == 0:
                    if i < len(best) and v in adj[best[0]]:
                        valid_positions.append(i)
                elif i == len(best):
                    if i > 0 and v in adj[best[-1]]:
                        valid_positions.append(i)
                else:
                    if v in adj[best[i-1]] and v in adj[best[i]]:
                        valid_positions.append(i)
            
            if valid_positions:
                pos = rng_local.choice(valid_positions)
                new_path = best[:pos] + [v] + best[pos:]
                new_seen = best_seen | {v}
                if len(new_path) > best_len:
                    best = new_path
                    best_seen = new_seen
                    best_len = len(best)
                    improved = True
            
            # Try 2-opt reversal
            for i in range(len(best) - 1):
                for j in range(i + 1, len(best)):
                    valid = True
                    if i > 0 and best[j] not in adj[best[i-1]]:
                        valid = False
                    if valid and j < len(best) - 1 and best[i] not in adj[best[j+1]]:
                        valid = False
                    if valid:
                        new_path = best[:i] + best[i:j+1][::-1] + best[j+1:]
                        # After reversal, try to extend
                        new_seen = set(new_path)
                        extended = try_extend(new_path, adj, new_seen)
                        if len(extended) > best_len:
                            best = extended
                            best_seen = set(best)
                            best_len = len(best)
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # Try extending from both ends
                extended = try_extend(best, adj, set(best))
                if len(extended) > best_len:
                    best = extended
                    best_seen = set(best)
                    best_len = len(best)
                    improved = True
            
            if not improved:
                break
        
        return best
    
    # Phase 1: Generate many candidate paths using different strategies
    candidates = []
    
    # Greedy from every vertex with different strategies
    for start in range(n):
        for strategy in ['min_degree', 'max_degree']:
            path = greedy_path(start, strategy)
            if path:
                candidates.append(path)
    
    # Random restarts
    rng = random.Random(42)
    for _ in range(200):
        start = rng.randint(0, n - 1)
        path = greedy_path(start, 'random')
        if path:
            candidates.append(path)
    
    # Phase 2: Try maximum weight matching approach
    if time.time() < deadline - 10:
        matching_path = max_weight_matching_path(adj, n)
        if matching_path and len(matching_path) > 1:
            candidates.append(matching_path)
    
    # Phase 3: Local improvement on top candidates
    candidates.sort(key=lambda p: len(p), reverse=True)
    
    best_path = candidates[0] if candidates else [0]
    best_len = len(best_path)
    
    # Improve top 30 candidates
    for path in candidates[:30]:
        if len(path) <= best_len - 5:  # Skip if much shorter
            continue
        improved = augmenting_path_improve(path, adj, n, max_iters=100)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)
        if time.time() > deadline - 5:
            break
    
    # Phase 4: Beam search
    beam_width = 50
    beam = []
    for start in range(n):
        beam.append(([start], {start}))
    
    beam.sort(key=lambda x: len(x[0]), reverse=True)
    beam = beam[:beam_width]
    
    for step in range(25):  # Max path length we search for
        new_beam = []
        for path, seen in beam:
            last = path[-1]
            for v in adj[last] - seen:
                new_path = path + [v]
                new_seen = seen | {v}
                new_beam.append((new_path, new_seen))
        
        if not new_beam:
            break
        
        new_beam.sort(key=lambda x: len(x[0]), reverse=True)
        beam = new_beam[:beam_width]
        
        # Update best
        if beam:
            if len(beam[0][0]) > best_len:
                best_path = beam[0][0]
                best_len = len(best_path)
        
        if time.time() > deadline - 5:
            break
    
    # Phase 5: Simulated annealing on path
    def anneal_path(initial_path, adj, iters=5000):
        current = list(initial_path)
        current_seen = set(current)
        best = list(current)
        best_seen = set(current)
        T = 1.0
        cooling = 0.995
        
        for _ in range(iters):
            if time.time() > deadline - 2:
                break
            
            # Move: try to insert an unvisited vertex
            unvisited = set(range(n)) - current_seen
            if unvisited:
                # Pick a random unvisited vertex
                v = rng.choice(list(unvisited))
                # Find a position to insert
                valid_positions = []
                for i in range(len(current) + 1):
                    if i == 0:
                        if i < len(current) and v in adj[current[0]]:
                            valid_positions.append(i)
                    elif i == len(current):
                        if i > 0 and v in adj[current[-1]]:
                            valid_positions.append(i)
                    else:
                        if v in adj[current[i-1]] and v in adj[current[i]]:
                            valid_positions.append(i)
                
                if valid_positions:
                    pos = rng.choice(valid_positions)
                    new_path = current[:pos] + [v] + current[pos:]
                    new_seen = current_seen | {v}
                    delta = 1  # Length increases by 1
                    if delta > 0 or rng.random() < pow(2.718, delta / T):
                        current = new_path
                        current_seen = new_seen
                        if len(current) > len(best):
                            best = list(current)
                            best_seen = set(current)
            
            # Move: try to remove a vertex (to allow reconfiguration)
            if len(current) > 2:
                i = rng.randint(1, len(current) - 2)  # Don't remove ends
                v = current[i]
                # Check if removing v keeps path valid
                if current[i-1] in adj[current[i+1]]:
                    new_path = current[:i] + current[i+1:]
                    new_seen = current_seen - {v}
                    delta = -1
                    if rng.random() < pow(2.718, delta / T):
                        current = new_path
                        current_seen = new_seen
            
            T *= cooling
        
        return best
    
    # Apply annealing to best path
    if best_path and len(best_path) < n and time.time() < deadline - 2:
        annealed = anneal_path(best_path, adj, iters=5000)
        if len(annealed) > best_len:
            best_path = annealed
            best_len = len(best_path)
    
    # Final validation
    if not is_valid_path(best_path):
        # Fallback to simple greedy
        best_path = greedy_path(0, 'min_degree')
    
    return best_path
# EVOLVE-BLOCK-END
