import time
import itertools
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    n = 16
    total_possible_edges = n * (n - 1) // 2  # 120
    
    # Seed: K_{3,13}
    def seed_edges():
        return [[a, b] for a in range(3) for b in range(3, 16)]
    
    best_edges = seed_edges()
    best_m = len(best_edges)
    
    # Q3 structure: 3-cube
    # Vertices 0..7, edges where hamming distance is 1
    Q3_EDGES = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    
    # Precompute Q3 adjacency
    q3_adj = [set() for _ in range(8)]
    for u, v in Q3_EDGES:
        q3_adj[u].add(v)
        q3_adj[v].add(u)
    
    def check_q3_free(adj_mask):
        """
        Check if the graph defined by adj_mask (list of 16 bitmasks) is Q3-free.
        Returns True if no Q3 is found, False otherwise.
        """
        # Iterate all 8-subsets
        for S in itertools.combinations(range(n), 8):
            S_set = set(S)
            
            # Quick filter: min degree in induced subgraph must be >= 3
            min_deg = 8
            for v in S:
                deg = 0
                mask = adj_mask[v]
                for u in S:
                    if u != v and (mask >> u) & 1:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
            if min_deg < 3:
                continue
                
            # Backtracking search for Q3 isomorphism
            # mapping[i] = vertex in S corresponding to Q3 vertex i
            mapping = [-1] * 8
            available = list(S)
            
            # Order Q3 vertices to maximize pruning?
            # Q3 is vertex transitive, so order doesn't matter much for the first step.
            # But we can fix the order of assignment to 0, 1, 2...
            
            def dfs(idx):
                if idx == 8:
                    return True
                
                # Try assigning Q3 vertex 'idx' to each available vertex
                # Pruning: Check constraints with already assigned neighbors
                # Neighbors of 'idx' in Q3 that are < idx (already assigned)
                assigned_neighbors = [nb for nb in q3_adj[idx] if nb < idx]
                
                for j in range(len(available)):
                    v = available[j]
                    valid = True
                    for nb in assigned_neighbors:
                        mapped_nb = mapping[nb]
                        if not (adj_mask[mapped_nb] >> v) & 1:
                            valid = False
                            break
                    
                    if valid:
                        mapping[idx] = v
                        if dfs(idx + 1):
                            return True
                        mapping[idx] = -1
                    # Optimization: remove v from available?
                    # The loop above uses index j, so we don't strictly need to modify list
                    # but standard backtracking removes it. Here we just skip.
                    
                return False
            
            if dfs(0):
                return False
        
        return True

    def get_edge_list(adj_mask):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    # Strategy 1: Greedy removal from K16
    # This might find a graph with > 39 edges if K16 is close to Q3-free? No, K16 is very dense.
    # But let's try.
    
    # Initialize with K16
    adj_mask = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
    
    current_edges = total_possible_edges
    
    # Greedy loop
    while time.time() < deadline - 5.0:
        # Check if current graph is Q3-free
        if check_q3_free(adj_mask):
            if current_edges > best_m:
                best_m = current_edges
                best_edges = get_edge_list(adj_mask)
            break
        
        # Find a Q3 to remove an edge from
        # We need to modify check_q3_free to return the Q3 edges if found, or just find one.
        # Let's refactor: find_one_q3(adj_mask) -> list of edges or None
        
        # For now, let's just pick a random edge to remove? No, that's slow.
        # We need to find a specific Q3.
        
        # Let's implement find_one_q3
        found = None
        for S in itertools.combinations(range(n), 8):
            S_set = set(S)
            min_deg = 8
            for v in S:
                deg = 0
                mask = adj_mask[v]
                for u in S:
                    if u != v and (mask >> u) & 1:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
            if min_deg < 3:
                continue
            
            mapping = [-1] * 8
            available = list(S)
            
            def dfs(idx):
                if idx == 8:
                    return True
                assigned_neighbors = [nb for nb in q3_adj[idx] if nb < idx]
                for j in range(len(available)):
                    v = available[j]
                    valid = True
                    for nb in assigned_neighbors:
                        mapped_nb = mapping[nb]
                        if not (adj_mask[mapped_nb] >> v) & 1:
                            valid = False
                            break
                    if valid:
                        mapping[idx] = v
                        if dfs(idx + 1):
                            return True
                        mapping[idx] = -1
                return False
            
            if dfs(0):
                # Found Q3
                edges_in_q3 = []
                for u, v in Q3_EDGES:
                    edges_in_q3.append((mapping[u], mapping[v]))
                found = edges_in_q3
                break
        
        if found is None:
            # Should have been caught by check_q3_free, but just in case
            if current_edges > best_m:
                best_m = current_edges
                best_edges = get_edge_list(adj_mask)
            break
            
        # Remove a random edge from the found Q3
        # To be deterministic or random? Random might escape local minima.
        random.shuffle(found)
        u, v = found[0]
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        current_edges -= 1
        
        # If we drop below best_m, we can stop?
        # Not necessarily, we might come back up? No, we only remove edges.
        # So if current_edges <= best_m, we can't beat the best.
        if current_edges <= best_m:
            break

    # Strategy 2: Local Search / Hill Climbing
    # Start from a random dense graph or the seed, and try adding edges?
    # Adding edges is risky.
    
    # Let's try a different construction:
    # Start with a bipartite graph K_{a, b} and add edges within parts.
    # We know K_{3,13} is safe.
    # Can we add edges within the 13-part?
    
    # Let's try to construct a specific candidate:
    # Take K_{4,12}. It has 48 edges. It contains Q3.
    # Remove edges to break all Q3s?
    
    # Let's try a randomized restart approach with a time limit.
    # We will try to build a graph by starting with K_{4,12} and removing edges.
    
    if time.time() < deadline - 10.0:
        # Try K_{4,12} based approach
        # Partition: A = {0,1,2,3}, B = {4..15}
        # Start with all edges between A and B.
        adj_mask2 = [0] * n
        for u in range(4):
            for v in range(4, 16):
                adj_mask2[u] |= (1 << v)
                adj_mask2[v] |= (1 << u)
        
        current_edges2 = 48
        
        # Check if K_{4,12} is Q3-free? No, it contains K_{4,4} which contains Q3.
        # So we must remove edges.
        
        # Greedy removal from K_{4,12}
        while time.time() < deadline - 5.0 and current_edges2 > best_m:
            # Check if Q3-free
            if check_q3_free(adj_mask2):
                if current_edges2 > best_m:
                    best_m = current_edges2
                    best_edges = get_edge_list(adj_mask2)
                break
            
            # Find Q3
            found = None
            for S in itertools.combinations(range(n), 8):
                S_set = set(S)
                min_deg = 8
                for v in S:
                    deg = 0
                    mask = adj_mask2[v]
                    for u in S:
                        if u != v and (mask >> u) & 1:
                            deg += 1
                    if deg < min_deg:
                        min_deg = deg
                if min_deg < 3:
                    continue
                
                mapping = [-1] * 8
                available = list(S)
                
                def dfs(idx):
                    if idx == 8:
                        return True
                    assigned_neighbors = [nb for nb in q3_adj[idx] if nb < idx]
                    for j in range(len(available)):
                        v = available[j]
                        valid = True
                        for nb in assigned_neighbors:
                            mapped_nb = mapping[nb]
                            if not (adj_mask2[mapped_nb] >> v) & 1:
                                valid = False
                                break
                        if valid:
                            mapping[idx] = v
                            if dfs(idx + 1):
                                return True
                            mapping[idx] = -1
                    return False
                
                if dfs(0):
                    edges_in_q3 = []
                    for u, v in Q3_EDGES:
                        edges_in_q3.append((mapping[u], mapping[v]))
                    found = edges_in_q3
                    break
            
            if found is None:
                if current_edges2 > best_m:
                    best_m = current_edges2
                    best_edges = get_edge_list(adj_mask2)
                break
                
            # Remove random edge
            random.shuffle(found)
            u, v = found[0]
            adj_mask2[u] &= ~(1 << v)
            adj_mask2[v] &= ~(1 << u)
            current_edges2 -= 1

    # Strategy 3: Try adding edges to the seed K_{3,13}
    # If we have time.
    if time.time() < deadline - 5.0 and best_m <= 39:
        # Start with K_{3,13}
        adj_mask3 = [0] * n
        for u in range(3):
            for v in range(3, 16):
                adj_mask3[u] |= (1 << v)
                adj_mask3[v] |= (1 << u)
        
        current_edges3 = 39
        
        # Candidate edges to add: within {3..15} or within {0,1,2}
        # Within {0,1,2} is small (3 edges).
        # Within {3..15} is large.
        
        # Let's try adding random edges within B and check.
        # This is a "add and check" loop.
        
        # To avoid infinite loops, we limit the number of additions.
        # But checking is expensive.
        
        # Actually, we can try specific structures.
        # E.g., add a matching within B.
        
        # For now, let's just return the best found so far.
        # The greedy removal from K16 or K4,12 is likely to find something > 39.
        # K4,12 has 48. Even if we remove 5-10 edges, we are at 38-43.
        # K16 has 120. We might remove many.
        
        # Let's do one more thing:
        # Try a random graph with p=0.8 and greedily remove.
        if time.time() < deadline - 5.0:
            random.seed(42)
            adj_mask4 = [0] * n
            current_edges4 = 0
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.8:
                        adj_mask4[u] |= (1 << v)
                        adj_mask4[v] |= (1 << u)
                        current_edges4 += 1
            
            while time.time() < deadline - 2.0 and current_edges4 > best_m:
                if check_q3_free(adj_mask4):
                    if current_edges4 > best_m:
                        best_m = current_edges4
                        best_edges = get_edge_list(adj_mask4)
                    break
                
                found = None
                for S in itertools.combinations(range(n), 8):
                    S_set = set(S)
                    min_deg = 8
                    for v in S:
                        deg = 0
                        mask = adj_mask4[v]
                        for u in S:
                            if u != v and (mask >> u) & 1:
                                deg += 1
                        if deg < min_deg:
                            min_deg = deg
                    if min_deg < 3:
                        continue
                    
                    mapping = [-1] * 8
                    available = list(S)
                    
                    def dfs(idx):
                        if idx == 8:
                            return True
                        assigned_neighbors = [nb for nb in q3_adj[idx] if nb < idx]
                        for j in range(len(available)):
                            v = available[j]
                            valid = True
                            for nb in assigned_neighbors:
                                mapped_nb = mapping[nb]
                                if not (adj_mask4[mapped_nb] >> v) & 1:
                                    valid = False
                                    break
                            if valid:
                                mapping[idx] = v
                                if dfs(idx + 1):
                                    return True
                                mapping[idx] = -1
                        return False
                    
                    if dfs(0):
                        edges_in_q3 = []
                        for u, v in Q3_EDGES:
                            edges_in_q3.append((mapping[u], mapping[v]))
                        found = edges_in_q3
                        break
                
                if found is None:
                    if current_edges4 > best_m:
                        best_m = current_edges4
                        best_edges = get_edge_list(adj_mask4)
                    break
                    
                random.shuffle(found)
                u, v = found[0]
                adj_mask4[u] &= ~(1 << v)
                adj_mask4[v] &= ~(1 << u)
                current_edges4 -= 1

    return best_edges
