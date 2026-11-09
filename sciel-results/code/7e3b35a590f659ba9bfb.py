import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 120s limit

    n = 16
    adj = [[0] * n for _ in range(n)]
    edges = set()

    def add_edge(u, v):
        if u == v:
            return
        if adj[u][v]:
            return
        adj[u][v] = 1
        adj[v][u] = 1
        edges.add((min(u, v), max(u, v)))

    def remove_edge(u, v):
        if not adj[u][v]:
            return
        adj[u][v] = 0
        adj[v][u] = 0
        edges.discard((min(u, v), max(u, v)))

    # Q3 is isomorphic to K_{4,4} minus a perfect matching.
    # A graph G contains Q3 if and only if there exists a set of 8 vertices S and a bipartition (A, B) of S
    # such that the missing edges between A and B in G form a matching (i.e., are a subset of a perfect matching).

    # Precompute all 35 bipartitions of 8 items into two sets of 4.
    all_splits = []
    seen = set()
    for A in combinations(range(8), 4):
        B = tuple(i for i in range(8) if i not in A)
        key = frozenset([frozenset(A), frozenset(B)])
        if key not in seen:
            seen.add(key)
            mask_a = 0
            mask_b = 0
            for i in A:
                mask_a |= (1 << i)
            for i in B:
                mask_b |= (1 << i)
            all_splits.append((mask_a, mask_b, A, B))
    
    # all_splits now has 35 entries.

    def contains_q3_in_subset(v_list):
        """
        v_list: tuple of 8 vertex IDs.
        Returns True if the induced subgraph on v_list contains a Q3.
        """
        # Precompute local adjacency masks for speed
        local_adj = [0] * 8
        for i in range(8):
            vi = v_list[i]
            mask = 0
            for j in range(8):
                if i != j and adj[vi][v_list[j]]:
                    mask |= (1 << j)
            local_adj[i] = mask

        for mask_a, mask_b, A_idx, B_idx in all_splits:
            covered = 0
            is_matching = True
            missing_count = 0
            
            for i in A_idx:
                for j in B_idx:
                    if not (local_adj[i] & (1 << j)):
                        missing_count += 1
                        if missing_count > 4:
                            is_matching = False
                            break
                        if (covered & (1 << i)) or (covered & (1 << j)):
                            is_matching = False
                            break
                        covered |= (1 << i)
                        covered |= (1 << j)
                if not is_matching:
                    break
            if is_matching:
                return True
        return False

    # Initialize with a good seed
    # K3,13 + K3 structure:
    # Vertices 0,1,2 form a triangle.
    # Vertices 3..15 are connected to 0,1,2.
    # No other edges.
    for a in range(3):
        for b in range(3, 16):
            add_edge(a, b)
    add_edge(0, 1)
    add_edge(0, 2)
    add_edge(1, 2)
    
    best_edges = len(edges)
    best_adj = [row[:] for row in adj]
    best_edge_set = set(edges)
    
    # Local Search
    # We want to add edges.
    # Strategy:
    # 1. Maintain a list of candidate missing edges.
    # 2. Pick one randomly.
    # 3. Add it.
    # 4. Check if any Q3 is created.
    #    - Only Q3s containing the new edge (u,v) can be new.
    #    - A Q3 containing (u,v) must include u and v.
    #    - We check all 8-vertex subsets containing u and v.
    #    - C(14, 6) = 3003 subsets.
    # 5. If Q3 found, remove the new edge.
    # 6. If no Q3, keep it.
    
    # To speed up, we can cache or limit attempts.
    # But 3003 * 35 splits * (small constant) is ~100k ops per edge addition attempt.
    # If we do 1000 attempts, that's 100M ops. Might be tight in Python.
    # Let's optimize contains_q3_in_subset further or limit attempts.
    
    # Optimization:
    # In contains_q3_in_subset, we precompute local adjacency.
    
    def contains_q3_in_subset_fast(v_list):
        # Build local adjacency masks
        local_adj = [0] * 8
        for i in range(8):
            vi = v_list[i]
            mask = 0
            for j in range(8):
                if i != j and adj[vi][v_list[j]]:
                    mask |= (1 << j)
            local_adj[i] = mask
            
        for mask_a, mask_b, A_idx, B_idx in all_splits:
            covered = 0
            is_matching = True
            missing_count = 0
            
            for i in A_idx:
                # Neighbors of i in B_idx
                # We want to find j in B_idx such that NOT (local_adj[i] has bit j)
                # i.e., j is in B_idx and (local_adj[i] & (1<<j)) == 0
                
                for j in B_idx:
                    if not (local_adj[i] & (1 << j)):
                        missing_count += 1
                        if missing_count > 4:
                            is_matching = False
                            break
                        if (covered & (1 << i)) or (covered & (1 << j)):
                            is_matching = False
                            break
                        covered |= (1 << i)
                        covered |= (1 << j)
                if not is_matching:
                    break
            if is_matching:
                return True
        return False

    # Main Loop
    iterations = 0
    max_iterations = 50000
    last_improvement_time = start_time
    
    # List of all possible missing edges
    # We can just pick random u, v
    
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        
        # Pick random missing edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v or adj[u][v]:
            continue
            
        add_edge(u, v)
        
        # Check for Q3 containing (u, v)
        # We need to check if there exists a set of 6 other vertices such that the 8-vertex set contains Q3.
        # And the Q3 must use edge (u,v).
        # Note: contains_q3_in_subset checks if the set contains *any* Q3.
        # If the set contained a Q3 before adding (u,v), it would have been invalid already.
        # So if it returns True now, it's likely a new Q3 using (u,v).
        # However, it's possible that the set contained a Q3 that didn't use (u,v) but we missed it?
        # No, we maintain a Q3-free invariant. So any Q3 found must use (u,v).
        
        found_q3 = False
        other_vertices = [i for i in range(n) if i != u and i != v]
        
        # Heuristic: Check only a subset of combinations if time is tight?
        # No, we must be exact.
        # But we can break early if found.
        
        for combo in combinations(other_vertices, 6):
            vertices = (u, v) + tuple(combo)
            if contains_q3_in_subset_fast(vertices):
                found_q3 = True
                break
                
        if found_q3:
            remove_edge(u, v)
        else:
            if len(edges) > best_edges:
                best_edges = len(edges)
                best_adj = [row[:] for row in adj]
                best_edge_set = set(edges)
                last_improvement_time = time.time()
                
        # Optional: If no improvement for a long time, try a different strategy?
        # For now, just keep going.
        
    return [list(e) for e in best_edge_set]

if __name__ == "__main__":
    res = construct()
    print(len(res))
