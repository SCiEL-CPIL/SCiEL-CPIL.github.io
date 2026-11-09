import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 112.0
    
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    best_edges_count = 0
    best_edge_set = set()
    
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

    def contains_q3_fast(v_list, adj):
        local_adj = [0] * 8
        for i in range(8):
            vi = v_list[i]
            mask = 0
            for j in range(8):
                if i != j and adj[vi][v_list[j]]:
                    mask |= (1 << j)
            local_adj[i] = mask
        
        for mask_a, mask_b, A_idx, B_idx in all_splits:
            missing_edges = []
            valid = True
            for i in A_idx:
                for j in B_idx:
                    if not (local_adj[i] & (1 << j)):
                        missing_edges.append((i, j))
                        if len(missing_edges) > 4:
                            valid = False
                            break
                if not valid:
                    break
            
            if not valid:
                continue
                
            if len(missing_edges) != 4:
                continue
                
            used = [0] * 8
            for i, j in missing_edges:
                if used[i] or used[j]:
                    valid = False
                    break
                used[i] = 1
                used[j] = 1
                
            if valid:
                return True
        return False

    # Run multiple greedy restarts
    while time.time() < deadline:
        adj = [[0] * n for _ in range(n)]
        current_edges = set()
        
        # Random order
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if adj[u][v]:
                continue
                
            # Try adding
            adj[u][v] = 1
            adj[v][u] = 1
            
            # Check Q3
            other_vertices = [i for i in range(n) if i != u and i != v]
            found_q3 = False
            for combo in combinations(other_vertices, 6):
                vertices = (u, v) + tuple(combo)
                if contains_q3_fast(vertices, adj):
                    found_q3 = True
                    break
                    
            if found_q3:
                adj[u][v] = 0
                adj[v][u] = 0
            else:
                current_edges.add((u, v))
                
        if len(current_edges) > best_edges_count:
            best_edges_count = len(current_edges)
            best_edge_set = set(current_edges)
            
    return [list(e) for e in best_edge_set]

if __name__ == "__main__":
    res = construct()
    print(len(res))
