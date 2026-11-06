import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    vertices = list(range(n))
    
    # Precompute all 8-vertex subsets
    subsets_8 = list(itertools.combinations(vertices, 8))
    
    # Precompute partitions for 8 positions
    # We need to check if a set of 8 vertices induces a Q3.
    # Q3 is bipartite with parts of size 4.
    # So we check if there is a partition of the 8 vertices into two sets X, Y of size 4
    # such that the induced bipartite graph has min degree >= 3.
    
    # Precompute all partitions of 8 items into two sets of 4
    # To avoid symmetry, we can just iterate all C(8,4) = 70 combinations.
    partition_indices = list(itertools.combinations(range(8), 4))
    
    def has_q3_full(adj_mask):
        """Check if graph contains Q3."""
        for s in subsets_8:
            # Quick check: count edges in s
            # If < 12, cannot be Q3 (Q3 has 12 edges)
            edge_count = 0
            for i in range(8):
                vi = s[i]
                mask = adj_mask[vi]
                for j in range(i+1, 8):
                    if mask & (1 << s[j]):
                        edge_count += 1
            if edge_count < 12:
                continue
            
            # Check partitions
            for x_combo in partition_indices:
                # x_combo is a tuple of indices in s that form X
                x_set = set(x_combo)
                y_set = tuple(j for j in range(8) if j not in x_set)
                
                valid = True
                # Check min degree from X side
                for i in x_combo:
                    v = s[i]
                    deg = 0
                    for j in y_set:
                        if adj_mask[v] & (1 << s[j]):
                            deg += 1
                    if deg < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check min degree from Y side
                for j in y_set:
                    u = s[j]
                    deg = 0
                    for i in x_combo:
                        if adj_mask[u] & (1 << s[i]):
                            deg += 1
                    if deg < 3:
                        valid = False
                        break
                
                if valid:
                    return True
        return False

    def has_q3_edge(adj_mask, u, v):
        """Check if adding edge (u,v) creates a Q3.
        Only check subsets containing both u and v."""
        for s in subsets_8:
            if u not in s or v not in s:
                continue
            
            # Count edges in s (u-v is already in adj_mask)
            edge_count = 0
            for i in range(8):
                vi = s[i]
                mask = adj_mask[vi]
                for j in range(i+1, 8):
                    if mask & (1 << s[j]):
                        edge_count += 1
            if edge_count < 12:
                continue
            
            for x_combo in partition_indices:
                x_set = set(x_combo)
                y_set = tuple(j for j in range(8) if j not in x_set)
                
                valid = True
                for i in x_combo:
                    vi = s[i]
                    deg = 0
                    for j in y_set:
                        if adj_mask[vi] & (1 << s[j]):
                            deg += 1
                    if deg < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                for j in y_set:
                    uj = s[j]
                    deg = 0
                    for i in x_combo:
                        if adj_mask[uj] & (1 << s[i]):
                            deg += 1
                    if deg < 3:
                        valid = False
                        break
                
                if valid:
                    return True
        return False
    
    def build_adj_mask(edge_set):
        adj_mask = [0] * n
        for u, v in edge_set:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        return adj_mask
    
    # Shrikhande graph construction
    def shrikhande_adj(u, v):
        a, b = divmod(u, 4)
        c, d = divmod(v, 4)
        da = (a - c) % 4
        db = (b - d) % 4
        if da == 0 and (db == 1 or db == 3): return True
        if db == 0 and (da == 1 or da == 3): return True
        if da == 2 and db == 2: return True
        return False
    
    shrikhande_edges = set()
    for u in range(16):
        for v in range(u+1, 16):
            if shrikhande_adj(u, v):
                shrikhande_edges.add((u, v))
    
    all_edges = set((u, v) for u in range(16) for v in range(u+1, 16))
    complement_edges = all_edges - shrikhande_edges
    
    # Try complement of Shrikhande (72 edges)
    adj_mask = build_adj_mask(complement_edges)
    if not has_q3_full(adj_mask):
        best_edges = set(complement_edges)
        best_count = len(best_edges)
    else:
        best_edges = set()
        best_count = 0
        
        # Try random dense graphs
        for attempt in range(20):
            if time.time() > deadline - 60:
                break
            all_edge_list = list(all_edges)
            random.shuffle(all_edge_list)
            current = set(all_edge_list[:60])
            adj_m = build_adj_mask(current)
            if not has_q3_full(adj_m):
                if len(current) > best_count:
                    best_count = len(current)
                    best_edges = set(current)
            
            # Greedy add
            for e in all_edge_list[60:]:
                if time.time() > deadline - 30:
                    break
                u, v = e
                if (u, v) in current:
                    continue
                adj_m[u] |= (1 << v)
                adj_m[v] |= (1 << u)
                if has_q3_edge(adj_m, u, v):
                    adj_m[u] &= ~(1 << v)
                    adj_m[v] &= ~(1 << u)
                else:
                    current.add((u, v))
            
            if len(current) > best_count:
                best_count = len(current)
                best_edges = set(current)
        
        # Fallback if still low
        if best_count < 50:
            candidate = set()
            for a in range(3):
                for b in range(3, 16):
                    candidate.add((a, b))
            for u, v in [(0,1), (1,2), (0,2)]:
                candidate.add((u, v))
            adj_m = build_adj_mask(candidate)
            if not has_q3_full(adj_m):
                if len(candidate) > best_count:
                    best_count = len(candidate)
                    best_edges = set(candidate)
            else:
                best_edges = set()
                best_count = 0
    
    # Local search
    adj_mask = build_adj_mask(best_edges)
    current_edges = set(best_edges)
    candidate_edges = list(all_edges - current_edges)
    
    for restart in range(30):
        if time.time() > deadline - 10:
            break
        
        random.shuffle(candidate_edges)
        current_edges = set(best_edges)
        adj_mask = build_adj_mask(current_edges)
        
        for u, v in candidate_edges:
            if time.time() > deadline - 5:
                break
            if (u, v) in current_edges:
                continue
            
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
            if has_q3_edge(adj_mask, u, v):
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
            else:
                current_edges.add((u, v))
            
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
            candidate_edges = list(all_edges - best_edges)
    
    # Final verification
    adj_mask = build_adj_mask(best_edges)
    if has_q3_full(adj_mask):
        edge_list = list(best_edges)
        random.shuffle(edge_list)
        current = set(best_edges)
        adj_m = build_adj_mask(current)
        for e in edge_list:
            if has_q3_full(adj_m):
                u, v = e
                adj_m[u] &= ~(1 << v)
                adj_m[v] &= ~(1 << u)
                current.discard(e)
        best_edges = current
        best_count = len(best_edges)
    
    result = [[u, v] for u, v in best_edges]
    return result

if __name__ == "__main__":
    start = time.time()
    edges = construct()
    print(f"Found {len(edges)} edges in {time.time() - start:.2f}s")
