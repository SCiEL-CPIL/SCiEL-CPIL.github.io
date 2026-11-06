import random
from itertools import combinations

def construct():
    """
    Construct a 5-uniform hypergraph with no proper 2-colouring, minimizing edges.
    Uses randomized greedy SAT reduction.
    """
    
    def try_construct(n, max_restarts=10):
        all_edges = list(combinations(range(n), 5))
        edge_masks = []
        for e in all_edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        edges_and_masks = list(zip(all_edges, edge_masks))
        
        best_edge_count = float('inf')
        best_result = None
        
        for restart in range(max_restarts):
            random.shuffle(edges_and_masks)
            valid_colorings = list(range(1 << n))
            selected_edges = []
            selected_indices = set()
            
            while valid_colorings:
                best_idx = -1
                best_count = 0
                
                # Optimization: If valid_colorings is small, this is fast.
                # If large, it's the bottleneck.
                vc_len = len(valid_colorings)
                
                for i, (e, mask) in enumerate(edges_and_masks):
                    if i in selected_indices:
                        continue
                    
                    count = 0
                    # Unroll or use fast path
                    for c in valid_colorings:
                        cm = c & mask
                        if cm == 0 or cm == mask:
                            count += 1
                            
                    if count > best_count:
                        best_count = count
                        best_idx = i
                        
                if best_idx == -1 or best_count == 0:
                    break
                
                selected_indices.add(best_idx)
                selected_edges.append(edges_and_masks[best_idx][0])
                
                mask = edges_and_masks[best_idx][1]
                # Filter
                new_valid = [c for c in valid_colorings if not ((c & mask) == 0 or (c & mask) == mask)]
                valid_colorings = new_valid
                
            if not valid_colorings:
                if len(selected_edges) < best_edge_count:
                    best_edge_count = len(selected_edges)
                    best_result = {'n': n, 'edges': [list(e) for e in selected_edges]}
                    if best_edge_count == 29:
                        return best_result
        
        return best_result

    # Try n=11 first (good balance of space and edge count)
    res = try_construct(11, max_restarts=15)
    if res:
        return res

    # Try n=10
    res = try_construct(10, max_restarts=15)
    if res:
        return res

    # Try n=12
    res = try_construct(12, max_restarts=5)
    if res:
        return res

    # Fallback
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
