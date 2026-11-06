import time
import random

def construct():
    deadline = time.time() + 110
    n = 20
    best_edges = None
    best_count = 0
    
    def greedy(neighbors, order):
        for (u, v) in order:
            if neighbors[u] & (1 << v):
                continue
            common_uv = neighbors[u] & neighbors[v]
            if common_uv.bit_count() > 2:
                continue
            valid = True
            w_mask = common_uv
            while w_mask:
                w_bit = w_mask & (-w_mask)
                w = w_bit.bit_length() - 1
                w_mask ^= w_bit
                cuw = (neighbors[u] & neighbors[w]).bit_count()
                if cuw + 1 > 2:
                    valid = False
                    break
                cvw = (neighbors[v] & neighbors[w]).bit_count()
                if cvw + 1 > 2:
                    valid = False
                    break
            if valid:
                neighbors[u] |= (1 << v)
                neighbors[v] |= (1 << u)
    
    def get_edge_list(neighbors):
        edges = []
        for u in range(n):
            mask = neighbors[u]
            for v in range(u+1, n):
                if mask & (1 << v):
                    edges.append([u, v])
        return edges
    
    def local_search(neighbors):
        # Try to add any missing edge
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if neighbors[u] & (1 << v):
                        continue
                    common_uv = neighbors[u] & neighbors[v]
                    if common_uv.bit_count() > 2:
                        continue
                    valid = True
                    w_mask = common_uv
                    while w_mask:
                        w_bit = w_mask & (-w_mask)
                        w = w_bit.bit_length() - 1
                        w_mask ^= w_bit
                        cuw = (neighbors[u] & neighbors[w]).bit_count()
                        if cuw + 1 > 2:
                            valid = False
                            break
                        cvw = (neighbors[v] & neighbors[w]).bit_count()
                        if cvw + 1 > 2:
                            valid = False
                            break
                    if valid:
                        neighbors[u] |= (1 << v)
                        neighbors[v] |= (1 << u)
                        improved = True
    
    for trial in range(3000):
        if time.time() > deadline:
            break
        
        random.seed(trial * 7919 + 1)
        neighbors = [0] * n
        
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        
        greedy(neighbors, all_edges)
        local_search(neighbors)
        
        edges = get_edge_list(neighbors)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
        if time.time() > deadline:
            break
    
    if best_edges is None:
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges
    
    return best_edges
