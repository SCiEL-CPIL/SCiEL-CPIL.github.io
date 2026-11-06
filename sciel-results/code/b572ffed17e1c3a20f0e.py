import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(all_edges)
    
    # Precompute edge to index mapping
    edge_to_idx = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_to_idx[(u, v)] = idx
    
    # Use bitmasks for fast adjacency checks
    # adj_mask[i] is a 16-bit integer where bit j is 1 if edge (i,j) exists
    
    def build_adj_mask(edge_set):
        masks = [0] * n
        for u, v in edge_set:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        return masks
    
    def find_q3_fast(adj_mask):
        """Find a Q3 subgraph using bitmask operations. Returns 12 edges or None."""
        for v in range(n):
            mv = adj_mask[v]
            if bin(mv).count('1') < 3:
                continue
            neighbors = [i for i in range(n) if (mv >> i) & 1]
            nn = len(neighbors)
            if nn < 3:
                continue
            for i in range(nn):
                a = neighbors[i]
                ma = adj_mask[a]
                for j in range(i + 1, nn):
                    b = neighbors[j]
                    mb = adj_mask[b]
                    for k in range(j + 1, nn):
                        c = neighbors[k]
                        mc = adj_mask[c]
                        
                        # x: neighbor of a and b, not v,a,b,c
                        xmask = (ma & mb) & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                        # y: neighbor of a and c, not v,a,b,c
                        ymask = (ma & mc) & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                        # z: neighbor of b and c, not v,a,b,c
                        zmask = (mb & mc) & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                        
                        xs = [x for x in range(n) if (xmask >> x) & 1]
                        ys = [y for y in range(n) if (ymask >> y) & 1]
                        zs = [z for z in range(n) if (zmask >> z) & 1]
                        
                        for x in xs:
                            mx = adj_mask[x]
                            for y in ys:
                                if x == y:
                                    continue
                                my = adj_mask[y]
                                for z in zs:
                                    if x == z or y == z:
                                        continue
                                    mz = adj_mask[z]
                                    
                                    # w: neighbor of x, y, z, not v,a,b,c,x,y,z
                                    wmask = (mx & my & mz) & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << x) & ~(1 << y) & ~(1 << z)
                                    ws = [w for w in range(n) if (wmask >> w) & 1]
                                    
                                    for w in ws:
                                        q3_edges = [
                                            (v, a), (v, b), (v, c),
                                            (a, x), (b, x),
                                            (a, y), (c, y),
                                            (b, z), (c, z),
                                            (x, w), (y, w), (z, w)
                                        ]
                                        return q3_edges
        return None
    
    def count_q3s_containing_edge(adj_mask, edge):
        """Count how many Q3s contain the given edge. Approximate by sampling."""
        u, v = edge
        # For each potential Q3 structure containing (u,v), check if it exists
        # This is expensive, so we use a heuristic: count common neighbors
        # A simpler heuristic: count 4-cycles through this edge
        common = adj_mask[u] & adj_mask[v]
        common &= ~(1 << u) & ~(1 << v)
        return bin(common).count('1')
    
    best_edges = []
    best_count = 0
    random.seed(42)
    
    # Multiple restarts with different strategies
    restart_configs = [
        {'density': 0.6, 'remove_heuristic': 'random'},
        {'density': 0.55, 'remove_heuristic': 'low_degree'},
        {'density': 0.5, 'remove_heuristic': 'random'},
        {'density': 0.65, 'remove_heuristic': 'low_degree'},
        {'density': 0.5, 'remove_heuristic': 'random'},
        {'density': 0.55, 'remove_heuristic': 'random'},
        {'density': 0.45, 'remove_heuristic': 'low_degree'},
        {'density': 0.6, 'remove_heuristic': 'random'},
    ]
    
    for config in restart_configs:
        if time.time() > deadline - 15:
            break
        
        p = config['density']
        heuristic = config['remove_heuristic']
        current_edges = set()
        for e in all_edges:
            if random.random() < p:
                current_edges.add(e)
                
        adj_mask = build_adj_mask(current_edges)
        
        # Phase 1: Remove Q3s until none found
        steps = 0
        max_steps = 500
        while steps < max_steps:
            if time.time() > deadline - 15:
                break
            q3_edges = find_q3_fast(adj_mask)
            if q3_edges is None:
                break
            
            if heuristic == 'random':
                e_to_remove = random.choice(q3_edges)
            else:
                # Remove edge with lowest degree in current graph
                degrees = [bin(adj_mask[u] & adj_mask[v]).count('1') for u, v in q3_edges]
                min_deg = min(degrees)
                candidates = [q3_edges[i] for i in range(len(q3_edges)) if degrees[i] == min_deg]
                e_to_remove = random.choice(candidates)
            
            if e_to_remove[0] > e_to_remove[1]:
                e_to_remove = (e_to_remove[1], e_to_remove[0])
            
            if e_to_remove in current_edges:
                current_edges.remove(e_to_remove)
                u, v = e_to_remove
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
            else:
                break
            steps += 1
        
        # Phase 2: Hill Climbing - try to add edges
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            missing = [e for e in all_edges if e not in current_edges]
            random.shuffle(missing)
            
            for e in missing:
                if time.time() > deadline - 15:
                    break
                u, v = e
                current_edges.add(e)
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                
                q3_edges = find_q3_fast(adj_mask)
                
                if q3_edges is not None:
                    # Q3 found, must remove e (since it was just added and no Q3 existed before)
                    current_edges.remove(e)
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
                else:
                    improved = True
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
            
    return [[u, v] for u, v in best_edges]

# EVOLVE-BLOCK-END
