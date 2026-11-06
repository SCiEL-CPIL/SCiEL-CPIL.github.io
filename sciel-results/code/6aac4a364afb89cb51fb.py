import time
import random
import itertools

def generate_q3_data():
    base_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            xor = i ^ j
            if xor == 1 or xor == 2 or xor == 4:
                base_edges.append((i, j))
                
    q3_masks = set()
    for perm in itertools.permutations(range(8)):
        mask = 0
        for (i, j) in base_edges:
            u, v = perm[i], perm[j]
            if u > v:
                u, v = v, u
            idx = u * (7 - u) + (v - u - 1)
            mask |= (1 << idx)
        q3_masks.add(mask)
        
    q3_list = list(q3_masks)
    
    # Precompute masks for each edge index
    # There are 28 edges in K8
    q3_by_edge = [[] for _ in range(28)]
    for mask in q3_list:
        for e in range(28):
            if mask & (1 << e):
                q3_by_edge[e].append(mask)
                
    return q3_list, q3_by_edge

Q3_LIST, Q3_BY_EDGE = generate_q3_data()

def get_edge_index_in_k8(u, v):
    # u, v are indices 0..7
    if u > v:
        u, v = v, u
    return u * (7 - u) + (v - u - 1)

def has_q3(adj, n=16, new_edge=None):
    if new_edge is None:
        # Full check (slow)
        for vertices in itertools.combinations(range(n), 8):
            mask = 0
            for i in range(8):
                u = vertices[i]
                for j in range(i+1, 8):
                    v = vertices[j]
                    if adj[u] & (1 << v):
                        idx = i * (7 - i) + (j - i - 1)
                        mask |= (1 << idx)
            if bin(mask).count('1') < 12:
                continue
            for q3_mask in Q3_LIST:
                if (mask & q3_mask) == q3_mask:
                    return True
        return False
    else:
        u, v = new_edge
        # Check only subsets containing u and v
        # And only Q3s containing the edge corresponding to (u,v) in that subset
        for vertices in itertools.combinations([x for x in range(n) if x != u and x != v], 6):
            vertices = tuple(sorted([u, v] + list(vertices)))
            # Map u, v to indices in vertices
            i_u = vertices.index(u)
            i_v = vertices.index(v)
            if i_u > i_v:
                i_u, i_v = i_v, i_u
                
            # Build mask
            mask = 0
            for i in range(8):
                uu = vertices[i]
                for j in range(i+1, 8):
                    vv = vertices[j]
                    if adj[uu] & (1 << vv):
                        idx = i * (7 - i) + (j - i - 1)
                        mask |= (1 << idx)
                        
            if bin(mask).count('1') < 12:
                continue
                
            # Get edge index for (u,v) in this K8
            idx_uv = i_u * (7 - i_u) + (i_v - i_u - 1)
            
            # Check only Q3s containing this edge
            for q3_mask in Q3_BY_EDGE[idx_uv]:
                if (mask & q3_mask) == q3_mask:
                    return True
        return False

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    edges = set()
    adj = [0] * n
    for a in range(3):
        for b in range(3, 16):
            edges.add((a, b))
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
            
    best_edges = set(edges)
    
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    candidates = [e for e in all_possible if e not in best_edges]
    
    while time.time() < deadline:
        random.shuffle(candidates)
        improved = False
        for e in candidates:
            if time.time() > deadline:
                break
            u, v = e
            best_edges.add(e)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if has_q3(adj, n, e):
                best_edges.remove(e)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                candidates.remove(e)
                improved = True
                
        if not improved:
            break
            
    return [list(e) for e in best_edges]
