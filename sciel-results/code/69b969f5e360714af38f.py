import random
import time

def construct():
    deadline = time.time() + 140
    d = 7
    n = 1 << d
    E = d * n // 2  # 448
    
    all_edges = []
    edge_idx = {}
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(all_edges)
                all_edges.append((u, v, i))
                edge_idx[(u, v)] = idx
    
    edge_face_others = [[] for _ in range(E)]
    for idx in range(E):
        u, v, i = all_edges[idx]
        for j in range(d):
            if j == i:
                continue
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            edge_face_others[idx].append((edge_idx[e1], edge_idx[e2], edge_idx[e3]))
    
    best_count = 0
    best_chosen = None
    attempt = 0
    
    while time.time() < deadline:
        rng = random.Random(attempt)
        chosen = bytearray(E)
        order = list(range(E))
        rng.shuffle(order)
        count = 0
        
        for idx in order:
            ok = True
            for (i1, i2, i3) in edge_face_others[idx]:
                if chosen[i1] and chosen[i2] and chosen[i3]:
                    ok = False
                    break
            if ok:
                chosen[idx] = 1
                count += 1
        
        # Local improvement: try to add skipped edges
        skipped = [idx for idx in range(E) if not chosen[idx]]
        rng.shuffle(skipped)
        for idx in skipped:
            ok = True
            for (i1, i2, i3) in edge_face_others[idx]:
                if chosen[i1] and chosen[i2] and chosen[i3]:
                    ok = False
                    break
            if ok:
                chosen[idx] = 1
                count += 1
        
        if count > best_count:
            best_count = count
            best_chosen = chosen.copy()
        
        attempt += 1
    
    out = []
    for idx in range(E):
        if best_chosen[idx]:
            u, v, i = all_edges[idx]
            out.append([u, v])
    return out
