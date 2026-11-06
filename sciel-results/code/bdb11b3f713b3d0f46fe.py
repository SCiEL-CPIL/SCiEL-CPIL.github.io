import random
import itertools

def construct():
    n = 16
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            edges.add((u, v))
            
    q3_edges = [
        (0, 1), (0, 2), (0, 4),
        (3, 1), (3, 2), (3, 7),
        (5, 1), (5, 4), (5, 7),
        (6, 2), (6, 4), (6, 7)
    ]
    
    # Precompute mappings
    mappings = []
    qa = [0, 3, 5, 6]
    qb = [1, 2, 4, 7]
    
    # Case 1: A -> 0-3, B -> 4-7
    for pa in itertools.permutations(range(4)):
        for pb in itertools.permutations(range(4, 8)):
            m = {}
            for i in range(4):
                m[qa[i]] = pa[i]
                m[qb[i]] = pb[i]
            sub_edges = []
            for u, v in q3_edges:
                su = m[u]
                sv = m[v]
                if su > sv:
                    su, sv = sv, su
                sub_edges.append((su, sv))
            mappings.append(sub_edges)
            
    # Case 2: A -> 4-7, B -> 0-3
    for pa in itertools.permutations(range(4, 8)):
        for pb in itertools.permutations(range(4)):
            m = {}
            for i in range(4):
                m[qa[i]] = pa[i]
                m[qb[i]] = pb[i]
            sub_edges = []
            for u, v in q3_edges:
                su = m[u]
                sv = m[v]
                if su > sv:
                    su, sv = sv, su
                sub_edges.append((su, sv))
            mappings.append(sub_edges)
            
    def has_q3(vertices):
        ind_edges = set()
        for i in range(8):
            for j in range(i+1, 8):
                u = vertices[i]
                v = vertices[j]
                if (u, v) in edges:
                    ind_edges.add((i, j))
        if len(ind_edges) < 12:
            return None
        deg = [0]*8
        for u, v in ind_edges:
            deg[u] += 1
            deg[v] += 1
        if min(deg) < 3:
            return None
        for m_edges in mappings:
            is_sub = True
            for e in m_edges:
                if e not in ind_edges:
                    is_sub = False
                    break
            if is_sub:
                return [ (vertices[u], vertices[v]) if vertices[u] < vertices[v] else (vertices[v], vertices[u]) for u, v in m_edges ]
        return None

    def find_q3():
        deg = [0]*n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        active = [i for i in range(n) if deg[i] >= 3]
        if len(active) < 8:
            return None
        # Increase samples if graph is sparse?
        # For now, 50 samples
        for _ in range(50):
            subset = random.sample(active, 8)
            q3 = has_q3(tuple(subset))
            if q3:
                return q3
        return None

    # Phase 1
    while True:
        q3 = find_q3()
        if not q3:
            break
        e_to_remove = random.choice(q3)
        edges.remove(e_to_remove)
        
    # Phase 2: Try to add edges
    all_edges = set((u, v) for u in range(n) for v in range(u+1, n))
    missing = list(all_edges - edges)
    random.shuffle(missing)
    
    for e in missing:
        edges.add(e)
        if find_q3() is None:
            continue
        else:
            edges.remove(e)
            
    return [[u, v] for u, v in edges]
