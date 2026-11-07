import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    
    best_score = float('inf')
    best_result = None
    
    def score(vals, edges):
        s = set()
        p = set()
        for u, v in edges:
            s.add(vals[u] + vals[v])
            p.add(vals[u] * vals[v])
        return max(len(s), len(p))
    
    def find_best_edges(vals):
        if time.time() > deadline:
            return None, None
        edges = list(all_edges)
        while len(edges) > m:
            if time.time() > deadline:
                return None, None
            best_i = -1
            best_s = score(vals, edges)
            for i in range(len(edges)):
                ne = edges[:i] + edges[i+1:]
                sc = score(vals, ne)
                if sc < best_s:
                    best_s = sc
                    best_i = i
            if best_i < 0:
                break
            edges.pop(best_i)
        if len(edges) != m:
            return None, None
        # Local search
        cur = score(vals, edges)
        for _ in range(30):
            if time.time() > deadline:
                break
            imp = False
            for i in range(len(edges)):
                if time.time() > deadline:
                    break
                for j in range(len(all_edges)):
                    if all_edges[j] in edges:
                        continue
                    ne = edges[:i] + [all_edges[j]] + edges[i+1:]
                    sc = score(vals, ne)
                    if sc < cur:
                        edges = ne
                        cur = sc
                        imp = True
                        break
                if imp:
                    break
            if not imp:
                break
        return cur, edges
    
    random.seed(123)
    
    # Try structured value sets
    for base in range(1, 300, 5):
        for step in range(1, 200, 7):
            if time.time() > deadline:
                break
            vals = [base + i * step for i in range(n)]
            if max(vals) > 10000 or len(set(vals)) != n:
                continue
            sc, edges = find_best_edges(vals)
            if sc is not None and sc < best_score:
                best_score = sc
                best_result = {'values': list(vals), 'edges': [list(e) for e in edges]}
    
    if time.time() > deadline:
        if best_result:
            return best_result
        return {'values': list(range(1, 15)), 'edges': [list(e) for e in all_edges[:m]]}
    
    for k in range(2, 40):
        if time.time() > deadline:
            break
        for _ in range(8):
            if time.time() > deadline:
                break
            mults = random.sample(range(1, 10000 // k + 1), n)
            vals = [x * k for x in mults]
            if len(set(vals)) != n or max(vals) > 10000:
                continue
            sc, edges = find_best_edges(vals)
            if sc is not None and sc < best_score:
                best_score = sc
                best_result = {'values': list(vals), 'edges': [list(e) for e in edges]}
    
    if time.time() > deadline:
        if best_result:
            return best_result
        return {'values': list(range(1, 15)), 'edges': [list(e) for e in all_edges[:m]]}
    
    for _ in range(20):
        if time.time() > deadline:
            break
        c = random.choice([2, 3, 4, 5, 6, 8, 10])
        vals = []
        while len(vals) < n:
            v = random.randint(1, 10000 // c) * c
            if v not in vals:
                vals.append(v)
        if len(set(vals)) != n:
            continue
        sc, edges = find_best_edges(vals)
        if sc is not None and sc < best_score:
            best_score = sc
            best_result = {'values': list(vals), 'edges': [list(e) for e in edges]}
    
    if best_result:
        return best_result
    return {'values': list(range(1, 15)), 'edges': [list(e) for e in all_edges[:m]]}
