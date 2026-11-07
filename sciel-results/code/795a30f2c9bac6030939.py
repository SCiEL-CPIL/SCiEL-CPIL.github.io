import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    deadline = time.time() + 85

    fallback = [[11, 1], [11, 4], [11, 7], [11, 15], [11, 18], [11, 21], [11, 23]]

    base_adj = [set() for _ in range(n)]
    base_edge_set = set()
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
        base_edge_set.add((min(u,v), max(u,v)))

    def is_valid_fast(added):
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added:
            if len(adj[u] & adj[v]) > 0:
                return False
            adj[u].add(v)
            adj[v].add(u)
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = deque([s])
            cnt = 1
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > 4:
                            return False
                        queue.append(v)
                        cnt += 1
            if cnt < n:
                return False
        return True

    best_solution = fallback
    best_len = len(fallback)

    # Try to find 6-edge solution
    # Strategy: enumerate candidate edge sets of size 6 using structured search
    
    # Precompute: for each pair (u,v), is it a valid added edge (no triangle with base)?
    # (u,v) is valid if |u-v| >= 3 (since |u-v|=1 is base, |u-v|=2 creates triangle)
    
    # For a set of added edges E, triangle-free means:
    # 1. No added edge (u,v) has a common neighbor in base graph
    # 2. No two added edges share a vertex and the other endpoints are adjacent
    # 3. No three added edges form a triangle
    
    # Simplified: for any two added edges (a,b) and (c,d), if they share a vertex,
    # the other endpoints must not be adjacent.
    # And for any added edge (u,v), u and v must not have a common base neighbor.
    
    def check_triangle_free(added):
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added:
            if len(adj[u] & adj[v]) > 0:
                return False
            adj[u].add(v)
            adj[v].add(u)
        return True

    def check_diameter(added):
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added:
            adj[u].add(v)
            adj[v].add(u)
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = deque([s])
            cnt = 1
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > 4:
                            return False
                        queue.append(v)
                        cnt += 1
            if cnt < n:
                return False
        return True

    # Structured search for 6-edge solutions
    # Pattern A: One hub with 6 spokes
    for c in range(n):
        if time.time() > deadline:
            break
        cands = [v for v in range(n) if v != c and abs(v - c) >= 3]
        # Select 6 non-adjacent from cands
        def gen_nonadj(cands, k, start=0, cur=None):
            if cur is None:
                cur = []
            if k == 0:
                yield list(cur)
                return
            if start >= len(cands):
                return
            for i in range(start, len(cands)):
                v = cands[i]
                # Check v is not adjacent to last in cur
                if cur and abs(v - cur[-1]) == 1:
                    continue
                cur.append(v)
                yield from gen_nonadj(cands, k-1, i+1, cur)
                cur.pop()
        
        found = False
        for sel in gen_nonadj(cands, 6):
            if time.time() > deadline:
                break
            edges = [(c, v) for v in sel]
            if check_triangle_free(edges) and check_diameter(edges):
                best_solution = edges
                best_len = 6
                found = True
                break
        if found:
            break

    # Pattern B: Two hubs
    if best_len > 6:
        for c1 in range(n):
            if time.time() > deadline:
                break
            for c2 in range(c1+3, n):
                if time.time() > deadline:
                    break
                # c1: 3 spokes, c2: 3 spokes
                c1_cands = [v for v in range(n) if v != c1 and v != c2 and abs(v-c1)>=3 and abs(v-c2)!=1]
                c2_cands = [v for v in range(n) if v != c1 and v != c2 and abs(v-c2)>=3 and abs(v-c1)!=1]
                
                def gen3(cands):
                    if len(cands) < 3:
                        return
                    for i in range(len(cands)):
                        for j in range(i+1, len(cands)):
                            if abs(cands[i]-cands[j])==1:
                                continue
                            for k in range(j+1, len(cands)):
                                if abs(cands[j]-cands[k])==1 or abs(cands[i]-cands[k])==1:
                                    continue
                                yield (cands[i], cands[j], cands[k])
                
                for s1 in gen3(c1_cands):
                    if time.time() > deadline:
                        break
                    for s2 in gen3(c2_cands):
                        if time.time() > deadline:
                            break
                        # Cross check
                        ok = True
                        for a in s1:
                            for b in s2:
                                if abs(a-b)==1:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            continue
                        edges = [(c1,v) for v in s1] + [(c2,v) for v in s2]
                        if check_triangle_free(edges) and check_diameter(edges):
                            best_solution = edges
                            best_len = 6
                            break
                    if best_len == 6:
                        break
                if best_len == 6:
                    break
            if best_len == 6:
                break

    # Pattern C: Three hubs with 2 spokes each
    if best_len > 6:
        for c1 in range(n):
            if time.time() > deadline:
                break
            for c2 in range(c1+3, n):
                if time.time() > deadline:
                    break
                for c3 in range(c2+3, n):
                    if time.time() > deadline:
                        break
                    hubs = (c1,c2,c3)
                    spokes = []
                    valid = True
                    for h in hubs:
                        cands = [v for v in range(n) if v!=h and all(v!=h2 for h2 in hubs) and abs(v-h)>=3 and all(abs(v-h2)!=1 for h2 in hubs if h2!=h)]
                        sel = None
                        for i in range(len(cands)):
                            for j in range(i+1, len(cands)):
                                if abs(cands[i]-cands[j])!=1:
                                    sel = (cands[i],cands[j])
                                    break
                            if sel:
                                break
                        if sel is None:
                            valid = False
                            break
                        spokes.append(sel)
                    if not valid:
                        continue
                    # Cross adjacency
                    ok = True
                    for i in range(3):
                        for j in range(i+1,3):
                            for a in spokes[i]:
                                for b in spokes[j]:
                                    if abs(a-b)==1:
                                        ok=False
                                        break
                                if not ok: break
                            if not ok: break
                        if not ok: break
                    if not ok:
                        continue
                    edges = []
                    for i,h in enumerate(hubs):
                        for v in spokes[i]:
                            edges.append((h,v))
                    if check_triangle_free(edges) and check_diameter(edges):
                        best_solution = edges
                        best_len = 6
                        break
                if best_len == 6:
                    break
            if best_len == 6:
                break

    # Pattern D: Mix - one hub with 4, one with 2
    if best_len > 6:
        for c1 in range(n):
            if time.time() > deadline:
                break
            for c2 in range(c1+3, n):
                if time.time() > deadline:
                    break
                c1_cands = [v for v in range(n) if v!=c1 and v!=c2 and abs(v-c1)>=3 and abs(v-c2)!=1]
                c2_cands = [v for v in range(n) if v!=c1 and v!=c2 and abs(v-c2)>=3 and abs(v-c1)!=1]
                
                def gen4(cands):
                    if len(cands)<4: return
                    for i in range(len(cands)):
                        for j in range(i+1,len(cands)):
                            if abs(cands[i]-cands[j])==1: continue
                            for k in range(j+1,len(cands)):
                                if abs(cands[j]-cands[k])==1 or abs(cands[i]-cands[k])==1: continue
                                for m in range(k+1,len(cands)):
                                    if abs(cands[k]-cands[m])==1 or abs(cands[j]-cands[m])==1 or abs(cands[i]-cands[m])==1: continue
                                    yield (cands[i],cands[j],cands[k],cands[m])
                
                def gen2(cands):
                    if len(cands)<2: return
                    for i in range(len(cands)):
                        for j in range(i+1,len(cands)):
                            if abs(cands[i]-cands[j])!=1:
                                yield (cands[i],cands[j])
                
                for s1 in gen4(c1_cands):
                    if time.time()>deadline: break
                    for s2 in gen2(c2_cands):
                        if time.time()>deadline: break
                        ok=True
                        for a in s1:
                            for b in s2:
                                if abs(a-b)==1: ok=False; break
                            if not ok: break
                        if not ok: continue
                        edges=[(c1,v) for v in s1]+[(c2,v) for v in s2]
                        if check_triangle_free(edges) and check_diameter(edges):
                            best_solution=edges; best_len=6; break
                    if best_len==6: break
                if best_len==6: break
            if best_len==6: break

    return [[int(u), int(v)] for u, v in best_solution]
