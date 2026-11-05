"""Optimized local search for Q3-free graph on 16 vertices."""
import time
import random
import itertools

Q3_EDGES = []
for i in range(8):
    for j in range(i+1, 8):
        if (i ^ j) in (1, 2, 4):
            Q3_EDGES.append((i, j))
Q3_ADJ = [set() for _ in range(8)]
for u, v in Q3_EDGES:
    Q3_ADJ[u].add(v)
    Q3_ADJ[v].add(u)

ALL_8_SUBSETS = list(itertools.combinations(range(16), 8))

def check_q3_subset_mask(subset, adj):
    sl = list(subset)
    for i in range(8):
        v = sl[i]
        mask = adj[v]
        d = 0
        for j in range(8):
            if i != j and (mask & (1 << sl[j])):
                d += 1
        if d < 3:
            return False
    assign = [-1]*8
    used = [False]*8
    def bt(qv):
        if qv == 8:
            for u, v in Q3_EDGES:
                if not (adj[assign[u]] & (1 << assign[v])):
                    return False
            return True
        for sv in range(8):
            if used[sv]:
                continue
            ok = True
            for nu in Q3_ADJ[qv]:
                if nu < qv and assign[nu] >= 0:
                    if not (adj[assign[nu]] & (1 << sl[sv])):
                        ok = False
                        break
            if ok:
                assign[qv] = sv
                used[sv] = True
                if bt(qv+1):
                    return True
                used[sv] = False
                assign[qv] = -1
        return False
    return bt(0)

def construct():
    start_time = time.time()
    deadline = start_time + 95
    edge_check_deadline = 0.1  # max time per edge check
    
    def build_adj(edges):
        adj = [0]*16
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    best_edges = set()
    for a in range(3):
        for b in range(3, 16):
            best_edges.add((a, b))
    best_count = len(best_edges)
    best_adj = build_adj(best_edges)
    
    all_pairs = [(i, j) for i in range(16) for j in range(i+1, 16)]
    
    for restart in range(5000):
        if time.time() > deadline:
            break
        
        if restart == 0:
            cur = set(best_edges)
        else:
            cur = set(best_edges)
            el = list(cur)
            random.shuffle(el)
            for _ in range(min(random.randint(1, 4), len(el))):
                cur.discard(el.pop())
            random.shuffle(all_pairs)
            added = 0
            for e in all_pairs:
                if added >= random.randint(1, 3):
                    break
                if e not in cur:
                    cur.add(e)
                    added += 1
        
        cur_adj = build_adj(cur)
        cur_count = len(cur)
        
        random.shuffle(all_pairs)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for e in all_pairs:
                if time.time() > deadline:
                    break
                if e in cur:
                    continue
                u, v = e
                found = False
                check_start = time.time()
                for subset in ALL_8_SUBSETS:
                    if time.time() - check_start > edge_check_deadline:
                        break
                    if u not in subset or v not in subset:
                        continue
                    old_u = cur_adj[u]
                    old_v = cur_adj[v]
                    cur_adj[u] |= (1 << v)
                    cur_adj[v] |= (1 << u)
                    if check_q3_subset_mask(subset, cur_adj):
                        cur_adj[u] = old_u
                        cur_adj[v] = old_v
                        found = True
                        break
                    cur_adj[u] = old_u
                    cur_adj[v] = old_v
                if not found:
                    cur.add(e)
                    cur_adj[u] |= (1 << v)
                    cur_adj[v] |= (1 << u)
                    cur_count += 1
                    changed = True
                    break
        
        if cur_count > best_count:
            best_count = cur_count
            best_edges = set(cur)
            best_adj = cur_adj
    
    return [[u, v] for u, v in best_edges if u < v]
