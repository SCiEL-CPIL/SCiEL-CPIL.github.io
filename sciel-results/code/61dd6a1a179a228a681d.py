import time
import itertools
import random

N = 16
ALL_PAIRS = list(itertools.combinations(range(N), 2))
NUM_PAIRS = len(ALL_PAIRS)

def _precompute_q3s():
    q3s = []
    for vs in itertools.combinations(range(N), 8):
        idx = list(vs)
        def gen_matchings(verts):
            if not verts:
                yield []
                return
            first = verts[0]
            rest = verts[1:]
            for i in range(len(rest)):
                partner = rest[i]
                new_rest = rest[:i] + rest[i+1:]
                for sub in gen_matchings(new_rest):
                    yield [(first, partner)] + sub
        for matching in gen_matchings(idx):
            edges = set()
            for a, b in matching:
                edges.add((a, b))
                edges.add((b, a))
            ok = True
            for v in idx:
                deg = 0
                for u in idx:
                    if u != v and (v, u) in edges:
                        deg += 1
                if deg != 3:
                    ok = False
                    break
            if not ok:
                continue
            color = {}
            start_v = idx[0]
            color[start_v] = 0
            stack = [start_v]
            bip = True
            while stack:
                cur = stack.pop()
                for v in idx:
                    if v != cur and (cur, v) in edges:
                        if v in color:
                            if color[v] == color[cur]:
                                bip = False
                                break
                        else:
                            color[v] = color[cur] ^ 1
                            stack.append(v)
                if not bip:
                    break
            if not bip:
                continue
            edge_indices = set()
            for u, v in ALL_PAIRS:
                if (u, v) in edges or (v, u) in edges:
                    edge_indices.add(ALL_PAIRS.index((u, v)))
            q3s.append(frozenset(edge_indices))
    return q3s

Q3S = _precompute_q3s()

def _build_conflicts():
    conflicts = [0] * NUM_PAIRS
    for q3 in Q3S:
        edges_in_q3 = list(q3)
        for a in range(len(edges_in_q3)):
            for b in range(a + 1, len(edges_in_q3)):
                i = edges_in_q3[a]
                j = edges_in_q3[b]
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    return conflicts

CONFLICTS = _build_conflicts()

def _count_bits(x):
    return bin(x).count('1')

def _greedy(order, time_limit=3.0):
    start = time.time()
    selected = 0
    for e in order:
        if time.time() - start > time_limit:
            break
        if not (selected >> e) & 1:
            if CONFLICTS[e] & selected:
                continue
            selected |= (1 << e)
    return selected

def _local_search(selected, time_limit=3.0):
    start = time.time()
    cur = selected
    cur_count = _count_bits(cur)
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        for e in range(NUM_PAIRS):
            if time.time() - start > time_limit:
                break
            if (cur >> e) & 1:
                continue
            to_remove = CONFLICTS[e] & cur
            removed = _count_bits(to_remove)
            if 1 - removed <= 0:
                continue
            new = (cur & ~to_remove) | (1 << e)
            if _count_bits(new) > cur_count:
                cur = new
                cur_count = _count_bits(cur)
                improved = True
    return cur

def construct():
    start = time.time()
    deadline = start + 140.0
    
    def time_left():
        return deadline - time.time()
    
    rng = random.Random(42)
    best = 0
    best_count = 0
    
    for trial in range(500):
        if time_left() < 10.0:
            break
        
        order = list(range(NUM_PAIRS))
        rng.shuffle(order)
        
        sel = _greedy(order, time_limit=3.0)
        cnt = _count_bits(sel)
        
        if cnt > best_count:
            best_count = cnt
            best = sel
        
        sel2 = _local_search(sel, time_limit=3.0)
        cnt2 = _count_bits(sel2)
        if cnt2 > best_count:
            best_count = cnt2
            best = sel2
    
    result = []
    for e in range(NUM_PAIRS):
        if (best >> e) & 1:
            u, v = ALL_PAIRS[e]
            result.append([u, v])
    
    return result
