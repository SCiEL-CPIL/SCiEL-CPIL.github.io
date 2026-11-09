import time
import random
from collections import defaultdict
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, deadline):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        def greedy_path_init(seed_offset=0):
            rng = random.Random(seed_offset)
            remaining = set(edges)
            paths = []
            
            while remaining:
                deg = defaultdict(int)
                for u, v in remaining:
                    deg[u] += 1
                    deg[v] += 1
                if not deg: break
                candidates = sorted(deg.keys(), key=lambda x: (-deg[x], rng.random()))
                start = candidates[0]
                
                path = [start]
                pverts = {start}
                pedges = set()
                
                while True:
                    best_next = None
                    best_score = float('inf')
                    nbrs = list(adj[start])
                    rng.shuffle(nbrs)
                    for v in nbrs:
                        e = (min(start, v), max(start, v))
                        if e in remaining and v not in pverts:
                            moves = 0
                            for w in adj[v]:
                                e2 = (min(v, w), max(v, w))
                                if e2 in remaining and w not in pverts and w != start:
                                    moves += 1
                            score = moves + rng.random() * 0.1
                            if score < best_score:
                                best_score = score
                                best_next = v
                    if best_next is None:
                        break
                    path.append(best_next)
                    pverts.add(best_next)
                    e = (min(start, best_next), max(start, best_next))
                    pedges.add(e)
                    start = best_next
                    
                if len(path) > 1:
                    paths.append(path)
                    for e in pedges:
                        remaining.discard(e)
                else:
                    if remaining:
                        e = next(iter(remaining))
                        paths.append(list(e))
                        remaining.discard(e)
            return paths

        def try_merge(paths):
            for i in range(len(paths)):
                for j in range(i+1, len(paths)):
                    p1 = paths[i]
                    p2 = paths[j]
                    if len(p1) < 2 or len(p2) < 2:
                        continue
                    set1 = set(p1)
                    set2 = set(p2)
                    common = set1 & set2
                    if len(common) != 1:
                        continue
                    shared = list(common)[0]
                    is_ep1 = (p1[0] == shared or p1[-1] == shared)
                    is_ep2 = (p2[0] == shared or p2[-1] == shared)
                    if is_ep1 and is_ep2:
                        if p1[-1] == shared and p2[0] == shared:
                            new_p = p1 + p2[1:]
                        elif p1[0] == shared and p2[-1] == shared:
                            new_p = p2 + p1[1:]
                        elif p1[0] == shared and p2[0] == shared:
                            new_p = p1[::-1] + p2[1:]
                        elif p1[-1] == shared and p2[-1] == shared:
                            new_p = p2 + p1[::-1][1:]
                        else:
                            continue
                        if len(new_p) == len(set(new_p)):
                            new_paths = [p for k, p in enumerate(paths) if k != i and k != j]
                            new_paths.append(new_p)
                            return new_paths
            return None

        def local_search(paths, deadline):
            changed = True
            while changed and time.time() < deadline:
                changed = False
                new_paths = try_merge(paths)
                if new_paths is not None:
                    paths = new_paths
                    changed = True
            return paths

        def recombine(paths, deadline):
            # Try to split paths and recombine to reduce count further
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for i in range(len(paths)):
                    if len(paths[i]) < 3: continue
                    p = paths[i]
                    # Try splitting at each possible internal point
                    for k in range(1, len(p)-1):
                        p1 = p[:k+1]
                        p2 = p[k:]
                        if len(p1) < 2 or len(p2) < 2: continue
                        
                        # Try to merge p1 with other paths
                        for j in range(len(paths)):
                            if j == i: continue
                            q = paths[j]
                            if len(q) < 2: continue
                            set_p1 = set(p1)
                            set_q = set(q)
                            common = set_p1 & set_q
                            if len(common) != 1: continue
                            shared = list(common)[0]
                            is_ep1 = (p1[0] == shared or p1[-1] == shared)
                            is_ep2 = (q[0] == shared or q[-1] == shared)
                            if is_ep1 and is_ep2:
                                if p1[-1] == shared and q[0] == shared:
                                    new_p = p1 + q[1:]
                                elif p1[0] == shared and q[-1] == shared:
                                    new_p = q + p1[1:]
                                elif p1[0] == shared and q[0] == shared:
                                    new_p = p1[::-1] + q[1:]
                                elif p1[-1] == shared and q[-1] == shared:
                                    new_p = q + p1[::-1][1:]
                                else: continue
                                if len(new_p) == len(set(new_p)):
                                    new_paths = [paths[x] for x in range(len(paths)) if x != i and x != j]
                                    new_paths.append(p2)
                                    new_paths.append(new_p)
                                    # Check if this leads to further merges
                                    temp = local_search(new_paths, deadline - 1)
                                    if len(temp) < len(paths):
                                        paths = temp
                                        improved = True
                                        break
                            # Also try merging p2 with other paths
                            if not improved:
                                set_p2 = set(p2)
                                common = set_p2 & set_q
                                if len(common) != 1: continue
                                shared = list(common)[0]
                                is_ep1 = (p2[0] == shared or p2[-1] == shared)
                                is_ep2 = (q[0] == shared or q[-1] == shared)
                                if is_ep1 and is_ep2:
                                    if p2[-1] == shared and q[0] == shared:
                                        new_p = p2 + q[1:]
                                    elif p2[0] == shared and q[-1] == shared:
                                        new_p = q + p2[1:]
                                    elif p2[0] == shared and q[0] == shared:
                                        new_p = p2[::-1] + q[1:]
                                    elif p2[-1] == shared and q[-1] == shared:
                                        new_p = q + p2[::-1][1:]
                                    else: continue
                                    if len(new_p) == len(set(new_p)):
                                        new_paths = [paths[x] for x in range(len(paths)) if x != i and x != j]
                                        new_paths.append(p1)
                                        new_paths.append(new_p)
                                        temp = local_search(new_paths, deadline - 1)
                                        if len(temp) < len(paths):
                                            paths = temp
                                            improved = True
                                            break
                    if improved:
                        break
            return paths

        # Main: run multiple random inits, keep best, apply recombine
        best_paths = None
        best_count = float('inf')
        
        for attempt in range(15):
            if time.time() > deadline - 3:
                break
            paths = greedy_path_init(attempt * 7 + 1)
            paths = local_search(paths, deadline - 2)
            paths = recombine(paths, deadline - 1)
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
                # Check lower bound
                deg = defaultdict(int)
                for u, v in edges:
                    deg[u] += 1
                    deg[v] += 1
                lb = sum(1 for d in deg.values() if d % 2 == 1) // 2
                if best_count <= lb:
                    break

        if best_paths is None:
            best_paths = greedy_path_init(0)
        return best_paths

    result = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges, DEADLINE)
        result.append(paths)
    return result
