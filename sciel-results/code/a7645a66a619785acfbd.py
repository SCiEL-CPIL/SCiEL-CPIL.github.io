"""Embed trees T1..T20 into K20 edge-disjointly, maximizing total edges."""
import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 140.0
    trees = instance_trees()
    n = 20
    
    # Order trees by size descending
    order = list(range(n))
    order.sort(key=lambda i: len(trees[i]), reverse=True)
    
    used_edges = set()
    result = [None] * n
    best_count = 0
    best_result = [None] * n
    
    def embed_tree(k, edges, used, rng=None):
        if time.time() > deadline:
            return None
        if k == 1:
            return [0]
        
        deg = [0] * k
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
        vorder = list(range(k))
        vorder.sort(key=lambda v: -deg[v])
        
        mapping = {}
        
        def dfs(pos):
            if time.time() > deadline:
                return False
            if pos == k:
                for a, b in edges:
                    e = frozenset((mapping[a], mapping[b]))
                    if e in used:
                        return False
                return True
            v = vorder[pos]
            candidates = list(range(n))
            if rng:
                rng.shuffle(candidates)
            for w in candidates:
                if w in mapping.values():
                    continue
                ok = True
                for a, b in edges:
                    if a == v and b in mapping:
                        e = frozenset((w, mapping[b]))
                        if e in used:
                            ok = False
                            break
                    elif b == v and a in mapping:
                        e = frozenset((w, mapping[a]))
                        if e in used:
                            ok = False
                            break
                if not ok:
                    continue
                mapping[v] = w
                if dfs(pos + 1):
                    return True
                del mapping[v]
            return False
        
        if dfs(0):
            return list(mapping[i] for i in range(k))
        return None
    
    def backtrack(idx, count, rng):
        nonlocal best_count, best_result
        if idx == n:
            if count > best_count:
                best_count = count
                best_result = result[:]
            return
        if time.time() > deadline:
            return
        
        tree_idx = order[idx]
        k = tree_idx + 1
        edges = trees[tree_idx]
        
        if k == 1:
            result[tree_idx] = None
            backtrack(idx + 1, count, rng)
            return
        
        res = embed_tree(k, edges, used_edges, rng)
        if res is not None:
            result[tree_idx] = res
            for a, b in edges:
                used_edges.add(frozenset((res[a], res[b])))
            backtrack(idx + 1, count + len(edges), rng)
            for a, b in edges:
                used_edges.discard(frozenset((res[a], res[b])))
            result[tree_idx] = None
        else:
            result[tree_idx] = None
            backtrack(idx + 1, count, rng)
    
    # Try multiple random orderings
    rng = Random(42)
    for attempt in range(50):
        if time.time() > deadline:
            break
        # Shuffle the order of trees (keeping larger ones first with some randomness)
        cur_order = order[:]
        # Add some randomness
        for i in range(len(cur_order) - 1, 0, -1):
            j = rng.randint(0, i)
            cur_order[i], cur_order[j] = cur_order[j], cur_order[i]
        # Re-sort but with random tie-breaking
        order = cur_order
        used_edges.clear()
        result = [None] * n
        backtrack(0, 0, rng)
    
    final = []
    for i in range(n):
        if best_result[i] is None:
            final.append(None)
        else:
            final.append([int(x) for x in best_result[i]])
    return final
