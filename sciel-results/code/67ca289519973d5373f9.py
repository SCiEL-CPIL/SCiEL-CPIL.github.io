"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    _t0 = time.time()
    _DEADLINE = 120.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def build_partition(n, k, p, seed, budget):
        # Cross edges: one clique per I-vertex w = {w} U N_C(w)  (covers all cross edges of w)
        neigh = instance(n, k, p, seed)  # sorted edge list
        # Build adjacency of C (u<k) to I (v>=k)
        adj = [set() for _ in range(k)]
        for (u, v) in neigh:
            if u < k and v >= k:
                adj[u].add(v)
        # N_C(w) for each w in I
        nc = [set() for _ in range(n - k)]
        for u in range(k):
            for v in adj[u]:
                nc[v - k].add(u)

        cliques = []
        for i in range(n - k):
            s = sorted(nc[i])
            if len(s) >= 1:
                cliques.append([i + k] + s)
            # if len(s)==0, w isolated -> no cross edges, no clique needed

        # Inner: partition C (0..k-1) into cliques (subsets) of size>=2 covering all inner edges exactly once
        def inner_count(parts):
            return sum(len(p) for p in parts)

        # Greedy partition of C into large cliques (size>=2)
        def greedy_partition():
            used = [False] * k
            parts = []
            for _ in range(k // 2):
                best = None
                bestsz = 1
                for i in range(k):
                    if used[i]:
                        continue
                    sz = 0
                    for j in range(i + 1, k):
                        if not used[j]:
                            sz += 1
                    if sz > bestsz:
                        bestsz = sz
                        best = i
                if best is None or bestsz < 2:
                    break
                part = []
                for j in range(best, k):
                    if not used[j]:
                        part.append(j)
                        used[j] = True
                parts.append(part)
            return parts

        parts = greedy_partition()
        rng2 = Random(seed + 100000)

        # Local search: minimize number of parts (each size>=2), keep a valid partition
        def valid_partition(parts):
            seen = [False] * k
            for p in parts:
                if len(p) < 2:
                    return False
                for x in p:
                    if seen[x]:
                        return False
                    seen[x] = True
            return all(seen)

        def try_move(parts):
            # move one vertex from a part (size>=3) to another part
            if len(parts) < 2:
                return False
            for a in range(len(parts)):
                if len(parts[a]) < 3:
                    continue
                for b in range(len(parts)):
                    if a == b:
                        continue
                    for idx in range(len(parts[a])):
                        x = parts[a][idx]
                        parts[a].pop(idx)
                        parts[b].append(x)
                        if valid_partition(parts):
                            return True
                        parts[b].pop()
                        parts[a].insert(idx, x)
            return False

        # local search within budget
        steps = 0
        max_steps = 400000
        while time.time() - _t0 < budget and steps < max_steps:
            if len(parts) <= 2:
                break
            if not try_move(parts):
                # try merging two parts if possible? (can't, union not necessarily clique, but C is clique so union IS a clique!)
                # Since C is a full clique, ANY union of parts is a clique. So merge any two parts -> fewer parts, still valid!
                merged = parts[0] + parts[1]
                parts = [merged] + parts[2:]
            steps += 1

        # Because C is a full clique, the optimal is: as few parts as possible with each size>=2.
        # Minimum parts = ceil(k/2) if we can make sizes >=2. But we must cover ALL inner edges exactly once,
        # which is satisfied by ANY partition into >=2 sets (since C clique, each part is a clique, disjoint, union C).
        # So optimal inner = ceil(k/2) parts. Achieve directly:
        best_parts = None
        m = (k + 1) // 2  # ceil(k/2)
        # build m parts each size >=2, sizes sum to k
        sizes = [2] * m
        rem = k - 2 * m
        for i in range(rem):
            sizes[i] += 1
        idx = 0
        pp = []
        for sz in sizes:
            part = list(range(idx, idx + sz))
            idx += sz
            pp.append(part)
        if valid_partition(pp):
            best_parts = pp

        if best_parts is not None:
            parts = best_parts

        for p in parts:
            cliques.append(sorted(p))

        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    for spec in specs:
        n, k, p, seed = spec
        rem = _DEADLINE - time.time()
        budget = max(1.0, rem - 2.0)
        cliques = build_partition(n, k, p, seed, budget)
        result.append(cliques)

    return result
# EVOLVE-BLOCK-END
