import time
import random
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

T0 = time.time()
DEADLINE = T0 + 80.0   # Safety margin vs 90s limit

def construct():
    t0 = time.time()
    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    adj_list = [sorted(adj[i]) for i in range(n)]

    def time_up():
        return time.time() > DEADLINE

    def fallback():
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    best = fallback()
    best_size = 2

    # BFS shortest path from s to t avoiding `blocked` (except s,t)
    def bfs_path(s, t, blocked):
        if s == t:
            return [s]
        parent = {s: None}
        q = deque([s])
        while q:
            curr = q.popleft()
            if curr == t:
                break
            for nb in adj_list[curr]:
                if nb in parent:
                    continue
                if nb != t and nb in blocked:
                    continue
                parent[nb] = curr
                if nb == t:
                    q.clear()
                    break
                q.append(nb)
        if t not in parent:
            return None
        path = []
        node = t
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path

    # Check if a set of branches has valid disjoint paths
    def find_paths(branches, max_backtrack=20000):
        bs = set(branches)
        sorted_bs = sorted(branches)
        pairs = list(combinations(sorted_bs, 2))
        if not pairs:
            return []

        # Heuristic ordering: pairs with fewer common neighbors are "tighter"
        def pair_tightness(p):
            u, v = p
            return len((adj[u] & adj[v]) - bs)
        
        # Sort by tightness (ascending) - try constrained ones first
        pairs.sort(key=pair_tightness)

        chosen = [None] * len(pairs)
        bt = [0]
        
        # Use a global counter for backtracking steps to enforce time/step limit
        def backtrack(idx, used_internal):
            if bt[0] > max_backtrack or time_up():
                return False
            if idx == len(pairs):
                return True
            u, v = pairs[idx]
            blocked = used_internal | bs
            path = bfs_path(u, v, blocked)
            if path is None:
                return False
            internal = set(path[1:-1])
            if internal & used_internal or internal & bs:
                return False
            
            chosen[idx] = path
            for vn in internal:
                used_internal.add(vn)
            
            if backtrack(idx + 1, used_internal):
                return True
            
            for vn in internal:
                used_internal.discard(vn)
            chosen[idx] = None
            bt[0] += 1
            return False

        if backtrack(0, set()):
            return chosen
        
        # If standard order failed, try a few random orders
        rng = random.Random(42)
        for _ in range(5):
            if time_up():
                return None
            pairs_copy = pairs[:]
            rng.shuffle(pairs_copy)
            # Re-sort by tightness but with random tie-breaking
            pairs_copy.sort(key=lambda p: (pair_tightness(p), rng.random()))
            
            chosen = [None] * len(pairs_copy)
            bt[0] = 0
            
            def backtrack_rand(idx, used_internal):
                if bt[0] > max_backtrack or time_up():
                    return False
                if idx == len(pairs_copy):
                    return True
                u, v = pairs_copy[idx]
                blocked = used_internal | bs
                path = bfs_path(u, v, blocked)
                if path is None:
                    return False
                internal = set(path[1:-1])
                if internal & used_internal or internal & bs:
                    return False
                
                chosen[idx] = path
                for vn in internal:
                    used_internal.add(vn)
                
                if backtrack_rand(idx + 1, used_internal):
                    return True
                
                for vn in internal:
                    used_internal.discard(vn)
                chosen[idx] = None
                bt[0] += 1
                return False

            if backtrack_rand(0, set()):
                # Map back to original pair order? No, we just need valid paths.
                # The returned paths list corresponds to pairs_copy order.
                # We need to return paths in the order of `pairs` (or any order, as long as endpoints match).
                # The task says "one simple path for each unordered pair". The order in the list doesn't matter for validity,
                # but we should return them clearly.
                return chosen
        return None

    # Bron-Kerbosch to find maximal cliques
    def bron_kerbosch(R, Pset, X, result, tlimit):
        if time.time() > tlimit:
            return
        if not Pset and not X:
            if len(R) > best_size:
                result.append(list(R))
            return
        candidates = Pset | X
        if candidates:
            pivot = max(candidates, key=lambda v: len(Pset & adj[v]))
            to_try = Pset - adj[pivot]
        else:
            to_try = Pset
        for v in list(to_try):
            bron_kerbosch(R | {v}, Pset & adj[v], X & adj[v], result, tlimit)
            Pset.discard(v)
            X.add(v)

    cliques = []
    # Allocate 10s for clique enumeration
    bron_kerbosch(set(), set(range(n)), set(), cliques, time.time() + 10.0)
    cliques.sort(key=len, reverse=True)

    # Try cliques first
    for clique in cliques:
        if time_up():
            break
        if len(clique) <= best_size:
            break
        paths = find_paths(clique)
        if paths is not None:
            best = {'branches': [int(x) for x in sorted(clique)],
                    'paths': [[int(x) for x in p] for p in paths]}
            best_size = len(clique)
            # If we found a valid clique, we can try to extend it
            # But first, let's see if we can beat it with non-clique extensions
            
    # Greedy Extension from the best found set
    if best_size < n and not time_up():
        current_branches = set(best['branches'])
        
        # Candidates sorted by connectivity to current branches
        def score_vertex(v):
            return sum(1 for b in current_branches if v in adj[b])
        
        candidates = [v for v in range(n) if v not in current_branches]
        candidates.sort(key=score_vertex, reverse=True)

        improved = True
        while improved and not time_up():
            improved = False
            for v in candidates:
                if time_up():
                    break
                new_set = current_branches | {v}
                paths = find_paths(new_set)
                if paths is not None:
                    best = {'branches': [int(x) for x in sorted(new_set)],
                            'paths': [[int(x) for x in p] for p in paths]}
                    best_size = len(new_set)
                    current_branches = new_set
                    # Re-sort candidates
                    candidates = [x for x in range(n) if x not in current_branches]
                    candidates.sort(key=score_vertex, reverse=True)
                    improved = True
                    break # Restart loop with new set

    # Randomized Restart to escape local optima
    if not time_up() and best_size < n:
        rng = random.Random(12345)
        for _ in range(100):
            if time_up():
                break
            # Start from a random subset of the current best or a random clique
            start_size = max(2, best_size - rng.randint(0, 2))
            # Pick a random clique or random set
            if cliques and len(cliques[0]) >= start_size:
                base_clique = cliques[0]
                start_set = set(rng.sample(base_clique, min(start_size, len(base_clique))))
            else:
                start_set = set(rng.sample(range(n), start_size))
            
            # Greedy fill
            local = start_set
            def sc(v):
                return sum(1 for b in local if v in adj[b])
            
            cands = [v for v in range(n) if v not in local]
            cands.sort(key=sc, reverse=True)
            rng.shuffle(cands) # Add randomness
            
            changed = True
            while changed and not time_up():
                changed = False
                for v in cands:
                    if time_up():
                        break
                    ns = local | {v}
                    paths = find_paths(ns)
                    if paths is not None:
                        local = ns
                        changed = True
                        if len(local) > best_size:
                            best = {'branches': [int(x) for x in sorted(local)],
                                    'paths': [[int(x) for x in p] for p in paths]}
                            best_size = len(local)
                        # Re-sort
                        cands = [x for x in range(n) if x not in local]
                        cands.sort(key=sc, reverse=True)
                        break

    return best

# Test (commented out for evaluation)
# print(construct())
