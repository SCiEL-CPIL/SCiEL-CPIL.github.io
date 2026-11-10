import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj):
        n = len(adj)
        def bk(R, P, X):
            if not P and not X:
                yield R
                return
            U = P | X
            if not U:
                return
            u = max(U, key=lambda x: len(P & adj[x]))
            for v in list(P - adj[u]):
                yield from bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        yield from bk(frozenset(), frozenset(range(n)), frozenset())

    def solve_instance(adj, n, time_limit):
        cliques = [frozenset(c) for c in bron_kerbosch(adj) if len(c) >= 2]
        if not cliques:
            return []
        
        m = len(cliques)
        all_mask = (1 << m) - 1
        
        # Precompute vertex coverage bitmasks
        vc = [0] * n
        for v in range(n):
            bit = 1 << v
            for i, c in enumerate(cliques):
                if v in c:
                    vc[v] |= (1 << i)
        
        # Greedy solution for upper bound
        def greedy():
            remaining = all_mask
            sol = []
            while remaining:
                best_v, best_cov = -1, 0
                for v in range(n):
                    cov = (vc[v] & remaining).bit_count()
                    if cov > best_cov:
                        best_cov, best_v = cov, v
                if best_cov == 0:
                    break
                sol.append(best_v)
                remaining &= ~vc[best_v]
            return sol
        
        best_sol = greedy()
        best_size = len(best_sol)
        
        # B&B with strong lower bound
        def lower_bound(rem):
            if rem == 0:
                return 0
            # Greedy packing: repeatedly pick vertex covering most, count
            lb = 0
            r = rem
            while r:
                best_v, best_cov = -1, 0
                for v in range(n):
                    cov = (vc[v] & r).bit_count()
                    if cov > best_cov:
                        best_cov, best_v = cov, v
                if best_cov == 0:
                    return float('inf')
                lb += 1
                r &= ~vc[best_v]
            return lb
        
        # For B&B: pick most constrained clique, try its vertices
        # Precompute clique membership as list of vertex lists
        clique_verts = [list(c) for c in cliques]
        
        def bb(rem, cur_size, cur_set):
            nonlocal best_sol, best_size
            if time.time() > time_limit:
                return
            if rem == 0:
                if cur_size < best_size:
                    best_size = cur_size
                    best_sol = list(cur_set)
                return
            if cur_size >= best_size:
                return
            lb = lower_bound(rem)
            if cur_size + lb >= best_size:
                return
            
            # Find most constrained uncovered clique
            best_idx, best_cnt = -1, n + 1
            for i in range(m):
                if (rem >> i) & 1:
                    cnt = len(clique_verts[i])
                    if cnt < best_cnt:
                        best_cnt = cnt
                        best_idx = i
                    if cnt == 2:
                        break
            
            if best_idx < 0:
                return
            
            verts = clique_verts[best_idx]
            # Sort by coverage descending
            verts.sort(key=lambda v: (vc[v] & rem).bit_count(), reverse=True)
            
            for v in verts:
                new_rem = rem & ~vc[v]
                cur_set.append(v)
                bb(new_rem, cur_size + 1, cur_set)
                cur_set.pop()
                if best_size <= cur_size:
                    break
        
        bb(all_mask, 0, [])
        
        # Local search: try removing vertices, then 1-for-1 swaps
        def is_valid(s):
            cov = 0
            for v in s:
                cov |= vc[v]
            return cov == all_mask
        
        if not is_valid(best_sol):
            best_sol = greedy()
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try removals
            for v in list(best_sol):
                trial = [x for x in best_sol if x != v]
                if is_valid(trial):
                    best_sol = trial
                    improved = True
                    break
            if improved:
                continue
            # Try 1-for-1 swaps
            for i, v in enumerate(best_sol):
                if time.time() > time_limit:
                    break
                others_cov = 0
                for j, w in enumerate(best_sol):
                    if j != i:
                        others_cov |= vc[w]
                unique = vc[v] & ~others_cov
                for w in range(n):
                    if w in best_sol:
                        continue
                    if (vc[w] & unique) == unique:
                        trial = best_sol[:i] + [w] + best_sol[i+1:]
                        if is_valid(trial):
                            best_sol = trial
                            improved = True
                            break
                if improved:
                    break
        
        return best_sol

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 5:
            adj = random_graph(n, p, seed)
            touched = set()
            for u in range(n):
                for v in adj[u]:
                    touched.add(u)
                    touched.add(v)
            out.append(sorted(touched) if touched else list(range(n)))
            continue
        
        adj = random_graph(n, p, seed)
        remaining_time = deadline - time.time() - 2
        sol = solve_instance(adj, n, time.time() + max(remaining_time, 1))
        out.append(sorted(sol))
    
    return out
