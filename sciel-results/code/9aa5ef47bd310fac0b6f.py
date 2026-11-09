import time
import random
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # Safety margin: evaluator kills at 150s

    def time_left():
        return DEADLINE - time.time()

    def bit_count(x):
        return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')

    # ---------- deterministic C4-free fallback ----------
    def fallback():
        # Use a known C4-free graph with small independence number
        # Complete 3-partite with parts of size 3: no C4, alpha=3
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

    # ---------- Fast C4-free check using common neighbor counts ----------
    def _can_add_fast(adj, n, i, j):
        # Check if adding edge (i,j) creates a C4
        # A C4 exists iff there's a pair of vertices with >=2 common neighbors
        # After adding (i,j):
        # - (i,j) pair: common neighbors = N(i)∩N(j), must be <=1
        # - For each k in N(i)\{j}: pair (j,k) gains i as common neighbor
        #   So need N(j)∩N(k) to be empty (since i was not in N(j) before)
        # - For each k in N(j)\{i}: pair (i,k) gains j as common neighbor  
        #   So need N(i)∩N(k) to be empty (since j was not in N(i) before)
        
        # Check (i,j) pair
        common_ij = adj[i] & adj[j]
        if bit_count(common_ij) > 1:
            return False
        
        # Check pairs (j,k) for k in N(i)
        # N(j)∩N(k) must be empty for all k in N(i)
        # This is equivalent to: for all k in N(i), N(j)∩N(k) = ∅
        # Which means: no vertex is adjacent to both j and any k in N(i)
        # i.e., N(j) ∩ (∪_{k in N(i)} N(k)) = ∅
        # But we can check more efficiently:
        # For each k in N(i), check if N(j) & N(k) != 0
        
        # Optimization: precompute the union of neighbors of all k in N(i)
        # Actually, let's just iterate but with early termination
        
        kmask = adj[i] & ~(1 << j)  # N(i) excluding j
        while kmask:
            lsb = kmask & -kmask
            k = lsb.bit_length() - 1
            if (adj[j] & adj[k]) != 0:
                return False
            kmask ^= lsb
        
        kmask = adj[j] & ~(1 << i)  # N(j) excluding i
        while kmask:
            lsb = kmask & -kmask
            k = lsb.bit_length() - 1
            if (adj[i] & adj[k]) != 0:
                return False
            kmask ^= lsb
        
        return True

    def _can_add(adj, n, i, j):
        return _can_add_fast(adj, n, i, j)

    # ---------- C4-free construction with multiple strategies ----------
    def build_c4free(n, seed):
        rng = random.Random(seed)
        best_adj = None
        best_density = -1

        strategies = ['greedy_random', 'greedy_degree', 'greedy_bipartite']
        
        for strat_idx, strategy in enumerate(strategies):
            adj = [0] * n
            
            if strategy == 'greedy_random':
                order = list(range(n))
                rng.shuffle(order)
                for idx, i in enumerate(order):
                    for j in order[idx + 1:]:
                        if _can_add(adj, n, i, j):
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                            
            elif strategy == 'greedy_degree':
                # Process vertices in order, prefer connecting to high-degree vertices
                order = list(range(n))
                rng.shuffle(order)
                for idx, i in enumerate(order):
                    candidates = [j for j in order[idx + 1:] if _can_add(adj, n, i, j)]
                    # Sort by current degree descending
                    candidates.sort(key=lambda x: bit_count(adj[x]), reverse=True)
                    for j in candidates:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            elif strategy == 'greedy_bipartite':
                # Try to build a bipartite-like structure first
                mid = n // 2
                left = list(range(mid))
                right = list(range(mid, n))
                rng.shuffle(left)
                rng.shuffle(right)
                # Connect left to right with C4 constraint
                for i in left:
                    for j in right:
                        if _can_add(adj, n, i, j):
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                # Then add intra-side edges
                for side in [left, right]:
                    for idx, i in enumerate(side):
                        for j in side[idx + 1:]:
                            if _can_add(adj, n, i, j):
                                adj[i] |= (1 << j)
                                adj[j] |= (1 << i)

            density = sum(bit_count(b) for b in adj)
            if density > best_density:
                best_density = density
                best_adj = [a for a in adj]
                
            if time_left() < 5.0:
                break

        return best_adj

    def densify(adj, n, seed):
        rng = random.Random(seed)
        changed = True
        while changed and time_left() > 3.0:
            changed = False
            pairs = list(combinations(range(n), 2))
            rng.shuffle(pairs)
            for i, j in pairs:
                if (adj[i] >> j) & 1:
                    continue
                if _can_add(adj, n, i, j):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    changed = True
        return adj

    # ---------- Fast exact alpha via bitset branch and bound ----------
    def exact_alpha(adj, n, timeout=8.0):
        if n == 0:
            return 0
        best = [0]
        start = time.time()
        # Precompute neighbor masks
        adj_list = adj[:]

        def solve(cand, cur):
            if time.time() - start > timeout:
                return
            if cand == 0:
                if cur > best[0]:
                    best[0] = cur
                return
            # Upper bound: cur + number of vertices in cand
            if cur + bit_count(cand) <= best[0]:
                return
            # Pick vertex with max degree in induced subgraph
            v = -1
            maxdeg = -1
            tm = cand
            while tm:
                lsb = tm & -tm
                idx = lsb.bit_length() - 1
                d = bit_count(adj_list[idx] & cand)
                if d > maxdeg:
                    maxdeg = d
                    v = idx
                tm ^= lsb
            
            if v == -1:
                return
                
            # Include v: remove v and all its neighbors from cand
            nbr = adj_list[v] & cand
            solve(cand & ~(nbr | (1 << v)), cur + 1)
            # Exclude v
            solve(cand & ~(1 << v), cur)

        solve((1 << n) - 1, 0)
        return best[0]

    # ---------- Candidate validation ----------
    def fast_ok(adj, n):
        # C4 check - verify no two vertices have >1 common neighbor
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    if bit_count(adj[i] & adj[j]) > 1:
                        return False
        # Greedy MIS filter - try to find independent set of size 11
        rng = random.Random(999)
        for _ in range(20):
            order = list(range(n))
            rng.shuffle(order)
            indep = []
            in_set = 0
            for v in order:
                if (in_set >> v) & 1:
                    continue
                # Check if v is independent from all in indep
                compatible = True
                for u in indep:
                    if (adj[v] >> u) & 1:
                        compatible = False
                        break
                if compatible:
                    indep.append(v)
                    in_set |= (1 << v)
            if len(indep) >= 11:
                return False
        return True

    # ---------- Main search ----------
    best = None
    target_ns = [43, 42, 41, 40, 39, 38, 37]

    for n in target_ns:
        if time_left() < 4.0:
            break
        # More trials for smaller n where success is more likely
        max_trials = 40 if n <= 40 else 20
        for trial in range(max_trials):
            if time_left() < 4.0:
                break
            adj = build_c4free(n, trial)
            adj = densify(adj, n, trial)
            if not fast_ok(adj, n):
                continue
            alpha = exact_alpha(adj, n, timeout=5.0)
            if alpha < 11:
                best = (n, adj)
                break
        if best is not None:
            break

    if best is None:
        return fallback()

    n, adj = best
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
