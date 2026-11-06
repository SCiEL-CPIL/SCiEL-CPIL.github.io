"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    n = 128
    # Precompute gcd table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Build conflict graph: edge (u,v) exists if there's w != u,v 
    # with gcd(u,v) == gcd(u,w) == gcd(v,w)
    # For efficiency, for each pair (u,v), check if the set of w 
    # satisfying the condition is non-empty
    # 
    # Actually, let's think differently. A triple (a,b,c) is forbidden if
    # gcd(a,b) = gcd(a,c) = gcd(b,c) = d.
    # This means a = d*a', b = d*b', c = d*c' where gcd(a',b')=gcd(a',c')=gcd(b',c')=1.
    # So we need to avoid: for any d, and any pairwise coprime triple (a',b',c'),
    # the numbers d*a', d*b', d*c' cannot all be in our set.
    #
    # Conflict graph approach: two numbers u,v conflict if they "can form a forbidden
    # triple with some w". But that's not quite right for MIS.
    #
    # Better: Build a hypergraph where each forbidden triple is a hyperedge.
    # We want the largest set with no hyperedge fully contained.
    # This is equivalent to: build a conflict graph where u~v if there EXISTS w
    # such that {u,v,w} is a forbidden triple. Then any independent set in this graph
    # has no forbidden triple (because if u,v,w were all in the set and formed a 
    # forbidden triple, then u~v, v~w, u~w would all be edges, contradicting independence).
    # Wait, that's not right either. An independent set means NO edges, so if u~v is an edge,
    # we can't have both u and v. But a forbidden triple {u,v,w} creates edges u-v, u-w, v-w.
    # An independent set can contain at most 2 of {u,v,w}, which is fine - we just need 
    # to not contain all 3.
    #
    # Actually, the conflict graph approach works: if {u,v,w} is a forbidden triple, 
    # we add edges (u,v), (u,w), (v,w). Then an independent set cannot contain all three
    # (since at least one edge would be violated). But it could contain 2, which is fine.
    # So maximum independent set in this conflict graph = maximum valid subset.
    
    # Build adjacency as bitmasks for speed
    adj = [0] * (n + 1)
    
    # For each pair (u,v), check if there exists w != u,v with gcd(u,v)=gcd(u,w)=gcd(v,w)
    # gcd(u,v) = d. We need w such that gcd(u,w) = d and gcd(v,w) = d.
    # This means d|u, d|v, d|w, and gcd(u/d, w/d) = 1, gcd(v/d, w/d) = 1.
    # So w = d*k where gcd(u/d, k) = 1 and gcd(v/d, k) = 1, and w != u, w != v.
    
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            d = gcd_table[u][v]
            ud = u // d
            vd = v // d
            # Find w = d*k where 1 <= w <= n, w != u, w != v,
            # gcd(ud, k) = 1, gcd(vd, k) = 1
            found = False
            for k in range(1, n // d + 1):
                w = d * k
                if w == u or w == v:
                    continue
                if gcd(ud, k) == 1 and gcd(vd, k) == 1:
                    found = True
                    break
            if found:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    if time.time() > deadline:
        # Fallback: greedy
        selected = []
        for x in range(1, n + 1):
            if all((adj[x] >> s) & 1 == 0 for s in selected):
                selected.append(x)
        return selected
    
    # Maximum independent set via branch and bound
    # Order vertices by degree (descending) for better pruning
    vertices = list(range(1, n + 1))
    vertices.sort(key=lambda x: bin(adj[x]).count('1'), reverse=True)
    
    # Map original vertex to position in ordering
    pos = [0] * (n + 1)
    for i, v in enumerate(vertices):
        pos[v] = i
    
    # Rebuild adjacency in new ordering as bitmasks
    # new_adj[i] has bit j set if vertices[i] and vertices[j] are adjacent
    new_adj = [0] * n
    for i in range(n):
        u = vertices[i]
        mask = 0
        for j in range(n):
            if j != i and (adj[u] >> vertices[j]) & 1:
                mask |= (1 << j)
        new_adj[i] = mask
    
    # Greedy coloring for upper bound (color classes = independent sets in complement)
    # For MIS, we use coloring on the graph: each color class is an independent set,
    # so |MIS| <= number of colors used (actually, we need coloring of the graph,
    # and MIS size <= max color class size... no.)
    # 
    # Standard approach: color the graph greedily. The chromatic number gives an upper
    # bound on clique size, but for MIS we want: |MIS| <= n - |min vertex cover|... 
    # 
    # Actually, for branch-and-bound MIS: at each node with candidate set S,
    # we compute a greedy coloring of the subgraph induced by S. The number of colors
    # is an upper bound on the size of a clique in the complement (i.e., MIS in original).
    # Wait, I'm confusing myself.
    #
    # Let me use a different approach: branch and bound where we either include or exclude
    # a vertex. Upper bound = current_size + |remaining candidates that are compatible|.
    # 
    # Simpler: use the fact that if we have a set of candidates, a greedy coloring of
    # the induced subgraph gives an upper bound on the clique number of the complement,
    # which equals the MIS size.
    
    best = [0]
    best_set = [set()]
    
    def greedy_color(candidates):
        """Greedy color the subgraph induced by candidates. Returns number of colors."""
        if not candidates:
            return 0
        # Order by degree within candidates
        cand_list = list(candidates)
        degs = []
        for v in cand_list:
            d = 0
            for u in cand_list:
                if u != v and (new_adj[v] >> u) & 1:
                    d += 1
            degs.append(d)
        order = sorted(range(len(cand_list)), key=lambda i: degs[i], reverse=True)
        
        colors = {}
        num_colors = 0
        for idx in order:
            v = cand_list[idx]
            # Find smallest available color
            used = set()
            for u in cand_list:
                if u != v and (new_adj[v] >> u) & 1 and u in colors:
                    used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
            num_colors = max(num_colors, c + 1)
        return num_colors
    
    def branch(candidates, current_size, current_set):
        nonlocal best, best_set
        if time.time() > deadline:
            return
        
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
                best_set[0] = set(current_set)
            return
        
        # Upper bound via greedy coloring
        ub = current_size + greedy_color(candidates)
        if ub <= best[0]:
            return
        
        # Pick a vertex to branch on (highest degree in candidates)
        v = max(candidates, key=lambda x: bin(new_adj[x] & ((1 << n) - 1)).count('1'))
        
        # Branch 1: include v
        new_candidates = {u for u in candidates if u != v and not ((new_adj[v] >> u) & 1)}
        branch(new_candidates, current_size + 1, current_set | {v})
        
        # Branch 2: exclude v
        new_candidates2 = candidates - {v}
        branch(new_candidates2, current_size, current_set)
    
    branch(set(range(n)), 0, set())
    
    if best[0] > 0:
        result = [vertices[i] for i in best_set[0]]
        return result
    
    # Fallback
    return [1]
# EVOLVE-BLOCK-END
