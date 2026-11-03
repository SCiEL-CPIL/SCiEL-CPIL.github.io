import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Moore bound for odd girth g, degree k
    if g % 2 == 1:
        half_g = (g - 3) // 2
        moore_bound = 1 + k * sum((k - 1) ** i for i in range(half_g + 1))
    else:
        half_g = g // 2 - 1
        moore_bound = 2 * sum((k - 1) ** i for i in range(half_g + 1))
    
    # For k=5, g=9: moore_bound = 1 + 5*(1 + 4 + 16) = 1 + 5*21 = 106
    # Best known: 1116 (Exoo et al. 2026)
    # LUW(5,4) gives 5^5 = 3125 vertices with girth 9
    
    # Strategy:
    # 1. Try circulant graphs C_n(S) with S = {a, -a, b, -b, n/2} for even n
    #    with fast short-cycle pre-filtering
    # 2. Fallback to LUW construction
    
    max_n = 5000
    
    # Fast short-cycle check for circulant graphs
    def check_circulant_short_cycles(S, n, max_cycle_len):
        """Check if circulant graph C_n(S) has any cycle of length <= max_cycle_len.
        A cycle of length L exists iff there are L elements s1,...,sL in S (with repetition allowed,
        but consecutive elements can't cancel for simple cycles) such that sum = 0 mod n.
        For simple cycles in circulant graphs, we need distinct steps that sum to 0.
        Actually, for a cycle v0 -> v1 -> ... -> vL -> v0, we need s1+...+sL = 0 mod n
        where si are in S. For a simple cycle, the partial sums must be distinct.
        
        For small L, we can enumerate all sequences of L elements from S and check if sum = 0 mod n.
        But we also need to ensure it's a simple cycle (no repeated vertices).
        For L=3: s1+s2+s3=0, and s1,s2,s3 not all same, and s1+s2 != 0 (else v1=v0).
        Actually for a 3-cycle: v0, v0+s1, v0+s1+s2, back to v0. Need s1+s2+s3=0,
        s1!=0, s2!=0, s3!=0, s1+s2!=0, s2+s3!=0, s1!=s2, s2!=s3, s3!=s1 (for distinct vertices).
        Wait, vertices are v0, v0+s1, v0+s1+s2. These are distinct iff s1!=0, s2!=0, s1+s2!=0.
        And s3 = -(s1+s2). We need s3 in S.
        
        Let me just do BFS-based girth check but with early termination.
        """
        pass
    
    # Better approach: for each candidate circulant, do a quick BFS girth check
    # but optimize with numpy if possible, or just use efficient Python
    
    def girth_check_fast(adj, n, target_girth):
        """Check if graph has girth >= target_girth. Returns True if valid."""
        # BFS from each vertex, but stop early if we find a short cycle
        # For target_girth=9, we need no cycles of length 3,4,5,6,7,8
        # BFS to depth 4 from each vertex: if we find a back edge to a node at depth d
        # where current_depth + 1 + d < target_girth, we have a short cycle.
        # Actually: when at node u at depth du, if neighbor v has depth dv in same BFS,
        # and v != parent[u], then cycle length = du + 1 + dv.
        # We need all such cycles to be >= target_girth.
        
        max_depth = target_girth // 2  # For girth 9, max_depth = 4
        
        for u in range(n):
            visited = [-1] * n
            parent = [-1] * n
            visited[u] = 0
            queue = [(u, 0)]
            
            while queue:
                node, dist = queue.pop(0)
                if dist >= max_depth:
                    continue
                for v in adj[node]:
                    if visited[v] == -1:
                        visited[v] = dist + 1
                        parent[v] = node
                        queue.append((v, dist + 1))
                    elif v != parent[node]:
                        cycle_len = dist + 1 + visited[v]
                        if cycle_len < target_girth:
                            return False
        return True
    
    # Try circulant graphs
    # For k=5, n must be even. S = {a, n-a, b, n-b, n//2}
    # We need to search over n and (a,b)
    
    # First, try to find a good n. The Moore bound is 106, but realistically
    # we need n >= 2*(1+4+16+64) = 2*85 = 170 for a bipartite-like structure.
    # Actually for 5-regular with girth 9, known smallest is 1116.
    # Let's search n from moore_bound to max_n, stepping by 2 (even only).
    
    # Pre-compute: for a given n, how many valid (a,b) pairs?
    # a in [1, n//2-1], b in [1, n//2-1], a != b, a != n-b, b != n-a
    # i.e., {a, n-a, b, n-b} has 4 distinct elements, and n//2 not in {a, n-a, b, n-b}
    # Since a < n/2 and b < n/2, n-a > n/2 and n-b > n/2, so n//2 can't equal a or b.
    # But n//2 could equal n-a if a = n/2, but a < n/2, so no.
    # So we just need a != b and a != n-b (i.e., a+b != n).
    
    # To speed up, let's first try a few specific n values that are known to be candidates
    # or just do a systematic search with time limits.
    
    # Let's be smarter: for each n, randomly sample (a,b) pairs and check girth.
    # Use a fast pre-check: for circulant C_n(S), a 4-cycle exists iff there are
    # s1,s2,s3,s4 in S with s1+s2+s3+s4=0 mod n and the partial sums are distinct.
    # For S = {a, -a, b, -b, n/2}, 4-cycles can be formed.
    # Let's just do the BFS check but with a time limit per graph.
    
    # Actually, let's try a different approach: use the known result that for k=5, g=9,
    # the LUW graph L(5,4) has 3125 vertices. Can we do better with circulants?
    # Probably not easily, but let's try.
    
    # Let's focus on the LUW fallback and make it efficient.
    # LUW(5,4): q=5, d=4, n=5^(d+1)=5^5=3125, girth=2d+1=9.
    
    # But wait - can we use a smaller LUW? L(5,3) has girth 7, not enough.
    # So LUW(5,4) is the smallest LUW for g=9.
    
    # Let's also try: what about non-circulant constructions?
    # The task says to keep the seed program's design family.
    # The seed uses LUW as fallback and circulant search.
    
    # Let me optimize the circulant search significantly:
    # 1. For each n, pre-compute which (a,b) pairs avoid 3-cycles and 4-cycles
    # 2. Only do full BFS for candidates that pass pre-filters
    
    def has_3_cycle_circulant(a, b, n):
        """Check if C_n({a,-a,b,-b,n/2}) has a 3-cycle.
        3-cycle: s1+s2+s3=0 mod n, s_i in S, partial sums distinct.
        Possible combinations (up to symmetry):
        a+a+(-2a)=0: need -2a in S, i.e., 2a = n/2 or 2a = a (no) or 2a = b or 2a = -b or 2a = n-a
        a+a+(-a)=a: not 0
        a+(-a)+s=s: not 0 unless s=0
        a+b+(-a-b)=0: need -a-b in S
        a+b+n/2: need a+b+n/2=0 mod n, i.e., a+b=n/2
        a+(-b)+... etc.
        Let's enumerate all ordered triples from S and check.
        """
        S = [a, n-a, b, n-b, n//2]
        for s1 in S:
            for s2 in S:
                s3 = (-s1 - s2) % n
                if s3 in S:
                    # Check partial sums distinct: s1 != 0, s1+s2 != 0
                    if s1 != 0 and (s1 + s2) % n != 0:
                        return True
        return False
    
    def has_4_cycle_circulant(a, b, n):
        """Check if C_n({a,-a,b,-b,n/2}) has a 4-cycle."""
        S = [a, n-a, b, n-b, n//2]
        # 4-cycle: s1+s2+s3+s4=0, partial sums distinct
        # s4 = -(s1+s2+s3)
        for s1 in S:
            for s2 in S:
                ps12 = (s1 + s2) % n
                if ps12 == 0:
                    continue
                for s3 in S:
                    ps123 = (ps12 + s3) % n
                    if ps123 == 0:
                        continue
                    s4 = (-ps123) % n
                    if s4 in S:
                        # Check all partial sums distinct and nonzero
                        # s1, s1+s2, s1+s2+s3, 0
                        # Need s1 != 0 (true), ps12 != 0 (checked), ps123 != 0 (checked)
                        # Also need s1 != ps12, s1 != ps123, ps12 != ps123
                        if s1 != ps12 and s1 != ps123 and ps12 != ps123:
                            return True
        return False
    
    # Now search for circulant graphs
    best_n = None
    best_edges = None
    
    # Try n values from moore_bound to max_n
    # For efficiency, step by 2 (even n only)
    # Limit the number of (a,b) samples per n
    
    max_samples_per_n = 200
    
    for n in range(moore_bound, max_n + 1, 2):
        if time.time() > deadline:
            break
        
        if n % 2 != 0:
            continue
        
        # Collect valid (a,b) pairs
        # a in [1, n//2 - 1], b in [1, n//2 - 1]
        # a != b, a + b != n (i.e., a != n - b)
        
        # Pre-filter: avoid 3-cycles and 4-cycles
        valid_pairs = []
        max_a = n // 2 - 1
        if max_a < 1:
            continue
        
        # Randomly sample to avoid O(n^2) enumeration
        # But for small n, enumerate all
        if n <= 200:
            for a in range(1, max_a + 1):
                for b in range(1, max_a + 1):
                    if a == b or a + b == n:
                        continue
                    # Quick 3-cycle check
                    if has_3_cycle_circulant(a, b, n):
                        continue
                    if has_4_cycle_circulant(a, b, n):
                        continue
                    valid_pairs.append((a, b))
        else:
            # Sample randomly
            for _ in range(max_samples_per_n):
                if time.time() > deadline:
                    break
                a = random.randint(1, max_a)
                b = random.randint(1, max_a)
                if a == b or a + b == n:
                    continue
                if has_3_cycle_circulant(a, b, n):
                    continue
                if has_4_cycle_circulant(a, b, n):
                    continue
                valid_pairs.append((a, b))
        
        # For each valid pair, do full girth check
        for a, b in valid_pairs:
            if time.time() > deadline:
                break
            
            S = [a, n - a, b, n - b, n // 2]
            
            # Build adjacency
            adj = [[] for _ in range(n)]
            for u in range(n):
                for s in S:
                    v = (u + s) % n
                    adj[u].append(v)
            
            # Full girth check
            if girth_check_fast(adj, n, g):
                # Found a valid graph!
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
    
    # Fallback: LUW construction
    q = k
    # Minimal odd d such that 2*d+1 >= g
    # For g=9, d=4 (2*4+1=9)
    d = (g - 1) // 2
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # LUW rule
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
