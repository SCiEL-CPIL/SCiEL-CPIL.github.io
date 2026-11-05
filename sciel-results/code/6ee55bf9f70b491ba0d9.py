"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _has_triangle_circulant(n, S):
    """Check if circulant graph on Z_n with symmetric connection set S is triangle-free.
    S should contain positive representatives. The graph connects i to i+s and i-s for s in S.
    Triangle exists if there are x, y, z in S (with possible signs) such that x+y+z = 0 mod n.
    Since S is symmetric (if x in S, -x in S is implied by construction of edges), 
    we check if there are a, b, c in S such that a+b+c is a multiple of n.
    Note: a, b, c can be the same or different, but vertices must be distinct.
    In a simple graph, a triangle is 3 distinct vertices u, v, w.
    u~v => v-u in S (or -S). v~w => w-v in S. w~u => u-w in S.
    Sum: (v-u) + (w-v) + (u-w) = 0.
    So we need x, y, z in S (the set of differences) such that x+y+z = 0 mod n.
    The set of differences is S ∪ -S. Let D = S ∪ -S.
    We need x, y, z in D such that x+y+z = 0 mod n.
    Since 0 is not in D (simple graph, no loops), and S is symmetric, D is symmetric.
    If x=y, then 2x+z=0 => z=-2x. So if 2x in D and -2x in D (always true if symmetric), we have a triangle?
    Wait, if x=y, the vertices are u, u+x, u+2x.
    Edges: (u, u+x) diff x. (u+x, u+2x) diff x. (u+2x, u) diff -2x.
    So we need x in D, x in D, -2x in D.
    Since D is symmetric, -2x in D iff 2x in D.
    So if 2x in D, we have a triangle (u, u+x, u+2x).
    If x!=y, we need x+y+z=0 with x!=y, y!=z, z!=x?
    Actually, just checking all triples in D is safe.
    """
    D = set(S)
    for s in S:
        D.add((-s) % n)
    
    # Check for x+y+z = 0 mod n with x,y,z in D
    # To avoid checking permutations, we can iterate
    for x in D:
        for y in D:
            z = (-x - y) % n
            if z in D:
                # Check if vertices are distinct
                # Vertices: 0, x, x+y
                # Distinct if x != 0, y != 0, x+y != 0
                # x,y are in D, so non-zero.
                # x+y == 0 => z=0, but 0 not in D. So x+y != 0.
                # x == y? If x==y, vertices 0, x, 2x. Distinct if 2x != 0.
                # If 2x == 0 mod n, then x = -x, so n is even and x=n/2.
                # Then 0, n/2, 0. Not distinct.
                # So if x==y and 2x % n == 0, it's not a triangle (it's a 2-cycle or loop, but simple graph).
                # If x!=y, vertices 0, x, x+y. Distinct if x != x+y => y!=0 (true) and x != 0 (true).
                # So we just need to ensure we don't count the "degenerate" case where 2x=0.
                if x == y:
                    if (2 * x) % n == 0:
                        continue
                return True
    return False

def _independence_number_circulant(n, S, time_limit=10):
    """Compute independence number of circulant graph using branch and bound with bitmasks."""
    start_time = time.time()
    
    # Build adjacency masks
    adj_mask = [0] * n
    for i in range(n):
        mask = 0
        for s in S:
            j1 = (i + s) % n
            j2 = (i - s) % n
            mask |= (1 << j1)
            mask |= (1 << j2)
        adj_mask[i] = mask
    
    full_mask = (1 << n) - 1
    # Non-neighbors mask for each vertex (excluding self)
    non_adj_mask = [0] * n
    for i in range(n):
        # All vertices except i and its neighbors
        non_adj_mask[i] = full_mask ^ (1 << i) ^ adj_mask[i]
    
    best = 0
    
    def dfs(candidates, current_size):
        nonlocal best
        if time.time() - start_time > time_limit:
            return
        
        # Pruning: if current_size + popcount(candidates) <= best, return
        if current_size + candidates.bit_count() <= best:
            return
            
        if candidates == 0:
            if current_size > best:
                best = current_size
            return
        
        # Heuristic: pick vertex with max degree in candidate subgraph? 
        # Or min degree? For independent set, picking a vertex with many non-neighbors is good?
        # Standard: pick vertex v.
        # Include v: candidates becomes candidates & non_adj_mask[v]
        # Exclude v: candidates becomes candidates & ~(1<<v)
        
        # Choose v to minimize branching or maximize pruning.
        # Let's pick the vertex with the smallest index in candidates for determinism, 
        # or use a greedy heuristic.
        # Greedy: pick v such that (candidates & non_adj_mask[v]) is smallest? 
        # This reduces the "include" branch size.
        
        cands_list = []
        tmp = candidates
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            cands_list.append(v)
            tmp ^= lsb
        
        # Find v that minimizes the size of the next candidate set when included
        best_v = cands_list[0]
        min_next_size = (candidates & non_adj_mask[best_v]).bit_count()
        
        for v in cands_list:
            next_size = (candidates & non_adj_mask[v]).bit_count()
            if next_size < min_next_size:
                min_next_size = next_size
                best_v = v
                if min_next_size == 0:
                    break
                    
        v = best_v
        
        # Include v
        dfs(candidates & non_adj_mask[v], current_size + 1)
        
        # Exclude v
        dfs(candidates & ~(1 << v), current_size)
        
    dfs(full_mask, 0)
    return best

def _verify_circulant(n, S, time_limit=15):
    if _has_triangle_circulant(n, S):
        return False
    alpha = _independence_number_circulant(n, S, time_limit)
    if alpha >= 10:
        return False
    return True

def _build_edges_circulant(n, S):
    edges = []
    S_set = set(S)
    for i in range(n):
        for s in S:
            j = (i + s) % n
            if j > i:
                edges.append([i, j])
    return edges

def _try_circulant(n, time_limit=20):
    """Try to find a valid circulant graph on n vertices."""
    # S is a set of positive integers < n/2 (or <= n/2 if n even)
    # We want to maximize edges (to minimize alpha) while keeping triangle-free.
    # Start with a random subset of {1, ..., n//2} and greedily add?
    # Or enumerate?
    # For n=40, n//2 = 20. 2^20 is too big.
    # Greedy: Start with empty S. Add x if triangle-free.
    # Order of adding matters.
    
    max_seed = 50
    for seed in range(max_seed):
        if time.time() > time_limit:
            break
        rng = random.Random(seed)
        candidates = list(range(1, n // 2 + 1))
        if n % 2 == 0:
            # n//2 is self-inverse, so -n/2 = n/2. 
            # Adding n/2 creates a perfect matching.
            # Triangle check: x, y, z. If x=n/2, y=n/2, z=0 (not in S). 
            # If x=n/2, y=s, z=-s-n/2?
            # Let's just include it in candidates and let the triangle check handle it.
            pass
        rng.shuffle(candidates)
        
        S = set()
        for x in candidates:
            new_S = S | {x}
            if not _has_triangle_circulant(n, new_S):
                S = new_S
        
        # Check if this S yields alpha < 10
        if S and _verify_circulant(n, S, time_limit=5):
            return _build_edges_circulant(n, S)
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_n = 0
    best_edges = []
    
    # Search n from 40 down to 20
    for n in range(40, 19, -1):
        if time.time() > deadline:
            break
            
        edges = _try_circulant(n, time_limit=10)
        if edges is not None:
            best_n = n
            best_edges = edges
            break # Found the largest n
    
    # Fallback if no circulant found
    if best_n == 0:
        # Use a known small valid graph: C5 blow-up by 4 (n=20, alpha=8)
        n0 = 20
        # C5: 0-1-2-3-4-0
        # Blow up by 4: 
        # Vertices 0-3 (twins of 0), 4-7 (twins of 1), 8-11 (twins of 2), 12-15 (twins of 3), 16-19 (twins of 4)
        # Edges between twin sets if base vertices are adjacent.
        base_adj = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        edges = []
        for u, v in base_adj:
            for i in range(4):
                for j in range(4):
                    a = u * 4 + i
                    b = v * 4 + j
                    if a < b:
                        edges.append([a, b])
        best_n = n0
        best_edges = edges
        
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
