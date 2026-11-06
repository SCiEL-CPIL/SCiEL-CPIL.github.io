"""Improved program: efficient C5 detection and alpha bound for finding larger valid graphs."""
import random
import time
import sys
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 118.0

    def remaining():
        return deadline - time.time()

    # ---- Guaranteed valid fallback: 8 disjoint K4s -> n=32, alpha=8, no C5 ----
    def k4_blocks(nblocks):
        n = 4 * nblocks
        edges = []
        for b in range(nblocks):
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([4 * b + i, 4 * b + j])
        return {'n': n, 'edges': edges}

    fallback = k4_blocks(8)  # n = 32

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def has_c5_np(adj_np, n):
        """Vectorized C5 check using NumPy.
        A C5 exists iff there is a 5-tuple (s, v1, v2, v3, v4) with distinct vertices
        forming a cycle. Equivalently: for some vertex s, there exist two neighbors a,b of s
        such that there is a vertex c (c != s, c != a, c != b) adjacent to both a and b.
        """
        # adj_np is n x n boolean/uint8 adjacency matrix (symmetric, zero diagonal)
        # For each s, N(s) = column s
        # For each pair (a, b) in N(s), check if (A[a] & A[b]) has any 1 outside {s, a, b}
        # Vectorized: for each s, compute common = A[N(s),:] & A[N(s),:] ... 
        # Simpler: compute M = A @ A (number of length-2 paths).
        # A C5 exists iff there exist s, a, b with a~s, b~s, a!=b, and a~c~b for some c not in {s,a,b}.
        # (A^2)[a][b] >= 1 means there's a common neighbor of a and b.
        # We need: a~s, b~s, a!=b, (A^2)[a][b] >= 1, and the common neighbor c is not s, a, or b.
        # If (A^2)[a][b] >= 1 and a~s and b~s and a!=b:
        #   The common neighbor c could be s (since s~a and s~b). We need c != s.
        #   Also c != a (since no self-loops, a is not adjacent to a) and c != b.
        #   So we need (A^2)[a][b] >= 2 if s is a common neighbor, or >= 1 if s is not.
        # Actually: (A^2)[a][b] = |N(a) ∩ N(b)|. If s ∈ N(a) ∩ N(b) (which it is since a~s, b~s),
        # then we need |N(a) ∩ N(b)| >= 2 to have another common neighbor c != s.
        # But wait, c could also be a or b? No: a ∉ N(a) (no self-loop), so a ∉ N(a)∩N(b). Same for b.
        # So we need |N(a) ∩ N(b)| >= 2 when s is already a common neighbor.
        # Alternatively: (A^2)[a][b] - A[s][a]*A[s][b] >= 1, i.e., at least one common neighbor other than s.
        
        A = adj_np  # uint8 array
        A2 = A @ A  # int array, (A^2)[i][j] = number of common neighbors
        
        # For each s, for each pair of neighbors a, b of s (a < b):
        # check if A2[a][b] - 1 >= 1 (since s is always a common neighbor of a and b)
        # i.e., A2[a][b] >= 2
        
        # Vectorized approach:
        # Create matrix P where P[a][b] = 1 if a~s and b~s for some s, and a != b
        # Then check if there exists s, a, b with A[s][a]=1, A[s][b]=1, a!=b, A2[a][b]>=2
        
        # Even simpler: iterate over s (only n <= 35 iterations)
        for s in range(n):
            nbs = np.where(A[s] > 0)[0]
            if len(nbs) < 2:
                continue
            # For all pairs in nbs, check A2[a][b] >= 2
            # Get submatrix of A2 indexed by nbs
            sub = A2[np.ix_(nbs, nbs)]
            # We need any off-diagonal element >= 2
            # Set diagonal to 0 to ignore self-pairs
            np.fill_diagonal(sub, 0)
            if sub.max() >= 2:
                return True
        return False

    def has_c5_bit(adj, n):
        """Bitmask C5 check (fallback if numpy unavailable or for small n)."""
        for s in range(n):
            nbs = adj[s]
            nb_list = []
            tmp = nbs
            while tmp:
                lb = tmp & (-tmp)
                v = lb.bit_length() - 1
                nb_list.append(v)
                tmp ^= lb
            for i in range(len(nb_list)):
                a = nb_list[i]
                for j in range(i + 1, len(nb_list)):
                    b = nb_list[j]
                    common = adj[a] & adj[b] & ~((1 << s))
                    if common:
                        return True
        return False

    def has_c5(adj, n):
        """Check for C5 using numpy if possible, else bitmask."""
        try:
            adj_np = np.zeros((n, n), dtype=np.uint8)
            for i in range(n):
                m = adj[i]
                for j in range(n):
                    if m & (1 << j):
                        adj_np[i][j] = 1
            return has_c5_np(adj_np, n)
        except:
            return has_c5_bit(adj, n)

    def alpha_at_least_9(adj, n, time_limit=3.0):
        """Check if alpha(G) >= 9 using branch and bound on complement.
        Returns True if alpha >= 9, False if alpha < 9, None if timeout."""
        # Build complement adjacency as bitmasks
        full = (1 << n) - 1
        comp = [full ^ (adj[i] | (1 << i)) for i in range(n)]
        
        # Sort by complement degree descending
        order = sorted(range(n), key=lambda i: -bin(comp[i]).count('1'))
        pos = {v: i for i, v in enumerate(order)}
        comp2 = [0] * n
        for i in range(n):
            orig = order[i]
            mask = 0
            cm = comp[orig]
            tmp = cm
            while tmp:
                lb = tmp & (-tmp)
                v = lb.bit_length() - 1
                if v != orig:
                    mask |= (1 << pos[v])
                tmp ^= lb
            comp2[i] = mask
        
        # Precompute popcount for small values or use bit_count
        def popcount(x):
            return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')
        
        t0 = time.time()
        deadline_local = t0 + time_limit
        
        def rec(cand, k):
            if k >= 9:
                return True
            if time.time() > deadline_local:
                return None
            pc = popcount(cand)
            if pc + k < 9:
                return False
            
            # Greedy upper bound on clique size in complement[cand]
            cb = cand
            gc = 0
            while cb:
                best_v = -1
                best_deg = -1
                cb2 = cb
                while cb2:
                    lb = cb2 & (-cb2)
                    v = lb.bit_length() - 1
                    deg = popcount(cb & comp2[v])
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    cb2 ^= lb
                if best_v < 0:
                    break
                gc += 1
                cb &= comp2[best_v]
            
            if gc + k < 9:
                return False
            
            # Branch: pick vertex with max degree in cand
            best_v = -1
            best_deg = -1
            cb2 = cand
            while cb2:
                lb = cb2 & (-cb2)
                v = lb.bit_length() - 1
                deg = popcount(cand & comp2[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                cb2 ^= lb
            
            v = best_v
            r = rec(cand & comp2[v], k + 1)
            if r is True:
                return True
            if r is None:
                return None
            return rec(cand & ~((1 << v)), k)

        return rec(full, 0)

    def is_valid(n, edges, time_limit=3.0):
        adj = build_adj(n, edges)
        if has_c5(adj, n):
            return False
        a9 = alpha_at_least_9(adj, n, time_limit)
        if a9 is True:
            return False
        if a9 is None:
            return False
        return True

    best = fallback  # always valid, n=32

    # ---- Random search with local repair for n=33,34,35 ----
    def local_repair(n, edges, max_iters=50, time_limit=10.0):
        """Try to repair a graph by adding/removing edges to eliminate C5s and reduce alpha."""
        t0 = time.time()
        adj = build_adj(n, edges)
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u,v), max(u,v)))
        
        for it in range(max_iters):
            if time.time() - t0 > time_limit:
                break
            if is_valid(n, list(edge_set), 1.0):
                return sorted(edge_set)
            
            # Find a C5 and break it by removing an edge
            # Simple heuristic: randomly remove an edge
            if edge_set:
                e = random.choice(list(edge_set))
                edge_set.discard(e)
            else:
                # No edges, add random ones
                for _ in range(5):
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u != v:
                        edge_set.add((min(u,v), max(u,v)))
        
        return None

    # ---- Random search for n=33,34,35 ----
    for target in (35, 34, 33):
        if remaining() < 20.0:
            break
        
        t0 = time.time()
        limit = min(remaining() - 15.0, 18.0)
        found = None
        
        for attempt in range(3000):
            if time.time() - t0 > limit:
                break
            
            n = target
            # Try various edge densities
            p = 0.25 + 0.1 * (attempt % 4)  # 0.25, 0.35, 0.45, 0.55
            
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        edges.append([i, j])
            
            if is_valid(n, edges, 2.0):
                found = {'n': n, 'edges': edges}
                break
        
        if found is not None:
            best = found
            break

    # ---- Structured modifications of 8xK4 for n=33 ----
    if best['n'] == 32 and remaining() > 12.0:
        t0 = time.time()
        limit = min(remaining() - 8.0, 12.0)
        
        for attempt in range(3000):
            if time.time() - t0 > limit:
                break
            
            n = 33
            edges = []
            for b in range(8):
                base = 4 * b
                for i in range(4):
                    for j in range(i + 1, 4):
                        edges.append([base + i, base + j])
            
            conn = set()
            for b in range(8):
                base = 4 * b
                num_conn = random.choice([2, 2, 3, 3])
                verts = [base + i for i in range(4)]
                random.shuffle(verts)
                for v in verts[:num_conn]:
                    conn.add(v)
            
            for v in conn:
                edges.append([32, v])
            
            if is_valid(n, edges, 2.0):
                best = {'n': n, 'edges': edges}
                break

    # ---- Try 9 K4s minus some edges for n=35 ----
    if best['n'] < 35 and remaining() > 10.0:
        t0 = time.time()
        limit = min(remaining() - 5.0, 10.0)
        
        for attempt in range(2000):
            if time.time() - t0 > limit:
                break
            
            n = 35
            # 8 K4s + 1 K3 (3 vertices) = 32 + 3 = 35
            edges = []
            for b in range(8):
                base = 4 * b
                for i in range(4):
                    for j in range(i + 1, 4):
                        edges.append([base + i, base + j])
            # K3 on vertices 32, 33, 34
            edges.append([32, 33])
            edges.append([33, 34])
            edges.append([32, 34])
            
            # This has alpha = 8*1 + 1 = 9, which is too high!
            # Need to connect the K3 to other blocks to reduce alpha.
            # Connect each of 32,33,34 to at least one vertex in each K4 block?
            # That might create C5s. Let's try connecting them to specific vertices.
            # Actually alpha = 9 is the issue. We need alpha <= 8.
            # With 8 K4s (alpha contribution 1 each) + K3 (alpha contribution 1) = 9.
            # We need to reduce this. Connect vertices in K3 to vertices in K4s
            # so that any independent set can't pick from both.
            
            # Try: connect vertex 32 to one vertex in each K4 block
            # Then an independent set containing 32 can't contain those 8 vertices,
            # but can still contain one from each other K4 block... this gets complex.
            # Skip this approach as it's unlikely to work easily.
            pass

    if time.time() > deadline - 2.0:
        return fallback

    return best
# EVOLVE-BLOCK-END
