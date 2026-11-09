"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph.

Strategy:
- Use structured 3-regular graphs (Q4-like, random 3-regular) as the sparse part
- Take complement to get dense K4-free candidates
- Fast branch-and-bound for triangle-free subgraph size
- Targeted local search with edge flips
"""
import time
import random
import sys

N = 18
DEADLINE = 140.0  # safety margin under 150s


def _is_k4_free(adj):
    """Check if the graph (adjacency bitmasks) is K4-free."""
    for i in range(N):
        for j in range(i + 1, N):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j]
            c = common
            while c:
                lsb = c & (-c)
                k = lsb.bit_length() - 1
                c ^= lsb
                if k > j and (adj[k] & (common & ~(1 << k))):
                    return False
    return True


def _max_triangle_free(adj):
    """Exact branch-and-bound for the size of the largest triangle-free induced subgraph."""
    # Precompute all triangles as bitmasks
    triangles = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i] & (1 << j):
                common = adj[i] & adj[j]
                c = common
                while c:
                    lsb = c & (-c)
                    k = lsb.bit_length() - 1
                    c ^= lsb
                    if k > j:
                        triangles.append((1 << i) | (1 << j) | (1 << k))
    
    if not triangles:
        return N
    
    # Order vertices by degree (ascending) for better pruning
    order = sorted(range(N), key=lambda v: bin(adj[v]).count('1'))
    
    best = 0
    sys.setrecursionlimit(10000)
    
    # Precompute conflict: for each vertex v, which vertices in S would prevent adding v
    # v can be added to S iff for all a in S: (adj[v] & adj[a] & S) == 0
    
    def search(S, candidates):
        nonlocal best
        s_count = bin(S).count('1')
        if s_count > best:
            best = s_count
            if best == N:
                return
        c_count = bin(candidates).count('1')
        if s_count + c_count <= best:
            return
        if candidates == 0:
            return
        
        # Try adding each candidate vertex
        c = candidates
        while c:
            lsb = c & (-c)
            v = lsb.bit_length() - 1
            c ^= lsb
            
            # Check if v can be added to S
            can_add = True
            s = S
            while s:
                lsb2 = s & (-s)
                a = lsb2.bit_length() - 1
                s ^= lsb2
                if adj[v] & adj[a] & S:
                    can_add = False
                    break
            
            if not can_add:
                continue
            
            new_S = S | lsb
            # New candidates: vertices in c that can also be added to new_S
            new_c = 0
            cc = c
            while cc:
                lsb3 = cc & (-cc)
                w = lsb3.bit_length() - 1
                cc ^= lsb3
                can_add_w = True
                s2 = new_S
                while s2:
                    lsb4 = s2 & (-s2)
                    a2 = lsb4.bit_length() - 1
                    s2 ^= lsb4
                    if adj[w] & adj[a2] & new_S:
                        can_add_w = False
                        break
                if can_add_w:
                    new_c |= lsb3
            
            search(new_S, new_c)
    
    search(0, (1 << N) - 1)
    return best


def _q4_extended(rng):
    """Generate a 3-regular-like graph on 18 vertices based on Q4 hypercube structure."""
    # Q4 has 16 vertices: 0-15, each connected to 3 others (differ in one bit)
    adj = [0] * N
    
    # Q4 edges
    for v in range(16):
        for bit in range(4):
            w = v ^ (1 << bit)
            if w > v:
                adj[v] |= (1 << w)
                adj[w] |= (1 << v)
    
    # Add vertices 16, 17
    # Connect each to 3 random vertices from 0-15, maintaining near-3-regular
    for new_v in [16, 17]:
        candidates = list(range(16))
        rng.shuffle(candidates)
        # Pick 3 vertices that currently have degree < 3
        picked = []
        for cand in candidates:
            if bin(adj[cand]).count('1') < 3:
                picked.append(cand)
                if len(picked) == 3:
                    break
        if len(picked) < 3:
            # Fill from remaining
            for cand in candidates:
                if cand not in picked:
                    picked.append(cand)
                    if len(picked) == 3:
                        break
        for u in picked[:3]:
            adj[new_v] |= (1 << u)
            adj[u] |= (1 << new_v)
    
    # Ensure symmetry and no self-loops
    for i in range(N):
        adj[i] &= ~(1 << i)
        for j in range(N):
            if adj[i] & (1 << j):
                adj[j] |= (1 << i)
    
    return adj


def _random_3_regular(rng):
    """Generate a random 3-regular graph on 18 vertices."""
    # Use configuration model
    stubs = []
    for v in range(N):
        for _ in range(3):
            stubs.append(v)
    rng.shuffle(stubs)
    
    adj = [0] * N
    used = [False] * len(stubs)
    
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        # Find an unused stub j != i, stubs[j] != u
        for j in range(i + 1, len(stubs)):
            if not used[j] and stubs[j] != u:
                v = stubs[j]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                used[j] = True
                break
    
    # Ensure no multi-edges (should be fine with random shuffle)
    for i in range(N):
        adj[i] &= ~(1 << i)
    
    return adj


def _complement(adj):
    """Return the complement of the graph."""
    comp = [0] * N
    for u in range(N):
        for v in range(u + 1, N):
            if not (adj[u] & (1 << v)):
                comp[u] |= (1 << v)
                comp[v] |= (1 << u)
    return comp


def construct():
    start_time = time.time()
    rng = random.Random(2024)
    
    def elapsed():
        return time.time() - start_time
    
    # Generate structured candidates
    candidates = []
    
    # Q4-based
    for _ in range(5):
        sparse = _q4_extended(rng)
        comp = _complement(sparse)
        if _is_k4_free(comp):
            candidates.append(comp)
    
    # Random 3-regular
    for _ in range(20):
        sparse = _random_3_regular(rng)
        comp = _complement(sparse)
        if _is_k4_free(comp):
            candidates.append(comp)
    
    # Also try some random dense K4-free as fallback
    for density in [0.5, 0.6, 0.7]:
        for _ in range(3):
            adj = [0] * N
            for i in range(N):
                for j in range(i + 1, N):
                    if rng.random() < density:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            # Remove edges to make K4-free
            changed = True
            while changed:
                changed = False
                for i in range(N):
                    for j in range(i + 1, N):
                        if adj[i] & (1 << j):
                            common = adj[i] & adj[j]
                            c = common
                            has_edge = False
                            while c:
                                lsb = c & (-c)
                                k = lsb.bit_length() - 1
                                c ^= lsb
                                if k > j and (adj[k] & (common & ~(1 << k))):
                                    has_edge = True
                                    break
                            if has_edge:
                                adj[i] &= ~(1 << j)
                                adj[j] &= ~(1 << i)
                                changed = True
            if _is_k4_free(adj):
                candidates.append(adj)
    
    if not candidates:
        # Fallback: 4-partite graph
        adj = [0] * N
        parts = [list(range(0, 5)), list(range(5, 9)), list(range(9, 13)), list(range(13, 18))]
        for i in range(4):
            for j in range(i + 1, 4):
                for u in parts[i]:
                    for v in parts[j]:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        candidates.append(adj)
    
    # Score all candidates
    best_adj = None
    best_score = N + 1
    
    for adj in candidates:
        if elapsed() > DEADLINE - 60:
            break
        s = _max_triangle_free(adj)
        if s < best_score:
            best_score = s
            best_adj = adj[:]
    
    if best_adj is None:
        best_adj = candidates[0]
        best_score = _max_triangle_free(best_adj)
    
    # Local search
    adj = best_adj[:]
    
    def get_edges(a):
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if a[i] & (1 << j):
                    edges.append((i, j))
        return edges
    
    def get_non_edges(a):
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if not (a[i] & (1 << j)):
                    edges.append((i, j))
        return edges
    
    improved = True
    while improved and elapsed() < DEADLINE - 15:
        improved = False
        
        # Try removing edges
        edges = get_edges(adj)
        rng.shuffle(edges)
        for (u, v) in edges:
            if elapsed() > DEADLINE - 15:
                break
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            s = _max_triangle_free(adj)
            if s < best_score:
                best_score = s
                best_adj = adj[:]
                improved = True
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Try adding edges (must maintain K4-free)
        non_edges = get_non_edges(adj)
        rng.shuffle(non_edges)
        for (u, v) in non_edges:
            if elapsed() > DEADLINE - 15:
                break
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if _is_k4_free(adj):
                s = _max_triangle_free(adj)
                if s < best_score:
                    best_score = s
                    best_adj = adj[:]
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
    
    # Convert to edge list
    result = []
    for i in range(N):
        for j in range(i + 1, N):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    return result


if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
