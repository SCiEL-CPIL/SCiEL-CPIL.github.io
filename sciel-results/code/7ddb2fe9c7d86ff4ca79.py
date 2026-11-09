"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
import time
import random
from itertools import combinations

N = 18
DEADLINE = 115.0  # safety margin under 150s


def _complement_of_3_regular():
    """Generate the complement of a random 3-regular graph on 18 vertices.
    
    A 3-regular graph on 18 vertices has no K4 (since K4 requires degree 3,
    but we need to verify the complement is K4-free).
    
    Actually, the complement of a 3-regular graph G is K4-free iff G has no
    independent set of size 4. For random 3-regular graphs, this is likely.
    
    Better: use a structured 3-regular graph like a cycle-based construction.
    """
    rng = random.Random(12345)
    
    # Generate a random 3-regular graph using configuration model
    # Each vertex has 3 stubs
    stubs = []
    for v in range(N):
        for _ in range(3):
            stubs.append(v)
    
    rng.shuffle(stubs)
    
    # Pair up stubs to form edges
    adj = [0] * N
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u != v:  # avoid self-loops
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    # If we got self-loops or multiple edges, try again or fix
    # For simplicity, let's use a deterministic construction instead
    
    # Use a circulant graph: connect each vertex to v+1, v-1, v+9 (mod 18)
    # This is 3-regular
    adj = [0] * N
    for v in range(N):
        for d in [1, 2, 9]:  # distances
            w = (v + d) % N
            if w != v:
                adj[v] |= (1 << w)
                adj[w] |= (1 << v)
    
    # Verify 3-regular
    for v in range(N):
        deg = bin(adj[v]).count('1')
        if deg != 3:
            # Fix by using a different construction
            break
    
    # Complement: original graph has edge (u,v) iff complement does NOT
    comp_adj = [0] * N
    for u in range(N):
        for v in range(u+1, N):
            if not (adj[u] & (1 << v)):
                comp_adj[u] |= (1 << v)
                comp_adj[v] |= (1 << u)
    
    return comp_adj


def _q4_extended():
    """Use Q4 (16-vertex hypercube) extended to 18 vertices.
    
    Q4 is 3-regular and bipartite (no triangles, no odd cycles).
    The complement of Q4 is dense.
    """
    # Q4: vertices 0..15, edge between u,v iff popcount(u^v)==1
    adj = [0] * 16
    for u in range(16):
        for v in range(16):
            if u < v:
                diff = u ^ v
                if diff and (diff & (diff - 1)) == 0:  # power of 2
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    
    # Extend to 18 vertices: add vertices 16, 17
    # Connect them to maintain 3-regularity
    # For simplicity, connect 16 to 0, 2, 4 and 17 to 1, 3, 5
    # But this breaks regularity. Let's just use the 16-vertex Q4 complement
    # and add 2 vertices with random connections
    
    full_adj = [0] * N
    for u in range(16):
        for v in range(16):
            if adj[u] & (1 << v):
                full_adj[u] |= (1 << v)
                full_adj[v] |= (1 << u)
    
    # Add vertices 16, 17 with some edges to make it work
    # For now, just return the 16-vertex part extended trivially
    rng = random.Random(999)
    for v in [16, 17]:
        for u in range(16):
            if rng.random() < 0.3:
                full_adj[v] |= (1 << u)
                full_adj[u] |= (1 << v)
    
    # Complement
    comp_adj = [0] * N
    for u in range(N):
        for v in range(u+1, N):
            if not (full_adj[u] & (1 << v)):
                comp_adj[u] |= (1 << v)
                comp_adj[v] |= (1 << u)
    
    return comp_adj


def _score_fast(adj):
    """Fast branch-and-bound for max triangle-free induced subgraph size using bitmasks."""
    # Precompute all triangles as bitmasks
    triangles = []
    for i in range(N):
        for j in range(i+1, N):
            if adj[i] & (1 << j):
                common = adj[i] & adj[j]
                c = common
                while c:
                    lsb = c & (-c)
                    k = lsb.bit_length() - 1
                    c ^= lsb
                    if k > j:
                        triangles.append((1 << i) | (1 << j) | (1 << k))
    
    best = 0
    full_mask = (1 << N) - 1
    
    # Precompute for each vertex v: which other vertices form a triangle with v
    # tri_neighbors[v] = set of vertices w such that there exists u with {v,w,u} a triangle
    # Actually, we need: when adding v to S, v conflicts with w if there's u in S with u~v, u~w
    # Precompute: conflict[v][w] = True if v and w share a common neighbor
    # But that's not quite right. Let me think differently.
    
    # For a set S to be triangle-free: for every vertex v in S, the neighbors of v within S form an independent set.
    # So when building S, if we add v, we need that no two neighbors of v in S are adjacent to each other.
    
    # Better approach: use the triangle list directly
    # A set S is triangle-free iff for every triangle T, S does not contain all of T.
    
    # Use branch and bound with vertex ordering by degree (ascending)
    order = sorted(range(N), key=lambda v: bin(adj[v]).count('1'))
    
    # For pruning: precompute for each triangle, which vertices it contains
    # When we decide to include a vertex, we can prune triangles that are now "impossible"
    
    # Simpler: just check triangles directly
    import sys
    sys.setrecursionlimit(10000)
    
    def search(S, candidates, remaining_triangles):
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
        
        # Get next vertex to try (from candidates, in order)
        # Try vertices that appear in more remaining triangles first (more constraining)
        # For simplicity, just pick the first available vertex in 'order'
        v = None
        for vertex in order:
            if candidates & (1 << vertex):
                v = vertex
                break
        
        if v is None:
            return
        
        v_bit = 1 << v
        
        # Option 1: exclude v
        search(S, candidates & ~v_bit, remaining_triangles)
        
        # Option 2: include v (if it doesn't complete any triangle)
        # Check: for each triangle containing v, if the other two vertices are in S, can't add v
        can_add = True
        for tri in remaining_triangles:
            if tri & v_bit:
                # Check if the other two vertices of this triangle are both in S
                other = tri & ~v_bit
                if (S & other) == other:
                    can_add = False
                    break
        
        if can_add:
            new_S = S | v_bit
            # Update remaining triangles: remove any triangle that has v and at least one vertex not in candidates
            # Actually, simpler: just pass the same triangle list and check at each step
            new_candidates = candidates & ~v_bit
            search(new_S, new_candidates, remaining_triangles)
    
    search(0, full_mask, triangles)
    return best


def _is_k4_free(adj):
    """Check if graph is K4-free using bitmask."""
    for i in range(N):
        for j in range(i+1, N):
            if adj[i] & (1 << j):
                common = adj[i] & adj[j]
                c = common
                while c:
                    lsb = c & (-c)
                    k = lsb.bit_length() - 1
                    c ^= lsb
                    if k > j and (adj[k] & common):
                        # Check if k is adjacent to another common neighbor
                        # Actually, K4 exists iff there's a triangle in the common neighborhood
                        # Simpler: just check if any edge exists in common
                        c2 = common
                        while c2:
                            lsb2 = c2 & (-c2)
                            m = lsb2.bit_length() - 1
                            c2 ^= lsb2
                            if m > k and (adj[k] & (1 << m)):
                                return False
    return True


def _is_k4_free_simple(adj):
    """Simple K4-free check."""
    for i in range(N):
        for j in range(i+1, N):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j]
            # Check if common neighborhood has any edge
            c = common
            while c:
                lsb = c & (-c)
                k = lsb.bit_length() - 1
                c ^= lsb
                if k > i and k > j:
                    # Check if k is connected to another vertex in common
                    c2 = common & ~(1 << k)
                    while c2:
                        lsb2 = c2 & (-c2)
                        m = lsb2.bit_length() - 1
                        c2 ^= lsb2
                        if m > k and (adj[k] & (1 << m)):
                            return False
    return True


def _is_k4_free_fast(adj):
    """Faster K4-free check using precomputed common neighborhoods."""
    for i in range(N):
        for j in range(i+1, N):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j]
            # Check if common has any edge: for each k in common, check if adj[k] & common has another vertex
            c = common
            while c:
                lsb = c & (-c)
                k = lsb.bit_length() - 1
                c ^= lsb
                # If k is connected to any other vertex in common, we have K4
                if adj[k] & (common & ~(1 << k)):
                    return False
    return True


def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # Strategy: Start with a good initial graph and do local search
    # Key: use a faster scoring function and smarter moves
    
    # Initial: try several random K4-free graphs and pick the best
    
    def random_k4_free():
        """Generate a random K4-free graph."""
        adj = [0] * N
        edges_to_try = [(i, j) for i in range(N) for j in range(i+1, N)]
        rng.shuffle(edges_to_try)
        
        for (u, v) in edges_to_try:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not _is_k4_free_fast(adj):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        return adj
    
    def structured_graph():
        """Try a structured K4-free graph."""
        adj = [0] * N
        for i in range(N):
            for j in range(i+1, N):
                if rng.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        changed = True
        while changed:
            changed = False
            for i in range(N):
                for j in range(i+1, N):
                    if adj[i] & (1 << j):
                        common = adj[i] & adj[j]
                        c = common
                        has_edge = False
                        while c:
                            lsb = c & (-c)
                            k = lsb.bit_length() - 1
                            c ^= lsb
                            if adj[k] & (common & ~(1 << k)):
                                has_edge = True
                                break
                        if has_edge:
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
                            changed = True
        return adj
    
    # Generate multiple candidates
    candidates = []
    
    candidates.append(_complement_of_3_regular())
    candidates.append(_q4_extended())
    candidates.extend(_try_more_structured())
    
    for _ in range(10):
        candidates.append(random_k4_free())
    for _ in range(5):
        candidates.append(structured_graph())
    
    # Score all candidates
    best_adj = None
    best_score = N + 1
    
    for adj in candidates:
        if (time.time() - start_time) > DEADLINE - 40:
            break
        if not _is_k4_free_fast(adj):
            continue
        s = _score_fast(adj)
        if s < best_score:
            best_score = s
            best_adj = adj[:]
    
    # Local search on best_adj
    if best_adj is not None:
        adj = best_adj[:]
        
        def get_edges(adj):
            edges = []
            for i in range(N):
                for j in range(i+1, N):
                    if adj[i] & (1 << j):
                        edges.append((i, j))
            return edges
        
        def get_non_edges(adj):
            edges = []
            for i in range(N):
                for j in range(i+1, N):
                    if not (adj[i] & (1 << j)):
                        edges.append((i, j))
            return edges
        
        improved = True
        while improved and (time.time() - start_time) < DEADLINE - 15:
            improved = False
            
            # Try removing edges - focus on edges that might be in critical triangles
            edges = get_edges(adj)
            rng.shuffle(edges)
            for (u, v) in edges:
                if (time.time() - start_time) > DEADLINE - 15:
                    break
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                s = _score_fast(adj)
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
                if (time.time() - start_time) > DEADLINE - 15:
                    break
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if _is_k4_free_fast(adj):
                    s = _score_fast(adj)
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
        for j in range(i+1, N):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    return result


def _try_more_structured():
    """Try additional structured K4-free graphs based on complement of sparse graphs."""
    rng = random.Random(777)
    candidates = []
    
    # Complement of a cycle C18 (2-regular)
    adj_cycle = [0] * N
    for v in range(N):
        adj_cycle[v] |= (1 << ((v+1) % N))
        adj_cycle[v] |= (1 << ((v-1) % N))
    comp = [0] * N
    for u in range(N):
        for v in range(u+1, N):
            if not (adj_cycle[u] & (1 << v)):
                comp[u] |= (1 << v)
                comp[v] |= (1 << u)
    if _is_k4_free_fast(comp):
        candidates.append(comp)
    
    # Complement of two disjoint 9-cycles
    adj_2cyc = [0] * N
    for v in range(9):
        adj_2cyc[v] |= (1 << ((v+1) % 9))
        adj_2cyc[v] |= (1 << ((v-1) % 9))
    for v in range(9, 18):
        adj_2cyc[v] |= (1 << ((v-9+1) % 9 + 9))
        adj_2cyc[v] |= (1 << ((v-9-1) % 9 + 9))
    comp = [0] * N
    for u in range(N):
        for v in range(u+1, N):
            if not (adj_2cyc[u] & (1 << v)):
                comp[u] |= (1 << v)
                comp[v] |= (1 << u)
    if _is_k4_free_fast(comp):
        candidates.append(comp)
    
    return candidates


if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
