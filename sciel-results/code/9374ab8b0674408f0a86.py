"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
import time
import random
from itertools import combinations

N = 18
DEADLINE = 115.0  # safety margin under 150s


def _score_fast(adj):
    """Fast branch-and-bound for max triangle-free induced subgraph size using bitmasks."""
    # Precompute: for each vertex v, adj[v] is bitmask of neighbors
    # A set S is triangle-free iff no triangle is fully contained in S
    # Triangle (u,v,w) exists iff u~v, v~w, w~u
    
    # Precompute triangle masks: for each edge (u,v), common neighbors
    # A set S contains a triangle iff there exists an edge (u,v) in S with a common neighbor in S
    
    # Better: precompute for each vertex v, the set of "forbidden pairs"
    # When building S incrementally, v can be added iff v doesn't complete a triangle with any pair in S
    
    # For vertex v and current set S: v can be added iff for all a in S, 
    # (adj[v] & adj[a] & S) == 0, i.e., v and a don't share a neighbor in S
    
    # Actually simpler: S is triangle-free iff for all v in S, (adj[v] & S) is independent
    # (adj[v] & S) is independent iff for all u in (adj[v] & S), adj[u] & (adj[v] & S) == 0
    
    # Let me use a different approach: enumerate all triangles and check none are in S
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
                    if k > j:  # avoid duplicates
                        triangles.append((1 << i) | (1 << j) | (1 << k))
    
    # Now use branch and bound to find max S with no triangle
    # Order vertices by degree for better pruning
    order = sorted(range(N), key=lambda v: bin(adj[v]).count('1'))
    order_set = set(order)
    
    best = 0
    
    def is_tf(S):
        """Check if set S (bitmask) is triangle-free by checking all triangles."""
        for tri in triangles:
            if S & tri == tri:
                return False
        return True
    
    # Branch and bound with vertex ordering
    # State: (current_set, available_mask, depth)
    # available_mask: vertices that can still be added
    
    import sys
    sys.setrecursionlimit(10000)
    
    # Use iterative approach with memoization-like pruning
    # Key insight: if we process vertices in a fixed order, we can use the fact that
    # once we skip a vertex, we never consider it again
    
    # Process vertices in 'order', at each step decide include or exclude
    # This gives 2^18 = 262144 possibilities, but we prune heavily
    
    # Better: use the triangle constraint to prune
    # When we include vertex v, we must exclude all vertices that form a triangle with two already-included vertices
    
    # Let me use a cleaner recursive approach
    def search(S, candidates):
        nonlocal best
        s_count = bin(S).count('1')
        if s_count > best:
            best = s_count
            if s_count == N:  # can't do better
                return
        c_count = bin(candidates).count('1')
        if s_count + c_count <= best:
            return
        if candidates == 0:
            return
        
        # Try to add vertices from candidates
        # For better pruning, try vertices that create more constraints first
        c = candidates
        while c:
            lsb = c & (-c)
            v = lsb.bit_length() - 1
            c ^= lsb
            
            # Check if v can be added to S
            # v can be added iff S ∪ {v} is triangle-free
            # This means: for all pairs (a,b) in S with a~b, v is not adjacent to both
            # Equivalently: for all a in S, (adj[v] & adj[a] & S) == 0
            
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
                # Check if w can be added to new_S
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


def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # Strategy: Start with a good initial graph and do local search
    # Key: use a faster scoring function and smarter moves
    
    # Initial: try several random K4-free graphs and pick the best
    
    def random_k4_free():
        """Generate a random K4-free graph."""
        adj = [0] * N
        # Start with no edges, add edges randomly while maintaining K4-free
        edges_to_try = [(i, j) for i in range(N) for j in range(i+1, N)]
        rng.shuffle(edges_to_try)
        
        for (u, v) in edges_to_try:
            # Try adding edge (u,v)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not _is_k4_free_simple(adj):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        return adj
    
    # Also try structured constructions
    def structured_graph():
        """Try a structured K4-free graph."""
        # Use a 4-partite graph with parts of sizes 5,4,4,5
        # This is automatically K4-free (no K4 in 4-partite with these sizes... wait, K4 needs 4 parts)
        # Actually a 4-partite graph can have K4 if we pick one vertex from each part
        # So this won't work directly
        
        # Instead: use a graph where neighborhoods are triangle-free
        # Start with a random graph and remove edges to make K4-free
        adj = [0] * N
        for i in range(N):
            for j in range(i+1, N):
                if rng.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Remove edges to make K4-free
        changed = True
        while changed:
            changed = False
            for i in range(N):
                for j in range(i+1, N):
                    if adj[i] & (1 << j):
                        common = adj[i] & adj[j]
                        # Check if common has an edge
                        c = common
                        has_edge = False
                        while c:
                            lsb = c & (-c)
                            k = lsb.bit_length() - 1
                            c ^= lsb
                            c2 = common & ~(1 << k)
                            while c2:
                                lsb2 = c2 & (-c2)
                                m = lsb2.bit_length() - 1
                                c2 ^= lsb2
                                if adj[k] & (1 << m):
                                    has_edge = True
                                    break
                            if has_edge:
                                break
                        if has_edge:
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
                            changed = True
        return adj
    
    # Generate multiple candidates
    candidates = []
    for _ in range(10):
        candidates.append(random_k4_free())
    for _ in range(5):
        candidates.append(structured_graph())
    
    # Score all candidates
    best_adj = None
    best_score = N + 1
    
    for adj in candidates:
        if (time.time() - start_time) > DEADLINE - 60:
            break
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
        
        # Also consider adding edges
        def get_non_edges(adj):
            edges = []
            for i in range(N):
                for j in range(i+1, N):
                    if not (adj[i] & (1 << j)):
                        edges.append((i, j))
            return edges
        
        improved = True
        while improved and (time.time() - start_time) < DEADLINE - 20:
            improved = False
            
            # Try removing edges
            edges = get_edges(adj)
            rng.shuffle(edges)
            for (u, v) in edges:
                if (time.time() - start_time) > DEADLINE - 20:
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
                if (time.time() - start_time) > DEADLINE - 20:
                    break
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if _is_k4_free_simple(adj):
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


if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
