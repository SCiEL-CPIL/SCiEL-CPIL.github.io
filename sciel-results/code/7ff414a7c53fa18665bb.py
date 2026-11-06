import random
import time
from itertools import combinations

N = 18
TOTAL_EDGES = N * (N - 1) // 2  # 153

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    best_edges = None
    best_score = 18  # worst case
    
    def edge_index(u, v):
        if u > v:
            u, v = v, u
        return u * N + v
    
    def make_adj_mask(edges_set):
        adj = [0] * N
        for (u, v) in edges_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def is_k4_free(adj):
        for i in range(N):
            for j in range(i+1, N):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    # check if any common neighbor forms K4
                    c = common
                    while c:
                        k = (c & -c).bit_length() - 1
                        c &= c - 1
                        if adj[k] & (1 << i) and adj[k] & (1 << j):
                            # i, j, k form a triangle; check if there's a 4th
                            # Actually K4 means 4 mutually connected vertices
                            # i-j edge, i-k edge, j-k edge. Need a 4th vertex connected to all three.
                            pass
                    # Better: check if the neighborhood of edge (i,j) contains a triangle
                    # i.e., is there a vertex k connected to both i and j, and another vertex l connected to i, j, and k?
                    # Simpler: for each pair (i,j) that is an edge, check if N(i) ∩ N(j) contains an edge
                    pass
        # Let me use a cleaner approach
        for i in range(N):
            for j in range(i+1, N):
                if not (adj[i] & (1 << j)):
                    continue
                # i and j are connected. Check if there exist k, l such that all 4 are mutually connected
                common = adj[i] & adj[j]
                # Remove i and j from common (they shouldn't be in each other's adj in the intersection sense... actually adj[i] has j, adj[j] has i, so common includes neither i nor j since adj[i] doesn't have i)
                # common = vertices connected to both i and j
                c = common
                while c:
                    k = (c & -c).bit_length() - 1
                    c &= c - 1
                    # Now check if k is connected to some l that is also in common and l != k
                    common_without_k = common & ~(1 << k)
                    if adj[k] & common_without_k:
                        return False
        return True
    
    def max_triangle_free_size(adj):
        """Find the largest subset of vertices with no triangles."""
        best = 0
        # Try all subsets? 2^18 = 262144, feasible
        for mask in range(1 << N):
            if bin(mask).count('1') <= best:
                continue
            # Check if this subset is triangle-free
            tri_free = True
            for i in range(N):
                if not (mask >> i) & 1:
                    continue
                for j in range(i+1, N):
                    if not (mask >> j) & 1:
                        continue
                    if not (adj[i] & (1 << j)):
                        continue
                    # i, j are connected in the subset. Check for common neighbor in subset
                    common = adj[i] & adj[j] & mask
                    if common:
                        tri_free = False
                        break
                if not tri_free:
                    break
            if tri_free:
                best = bin(mask).count('1')
        return best
    
    def max_triangle_free_size_fast(adj, adj_masks):
        """Faster version using precomputed adjacency."""
        best = 0
        # Use branch and bound or just check all subsets
        # 2^18 = 262144 subsets, for each check triangles
        # Precompute: for each pair, whether they have a common neighbor
        # A subset is triangle-free iff for every edge (i,j) in the subset, no common neighbor of i,j is in the subset
        
        # Better: iterate over subsets in decreasing size order and stop at first triangle-free
        # But that's expensive. Let's just check all.
        
        # Actually, let's use a smarter approach:
        # A set S is triangle-free iff for all i in S, the neighbors of i in S form an independent set
        # Or: for all edges (i,j) in G[S], N(i) ∩ N(j) ∩ S = ∅
        
        for mask in range(1 << N):
            sz = bin(mask).count('1')
            if sz <= best:
                continue
            tri_free = True
            for i in range(N):
                if not (mask >> i) & 1:
                    continue
                # Get neighbors of i within the subset
                ni = adj[i] & mask
                # Check if any two neighbors of i in the subset are connected
                # i.e., is there j, k in ni such that j-k is an edge?
                c = ni
                while c:
                    j = (c & -c).bit_length() - 1
                    c &= c - 1
                    if adj[j] & c:
                        tri_free = False
                        break
                if not tri_free:
                    break
            if tri_free:
                best = sz
        return best
    
    def generate_random_k4_free():
        """Generate a random K4-free graph."""
        # Start with random edges, then remove edges that create K4
        edges = set()
        adj = [0] * N
        
        # Add random edges
        order = list(range(TOTAL_EDGES))
        random.shuffle(order)
        
        for idx in order:
            u = idx // (N - 1 - idx // (N - 1))  # This is wrong
            # Let me just iterate over all pairs
            pass
        
        # Simpler: randomly decide for each edge
        edges_set = set()
        adj = [0] * N
        for i in range(N):
            for j in range(i+1, N):
                if random.random() < 0.5:
                    edges_set.add((i, j))
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Remove edges that create K4
        changed = True
        while changed:
            changed = False
            for i in range(N):
                for j in range(i+1, N):
                    if (i, j) not in edges_set:
                        continue
                    common = adj[i] & adj[j]
                    c = common
                    while c:
                        k = (c & -c).bit_length() - 1
                        c &= c - 1
                        common_without_k = common & ~(1 << k)
                        if adj[k] & common_without_k:
                            # Removing edge (i,j) breaks this K4
                            edges_set.remove((i, j))
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
        
        return edges_set, adj
    
    def local_search(edges_set, adj):
        """Try to improve by flipping edges."""
        current_score = max_triangle_free_size_fast(adj, None)
        
        for _ in range(200):  # iterations
            if time.time() > deadline:
                break
            
            # Try random edge flips
            improved = False
            for _ in range(50):
                i, j = random.sample(range(N), 2)
                if i > j:
                    i, j = j, i
                
                if (i, j) in edges_set:
                    # Remove edge
                    edges_set.remove((i, j))
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                else:
                    # Add edge, but check K4-free
                    common = adj[i] & adj[j]
                    c = common
                    ok = True
                    while c:
                        k = (c & -c).bit_length() - 1
                        c &= c - 1
                        common_without_k = common & ~(1 << k)
                        if adj[k] & common_without_k:
                            ok = False
                            break
                    if not ok:
                        continue
                    edges_set.add((i, j))
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                
                new_score = max_triangle_free_size_fast(adj, None)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    break
                else:
                    # Revert
                    if (i, j) in edges_set:
                        edges_set.remove((i, j))
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                    else:
                        edges_set.add((i, j))
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            if not improved:
                break
        
        return edges_set, adj, current_score
    
    # Main loop: random restarts with local search
    for restart in range(50):
        if time.time() > deadline:
            break
        
        edges_set, adj = generate_random_k4_free()
        edges_set, adj, score = local_search(edges_set, adj)
        
        if score < best_score:
            best_score = score
            best_edges = list(edges_set)
            if best_score <= 10:
                break
    
    if best_edges is None:
        # Fallback: use the tripartite graph
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        best_edges = [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    
    return [[u, v] for u, v in best_edges]
