import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 18
    all_vertices_mask = (1 << n) - 1
    
    def is_k4_free(adj):
        # adj is a list of bitmasks
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    continue
                for k in range(j + 1, n):
                    if not (adj[i] & (1 << k)) or not (adj[j] & (1 << k)):
                        continue
                    # Check if any l > k is common neighbor of i, j, k
                    candidates = adj[i] & adj[j] & adj[k]
                    candidates &= ~(1 << i) & ~(1 << j) & ~(1 << k)
                    # Only need to check l > k
                    candidates &= ((1 << n) - 1) ^ ((1 << (k + 1)) - 1)
                    if candidates:
                        return False
        return True

    def get_triangles(adj):
        triangles = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    continue
                common = adj[i] & adj[j] & ~(1 << i) & ~(1 << j)
                # k must be > j
                k_mask = common & (((1 << n) - 1) ^ ((1 << (j + 1)) - 1))
                while k_mask:
                    lsb = k_mask & -k_mask
                    k = lsb.bit_length() - 1
                    triangles.append((i, j, k))
                    k_mask ^= lsb
        return triangles

    def max_triangle_free_size(adj, triangles):
        # Check subsets of size k from n down to 1
        if not triangles:
            return n
        
        # Precompute triangle bitmasks
        tri_masks = [0] * len(triangles)
        for idx, (i, j, k) in enumerate(triangles):
            tri_masks[idx] = (1 << i) | (1 << j) | (1 << k)
            
        for k in range(n, 0, -1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                contains_tri = False
                for tm in tri_masks:
                    if (mask & tm) == tm:
                        contains_tri = True
                        break
                if not contains_tri:
                    return k
        return 0

    def max_triangle_free_size_fast(adj):
        """
        Faster check: use the fact that a set is triangle-free iff it contains no triangle.
        We can use a branch-and-bound or just iterate but with early termination.
        For n=18, checking all subsets of size 12 is C(18,12) = 18564.
        Checking all subsets of size 11 is C(18,11) = 31824.
        This is small enough to do directly if we optimize the triangle check.
        """
        # Precompute for each vertex, the set of triangles it's in?
        # Or just precompute triangle masks.
        triangles = get_triangles(adj)
        if not triangles:
            return n
        
        tri_masks = []
        for (i, j, k) in triangles:
            tri_masks.append((1 << i) | (1 << j) | (1 << k))
        
        # Sort triangles by "difficulty" or just use as is
        # To check if a mask contains a triangle, we can iterate triangles.
        # But we can be smarter: if we have a list of triangles, for a given mask,
        # we check if any triangle is a subset.
        
        # Optimization: If we are checking size k, and we know that any triangle must have
        # at least 2 vertices in the set (impossible, triangle has 3).
        # So if the set contains all 3 vertices of any triangle, it's not triangle-free.
        
        # Let's try to prune:
        # If a vertex v is in many triangles, maybe avoid it?
        
        # Simple approach: iterate combinations.
        # To speed up, we can precompute for each vertex the list of triangles containing it.
        # Then for a subset, we can check if any triangle is fully contained.
        
        # Even faster: 
        # A set S is triangle-free iff for every edge (u,v) in S, N(u) ∩ N(v) ∩ S is empty.
        # Or: for every triangle, not all 3 vertices are in S.
        
        # Let's stick to the triangle mask approach but optimize the inner loop.
        # We can precompute the list of triangles.
        # For a given mask, we check:
        # for tm in tri_masks: if mask & tm == tm: return False
        
        # To make this faster, we can group triangles by their smallest vertex?
        # Or just use the fact that n=18 is small.
        
        # Let's try a different approach: 
        # Find the maximum independent set in the "triangle hypergraph"?
        # No, just check sizes.
        
        # We can use a heuristic to find a large triangle-free set first,
        # then prove no larger one exists?
        # Or just check from n down.
        
        # Given the small n, the combination iteration is the bottleneck.
        # Let's try to optimize the triangle check.
        
        # Precompute: for each pair of vertices (i,j) that form an edge,
        # the common neighbors.
        # A triangle (i,j,k) exists if k is a common neighbor of i,j.
        
        # Alternative:
        # For a candidate set S (mask), check if it contains a triangle.
        # We can iterate over edges in S. For each edge (u,v) in S,
        # check if there is a w in S such that w is neighbor of u and v.
        
        # This might be faster if S is small, but S is large (size ~12).
        
        # Let's stick to the triangle mask method but optimize it.
        # We can sort triangles by frequency or something?
        
        # Actually, for n=18, the number of triangles in a K4-free graph is limited.
        # Max triangles in K4-free graph on 18 vertices?
        # By Turan, K4-free max edges is 2 * C(9,2) + ... no, Turan for K4 is complete tripartite.
        # T3(18) = 6+6+6. Edges = 3*36 = 108.
        # Triangles in K6,6,6: 0.
        # So random K4-free graphs might have few triangles.
        
        # Let's just run the combination check.
        
        for k in range(n, 0, -1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                # Check if mask contains any triangle
                contains_tri = False
                for tm in tri_masks:
                    if (mask & tm) == tm:
                        contains_tri = True
                        break
                if not contains_tri:
                    return k
        return 0

    def random_k4_free_graph():
        # Start with random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Remove K4s greedily
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        continue
                    for k in range(j + 1, n):
                        if not (adj[i] & (1 << k)) or not (adj[j] & (1 << k)):
                            continue
                        for l in range(k + 1, n):
                            if (adj[i] & (1 << l)) and (adj[j] & (1 << l)) and (adj[k] & (1 << l)):
                                # Remove edge (i, l)
                                adj[i] &= ~(1 << l)
                                adj[l] &= ~(1 << i)
                                changed = True
                                break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
        return adj

    def local_search(adj):
        # Try to improve the graph by adding/removing edges
        # Goal: minimize max_triangle_free_size
        # We can try to add edges to break large triangle-free sets,
        # or remove edges to break K4s (if we added too many).
        
        best_adj = adj[:]
        best_score = max_triangle_free_size_fast(best_adj)
        
        for _ in range(100):  # Limit iterations
            if time.time() > deadline:
                break
            
            # Try adding a random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if not (best_adj[i] & (1 << j)):
                new_adj = best_adj[:]
                new_adj[i] |= (1 << j)
                new_adj[j] |= (1 << i)
                if is_k4_free(new_adj):
                    score = max_triangle_free_size_fast(new_adj)
                    if score < best_score:
                        best_adj = new_adj
                        best_score = score
            
            # Try removing a random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if best_adj[i] & (1 << j):
                new_adj = best_adj[:]
                new_adj[i] &= ~(1 << j)
                new_adj[j] &= ~(1 << i)
                # Removing edge always keeps K4-free
                score = max_triangle_free_size_fast(new_adj)
                if score < best_score:
                    best_adj = new_adj
                    best_score = score
        
        return best_adj, best_score

    best_score = 12
    best_edges = None
    
    # Seed construction: K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    seed_edges = [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    
    # Try to find a better graph
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        adj = random_k4_free_graph()
        
        if not is_k4_free(adj):
            continue
        
        # Local search
        adj, score = local_search(adj)
        
        if score < best_score:
            best_score = score
            # Convert adj to edges
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            best_edges = edges
            
        if best_score <= 10:
            break
            
    if best_edges is not None:
        return best_edges
    else:
        return seed_edges
