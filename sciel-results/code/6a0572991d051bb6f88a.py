import time
import itertools

def construct():
    n = 17
    minv = 8
    edges = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], 
             [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], 
             [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], 
             [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], 
             [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], 
             [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], 
             [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], 
             [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], 
             [8, 9], [8, 10], [8, 12], [8, 16], 
             [9, 10], [9, 11], [9, 13], 
             [10, 11], [10, 12], [10, 14], 
             [11, 12], [11, 13], [11, 15], 
             [12, 13], [12, 14], [12, 16], 
             [13, 14], [13, 15], 
             [14, 15], [14, 16], 
             [15, 16]]

    start = time.time()
    deadline = start + 55.0

    # Build adjacency list and bitmask representations
    adj = [[] for _ in range(n)]
    adj_mask = [0] * n
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        adj_mask[a] |= (1 << b)
        adj_mask[b] |= (1 << a)

    def distinct_degree_count(vertex_list):
        """Return the number of distinct degrees in the induced subgraph."""
        vs = set(vertex_list)
        degs = set()
        for v in vertex_list:
            deg = sum(1 for u in vertex_list if u in adj_mask[v] or (v != u and adj_mask[v] >> u & 1))
            # Simpler: count neighbors within the set
            deg = 0
            for u in adj[v]:
                if u in vs:
                    deg += 1
            degs.add(deg)
        return len(degs)

    def distinct_degree_count_mask(mask):
        """Return the number of distinct degrees using bitmask representation."""
        degs = set()
        m = mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            deg = (adj_mask[v] & mask).bit_count()
            degs.add(deg)
            m ^= lb
        return len(degs)

    best_count = -1
    best_vertices = None

    # Phase 1: Exhaustive search (fast, 2^17 subsets)
    for mask in range(1 << n):
        if time.time() > deadline:
            break
        sz = mask.bit_count()
        if sz < minv:
            continue
        nd = distinct_degree_count_mask(mask)
        if nd > best_count:
            best_count = nd
            best_vertices = [int(i) for i in range(n) if mask & (1 << i)]
            if nd >= 9:
                break  # Theoretical max for 8-regular graph

    if best_vertices is None:
        # Fallback
        return [int(i) for i in range(minv)]

    # Phase 2: Greedy with multiple random seeds
    # This is redundant if exhaustive search found the global optimum,
    # but we add it as a distinct algorithmic approach.
    import random
    random.seed(42)

    def greedy_build(seed_vertices, max_iter=1000):
        """Greedy: start from seed, iteratively add the vertex that maximizes distinct degrees."""
        current = list(seed_vertices)
        current_set = set(current)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if len(current) >= n:
                break
            
            best_gain = -1
            best_v = None
            best_new_count = distinct_degree_count(current)
            
            for v in range(n):
                if v in current_set:
                    continue
                # Tentatively add v
                test_set = current_set | {v}
                test_list = current + [v]
                # Compute distinct degrees
                degs = set()
                for u in test_list:
                    deg = 0
                    for w in adj[u]:
                        if w in test_set:
                            deg += 1
                    degs.add(deg)
                nd = len(degs)
                if nd > best_gain:
                    best_gain = nd
                    best_v = v
            
            if best_v is None:
                break
            
            current.append(best_v)
            current_set.add(best_v)
            
            if best_gain >= 9:
                break
        
        return current, distinct_degree_count(current)

    # Try different seed sizes
    for sz in range(3, 8):
        if time.time() > deadline:
            break
        # Sample multiple random seeds of this size
        for trial in range(50):
            if time.time() > deadline:
                break
            seed = random.sample(range(n), sz)
            verts, nd = greedy_build(seed)
            if nd > best_count:
                best_count = nd
                best_vertices = verts
                if nd >= 9:
                    break

    return [int(v) for v in best_vertices]
