import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def popcount(x):
        return bin(x).count('1')

    def build_adj_bitset(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def is_triangle_free(adj, n):
        for i in range(n):
            neighbors = adj[i]
            while neighbors:
                lsb = neighbors & (-neighbors)
                j = lsb.bit_length() - 1
                neighbors ^= lsb
                common = adj[j] & adj[i] & ~(1 << i) & ~(1 << j)
                if common:
                    return False
        return True

    def find_independent_set(adj, n, k, time_limit=2.0):
        """
        Returns an independent set of size k if found, else None.
        Uses randomized greedy + backtracking if needed, but greedy is usually fast for detection.
        """
        start_t = time.time()
        # Try a few random orderings for greedy
        for _ in range(10):
            if time.time() - start_t > time_limit:
                break
            candidates = (1 << n) - 1
            current_set = 0
            while candidates and popcount(current_set) < k:
                # Pick vertex with min degree in remaining subgraph (greedy for independent set)
                best_v = -1
                best_deg = float('inf')
                temp = candidates
                # Randomize tie-breaking
                vertices = []
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    temp ^= lsb
                    deg = popcount(adj[v] & candidates)
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                        vertices = [v]
                    elif deg == best_deg:
                        vertices.append(v)
                
                if best_v == -1:
                    break
                
                # Random pick among min-degree vertices
                v = random.choice(vertices)
                current_set |= (1 << v)
                candidates &= ~adj[v] & ~(1 << v)
                
            if popcount(current_set) >= k:
                return current_set
        return None

    def verify_graph(n, edges):
        adj = build_adj_bitset(n, edges)
        if not is_triangle_free(adj, n):
            return False
        # Check if independent set of size 10 exists
        # If it does, invalid. If not, valid.
        if find_independent_set(adj, n, 10, time_limit=5.0):
            return False
        return True

    def generate_random_triangle_free(n, seed_val, density=0.4):
        random.seed(seed_val)
        adj = [0] * n
        # Start with random graph
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Remove triangles
        for _ in range(200):
            if time.time() - start_time > deadline - 5:
                break
            found = False
            for i in range(n):
                ni = adj[i]
                while ni:
                    lsb1 = ni & (-ni)
                    j = lsb1.bit_length() - 1
                    ni ^= lsb1
                    common = adj[j] & adj[i] & ~(1 << i) & ~(1 << j)
                    if common:
                        lsb2 = common & (-common)
                        k = lsb2.bit_length() - 1
                        # Remove random edge from triangle (i,j,k)
                        choice = random.randint(0, 2)
                        if choice == 0:
                            adj[i] &= ~(1 << j); adj[j] &= ~(1 << i)
                        elif choice == 1:
                            adj[i] &= ~(1 << k); adj[k] &= ~(1 << i)
                        else:
                            adj[j] &= ~(1 << k); adj[k] &= ~(1 << j)
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        return adj

    def reduce_alpha_local_search(adj, n, max_iter=500):
        """
        Add edges to reduce independence number while maintaining triangle-free.
        """
        for _ in range(max_iter):
            if time.time() - start_time > deadline - 5:
                break
            
            # Check if we already have alpha < 10
            if not find_independent_set(adj, n, 10, time_limit=1.0):
                break
            
            # Find an independent set I of size >= 10
            I = find_independent_set(adj, n, 10, time_limit=1.0)
            if I is None:
                break
                
            # Try to add an edge within I
            added = False
            # Get list of vertices in I
            verts = []
            temp = I
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                temp ^= lsb
                verts.append(v)
                
            random.shuffle(verts)
            for idx in range(len(verts)):
                for jdx in range(idx + 1, len(verts)):
                    u = verts[idx]
                    v = verts[jdx]
                    if not (adj[u] & (1 << v)):
                        # Check if adding (u,v) creates a triangle
                        # Common neighbors of u and v must be empty
                        if not (adj[u] & adj[v]):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            added = True
                            break
                if added:
                    break
            
            if not added:
                # Perturbation: remove a random edge to allow new structure?
                # Or just break and try a new random graph
                break
                
        return adj

    best = {'n': 27, 'edges': []}
    
    # Baseline: Andrásfai A_9 (n=27, alpha=9)
    def andrasfai_graph(k):
        n = 3 * k
        edges = []
        for a in range(3):
            for i in range(k):
                for b in range(3):
                    if (b - a) % 3 == 1:
                        for j in range(k):
                            if (j - i) % k in (0, 1):
                                u = a * k + i
                                v = b * k + j
                                if u < v:
                                    edges.append([u, v])
        return {'n': n, 'edges': edges}

    ag = andrasfai_graph(9)
    if verify_graph(ag['n'], ag['edges']):
        best = ag

    # Try to find larger valid graphs
    for target_n in range(28, 41):
        if time.time() - start_time > deadline - 10:
            break

        found = False
        # Multiple attempts with different seeds and densities
        for attempt in range(30):
            if time.time() - start_time > deadline - 5:
                break
            
            seed_val = attempt * 100003 + target_n * 7
            density = 0.3 + (attempt % 5) * 0.05
            
            adj = generate_random_triangle_free(target_n, seed_val, density)
            if not is_triangle_free(adj, target_n):
                continue
            
            adj = reduce_alpha_local_search(adj, target_n)
            
            if is_triangle_free(adj, target_n) and not find_independent_set(adj, target_n, 10, time_limit=2.0):
                edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                best = {'n': target_n, 'edges': edges}
                found = True
                break
        
        if found:
            continue
        else:
            # If failed, try a few more with lower density
            for attempt in range(30, 50):
                if time.time() - start_time > deadline - 5:
                    break
                seed_val = attempt * 7919 + target_n * 31 + 42
                adj = generate_random_triangle_free(target_n, seed_val, 0.25)
                if not is_triangle_free(adj, target_n):
                    continue
                adj = reduce_alpha_local_search(adj, target_n)
                if is_triangle_free(adj, target_n) and not find_independent_set(adj, target_n, 10, time_limit=2.0):
                    edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    best = {'n': target_n, 'edges': edges}
                    found = True
                    break

    return best
