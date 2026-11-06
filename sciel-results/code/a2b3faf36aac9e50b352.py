"""K4-free graphs on 18 vertices minimizing largest triangle-free induced subgraph."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _is_k4_free(adj):
    """Check if graph is K4-free using bitmask adjacency."""
    n = 18
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] >> j) & 1:
                continue
            for k in range(j + 1, n):
                if not (adj[i] >> k) & 1:
                    continue
                for l in range(k + 1, n):
                    if (adj[i] >> l) & 1 and (adj[j] >> l) & 1 and (adj[k] >> l) & 1:
                        return False
    return True

def _max_triangle_free_size(adj):
    """Find largest subset of vertices inducing a triangle-free subgraph via branch and bound."""
    n = 18
    # Precompute: for each vertex, the set of vertices that form a triangle with it
    # tri_mask[i] = bitmask of vertices j such that i,j and their common neighbor form triangle
    # Actually we need: for a set S to be triangle-free, no three vertices in S form a triangle.
    # We'll use branch and bound: try to include vertices, prune if adding v creates a triangle.
    
    # Precompute triangle membership: tri[i][j] = set of k > j such that i,j,k form a triangle
    # For B&B: maintain current set, and for each candidate v, check if v with any two in current set forms triangle.
    
    # Faster approach: precompute for each pair (i,j) the set of common neighbors
    common = [0] * (n * n)
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                common[i * n + j] = adj[i] & adj[j] & ((1 << n) - 1)
            else:
                common[i * n + j] = 0
    
    best = [0]
    
    # Branch and bound: try to build a triangle-free set
    # State: (vertex_index_to_consider, current_set_mask, current_size)
    # Prune: if current_size + remaining <= best[0], return
    
    def bb(pos, cur_mask, cur_size):
        if cur_size + (n - pos) <= best[0]:
            return
        if pos == n:
            if cur_size > best[0]:
                best[0] = cur_size
            return
        
        v = pos
        # Option 1: exclude v
        bb(pos + 1, cur_mask, cur_size)
        
        # Option 2: include v, if it doesn't create a triangle with any pair in cur_mask
        # Check: for each pair (a,b) in cur_mask where a < b, if a,b,v form a triangle, can't include v
        ok = True
        m = cur_mask
        while m:
            lb = m & (-m)
            a = lb.bit_length() - 1
            m ^= lb
            # For each b in cur_mask with b > a
            m2 = m  # vertices after a in cur_mask
            while m2:
                lb2 = m2 & (-m2)
                b = lb2.bit_length() - 1
                m2 ^= lb2
                if a < b and (common[a * n + b] >> v) & 1:
                    ok = False
                    break
            if not ok:
                break
        
        if ok:
            bb(pos + 1, cur_mask | (1 << v), cur_size + 1)
    
    bb(0, 0, 0)
    return best[0]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n = 18
    all_edges_mask = (1 << n) - 1
    
    # Start with random K4-free graph
    # Generate random graph and check K4-free, or use tripartite as seed
    # Let's start with the tripartite graph as initial solution
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    
    def make_adj_from_parts(parts):
        adj = [0] * n
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        return adj
    
    adj = make_adj_from_parts(parts)
    current_score = _max_triangle_free_size(adj)
    best_adj = adj[:]
    best_score = current_score
    
    # Simulated annealing
    temp = 50.0
    cooling = 0.9995
    min_temp = 0.1
    rng = random.Random(42)
    
    def random_neighbor(adj):
        """Flip a random edge, check K4-free."""
        new_adj = adj[:]
        # Pick random non-loop edge
        while True:
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            break
        
        # Flip edge (i,j)
        bit_i = (1 << j)
        bit_j = (1 << i)
        if (adj[i] >> j) & 1:
            # Remove edge
            new_adj[i] &= ~bit_i
            new_adj[j] &= ~bit_j
        else:
            # Add edge
            new_adj[i] |= bit_i
            new_adj[j] |= bit_j
        
        # Check K4-free
        if not _is_k4_free(new_adj):
            return None
        return new_adj
    
    iterations = 0
    while time.time() < deadline:
        new_adj = random_neighbor(adj)
        if new_adj is None:
            temp *= cooling
            continue
        
        new_score = _max_triangle_free_size(new_adj)
        delta = new_score - current_score
        
        if delta <= 0:
            # Improvement or equal
            adj = new_adj
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = adj[:]
                print(f"New best: {best_score} at iter {iterations}")
        else:
            # Worse: accept with probability exp(-delta/temp)
            prob = np.exp(-delta / max(temp, 1e-10))
            if rng.random() < prob:
                adj = new_adj
                current_score = new_score
        
        iterations += 1
        temp *= cooling
        if temp < min_temp:
            temp = min_temp
    
    # Return edges from best_adj
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    return edges

# EVOLVE-BLOCK-END
