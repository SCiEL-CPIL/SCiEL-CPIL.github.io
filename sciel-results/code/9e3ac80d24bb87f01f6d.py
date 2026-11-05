import random
import time
import math

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    random.seed(42)
    
    n = 18
    
    def build_adj_from_edges(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def is_k4_free(adj):
        # Check: for every edge (u,v), N(u) ∩ N(v) must be independent
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    common = adj[u] & adj[v]
                    for a in common:
                        for b in common:
                            if a < b and b in adj[a]:
                                return False
        return True
    
    def max_triangle_free(adj):
        # Find largest set of vertices inducing no triangle
        # Use branch and bound
        best = [0]
        
        def has_triangle(vertices):
            # Check if any triangle exists among vertices
            verts = list(vertices)
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    for k in range(j+1, len(verts)):
                        a, b, c = verts[i], verts[j], verts[k]
                        if b in adj[a] and c in adj[a] and c in adj[b]:
                            return True
            return False
        
        def branch(remaining, current_size, current_set):
            if current_size + len(remaining) <= best[0]:
                return
            if not remaining:
                if current_size > best[0]:
                    best[0] = current_size
                return
            # Pick a vertex from remaining
            v = remaining[0]
            rest = remaining[1:]
            
            # Option 1: include v
            new_set = current_set | {v}
            if not has_triangle(new_set):
                branch(rest, current_size + 1, new_set)
            
            # Option 2: exclude v
            branch(rest, current_size, current_set)
        
        branch(set(range(n)), 0, set())
        return best[0]
    
    def max_triangle_free_fast(adj):
        # Faster: use bitmasks for triangle checking
        # adj_mask[i] = bitmask of neighbors of i
        adj_mask = [0] * n
        for i in range(n):
            for j in adj[i]:
                adj_mask[i] |= (1 << j)
        
        best = [0]
        
        def has_triangle_mask(verts_mask):
            # Check if any triangle in the set
            verts = [i for i in range(n) if (verts_mask >> i) & 1]
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    a, b = verts[i], verts[j]
                    if (adj_mask[a] >> b) & 1:
                        # a-b edge exists, check common neighbors in set
                        common = adj_mask[a] & adj_mask[b] & verts_mask
                        if common:
                            return True
            return False
        
        def branch(remaining_mask, current_mask, current_size):
            if current_size + bin(remaining_mask).count('1') <= best[0]:
                return
            if remaining_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Pick lowest set bit
            v = (remaining_mask & (-remaining_mask)).bit_length() - 1
            rest_mask = remaining_mask & ~(1 << v)
            
            # Include v
            new_mask = current_mask | (1 << v)
            if not has_triangle_mask(new_mask):
                branch(rest_mask, new_mask, current_size + 1)
            
            # Exclude v
            branch(rest_mask, current_mask, current_size)
        
        branch((1 << n) - 1, 0, 0)
        return best[0]
    
    # Start with two C9 cycles
    def make_c9_edges(offset):
        edges = []
        for i in range(9):
            u = offset + i
            v = offset + (i + 1) % 9
            edges.append([u, v])
        return edges
    
    base_edges = make_c9_edges(0) + make_c9_edges(9)
    adj = build_adj_from_edges(base_edges)
    current_score = max_triangle_free_fast(adj)
    best_edges = list(base_edges)
    best_score = current_score
    
    # Simulated annealing: try adding edges
    all_possible_edges = set()
    for i in range(n):
        for j in range(i+1, n):
            all_possible_edges.add((i, j))
    
    current_edges = set(tuple(e) for e in base_edges)
    current_adj = build_adj_from_edges(list(current_edges))
    
    def can_add_edge(adj, u, v):
        if v in adj[u]:
            return False
        # Check K4 constraint: common neighbors must be independent
        common = adj[u] & adj[v]
        for a in common:
            for b in common:
                if a < b and b in adj[a]:
                    return False
        return True
    
    def try_add_edge(adj, u, v):
        if not can_add_edge(adj, u, v):
            return None, None
        new_adj = [set(s) for s in adj]
        new_adj[u].add(v)
        new_adj[v].add(u)
        return new_adj, (u, v)
    
    def try_remove_edge(adj, u, v):
        new_adj = [set(s) for s in adj]
        new_adj[u].discard(v)
        new_adj[v].discard(u)
        return new_adj
    
    # Simulated annealing
    T = 1.0
    cooling = 0.999
    best_adj = [set(s) for s in current_adj]
    best_edges_set = set(current_edges)
    
    while time.time() < deadline and T > 0.01:
        # Pick a random edge to toggle
        if random.random() < 0.5:
            # Try adding an edge
            candidates = [e for e in all_possible_edges if e not in current_edges]
            if not candidates:
                T *= cooling
                continue
            u, v = random.choice(candidates)
            new_adj, _ = try_add_edge(current_adj, u, v)
            if new_adj is None:
                T *= cooling
                continue
            new_score = max_triangle_free_fast(new_adj)
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_adj = new_adj
                current_edges.add((u, v))
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = [set(s) for s in current_adj]
                    best_edges_set = set(current_edges)
        else:
            # Try removing an edge
            if not current_edges:
                T *= cooling
                continue
            u, v = random.choice(list(current_edges))
            new_adj = try_remove_edge(current_adj, u, v)
            new_score = max_triangle_free_fast(new_adj)
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_adj = new_adj
                current_edges.discard((u, v))
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = [set(s) for s in current_adj]
                    best_edges_set = set(current_edges)
        
        T *= cooling
    
    # Convert best_adj back to edge list
    result_edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                result_edges.append([u, v])
    
    return result_edges
