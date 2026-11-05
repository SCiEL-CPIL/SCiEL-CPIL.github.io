import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s buffer for safety
    
    n = 18

    # Helper: Get edges from adjacency list (list of bitmasks)
    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append([i, j])
                m ^= lsb
        return edges

    # Helper: Check if the subgraph induced by a set of vertices (mask) is triangle-free
    # Returns True if no triangle exists within 'mask'
    def is_triangle_free(adj, mask):
        m = mask
        while m:
            lsb = m & -m
            m ^= lsb
            u = lsb.bit_length() - 1
            # Check if u has two neighbors in the remaining mask that are adjacent
            neighbors_in_mask = adj[u] & m
            # If neighbors_in_mask contains an edge, there is a triangle
            # Iterate through neighbors of u in m
            w = neighbors_in_mask
            while w:
                lsb_w = w & -w
                w ^= lsb_w
                v = lsb_w.bit_length() - 1
                # Check if v has any neighbor in (neighbors_in_mask & ~lsb_w)
                if adj[v] & (neighbors_in_mask & ~lsb_w):
                    return False
        return True

    # Helper: Find maximum triangle-free induced subgraph size
    # Uses bitmask Branch and Bound
    def max_triangle_free_size(adj):
        best = 0
        
        def dfs(current_set, candidates):
            nonlocal best
            if not candidates:
                if current_set.bit_count() > best:
                    best = current_set.bit_count()
                return
            
            # Upper bound
            if current_set.bit_count() + candidates.bit_count() <= best:
                return
            
            # Pick a vertex from candidates to branch on
            # Heuristic: pick the vertex with the lowest index
            v = (candidates & -candidates).bit_length() - 1
            v_mask = 1 << v
            
            # Branch 1: Exclude v
            dfs(current_set, candidates & ~v_mask)
            
            # Branch 2: Include v
            # Check if v can be added to current_set without creating a triangle.
            neighbors_in_current = adj[v] & current_set
            
            valid = True
            check = neighbors_in_current
            while check:
                lsb = check & -check
                check ^= lsb
                ui = lsb.bit_length() - 1
                if adj[ui] & (neighbors_in_current & ~lsb):
                    valid = False
                    break
            
            if valid:
                new_current = current_set | v_mask
                # Update candidates: remove v and any w that forms a triangle with v and some u in current_set
                # w is forbidden if w is neighbor of v AND w is neighbor of some u in neighbors_in_current
                forbidden = 0
                check_u = neighbors_in_current
                while check_u:
                    lsb_u = check_u & -check_u
                    check_u ^= lsb_u
                    ui = lsb_u.bit_length() - 1
                    forbidden |= adj[ui]
                    
                forbidden &= adj[v]
                
                new_candidates = (candidates & ~v_mask) & ~forbidden
                dfs(new_current, new_candidates)
                
        dfs(0, (1 << n) - 1)
        return best

    # Strategy: Greedy Maximal K4-free Graph Construction
    # We define the graph by a permutation of vertices.
    # We iterate through vertices in the order given by the permutation.
    # For each vertex u, we connect it to all previous vertices v such that
    # adding (u, v) does not create a K4.
    # Condition for adding (u, v): The common neighborhood of u and v must be independent (triangle-free).
    # Since we are building the graph, the common neighborhood consists of vertices w < u (in permutation order)
    # that are already connected to both u and v.
    # Wait, if we process in order, when we consider edge (u, v) where v < u:
    # Common neighbors are w < v such that (w, v) exists and (w, u) exists?
    # Actually, if we just want to ensure the graph is K4-free, we can check if the set of common neighbors of u and v
    # contains an edge.
    
    def generate_greedy_adj(permutation):
        adj = [0] * n
        # Map vertex index to position in permutation
        # pos[vertex] = index in permutation
        pos = [0] * n
        for i, v in enumerate(permutation):
            pos[v] = i
            
        # Iterate through pairs (u, v) where pos[u] > pos[v]
        # u is "later" in the permutation than v
        for i in range(n):
            u = permutation[i]
            for j in range(i):
                v = permutation[j]
                # Check if adding (u, v) creates a K4
                # Common neighbors of u and v in the current graph
                # Since we are building, only vertices with pos < pos[v] or pos < pos[u] can be neighbors?
                # Actually, neighbors of u are only among vertices with pos < pos[u] (already processed as 'previous' to u)
                # Neighbors of v are among vertices with pos < pos[v]
                # So common neighbors must be in the set of vertices with pos < pos[v]
                # Let's compute common neighbors
                common = adj[u] & adj[v]
                # Remove u and v themselves (though they shouldn't be in each other's adj yet if we are checking)
                # adj[u] and adj[v] only contain vertices with lower pos.
                # Check if common contains an edge
                w = common
                valid = True
                while w:
                    lsb = w & -w
                    w ^= lsb
                    wi = lsb.bit_length() - 1
                    if adj[wi] & (common & ~lsb):
                        valid = False
                        break
                
                if valid:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
        return adj

    # Initial permutation: Random
    base_perm = list(range(n))
    random.shuffle(base_perm)
    current_perm = base_perm[:]
    current_adj = generate_greedy_adj(current_perm)
    current_score = max_triangle_free_size(current_adj)
    
    best_score = current_score
    best_perm = current_perm[:]
    best_adj = current_adj[:]
    
    # Simulated Annealing on Permutation
    # Move: Swap two elements in the permutation
    temp = 1.0
    min_temp = 0.01
    no_improve_count = 0
    
    while time.time() < deadline - 5:
        # Pick two indices to swap
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j:
            continue
            
        new_perm = current_perm[:]
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        
        new_adj = generate_greedy_adj(new_perm)
        new_score = max_triangle_free_size(new_adj)
        
        if new_score is None:
            continue
            
        delta = new_score - current_score
        
        if new_score < best_score:
            best_score = new_score
            best_perm = new_perm
            best_adj = new_adj
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Acceptance
        if delta < 0:
            accept = True
        else:
            if delta == 0:
                accept = random.random() < 0.5
            else:
                # Probability of accepting worse solution
                accept = random.random() < (2.71828 ** (-delta / temp))
                
        if accept:
            current_perm = new_perm
            current_adj = new_adj
            current_score = new_score
            
        temp = max(min_temp, temp * 0.99)
        
        # Restart if stuck
        if no_improve_count > 200:
            # Random restart
            random.shuffle(current_perm)
            current_adj = generate_greedy_adj(current_perm)
            current_score = max_triangle_free_size(current_adj)
            temp = 1.0
            no_improve_count = 0

    return get_edges_from_adj(best_adj)

# Fallback if construct fails or returns empty
if __name__ == "__main__":
    pass
