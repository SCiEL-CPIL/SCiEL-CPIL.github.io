"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    random.seed(42)
    
    n = 18
    
    # Use bitmask representation for speed
    # adj[i] is a bitmask of neighbors of vertex i
    def is_k4_free_mask(adj_mask):
        """Check if graph is K4-free using bitmask adjacency."""
        for a in range(n):
            # For each pair (a,b) that is an edge
            b_mask = adj_mask[a]
            b = 0
            while b_mask:
                lsb = b_mask & (-b_mask)
                b = lsb.bit_length() - 1
                b_mask ^= lsb
                if b <= a:
                    continue
                # Common neighbors of a and b
                common = adj_mask[a] & adj_mask[b]
                # For each common neighbor c > b, check if any common neighbor d of a,b,c exists
                c_mask = common
                while c_mask:
                    lsb_c = c_mask & (-c_mask)
                    c = lsb_c.bit_length() - 1
                    c_mask ^= lsb_c
                    if c <= b:
                        continue
                    # Check if c has a common neighbor with a and b that is > c
                    # Actually, we need to check if there's a d such that d is adjacent to a, b, c, and d > c
                    # Or more simply: for each d in common(a,b,c), if d forms K4 with a,b,c
                    # common(a,b,c) = adj_mask[a] & adj_mask[b] & adj_mask[c]
                    abc_common = adj_mask[a] & adj_mask[b] & adj_mask[c]
                    if abc_common:
                        # There exists d adjacent to all of a,b,c
                        # Check if any such d creates a K4 (it does if d != a,b,c)
                        # Since d is in abc_common and d is not a,b,c (they're not self-adjacent),
                        # {a,b,c,d} forms a K4
                        return False
        return True
    
    def max_triangle_free_mask(adj_mask):
        """Find the maximum number of vertices inducing a triangle-free subgraph using bitmask."""
        # Precompute triangles: for each pair (u,v) that is an edge, common neighbors
        # A set S is triangle-free iff no triangle is fully contained in S
        
        # Branch and bound with bitmask
        best = [0]
        
        def has_triangle_in_mask(mask):
            """Check if the vertex set represented by mask contains a triangle."""
            # For each vertex in mask, check if its neighbors within mask form an edge
            m = mask
            while m:
                lsb = m & (-m)
                u = lsb.bit_length() - 1
                m ^= lsb
                # Neighbors of u within the set
                neighbors_in_set = adj_mask[u] & mask
                # Check if any edge exists among neighbors_in_set
                nm = neighbors_in_set
                while nm:
                    lsb_n = nm & (-nm)
                    v = lsb_n.bit_length() - 1
                    nm ^= lsb_n
                    # Check if v has another neighbor in set that is also neighbor of u
                    # i.e., is there w in (adj_mask[v] & adj_mask[u] & mask) with w != u, w != v?
                    # Actually, we need w such that w is adjacent to both u and v, and w is in mask
                    common = adj_mask[u] & adj_mask[v] & mask
                    if common:
                        return True
            return False
        
        def max_tf_recursive(candidates, current_mask, current_size):
            if time.time() > deadline:
                return
            if not candidates:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Upper bound
            if current_size + len(candidates) <= best[0]:
                return
            
            for idx, v in enumerate(candidates):
                v_bit = 1 << v
                new_candidates = candidates[idx+1:]
                
                # Try including v: check if current_mask | {v} is triangle-free
                new_mask = current_mask | v_bit
                if not has_triangle_in_mask(new_mask):
                    max_tf_recursive(new_candidates, new_mask, current_size + 1)
                
                # Excluding v is implicit by continuing
        
        all_vertices = list(range(n))
        max_tf_recursive(all_vertices, 0, 0)
        return best[0]
    
    def score_graph_mask(adj_mask):
        """Compute the score (max triangle-free induced subgraph size)."""
        return max_triangle_free_mask(adj_mask)
    
    # 3-partite structure: parts of size 6 each
    # Part 0: vertices 0-5
    # Part 1: vertices 6-11
    # Part 2: vertices 12-17
    # Edges only between parts (no edges within a part)
    # This guarantees K4-free since K4 would need 4 vertices, but by pigeonhole
    # at least 2 are in the same part, and there are no edges within a part.
    
    def random_3partite():
        """Generate a random 3-partite graph."""
        adj_mask = [0] * n
        # Edges between part 0 and part 1
        for i in range(6):
            for j in range(6, 12):
                if random.random() < 0.5:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        # Edges between part 0 and part 2
        for i in range(6):
            for j in range(12, 18):
                if random.random() < 0.5:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        # Edges between part 1 and part 2
        for i in range(6, 12):
            for j in range(12, 18):
                if random.random() < 0.5:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        return adj_mask
    
    def is_3partite_mask(adj_mask):
        """Verify the graph is 3-partite with the given partition."""
        # No edges within part 0
        for i in range(6):
            for j in range(i+1, 6):
                if adj_mask[i] & (1 << j):
                    return False
        # No edges within part 1
        for i in range(6, 12):
            for j in range(i+1, 12):
                if adj_mask[i] & (1 << j):
                    return False
        # No edges within part 2
        for i in range(12, 18):
            for j in range(i+1, 18):
                if adj_mask[i] & (1 << j):
                    return False
        return True
    
    def local_search_3partite(adj_mask, max_iters=200):
        """Local search that maintains 3-partite structure."""
        current_score = score_graph_mask(adj_mask)
        best_adj = list(adj_mask)
        best_score = current_score
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try flipping a random inter-part edge
            # Choose two parts randomly
            part_pair = random.choice([(0,1), (0,2), (1,2)])
            p1, p2 = part_pair
            if p1 == 0 and p2 == 1:
                i = random.randint(0, 5)
                j = random.randint(6, 11)
            elif p1 == 0 and p2 == 2:
                i = random.randint(0, 5)
                j = random.randint(12, 17)
            else:
                i = random.randint(6, 11)
                j = random.randint(12, 17)
            
            # Flip edge
            adj_mask[i] ^= (1 << j)
            adj_mask[j] ^= (1 << i)
            
            # Verify still 3-partite (should be, since we only flip inter-part edges)
            if not is_3partite_mask(adj_mask):
                adj_mask[i] ^= (1 << j)
                adj_mask[j] ^= (1 << i)
                continue
            
            # Verify K4-free (should be for 3-partite, but double-check)
            if not is_k4_free_mask(adj_mask):
                adj_mask[i] ^= (1 << j)
                adj_mask[j] ^= (1 << i)
                continue
            
            new_score = score_graph_mask(adj_mask)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = list(adj_mask)
            else:
                # Revert
                adj_mask[i] ^= (1 << j)
                adj_mask[j] ^= (1 << i)
        
        return best_adj, best_score
    
    # Try multiple random 3-partite starts
    best_adj = None
    best_score = n  # worst case
    
    for attempt in range(80):
        if time.time() > deadline:
            break
        
        adj_mask = random_3partite()
        
        # Verify it's valid
        if not is_3partite_mask(adj_mask) or not is_k4_free_mask(adj_mask):
            continue
        
        adj_mask, score = local_search_3partite(adj_mask, max_iters=150)
        if score < best_score:
            best_score = score
            best_adj = list(adj_mask)
        
        # If we've found a good score, do more polish
        if best_score <= 10 and time.time() < deadline - 30:
            adj2, score2 = local_search_3partite(best_adj, max_iters=300)
            if score2 < best_score:
                best_score = score2
                best_adj = list(adj2)
    
    # Final polish pass on the best solution
    if best_adj is not None and time.time() < deadline - 10:
        adj_final, score_final = local_search_3partite(best_adj, max_iters=200)
        if score_final < best_score:
            best_score = score_final
            best_adj = list(adj_final)
    
    # Convert adjacency mask to edge list
    if best_adj is None:
        # Fallback: complete 3-partite graph
        adj_mask = [0] * n
        for i in range(6):
            for j in range(6, 12):
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
            for j in range(12, 18):
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
        for i in range(6, 12):
            for j in range(12, 18):
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
        best_adj = adj_mask
    
    edges = []
    for i in range(n):
        m = best_adj[i]
        while m:
            lsb = m & (-m)
            j = lsb.bit_length() - 1
            m ^= lsb
            if j > i:
                edges.append([i, j])
    
    return edges

# EVOLVE-BLOCK-END
