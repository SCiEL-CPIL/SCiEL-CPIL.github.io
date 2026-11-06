"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 115.0
    n = 18
    all_vertices_mask = (1 << n) - 1
    
    def is_k4_free(masks):
        """Check if graph with adjacency masks is K4-free using bitmasks."""
        for a in range(n):
            for b in range(a + 1, n):
                if not (masks[a] >> b) & 1:
                    continue
                common = masks[a] & masks[b] & ~(1 << a) & ~(1 << b)
                c_bits = common
                while c_bits:
                    c_bit = c_bits & (-c_bits)
                    c = c_bit.bit_length() - 1
                    c_bits ^= c_bit
                    # Check if any d in common (d > c) is connected to c
                    d_bits = common & (~((1 << (c + 1)) - 1))
                    while d_bits:
                        d_bit = d_bits & (-d_bits)
                        d = d_bit.bit_length() - 1
                        d_bits ^= d_bit
                        if (masks[c] >> d) & 1:
                            return False
        return True
    
    def max_triangle_free_induced(masks):
        """Find largest triangle-free induced subgraph using branch and bound."""
        best = 0
        best_set = 0
        
        # Precompute neighbor masks
        def can_add(current_mask, v):
            """Check if vertex v can be added to current_mask without creating a triangle."""
            v_neigh_in_current = masks[v] & current_mask
            # Check if v_neigh_in_current is an independent set
            bits = v_neigh_in_current
            while bits:
                b = bits & (-bits)
                u = b.bit_length() - 1
                bits ^= b
                # If u has any neighbor in v_neigh_in_current (other than itself), not independent
                if (masks[u] & v_neigh_in_current & ~b) != 0:
                    return False
            return True
        
        def search(current_mask, candidates_mask):
            nonlocal best, best_set
            if time.time() > deadline - 3:
                return
            
            cur_size = current_mask.bit_count()
            cand_size = candidates_mask.bit_count()
            
            # Upper bound
            if cur_size + cand_size <= best:
                return
            
            if candidates_mask == 0:
                if cur_size > best:
                    best = cur_size
                    best_set = current_mask
                return
            
            # Choose vertex with highest degree in candidates for better pruning
            # Use heuristic: pick vertex that appears in most edges within candidates
            v_bit = candidates_mask & (-candidates_mask)
            v = v_bit.bit_length() - 1
            remaining = candidates_mask ^ v_bit
            
            # Option 1: include v if possible
            if can_add(current_mask, v):
                new_current = current_mask | v_bit
                # Filter candidates: only those compatible with new_current
                new_cands = 0
                temp = remaining
                while temp:
                    b = temp & (-temp)
                    idx = b.bit_length() - 1
                    temp ^= b
                    if can_add(new_current, idx):
                        new_cands |= b
                search(new_current, new_cands)
            
            # Option 2: exclude v
            search(current_mask, remaining)
        
        search(0, all_vertices_mask)
        return best
    
    def random_k4_free_graph():
        """Generate a random K4-free graph."""
        # Start with empty graph, add edges randomly while maintaining K4-free
        masks = [0] * n
        random.seed(random.randint(0, 2**31))
        
        # Try to add edges in random order
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            # Check if adding (u,v) would create a K4
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
            if not is_k4_free(masks):
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
        
        return masks
    
    def improve_graph(masks, time_limit):
        """Improve graph by local search."""
        current_best = max_triangle_free_induced(masks)
        current_masks = masks[:]
        start_time = time.time()
        
        while time.time() < time_limit:
            # Try random edge additions
            improved = False
            for _ in range(50):
                if time.time() > time_limit:
                    break
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v or (current_masks[u] >> v) & 1:
                    continue
                
                new_masks = current_masks[:]
                new_masks[u] |= (1 << v)
                new_masks[v] |= (1 << u)
                
                if is_k4_free(new_masks):
                    score = max_triangle_free_induced(new_masks)
                    if score < current_best:
                        current_masks = new_masks
                        current_best = score
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try random edge removals
            for _ in range(50):
                if time.time() > time_limit:
                    break
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v or not ((current_masks[u] >> v) & 1):
                    continue
                
                new_masks = current_masks[:]
                new_masks[u] &= ~(1 << v)
                new_masks[v] &= ~(1 << u)
                
                score = max_triangle_free_induced(new_masks)
                if score < current_best:
                    current_masks = new_masks
                    current_best = score
                    improved = True
                    break
            
            if not improved:
                # Try more aggressive: remove multiple edges
                for _ in range(10):
                    if time.time() > time_limit:
                        break
                    new_masks = current_masks[:]
                    for _ in range(random.randint(1, 3)):
                        u = random.randint(0, n - 1)
                        v = random.randint(0, n - 1)
                        if u == v:
                            continue
                        new_masks[u] &= ~(1 << v)
                        new_masks[v] &= ~(1 << u)
                    
                    score = max_triangle_free_induced(new_masks)
                    if score < current_best:
                        current_masks = new_masks
                        current_best = score
                        improved = True
                        break
        
        return current_masks, current_best
    
    # Multiple restarts with different strategies
    best_overall = 18  # worst case
    best_masks = None
    
    # Strategy 1: Random K4-free graphs
    for restart in range(15):
        if time.time() > deadline - 10:
            break
        
        masks = random_k4_free_graph()
        score = max_triangle_free_induced(masks)
        if score < best_overall:
            best_overall = score
            best_masks = masks[:]
        
        # Improve this graph
        if time.time() < deadline - 15:
            improved_masks, improved_score = improve_graph(masks, deadline - 12)
            if improved_score < best_overall:
                best_overall = improved_score
                best_masks = improved_masks
    
    # Strategy 2: Start from complete tripartite and modify
    if time.time() < deadline - 15:
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        masks = [0] * n
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        masks[a] |= (1 << b)
                        masks[b] |= (1 << a)
        
        score = max_triangle_free_induced(masks)
        if score < best_overall:
            best_overall = score
            best_masks = masks[:]
        
        improved_masks, improved_score = improve_graph(masks, deadline - 12)
        if improved_score < best_overall:
            best_overall = improved_score
            best_masks = improved_masks
    
    # Strategy 3: Start from random dense graph and prune
    if time.time() < deadline - 15:
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Start with complete graph, remove edges to make K4-free
            masks = [(1 << n) - 1 ^ (1 << i) for i in range(n)]
            
            # Remove edges to break K4s
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((masks[u] >> v) & 1):
                        continue
                    # Check if this edge is in a K4
                    common = masks[u] & masks[v] & ~(1 << u) & ~(1 << v)
                    c_bits = common
                    has_k4 = False
                    while c_bits and not has_k4:
                        c_bit = c_bits & (-c_bits)
                        c = c_bit.bit_length() - 1
                        c_bits ^= c_bit
                        d_bits = common & (~((1 << (c + 1)) - 1))
                        while d_bits:
                            d_bit = d_bits & (-d_bits)
                            d = d_bit.bit_length() - 1
                            d_bits ^= d_bit
                            if (masks[c] >> d) & 1:
                                has_k4 = True
                                break
                    if has_k4:
                        masks[u] &= ~(1 << v)
                        masks[v] &= ~(1 << u)
            
            if not is_k4_free(masks):
                continue
            
            score = max_triangle_free_induced(masks)
            if score < best_overall:
                best_overall = score
                best_masks = masks[:]
            
            improved_masks, improved_score = improve_graph(masks, deadline - 12)
            if improved_score < best_overall:
                best_overall = improved_score
                best_masks = improved_masks
    
    # If no good graph found, use complete tripartite
    if best_masks is None:
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        best_masks = [0] * n
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        best_masks[a] |= (1 << b)
                        best_masks[b] |= (1 << a)
    
    # Convert to edge list
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_masks[u] >> v) & 1:
                result.append([u, v])
    
    return result
# EVOLVE-BLOCK-END
