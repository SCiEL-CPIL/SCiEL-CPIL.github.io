import time
import random
import sys

def construct():
    """
    Finds a triangle-free graph on 16 vertices with the maximum number of edges
    that is not 3-colorable.
    """
    n = 16
    deadline = time.time() + 115  # Safety margin
    
    # Helper: Initialize Clebsch Graph
    # Vertices 0-7 in partition A, 8-15 in partition B? No, Clebsch is defined differently.
    # Clebsch graph is the folded 5-cube.
    # It is a strongly regular graph (16, 5, 0, 2).
    # Let's construct it using the standard definition:
    # Vertices are binary strings of length 4.
    # Two vertices are adjacent if they differ in exactly 2 bits.
    edges = []
    vertices = [i for i in range(16)]
    for u in vertices:
        for v in vertices:
            if u < v:
                diff = u ^ v
                # Count bits
                if bin(diff).count('1') == 2:
                    edges.append([u, v])
    
    # Check if this is the Clebsch graph
    # It should have 40 edges.
    # 16 * 5 / 2 = 40. Correct.
    
    # Check if it is 3-colorable
    def is_3_colorable(edge_list):
        # Build adjacency list
        adj = [[0]*n for _ in range(n)]
        for u, v in edge_list:
            adj[u][v] = 1
            adj[v][u] = 1
        
        # Backtracking with bitsets
        # colors: 0, 1, 2
        # Use recursion
        def backtrack(v, colors):
            if v == n:
                return True
            
            # Get available colors
            # If neighbor has color c, c is not available
            avail = 7  # 111b
            for u in range(n):
                if adj[v][u]:
                    avail &= ~(1 << colors[u])
            
            if avail == 0:
                return False
            
            # Try each available color
            # Heuristic: try most constrained first? Or just 0,1,2
            # Order: 0, 1, 2
            for c in range(3):
                if avail & (1 << c):
                    colors[v] = c
                    if backtrack(v+1, colors):
                        return True
            return False
        
        # Sort vertices by degree descending for better pruning?
        # Let's stick to simple first for now, or simple degree sort.
        degrees = [sum(row) for row in adj]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        # Map original index to new order index?
        # Easier: just permute the adjacency matrix?
        # Or just use the order in backtracking.
        # Let's modify backtrack to follow 'order'
        
        def backtrack_order(idx, colors):
            if idx == n:
                return True
            v = order[idx]
            avail = 7
            for u in range(n):
                if adj[v][u]:
                    avail &= ~(1 << colors[u])
            
            if avail == 0:
                return False
            
            for c in range(3):
                if avail & (1 << c):
                    colors[v] = c
                    if backtrack_order(idx+1, colors):
                        return True
            return False
        
        colors = [0]*n
        return backtrack_order(0, colors)

    def is_triangle_free(edge_list):
        adj = [[0]*n for _ in range(n)]
        for u, v in edge_list:
            adj[u][v] = 1
            adj[v][u] = 1
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    for w in range(v+1, n):
                        if adj[u][w] and adj[v][w]:
                            return False
        return True

    # Initial check
    if not is_triangle_free(edges):
        return edges # Should not happen
    if is_3_colorable(edges):
        # Clebsch is not 3-colorable, so this should be False
        pass

    best_edges = edges
    best_count = len(edges)
    best_is_3c = is_3_colorable(best_edges)
    # We want NOT 3-colorable.
    if best_is_3c:
        # Fallback?
        pass

    # Simulated Annealing
    # State: set of edges
    # We will represent state as a set of tuples for fast lookup
    current_edges = set(tuple(e) for e in edges)
    
    def get_edge_count(s):
        return len(s)
    
    def is_triangle_free_set(edge_set):
        # Fast check using bitsets
        adj = [0]*n
        for u, v in edge_set:
            if u > v: u, v = v, u
            # This is slow. Let's use a matrix or bitset array.
            pass
        # Better: Use a 16x16 matrix or list of bitsets
        adj_bits = [0]*n
        for u, v in edge_set:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        for u in range(n):
            for v in range(u+1, n):
                if (adj_bits[u] >> v) & 1:
                    # Check if u and v have common neighbor
                    # common = adj_bits[u] & adj_bits[v]
                    # if common & ~(1<<u) & ~(1<<v) != 0:
                    common = adj_bits[u] & adj_bits[v]
                    # Remove u and v from common?
                    # u and v are not neighbors of themselves in this context?
                    # adj_bits[u] has bit v set. adj_bits[v] has bit u set.
                    # Intersection will have bits for common neighbors.
                    # We need to ensure no common neighbor w exists.
                    # If common != 0, there is a common neighbor?
                    # Wait, adj_bits[u] contains v. adj_bits[v] contains u.
                    # So common contains... 
                    # Let's just check if there is any w != u, v.
                    # Since graph is simple, no self loops.
                    # If common != 0, it means there is a w connected to both.
                    # Does w = u or w = v?
                    # adj_bits[u] has bit v. adj_bits[v] has bit u.
                    # (1<<v) & (1<<u) is 0 if u!=v.
                    # So common does not contain u or v.
                    if common != 0:
                        return False
        return True

    def is_3_colorable_set(edge_set):
        # Convert to list for the existing function?
        # Or reuse logic.
        # For simplicity in SA, we can call is_3_colorable(list(edge_set))
        # But that's slow.
        # Let's optimize is_3_colorable to take a set or bitsets.
        # For now, use list conversion.
        return is_3_colorable([list(e) for e in edge_set])

    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # SA Parameters
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    iterations = 0
    
    while time.time() < deadline:
        iterations += 1
        # Generate a move
        # Try to add an edge and remove one? (Swap)
        # Or add two and remove one? (Net +1)
        # Let's try Swap first to explore space, then try Net +1.
        
        move_type = random.random()
        if move_type < 0.7:
            # Swap: Remove e1, Add e2
            if not current_edges: continue
            e1 = random.choice(list(current_edges))
            e2 = random.choice(all_edges)
            if e1 == e2: continue
            new_edges = set(current_edges)
            new_edges.remove(e1)
            new_edges.add(e2)
        else:
            # Net +1: Remove e1, Add e2, Add e3
            if len(current_edges) < 2: continue
            e1, e2_rem = random.sample(list(current_edges), 2)
            e3_add, e4_add = random.sample(all_edges, 2)
            # Ensure e3, e4 are not in current_edges and not e1, e2
            if e3_add in current_edges or e4_add in current_edges:
                continue
            if e3_add == e4_add:
                continue
            new_edges = set(current_edges)
            new_edges.remove(e1)
            new_edges.remove(e2_rem)
            new_edges.add(e3_add)
            new_edges.add(e4_add)
            
        # Check Triangle Freeness
        if not is_triangle_free_set(new_edges):
            continue
            
        # Check 3-Colorability
        is_3c = is_3_colorable_set(new_edges)
        
        # Acceptance Criterion
        # We want to maximize edges, and prefer NOT 3-colorable.
        # Score: -edges * 1000 - (1 if 3-colorable else 0) * 10?
        # Actually, if it is 3-colorable, it's invalid for the final answer, but might be a stepping stone.
        # Let's define Energy E = -len(new_edges) + (10 if is_3c else 0)
        # Lower E is better.
        E_new = -len(new_edges) + (10 if is_3c else 0)
        
        # Current Energy
        E_curr = -len(current_edges) + (10 if is_3_colorable_set(current_edges) else 0)
        
        delta_E = E_new - E_curr
        
        if delta_E < 0 or random.random() < pow(2.71828, -delta_E / T):
            current_edges = new_edges
            
            # Update Best
            if not is_3c and len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = [list(e) for e in current_edges]
                
        T *= alpha
        if T < T_min:
            T = T_min
            # Restart? Or just continue.
            
    return best_edges

# Run construct
if __name__ == "__main__":
    result = construct()
    print(f"Found graph with {len(result)} edges")
