import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for safety
    
    N = 256
    # 1. Build Conflict Graph
    # conflict[i] is a bitmask of vertices that conflict with i
    conflict = [0] * (N + 1)
    
    # Precompute conflicts
    # a, b in 1..N
    # Condition: (a*b) % (a+b) == 0
    # Optimization: Iterate a, then b > a
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                # Set bit b in conflict[a] and bit a in conflict[b]
                conflict[a] |= (1 << b)
                conflict[b] |= (1 << a)
                
    # Bitmask of all vertices 1..N
    # We use bits 1..N. Bit 0 is unused.
    all_vertices = (1 << (N + 1)) - 2  # Bits 1 to N set
    
    best_size = 0
    best_set = set()
    
    # Tomita's Algorithm for Maximum Independent Set
    # P: bitmask of available vertices
    # I: list of vertices in current independent set (or just count for bound, but we need the set for result)
    # We will maintain the set of vertices in the best solution found.
    
    def greedy_color_bound(P):
        """
        Returns the number of colors needed to color the subgraph induced by P.
        This is an upper bound on the size of the MIS in P.
        """
        colors = 0
        remaining = P
        while remaining:
            colors += 1
            # Find a maximal independent set in the remaining graph to assign this color
            # Greedy: pick a vertex, remove it and its neighbors
            color_class = 0
            temp_remaining = remaining
            while temp_remaining:
                # Pick the lowest set bit vertex
                # v = (temp_remaining & -temp_remaining).bit_length() - 1
                v_bit = temp_remaining & -temp_remaining
                v = v_bit.bit_length() - 1
                
                color_class |= v_bit
                # Remove v and its neighbors from temp_remaining
                # Neighbors of v in the context of 'remaining'
                # We need to remove v and N(v) from temp_remaining
                # Note: conflict[v] contains neighbors in the full graph.
                # We only care about those in 'remaining'.
                temp_remaining &= ~(v_bit | conflict[v])
            
            remaining &= ~color_class
        return colors

    def branch_and_bound(P, I_count, I_set):
        nonlocal best_size, best_set, deadline
        
        if time.time() > deadline:
            return
            
        if P == 0:
            if I_count > best_size:
                best_size = I_count
                best_set = set(I_set)
            return
            
        # Pruning 1: Simple bound
        if I_count + P.bit_count() <= best_size:
            return
            
        # Pruning 2: Coloring bound
        # The coloring bound is expensive, so maybe check simple bound first.
        # Let's try to get the coloring bound.
        # To speed up, we can order vertices by degree or just use the bitmask as is.
        # Tomita's algorithm usually sorts vertices by degree descending.
        # For bitmasks, sorting is tricky. We can just pick a vertex.
        
        # A simpler upper bound is the size of the largest independent set in a clique cover? No.
        # Let's use the greedy coloring bound.
        # However, running greedy coloring on every node might be slow.
        # We can use a cheaper bound first, e.g., just P.bit_count().
        # If that doesn't prune, then use coloring.
        
        # Let's pick a vertex v to branch on.
        # Heuristic: Pick the vertex with the highest degree in P.
        # Calculating degree for all v in P is O(|P|^2).
        # Alternative: Just pick the lowest set bit vertex.
        # To improve performance, we can select v such that it has many neighbors in P.
        
        # Let's try selecting the vertex with the maximum degree in the subgraph P.
        # This requires iterating through P.
        # If P is large, this is costly.
        # Given N=256, P starts at 256.
        
        # Let's use a simpler heuristic: Pick the vertex v in P that has the most neighbors in P.
        # If we can't afford that, pick the first one.
        
        # Let's optimize the vertex selection.
        # We can precompute degrees? No, degrees change as P changes.
        
        # Let's stick to picking the lowest set bit vertex for now, but apply the coloring bound.
        # Actually, the quality of the coloring bound depends on the order of vertices.
        # A better approach:
        # 1. Get a list of vertices in P.
        # 2. Sort them by degree in P descending.
        # 3. Use this order for coloring.
        
        # For now, let's just use the bitmask order (lowest to highest) for coloring.
        # This is deterministic.
        
        ub = greedy_color_bound(P)
        
        if I_count + ub <= best_size:
            return
        
        # Select vertex v
        # Let's pick the vertex with the highest degree in P to branch on.
        # This helps prune faster.
        max_deg = -1
        v = -1
        temp_P = P
        while temp_P:
            v_bit = temp_P & -temp_P
            v_candidate = v_bit.bit_length() - 1
            # Count neighbors of v_candidate in P
            deg = (conflict[v_candidate] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = v_candidate
                v_bit = v_bit # keep track
            temp_P &= temp_P - 1
            
        if v == -1:
            return
            
        v_bit = 1 << v
        
        # Branch 1: Include v
        # New P is P without v and its neighbors
        P1 = P & ~(v_bit | conflict[v])
        I_set.append(v)
        branch_and_bound(P1, I_count + 1, I_set)
        I_set.pop()
        
        # Branch 2: Exclude v
        P2 = P & ~v_bit
        branch_and_bound(P2, I_count, I_set)

    # Initial call
    # To help the search, we can start with a greedy solution.
    # Greedy MIS:
    # 1. Sort vertices by degree ascending (in conflict graph).
    # 2. Iterate, if no conflict with selected, add to set.
    
    degrees = [0] * (N + 1)
    for i in range(1, N + 1):
        degrees[i] = conflict[i].bit_count()
        
    sorted_vertices = sorted(range(1, N + 1), key=lambda x: degrees[x])
    
    greedy_set = set()
    greedy_mask = 0
    for v in sorted_vertices:
        if (conflict[v] & greedy_mask) == 0:
            greedy_set.add(v)
            greedy_mask |= (1 << v)
            
    if len(greedy_set) > best_size:
        best_size = len(greedy_set)
        best_set = greedy_set
        
    # Run B&B
    # We pass the full set of vertices
    branch_and_bound(all_vertices, 0, [])
    
    # Return the best set found
    return list(best_set)

if __name__ == "__main__":
    # Quick test
    s = construct()
    print(len(s))
