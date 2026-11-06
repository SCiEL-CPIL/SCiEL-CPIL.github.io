import time
import sys

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit

    n = 256
    max_sum = 2 * n

    # Identify squares
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1

    # Filter valid numbers (2*a not square)
    valid_numbers = []
    for a in range(1, n + 1):
        if 2 * a not in squares:
            valid_numbers.append(a)

    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}

    # Build COMPATIBILITY graph
    # compat[i] is a bitmask of indices j such that valid_numbers[i] + valid_numbers[j] is NOT a square
    # We want Max Clique in this graph.
    # Note: A node is compatible with itself if 2*a is not a square (already filtered).
    # So we include self in the mask? 
    # In Max Clique, vertices are distinct. The condition "a+b is never a square for a<=b" includes a=b.
    # We already filtered out a where 2a is square.
    # For a != b, we check if a+b is square. If not, they are compatible.
    # So compat[i] should contain j if i!=j and (valid_numbers[i] + valid_numbers[j]) is not square.
    # And we treat the graph as simple (no self-loops in the clique sense, but we pick distinct vertices).
    
    compat = [0] * num_valid
    for i in range(num_valid):
        a = valid_numbers[i]
        mask = 0
        for j in range(num_valid):
            if i == j:
                continue
            b = valid_numbers[j]
            if (a + b) not in squares:
                mask |= (1 << j)
        compat[i] = mask

    # Tomita's Max-Clique algorithm with greedy coloring
    # We represent the candidate set P as a list of vertices (indices) and use bitmasks for efficiency if possible,
    # but standard Tomita uses lists. Given N~250, lists are fine.
    # However, to make it faster, we can use bitmasks for the candidate set and precomputed colors?
    # Let's stick to a robust implementation using bitmasks for the candidate set P and a list for the order.
    
    # Actually, a pure bitmask Tomita is tricky because coloring depends on the order.
    # Let's use the standard list-based Tomita but optimize the inner loops.
    
    best_clique = []
    
    def color_sort(P):
        """
        Greedy coloring of P. Returns (ordered_P, colors) where ordered_P is the list of vertices
        in the order they were colored, and colors[i] is the color of ordered_P[i].
        The colors are non-decreasing.
        """
        if not P:
            return [], []
        
        # We will assign colors greedily.
        # To speed up, we can sort P by degree in P? 
        # Standard Tomita doesn't pre-sort by degree, but it's a good heuristic.
        # Let's just iterate through P.
        
        uncolored = list(P)
        colors = []
        ordered = []
        color_classes = []  # list of lists, each is a set of vertices with that color
        
        while uncolored:
            # Pick a vertex from uncolored
            # Heuristic: pick vertex with max degree in uncolored?
            # Or just the first one?
            # Let's try max degree in uncolored to improve bound tightness.
            # This is O(N^2) per coloring step which might be slow.
            # Let's stick to first available for speed, or simple heuristic.
            
            # Simple heuristic: pick vertex with max degree in the remaining uncolored set
            # Actually, let's just pick the first one to keep it fast, as N is small.
            # Wait, the quality of the coloring bound is crucial.
            # Let's use a simple greedy: iterate through uncolored, assign smallest possible color.
            
            # To make it efficient, we'll maintain a list of vertices for each color.
            # But we need the ordering.
            
            # Let's use a different approach:
            # 1. Sort P by degree in P (descending) - optional but helps.
            # 2. Greedily assign colors.
            
            # For now, let's just pick the first vertex in uncolored.
            v = uncolored[0]
            uncolored.pop(0)
            
            # Find smallest color c such that v has no neighbors in color_class[c]
            # color_classes[c] is a bitmask of vertices with color c
            assigned_color = 0
            while assigned_color < len(color_classes):
                # Check if v has any neighbor in color_classes[assigned_color]
                if compat[v] & color_classes[assigned_color]:
                    assigned_color += 1
                else:
                    break
            
            if assigned_color == len(color_classes):
                color_classes.append(1 << v)
            else:
                color_classes[assigned_color] |= (1 << v)
            
            ordered.append(v)
            colors.append(assigned_color + 1) # 1-indexed colors
            
        return ordered, colors

    def expand(R, P):
        """
        R: current clique (list of vertices)
        P: candidate set (list of vertices)
        """
        nonlocal best_clique, start_time, deadline
        
        if time.time() > deadline:
            return
        
        if not P:
            if len(R) > len(best_clique):
                best_clique = R[:]
            return
        
        # Color P
        ordered_P, colors = color_sort(P)
        
        # Process in reverse order of colors (highest color first)
        # This is the standard Tomita optimization.
        for i in range(len(ordered_P) - 1, -1, -1):
            v = ordered_P[i]
            c = colors[i]
            
            # Upper bound: len(R) + c
            if len(R) + c <= len(best_clique):
                return
            
            # Branch: include v
            new_R = R + [v]
            
            # New P: vertices in P[0..i-1] that are neighbors of v
            # In ordered_P, vertices before i are those with color <= c.
            # We need to intersect P (which is effectively ordered_P[0..i]) with neighbors of v.
            # But note: P in the recursive call should only contain vertices that are compatible with all in R.
            # Since R is a clique, and we are adding v, the new candidates must be neighbors of v AND in the previous P.
            
            # The previous P was ordered_P.
            # The new P is { u in ordered_P[0..i-1] | u is neighbor of v }
            
            new_P = []
            # We can iterate through ordered_P[0..i-1]
            for j in range(i):
                u = ordered_P[j]
                if compat[v] & (1 << u):
                    new_P.append(u)
            
            expand(new_R, new_P)
            
            # Remove v from P for subsequent iterations
            # This is implicitly handled by the loop index decreasing and the fact that
            # we are iterating over the fixed ordered_P list.
            # However, we must ensure that when we call expand, we are passing the correct subset.
            # The logic above constructs new_P from the original ordered_P prefix.
            # This is correct.

    # Initial P is all valid numbers
    # To improve performance, we can order the initial P by degree
    degrees = [bin(compat[i]).count('1') for i in range(num_valid)]
    initial_P = sorted(range(num_valid), key=lambda x: degrees[x], reverse=True)
    
    # Run Tomita
    expand([], initial_P)
    
    # Extract result
    result = []
    for v in best_clique:
        result.append(valid_numbers[v])
        
    if not result:
        # Fallback
        result = [1]
        
    return result
