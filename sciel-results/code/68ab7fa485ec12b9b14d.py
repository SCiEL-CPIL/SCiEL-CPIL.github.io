import random
import time

def construct():
    """
    Edge-colourings of K16 in which every K4 sees five colours.
    
    We use a randomised local search to minimize the number of colours used
    while ensuring every K4 has at least 5 distinct edge colours.
    
    Strategy:
    1. Start with a random valid colouring (using many colours).
    2. Iteratively try to reduce colours by merging colour classes
       (i.e., recoloring edges from colour c2 to c1) and check validity.
    3. Use a greedy approach to find a good solution within time limit.
    """
    random.seed(42)
    
    n = 16
    num_edges = n * (n - 1) // 2  # 120 edges
    
    # Precompute all K4s and their edge indices
    all_k4s = []
    edge_indices = {}
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_indices[(i, j)] = len(edge_list)
            edge_list.append((i, j))
    
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    # 6 edges in K4
                    edges = [
                        (a, b), (a, c), (a, d),
                        (b, c), (b, d), (c, d)
                    ]
                    edge_idxs = [edge_indices[e] for e in edges]
                    all_k4s.append(edge_idxs)
    
    def count_k4s_with_fewer_than_5(colours):
        """Count K4s with fewer than 5 distinct colours."""
        bad_count = 0
        for k4 in all_k4s:
            distinct = len(set(colours[k4[0]], colours[k4[1]], colours[k4[2]],
                               colours[k4[3]], colours[k4[4]], colours[k4[5]]))
            if distinct < 5:
                bad_count += 1
        return bad_count
    
    def is_valid(colours):
        """Check if every K4 has at least 5 distinct colours."""
        for k4 in all_k4s:
            distinct = len(set(colours[k4[0]], colours[k4[1]], colours[k4[2]],
                               colours[k4[3]], colours[k4[4]], colours[k4[5]]))
            if distinct < 5:
                return False
        return True
    
    # Start with a simple valid colouring: each edge gets a unique colour
    # This is valid (6 distinct colours per K4) but uses 120 colours
    colours = list(range(num_edges))
    
    # Now try to reduce the number of colours
    # We'll use a greedy approach: try to merge colour classes
    
    start_time = time.time()
    time_limit = 100  # seconds
    
    # Count colours used
    def num_colours_used(colours):
        return len(set(colours))
    
    # Try to reduce colours by assigning edges to a smaller palette
    # Use a different approach: start with a random assignment to 15 colours
    # and fix violations
    
    best_solution = None
    best_num_colours = 120
    
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
        
        # Random assignment to a small number of colours
        # Start with 15 colours (upper bound from problem statement)
        num_c = 15
        curr_colours = [random.randint(0, num_c - 1) for _ in range(num_edges)]
        
        # Check validity and fix
        # Use local search: if a K4 has < 5 colours, recolor one of its edges
        for iteration in range(10000):
            if time.time() - start_time > time_limit:
                break
            
            # Find a bad K4
            bad_k4 = None
            for k4 in all_k4s:
                distinct = set(curr_colours[k4[0]], curr_colours[k4[1]], curr_colours[k4[2]],
                               curr_colours[k4[3]], curr_colours[k4[4]], curr_colours[k4[5]])
                if len(distinct) < 5:
                    bad_k4 = k4
                    break
            
            if bad_k4 is None:
                # Valid!
                nc = num_colours_used(curr_colours)
                if nc < best_num_colours:
                    best_num_colours = nc
                    best_solution = curr_colours[:]
                break
            
            # Recolor one edge in the bad K4 to a new colour
            # Choose an edge that appears in the most bad K4s (approximate)
            # For now, just pick randomly
            edge_idx = bad_k4[random.randint(0, 5)]
            
            # Assign a colour that is not used by the other 5 edges in this K4
            other_colours = set()
            for j, ei in enumerate(bad_k4):
                if j != list(bad_k4).index(edge_idx):
                    other_colours.add(curr_colours[ei])
            
            # Find a colour not in other_colours
            new_colour = 0
            while new_colour in other_colours:
                new_colour += 1
            curr_colours[edge_idx] = new_colour
        
        if time.time() - start_time > time_limit:
            break
    
    # If we found a solution, use it; otherwise fall back to the (i+j) mod 17 scheme
    if best_solution is not None:
        return best_solution
    
    # Fallback: (i+j) mod 17
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
