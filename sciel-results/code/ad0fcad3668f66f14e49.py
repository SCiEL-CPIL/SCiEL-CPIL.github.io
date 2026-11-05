# EVOLVE-BLOCK-START
import itertools

def construct():
    n = 20
    A = list(range(10))
    B = list(range(10, 20))
    
    # Map B indices to 0..9 for calculation
    b_map = {b: i for i, b in enumerate(B)}
    
    # Initialize all edges
    edges = {}
    for a in A:
        for b in B:
            edges[(a, b)] = True
            
    # Define coloring
    # Use mod 19 to avoid trivial wrap-around issues with small indices
    def get_color(a, b):
        b_idx = b_map[b]
        return (a + 2 * b_idx) % 19

    # Precompute colors for all edges
    colors = {}
    for (a, b) in edges:
        colors[(a, b)] = get_color(a, b)

    # Function to check if a 4-cycle is rainbow
    def is_rainbow(a1, b1, a2, b2):
        # Cycle: a1-b1-a2-b2-a1
        # Edges: (a1,b1), (a2,b1), (a2,b2), (a1,b2)
        e1 = (a1, b1)
        e2 = (a2, b1)
        e3 = (a2, b2)
        e4 = (a1, b2)
        
        if not (edges[e1] and edges[e2] and edges[e3] and edges[e4]):
            return True # Not a cycle in current graph
        
        c1 = colors[e1]
        c2 = colors[e2]
        c3 = colors[e3]
        c4 = colors[e4]
        
        return len({c1, c2, c3, c4}) == 4

    # Find and remove edges from bad 4-cycles
    # We iterate until no bad 4-cycles exist
    # To be efficient, we can iterate through all potential 4-cycles
    # and if one is bad, remove an edge and restart/continue
    
    # Greedy approach:
    # Keep a list of "active" 4-cycles?
    # Simpler: Just loop.
    
    max_iterations = 1000
    for _ in range(max_iterations):
        found_bad = False
        # Iterate through all pairs of distinct a's and b's
        for a1, a2 in itertools.combinations(A, 2):
            for b1, b2 in itertools.combinations(B, 2):
                # Check cycle a1-b1-a2-b2
                if is_rainbow(a1, b1, a2, b2):
                    continue
                
                # Found a bad 4-cycle
                found_bad = True
                # Remove one edge. Let's remove (a1, b1)
                edges[(a1, b1)] = False
                # Break out to restart the search (or just continue, but restart is safer for correctness if logic is complex)
                # Actually, we can just break the inner loops and restart the whole iteration
                # To save time, we can just break and let the outer loop catch it again
                # But we need to exit all loops.
                break
            if found_bad:
                break
        if not found_bad:
            break

    # Construct result
    result = []
    for (a, b) in edges:
        if edges[(a, b)]:
            result.append([a, b, colors[(a, b)]])
            
    return result
# EVOLVE-BLOCK-END
