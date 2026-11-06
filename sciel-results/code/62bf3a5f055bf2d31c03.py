import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    N = 16
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    num_edges = len(edges)
    
    # Precompute all K4s and the indices of edges in each K4
    # A K4 is defined by 4 vertices (a, b, c, d) with a<b<c<d
    # Edges are (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
    k4s = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    # Find indices of edges in the global list
                    # We can map (u,v) to index for faster lookup
                    pass
                    
    # Map edge (u,v) to index
    edge_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx
        
    k4_edge_indices = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    e1 = edge_idx[(a,b)]
                    e2 = edge_idx[(a,c)]
                    e3 = edge_idx[(a,d)]
                    e4 = edge_idx[(b,c)]
                    e5 = edge_idx[(b,d)]
                    e6 = edge_idx[(c,d)]
                    k4_edge_indices.append((e1, e2, e3, e4, e5, e6))
                    
    # Initialize with seed: (i + j) % 17
    # This uses colours 0..16? No, max sum is 15+14=29, 29%17=12.
    # Actually, for i,j < 16, max i+j is 29. 
    # The seed in the prompt returns [(i + j) % 17 ...]
    # Let's trace: i=0, j=15 -> 15. i=1, j=15 -> 16. i=2, j=15 -> 17%17=0.
    # So colours 0..16 are potentially used?
    # Wait, the prompt says "120 integer colours in 0..119".
    # And "Minimize the number of colours used".
    # The seed uses (i+j)%17.
    # Let's check the max colour used.
    # i=0, j=1..15 -> 1..15
    # i=1, j=2..15 -> 3..16, 1..14 (1+15=16, 2+15=17->0)
    # So 16 is used. 0 is used.
    # Is 17 used? No, mod 17.
    # So colours are 0..16. Total 17 colours?
    # Let's re-read the seed code comment: "Colour {i, j} by (i + j) mod 17".
    # If the seed uses 17 colours, the score is 1.0.
    # But the text says "13 <= f(16) <= 15".
    # If the seed uses 17, then 17 is not minimal.
    # Let's check if the seed is valid.
    # If a K4 has edges with colours c1..c6.
    # In a cyclic group Z_17, the sum of differences...
    # Actually, let's just run the seed logic and count distinct colours.
    
    current_colours = [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]
    
    def get_distinct_count(colours):
        return len(set(colours))
        
    # Helper to check validity
    def is_valid(colours):
        for e_tuple in k4_edge_indices:
            c_set = set(colours[i] for i in e_tuple)
            if len(c_set) < 5:
                return False
        return True
        
    # Helper to check if a specific merge is valid
    # Instead of checking all K4s, we can optimize, but N=16 is small.
    # Number of K4s is C(16,4) = 1820.
    # Checking validity is 1820 * 6 lookups. Very fast.
    
    best_colours = current_colours[:]
    best_count = get_distinct_count(current_colours)
    
    # Random seed for reproducibility in evaluation? 
    # The prompt implies deterministic or seeded. 
    # Using a fixed seed ensures consistent score if the search is deterministic.
    # But local search might get stuck. 
    # Let's use a fixed seed for the random number generator to make the program deterministic.
    random.seed(42)
    
    while time.time() < deadline:
        # Check if current state is valid
        if not is_valid(current_colours):
            # If we got into an invalid state (shouldn't happen if we only apply valid moves), 
            # revert to best
            current_colours = best_colours[:]
            continue
            
        current_count = get_distinct_count(current_colours)
        if current_count < best_count:
            best_count = current_count
            best_colours = current_colours[:]
            
        # Try to reduce colour count by merging
        # Get list of colours present
        present_colours = sorted(set(current_colours))
        
        merged = False
        # Try to remove the largest colour first
        for c_target in reversed(present_colours):
            # Try to merge c_target into any other colour c_src
            # We only need to try merging into colours that are present
            # Or we can just try merging into 0, 1, etc.
            # It's better to try merging into existing colours.
            for c_src in present_colours:
                if c_src == c_target:
                    continue
                
                # Create a copy to test
                test_colours = current_colours[:]
                for i in range(num_edges):
                    if test_colours[i] == c_target:
                        test_colours[i] = c_src
                        
                if is_valid(test_colours):
                    # Merge successful
                    current_colours = test_colours
                    merged = True
                    break
            if merged:
                break
                
        if merged:
            continue
            
        # If no merge possible, perform random local moves
        # Move 1: Change colour of a single edge
        # Move 2: Swap colours of two edges
        
        # Try single edge change
        moved = False
        for _ in range(100): # Limit random moves per iteration to avoid stalling
            e_idx = random.randint(0, num_edges - 1)
            old_c = current_colours[e_idx]
            new_c = random.randint(0, best_count - 1) # Only use existing colours
            if new_c == old_c:
                continue
                
            current_colours[e_idx] = new_c
            if is_valid(current_colours):
                moved = True
                break
            else:
                current_colours[e_idx] = old_c
                
        if not moved:
            # Try swapping two edges
            e1 = random.randint(0, num_edges - 1)
            e2 = random.randint(0, num_edges - 1)
            if e1 == e2:
                continue
            c1 = current_colours[e1]
            c2 = current_colours[e2]
            if c1 == c2:
                continue
                
            current_colours[e1] = c2
            current_colours[e2] = c1
            if is_valid(current_colours):
                moved = True
            else:
                current_colours[e1] = c1
                current_colours[e2] = c2

    # Final check
    if not is_valid(best_colours):
        # Fallback to seed if something went wrong
        return [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]
        
    return best_colours

