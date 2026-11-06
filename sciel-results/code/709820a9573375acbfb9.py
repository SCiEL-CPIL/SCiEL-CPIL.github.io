import itertools
import random

def construct():
    # Initialize with the seed solution
    # Colour (i, j) by (i + j) % 17
    n = 16
    colors = {}
    for i in range(n):
        for j in range(i + 1, n):
            colors[(i, j)] = (i + j) % 17
            
    # List of all 4-subsets
    subsets = list(itertools.combinations(range(n), 4))
    
    # Function to check if a merge is valid
    # Merging c1 and c2 is valid if no subset has exactly 5 distinct colors 
    # and contains both c1 and c2.
    
    # To speed up, we can precompute the set of colors for each subset?
    # Or just check on the fly. 1820 subsets * 6 edges is small.
    
    def is_merge_valid(c1, c2):
        for sub in subsets:
            # Get colors of edges in this subset
            edge_colors = set()
            has_c1 = False
            has_c2 = False
            
            for u, v in itertools.combinations(sub, 2):
                c = colors[(u, v)]
                if c == c1:
                    has_c1 = True
                elif c == c2:
                    has_c2 = True
                edge_colors.add(c)
                
            if len(edge_colors) == 5 and has_c1 and has_c2:
                return False
        return True

    # Try to merge colors
    # We want to minimize the number of colors used.
    # The colors are in 0..16.
    # We can try to merge color k into color k-1 (or 0) repeatedly?
    # Or just find any valid merge.
    
    used_colors = set(colors.values())
    # Sort colors to have a deterministic order, or randomize?
    # Let's try to merge the highest color into the lowest? 
    # Actually, merging any valid pair reduces the count.
    
    # We need to keep track of which colors are "active".
    # If we merge c1 into c2, c1 is no longer used.
    
    # To be safe and potentially find a better local minimum, we can shuffle the order of attempts.
    # But deterministic is safer for reproducibility if not specified. 
    # However, the task says "Minimize", implying we should try hard.
    
    # Let's try a loop:
    # While there exists a valid merge, do it.
    # If we get stuck, we might try a random restart or a different merge order.
    # Given 120s, we can afford multiple attempts or a more complex search.
    
    # Simple approach: 
    # Repeatedly scan for a valid merge. 
    # To avoid bias, we can shuffle the list of color pairs to try.
    
    current_colors = list(range(17)) # 0 to 16
    
    # We will maintain a list of "dead" colors.
    # Actually, easier to just update the `colors` dict.
    
    # Optimization: Pre-calculate the color sets for all subsets initially?
    # As we merge, the sets change.
    
    # Let's just run the check.
    
    max_iterations = 100
    for _ in range(max_iterations):
        merged = False
        # Try all pairs of currently used colors
        active_colors = sorted(list(set(colors.values())))
        # Randomize the order to escape local minima? 
        # The problem is symmetric, so order might not matter much, but let's shuffle.
        random.shuffle(active_colors)
        
        for i in range(len(active_colors)):
            for j in range(i + 1, len(active_colors)):
                c1 = active_colors[i]
                c2 = active_colors[j]
                if is_merge_valid(c1, c2):
                    # Merge c1 into c2
                    for key, val in colors.items():
                        if val == c1:
                            colors[key] = c2
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
            
    # Return the colors in lexicographic order of edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            result.append(colors[(i, j)])
            
    # The evaluator wants 120 integer colours in 0..119.
    # "Minimize the number of colours used."
    # The raw value is the number of colours used.
    # Wait, the contract says "construct() -> 120 integer colours ... Minimize the number of colours used."
    # And "evaluator measures the raw value".
    # Does "raw value" mean the list? Or the count?
    # "Minimize the number of colours used. A valid colouring certifies f(16) <= colours used."
    # Usually in these tasks, the returned list is parsed to count distinct colours.
    # So returning the list is correct.
    
    return result
