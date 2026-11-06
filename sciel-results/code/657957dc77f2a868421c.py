"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _validate(coloring):
    """Check that every K4 has at least 5 distinct colors. Returns True if valid."""
    n = 16
    # Build edge color lookup: edge_color[i][j] for i<j
    edge_color = [[-1]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_color[i][j] = coloring[idx]
            edge_color[j][i] = coloring[idx]
            idx += 1
    # Check all K4s
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    colors = set([
                        edge_color[a][b], edge_color[a][c], edge_color[a][d],
                        edge_color[b][c], edge_color[b][d], edge_color[c][d]
                    ])
                    if len(colors) < 5:
                        return False
    return True

def _count_colors(coloring):
    return len(set(coloring))

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = 16
    # Start with the additive scheme mod 17
    coloring = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    best_coloring = coloring[:]
    best_count = _count_colors(best_coloring)  # should be 16
    
    # Try to reduce colors by merging color classes
    # Greedy: try to merge each pair of colors, check validity
    # But that's O(C^2 * #K4s) which is expensive. Let's be smarter.
    
    # Alternative: local search - randomly recolor edges and try to reduce max color used
    # while maintaining validity.
    
    # First, let's try a systematic approach: try to see if 15 colors work
    # by trying to eliminate one color from the additive scheme.
    
    # The additive scheme uses colors 0..15. Let's see which color appears least
    # and try to recolor those edges.
    
    # Actually, let's try a different approach: use a randomized local search
    # that tries to minimize the number of distinct colors.
    
    random.seed(42)
    
    # Helper to get the set of edges with a given color
    def edges_with_color(c, col):
        return [idx for idx, v in enumerate(col) if v == c]
    
    # Try merging: pick two colors, try to recolor all edges of one to the other
    # Check if result is valid.
    
    # This is expensive. Let's use a smarter validation.
    
    # For the additive scheme, we know the structure. Let's try to find
    # if there's a valid coloring with fewer colors using a more direct approach.
    
    # Try: can we use colors 0..14 (15 colors)?
    # Start from additive mod 17, and try to map color 15 to some other color.
    
    # Actually, let's just do a simple local search:
    # Objective: minimize max color index (or number of distinct colors)
    # Move: recolor a single edge
    # Constraint: every K4 must have >= 5 distinct colors
    
    # This is a constraint satisfaction + optimization problem.
    
    # Given the time budget, let's try a few strategies:
    
    # Strategy 1: Try to find if 15 colors suffice by systematic search
    # Strategy 2: Random restarts with local search
    
    # For now, let's try to improve on the additive scheme by checking
    # if we can eliminate a color.
    
    # Try each color to eliminate: recolor all edges of that color to another color
    # and check validity.
    
    current = coloring[:]
    current_count = _count_colors(current)
    
    # Try to merge each color pair
    for target_color in range(16):
        for source_color in range(16):
            if source_color == target_color:
                continue
            if time.time() > deadline:
                break
            
            # Try recoloring all edges of source_color to target_color
            test = current[:]
            for idx in range(len(test)):
                if test[idx] == source_color:
                    test[idx] = target_color
            
            new_count = _count_colors(test)
            if new_count >= current_count:
                continue
            
            # Check validity
            if _validate(test):
                current = test
                current_count = new_count
                if current_count < best_count:
                    best_count = current_count
                    best_coloring = test[:]
        
        if time.time() > deadline:
            break
    
    # Try more aggressive merging: merge multiple colors at once
    # This is harder, so let's do a few iterations of pairwise merging
    
    for iteration in range(5):
        if time.time() > deadline:
            break
        improved = False
        for target_color in range(16):
            if time.time() > deadline:
                break
            for source_color in range(16):
                if source_color == target_color:
                    continue
                # Only try if source_color exists in current
                if source_color not in set(current):
                    continue
                if target_color not in set(current):
                    continue
                
                test = current[:]
                for idx in range(len(test)):
                    if test[idx] == source_color:
                        test[idx] = target_color
                
                new_count = _count_colors(test)
                if new_count >= current_count:
                    continue
                
                if _validate(test):
                    current = test
                    current_count = new_count
                    improved = True
                    if current_count < best_count:
                        best_count = current_count
                        best_coloring = test[:]
        
        if not improved:
            break
    
    return best_coloring
# EVOLVE-BLOCK-END
