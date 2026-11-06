import numpy as np
import time
import random

def construct():
    # Deadline guard: 110 seconds to be safe
    deadline = time.time() + 110
    
    # Pre-compute all K4s as indices into the 120-edge array
    # Edge index for (i, j) where i < j:
    # sum_{k=0}^{i-1} (15-k) + (j-i-1)
    # Let's precompute edge indices for all pairs
    edge_idx = np.zeros((16, 16), dtype=np.int32)
    idx = 0
    for i in range(16):
        for j in range(i+1, 16):
            edge_idx[i, j] = idx
            edge_idx[j, i] = idx
            idx += 1
            
    # List of K4s as arrays of 6 edge indices
    # There are C(16, 4) = 1820 K4s
    k4s = []
    for a in range(16):
        for b in range(a+1, 16):
            for c in range(b+1, 16):
                for d in range(c+1, 16):
                    # Edges: ab, ac, ad, bc, bd, cd
                    k4s.append([
                        edge_idx[a, b], edge_idx[a, c], edge_idx[a, d],
                        edge_idx[b, c], edge_idx[b, d], edge_idx[c, d]
                    ])
    k4s = np.array(k4s, dtype=np.int32)
    num_k4s = len(k4s)
    
    # Objective: count K4s with < 5 distinct colors
    # A K4 has 6 edges.
    # If all 6 are same color -> 1 color (Bad)
    # If 5 same, 1 diff -> 2 colors (Bad)
    # If 4 same, 2 diff (same diff) -> 2 colors (Bad)
    # If 4 same, 2 diff (diff diff) -> 3 colors (Bad)
    # If 3 same, 3 same (diff) -> 2 colors (Bad)
    # If 3 same, 2 same, 1 same (all diff) -> 3 colors (Bad)
    # If 3 same, 2 same, 1 same (2nd pair same as 1st? No, 3-2-1 partition)
    # Basically, we need distinct colors >= 5.
    # This means the partition of 6 edges into color classes must have size >= 5.
    # Possible partitions of 6 with size >= 5:
    # 1-1-1-1-1-1 (6 colors)
    # 2-1-1-1-1 (5 colors)
    # Any other partition has max 4 parts (e.g., 2-2-1-1, 3-1-1-1, etc.)
    
    # So a K4 is GOOD if:
    # 1. All 6 colors distinct
    # 2. Exactly one pair matches, others distinct
    
    def count_bad_colors(colors):
        # colors: array of shape (120,)
        # Get colors for all edges in all K4s
        # k4s shape: (1820, 6)
        k4_colors = colors[k4s] # shape (1820, 6)
        
        # Check for all distinct
        # Sort each row
        sorted_k4 = np.sort(k4_colors, axis=1)
        
        # Check if all 6 are distinct
        all_distinct = np.all(sorted_k4[:, 1:] != sorted_k4[:, :-1], axis=1)
        
        # Check if exactly one pair matches
        # This is tricky with vectorization.
        # Alternative: Count distinct values.
        # np.unique is slow.
        
        # Let's use the property:
        # Good if (All distinct) OR (One pair equal, rest distinct from each other and the pair)
        
        # Let's just compute the number of distinct colors per K4.
        # We can do this by checking equality patterns.
        
        # A simpler, robust method for small N:
        # For each K4, check if the set of colors has size >= 5.
        
        # Vectorized distinct count is hard. Let's try a different approach.
        # If we assume the palette is small (<= 15), we can use one-hot encoding? No, too big.
        
        # Let's stick to the logic:
        # Bad if distinct < 5.
        
        # Let's compute pairwise differences or equalities.
        # There are 15 pairs of edges in a K4.
        # If any 5 edges are the same color, it's bad.
        # If 4 edges are same, bad.
        
        # Let's just iterate over the 1820 K4s in a loop if vectorization is too complex?
        # 1820 * 6 operations is small. But called in SA loop (thousands of times).
        # We need speed.
        
        # Let's try:
        # 1. All distinct?
        # 2. If not, check if exactly one pair is equal.
        
        # Actually, let's just compute the number of distinct colors using a trick.
        # Since colors are integers 0..K-1, we can't easily use bitmasks if K is large.
        # But K is small (<= 15).
        
        # Let's use a Python loop for the bad count, but optimized?
        # No, pure Python loop over 1820 items is ~1ms.
        # 100,000 iterations -> 100s. Too slow.
        
        # We need a vectorized solution.
        
        # Let's check the condition: "at least 5 distinct colors".
        # This is equivalent to: NOT (distinct colors <= 4).
        
        # Let's compute the number of distinct colors for each K4.
        # We can do this by:
        # 1. Sort the 6 colors.
        # 2. Count how many elements are different from their predecessor.
        # 3. Add 1.
        
        diff = np.diff(sorted_k4, axis=1)
        # diff has shape (1820, 5)
        # Number of distinct = 1 + sum(diff != 0)
        distinct_counts = 1 + np.sum(diff != 0, axis=1)
        
        bad_mask = distinct_counts < 5
        return np.sum(bad_mask)

    # State: array of 120 colors
    # Start with random 13-coloring
    palette_size = 13
    colors = np.random.randint(0, palette_size, 120)
    
    best_bad = count_bad_colors(colors)
    best_colors = colors.copy()
    
    # SA Parameters
    T = 10.0
    cooling = 0.999
    min_T = 0.1
    
    # If we start with 13 and it's invalid, we might need to expand palette temporarily?
    # No, the constraint is on the structure.
    # If 13 is not possible, we'll find 14 or 15.
    # But we want to MINIMIZE colors.
    # So we should try 13 first. If we find a valid 13-coloring, return it.
    # If not, try 14?
    
    # Strategy:
    # 1. Try to find valid 13-coloring.
    # 2. If fail, try 14.
    # 3. If fail, try 15.
    
    # We will run SA for each palette size.
    
    for p in [13, 14, 15]:
        if time.time() > deadline - 5:
            break
            
        colors = np.random.randint(0, p, 120)
        current_bad = count_bad_colors(colors)
        best_bad = current_bad
        best_colors = colors.copy()
        
        T = 10.0
        steps = 0
        max_steps = 500000 # Adjust based on time
        
        while time.time() < deadline - 5 and T > min_T:
            # Pick a random edge
            edge_i = np.random.randint(0, 120)
            old_c = colors[edge_i]
            
            # Pick a new color
            if p > 1:
                new_c = np.random.randint(0, p)
                if new_c == old_c:
                    continue
            else:
                continue
                
            colors[edge_i] = new_c
            new_bad = count_bad_colors(colors)
            
            delta = new_bad - current_bad
            
            if delta <= 0 or np.random.random() < np.exp(-delta / T):
                current_bad = new_bad
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_colors = colors.copy()
                    if best_bad == 0:
                        # Found valid coloring with p colors
                        # Try to squeeze?
                        # For now, return it.
                        # We can check if we can map to fewer colors later, but 
                        # since we iterate p=13,14,15, finding 13 is optimal.
                        # Finding 14 is better than 15.
                        return best_colors.tolist()
            else:
                colors[edge_i] = old_c
                
            T *= cooling
            steps += 1
            
    # If no valid coloring found with 13, 14, or 15 in time, 
    # return the best found (might be invalid, but score 0) 
    # or fallback to a known valid one (17 colors).
    
    # Fallback: The modular construction (17 colors) is always valid.
    # Let's generate it.
    fallback = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    # Check if best_colors is valid (just in case)
    if count_bad_colors(best_colors) == 0:
        return best_colors.tolist()
        
    return fallback
