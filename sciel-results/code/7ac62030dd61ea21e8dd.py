import numpy as np
import time
import itertools

def construct(q, n, radius):
    # q=3, n=8, radius=1
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters, though task is specific
        # Use simple greedy for general case
        pass
        
    # Specific optimization for 3^8, radius 1
    num_points = 3**8
    # Map point index to tuple
    # We can work with indices 0..6560
    # Convert index to base-3 tuple
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % 3)
            idx //= 3
        return tuple(t)
        
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * 3 + t[i]
        return idx
        
    # Precompute neighbors for each point (candidate codeword)
    # neighbors[c] = list of points covered by c
    # We need to check coverage efficiently.
    # Let's represent the set of uncovered points as a boolean array or set.
    # Given the size (6561), a set of integers is fine.
    
    # Generate all points
    all_points = list(range(num_points))
    
    # Precompute coverage sets for each candidate codeword
    # coverage[c] = set of points p such that d(c, p) <= 1
    coverages = [None] * num_points
    for c in all_points:
        ct = idx_to_tuple(c)
        pts = {c}
        for i in range(n):
            for val in range(3):
                if ct[i] == val: continue
                new_t = list(ct)
                new_t[i] = val
                pts.add(tuple_to_idx(tuple(new_t)))
        coverages[c] = pts
        
    # Greedy Set Cover
    uncovered = set(all_points)
    selected = []
    
    # To speed up, we can pre-compute the size of intersection with uncovered?
    # No, uncovered changes.
    # We just iterate.
    
    # Optimization: If time is tight, we might not finish.
    # 6561 candidates, ~500 iterations. 3M set intersections.
    # Python sets are fast.
    
    start_time = time.time()
    deadline = 100.0 # Leave margin
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_c = -1
        max_gain = 0
        # Randomize tie-breaking?
        for c in all_points:
            if c in selected: continue # Not necessary if we just pick best
            # Calculate gain
            # Intersection of coverages[c] and uncovered
            # To speed up, we can check if len(coverages[c] & uncovered) > max_gain
            # This is expensive.
            # Alternative: iterate through coverages[c] and count how many are in uncovered
            # This is also O(17).
            
            # Let's try counting
            cnt = 0
            for p in coverages[c]:
                if p in uncovered:
                    cnt += 1
                    if cnt <= max_gain:
                        break
            if cnt > max_gain:
                max_gain = cnt
                best_c = c
                if max_gain == 17: # Max possible
                    break
                    
        if best_c == -1:
            break
            
        selected.append(best_c)
        # Remove covered points
        for p in coverages[best_c]:
            if p in uncovered:
                uncovered.remove(p)
                
    # Peeling phase: Remove redundant codewords
    # We have `selected`.
    # For each c in selected, check if it's needed.
    # A codeword c is needed if there exists a point p in B_1(c) that is ONLY covered by c in `selected`.
    # i.e., for all other c' in selected, p not in B_1(c').
    
    # To do this efficiently:
    # 1. Count for each point p, how many codewords in `selected` cover it.
    # 2. If count[p] == 1, the unique codeword covering p is essential.
    # 3. Remove non-essential codewords.
    
    # Step 1: Count coverage
    cover_count = np.zeros(num_points, dtype=np.int8)
    for c in selected:
        for p in coverages[c]:
            cover_count[p] += 1
            
    # Step 2: Identify essential codewords
    essential = set()
    for c in selected:
        for p in coverages[c]:
            if cover_count[p] == 1:
                essential.add(c)
                break
                
    # Step 3: Construct final list
    # We can keep only essential ones?
    # If we remove a non-essential one, does it become essential for others?
    # No, because if it wasn't essential, every point it covered was covered by at least one other.
    # Removing it doesn't uncover any point.
    # So we can remove ALL non-essential ones.
    
    final_code = [c for c in selected if c in essential]
    
    # Convert indices to lists of integers
    result = [list(idx_to_tuple(c)) for c in final_code]
    
    return result
