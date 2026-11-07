import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute comparability matrix
    # comp[a][b] = 1 if a is subset of b (strict), 0 otherwise
    # We only need strict subset.
    # To save memory, we can store as list of bitsets or just compute on the fly if optimized.
    # But 4096x4096 is small enough for a list of lists of ints.
    # comp[a] is a bitmask of b such that a < b? No, b is index.
    # Let's use a list of sets of supersets.
    # supersets[a] = set of b such that a is strict subset of b.
    
    # This might be slow to build.
    # Alternative: compute gain on the fly using bit operations.
    # Gain of adding c to family F:
    # count = sum(1 for f in F if c < f or f < c)
    # This is O(m) per candidate.
    # With m=40, and checking 4096 candidates, that's 160k ops per step.
    # SA needs many steps. 55s / 160k is ~300 steps? Too slow.
    
    # We need a faster evaluation.
    # Precompute "subset_count" for each mask against the current family?
    # Let count_sup[c] = number of f in F such that c < f.
    # Let count_sub[c] = number of f in F such that f < c.
    # When we remove x and add y:
    # For each f in F \ {x}:
    #   if y < f or f < y: gain += 1
    #   if x < f or f < x: loss += 1
    # Also check y vs x and x vs y.
    
    # We can maintain an array `comp_with_family` of size 4096.
    # comp_with_family[c] = number of f in F such that c and f are comparable (c!=f).
    # When F changes, updating this array is O(m) per change?
    # If we remove x:
    #   for all c: if c and x comparable, comp_with_family[c] -= 1
    #   This is O(4096) per removal.
    # If we add y:
    #   for all c: if c and y comparable, comp_with_family[c] += 1
    #   This is O(4096) per addition.
    # Total O(8192) per swap.
    # 55s / 8192 is ~2000 swaps. Still might be low.
    
    # Can we do better?
    # Only update candidates?
    # We need to find the best y to add.
    # We want to maximize comp_with_family[y] - (loss from removing x).
    # Loss from removing x is fixed for a given x.
    # So we want to maximize comp_with_family[y].
    # But comp_with_family[y] depends on F.
    
    # Let's try to optimize the inner loop in Python.
    # Use NumPy?
    # Represent family as a set of masks.
    # For a candidate y, count comparable in F.
    # This is O(m) per candidate.
    # To find best y, we scan all 4096 candidates. O(4096 * m).
    # 4096 * 40 = 163k ops.
    # In Python, 163k ops takes ~10-20ms?
    # 55s / 20ms = 2750 iterations.
    
    # Let's try to speed up the "count comparable" step.
    # Precompute a matrix `is_subset[a][b]`?
    # 4096 x 4096 bool matrix.
    # Using NumPy, we can vectorize.
    
    import numpy as np
    
    # Build subset matrix
    # A[i, j] = 1 if i is subset of j
    # i < j implies i subset j and i != j.
    # We want strict subset.
    
    # This matrix is 4096x4096.
    # Memory: 16M bytes if uint8.
    
    # Let's build it.
    # masks = np.arange(4096)
    # A = np.zeros((4096, 4096), dtype=np.uint8)
    # for i in range(4096):
    #   for j in range(i):
    #     if (i & j) == i: A[i, j] = 1 # i subset j? No, i & j == i means i subset j.
    #     # Wait, if i is subset of j, then i & j = i.
    #     # So A[i, j] = 1 if i subset j.
    #     # We want strict. So i != j.
    #     # Since j < i, i != j is always true.
    #     # But we need i subset j.
    #     # If i subset j, then popcount(i) <= popcount(j).
    #     # So we only need to check j with popcount >= popcount(i).
    
    # This build time might be significant.
    # 4096 * 4096 / 2 = 8M checks.
    # In Python loop, 8M checks is ~1-2s. Acceptable.
    
    # However, we can avoid building the full matrix if we use bit tricks.
    # But NumPy vectorization is fast.
    
    # Let's try a simpler approach:
    # Greedy + Local Search with fast evaluation.
    
    # 1. Initialize with a good heuristic.
    # 2. Local search: 
    #    - Try all swaps (i, j) where i is in F, j is not.
    #    - Evaluate delta.
    #    - Accept if delta > 0.
    
    # To make it fast, we will use the `comp_with_family` array idea.
    # `score[c]` = number of f in F such that c and f are comparable.
    # When we swap x (in F) with y (not in F):
    #   New score for y in F' = score[y] - (1 if y comparable to x else 0) + (1 if y comparable to x else 0)?
    #   No.
    #   Let F be current family.
    #   score[c] = |{f in F : c comparable to f}|.
    #   If we remove x:
    #     F_new = F \ {x}.
    #     score_new[c] = score[c] - (1 if c comparable to x else 0).
    #   If we add y to F_new:
    #     F_final = F_new U {y}.
    #     The gain in total comparable pairs is:
    #     (Number of pairs in F_final) - (Number of pairs in F)
    #     = (Pairs in F_new) + (Pairs involving y in F_final) - (Pairs in F)
    #     Pairs in F = Pairs in F_new + (Pairs involving x in F)
    #     So Gain = Pairs involving y in F_final - Pairs involving x in F
    #     Pairs involving x in F = score[x] - 1 (since x is in F, it's comparable to itself? No, strict. So score[x] counts f in F, f!=x. So score[x] is exactly the number of pairs involving x.)
    #     Wait, score[x] as defined above counts f in F such that x comparable to f. Since x in F, and we exclude self, score[x] is the number of comparable partners of x in F.
    #     So Loss = score[x].
    #     
    #     Pairs involving y in F_final:
    #     y is comparable to f in F_new.
    #     Number of such f = score[y] - (1 if y comparable to x else 0).
    #     (Because score[y] counts f in F. If x is comparable to y, we must subtract it because x is removed.)
    #     
    #     So Gain = (score[y] - (1 if y comparable to x else 0)) - score[x].
    #     
    #     We want to maximize Gain.
    #     This is O(1) per candidate y, IF we have score[] up to date.
    #     
    #     Updating score[] after a swap:
    #     We removed x, added y.
    #     For all c:
    #       score[c] changes by:
    #       -1 if c comparable to x (x removed)
    #       +1 if c comparable to y (y added)
    #     
    #     So we need to update score[] for all 4096 c.
    #     This is O(4096) per swap.
    #     Finding best y is O(4096) per swap.
    #     Total O(8192) per swap.
    #     As estimated, ~2000-3000 swaps in 55s.
    
    # This is a solid local search. To escape local optima, we can add random restarts or simulated annealing (accept negative gain with prob exp(-delta/T)).
    
    # Let's implement this.

    # Precompute comparability:
    # comp[a][b] = 1 if a comparable to b (a!=b)
    # We can store this as a list of lists of booleans, or bitsets.
    # Bitsets are faster for "is comparable" check?
    # Actually, we just need to check "is comparable" for a specific pair (x, y).
    # (x & y) == x or (y & x) == y.
    
    # But for updating score[], we need to iterate c and check if c comparable to x.
    # This is 4096 checks.
    
    # Let's optimize the update.
    # Precompute `supersets[a]` as a list of masks?
    # Or just use the bit check.
    
    # Initialize F with a good guess.
    # Chain of 12 + 12 shifted + 16 random?
    # Let's use the 24-set construction from above + 16 sets.
    
    random.seed(42)
    
    # Construct initial family
    # Chain C0..C11
    # D0..D11 (C_i U {11})
    # E0..E11 (C_i U {10})? 
    # Let's just take the first 40 from a lex-ordered list of "good" sets?
    # Or simply:
    # F = []
    # for i in range(12):
    #   mask = (1 << i) - 1
    #   F.append(mask)
    #   F.append(mask | (1 << 11))
    #   F.append(mask | (1 << 10))
    # This gives 36 sets.
    # Add 4 more.
    
    # Let's use a greedy initialization.
    F = []
    F_set = set()
    
    # Add chain
    for i in range(12):
        mask = (1 << i) - 1
        F.append(mask)
        F_set.add(mask)
        
    # Add shifted chains
    for i in range(12):
        mask = ((1 << i) - 1) | (1 << 11)
        if mask not in F_set:
            F.append(mask)
            F_set.add(mask)
            
    # Fill rest with random
    while len(F) < m:
        mask = random.randint(0, total_masks - 1)
        if mask not in F_set:
            F.append(mask)
            F_set.add(mask)
            
    # Calculate initial score array
    # score[c] = number of f in F such that c comparable to f
    score = [0] * total_masks
    for c in range(total_masks):
        cnt = 0
        for f in F:
            if c == f:
                continue
            if (c & f) == c or (f & c) == f:
                cnt += 1
        score[c] = cnt
        
    # Local Search
    best_score_val = sum(score[f] for f in F) // 2
    
    while time.time() < deadline - 5.0:
        improved = False
        
        # Find best swap
        best_gain = -1
        best_x = -1
        best_y = -1
        
        # We can iterate x in F, and for each x, find best y not in F.
        # Or iterate all pairs?
        # Iterating x (40) and y (4096) is 160k checks.
        # For each pair, Gain = score[y] - (1 if y comp x else 0) - score[x].
        # score[x] is constant for a given x.
        # So for a fixed x, we want to maximize score[y] - (1 if y comp x else 0).
        
        for x in F:
            sx = score[x]
            # Find y not in F maximizing score[y] - (1 if y comp x else 0)
            best_y_for_x = -1
            best_val = -1
            for y in range(total_masks):
                if y in F_set:
                    continue
                val = score[y]
                if (x & y) == x or (y & x) == y:
                    val -= 1
                if val > best_val:
                    best_val = val
                    best_y_for_x = y
            
            gain = best_val - sx
            if gain > best_gain:
                best_gain = gain
                best_x = x
                best_y = best_y_for_x
                
        if best_gain > 0:
            # Perform swap
            F.remove(best_x)
            F_set.remove(best_x)
            F.append(best_y)
            F_set.add(best_y)
            
            # Update score[]
            # Remove best_x: for all c, if c comp best_x, score[c] -= 1
            # Add best_y: for all c, if c comp best_y, score[c] += 1
            
            for c in range(total_masks):
                # Check c comp best_x
                if c != best_x and ((c & best_x) == c or (best_x & c) == best_x):
                    score[c] -= 1
                # Check c comp best_y
                if c != best_y and ((c & best_y) == c or (best_y & c) == best_y):
                    score[c] += 1
            
            # Update best_score_val
            best_score_val += best_gain
            improved = True
            
        if not improved:
            # Stuck in local optimum. Perturb.
            # Randomly swap 2 sets.
            if len(F) >= 2:
                i, j = random.sample(range(m), 2)
                x = F[i]
                y = F[j]
                # Find replacements
                # Just pick random ones?
                # Better: pick random y1, y2 not in F.
                # And accept regardless of gain? (Simulated Annealing step)
                # For simplicity, just restart or random walk.
                # Let's do a random walk: replace one set with a random one.
                idx = random.randint(0, m-1)
                old = F[idx]
                new = random.randint(0, total_masks - 1)
                while new in F_set:
                    new = random.randint(0, total_masks - 1)
                
                F[idx] = new
                F_set.remove(old)
                F_set.add(new)
                
                # Update score
                for c in range(total_masks):
                    if c != old and ((c & old) == c or (old & c) == old):
                        score[c] -= 1
                    if c != new and ((c & new) == c or (new & c) == new):
                        score[c] += 1
                        
                # Recalc best_score_val
                best_score_val = sum(score[f] for f in F) // 2

    # Convert to lists
    result = []
    for mask in F:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
        
    return result
