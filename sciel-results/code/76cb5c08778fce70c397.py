import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 118  # 2s margin for safety
    
    points = list(range(16))
    
    # --- Fast Independence Number Calculation ---
    # We represent a set of points as a 16-bit integer.
    # A triple is valid if it is not a subset of the independent set.
    # We want to find the largest subset S such that no triple is contained in S.
    
    def compute_alpha(triples):
        """
        Computes the independence number of the hypergraph defined by 'triples'.
        triples: list of (a, b, c)
        """
        if not triples:
            return 16
            
        # Precompute masks for the triples
        masks = []
        for a, b, c in triples:
            masks.append((1 << a) | (1 << b) | (1 << c))
        
        # Optimization: 
        # 1. Check if there is a large independent set.
        # 2. Since we start with a good construction (alpha ~ 6), we can check 
        #    downwards from a reasonable bound or use a smarter pruning.
        # 
        # Efficient approach:
        # Iterate k from 16 down to 1.
        # For each k, iterate all subsets of size k.
        # Check if any triple is contained in the subset.
        # 
        # C(16, 8) = 12870. C(16, 9) = 11440. 
        # Checking ~25k subsets * ~50 triples (avg) = 1.25M ops. 
        # In Python, this might take ~0.5-1s if not optimized.
        # Let's optimize the check:
        # For a given subset mask S, we need to check if any triple mask T 
        # satisfies (T & S) == T.
        # 
        # Better approach for speed:
        # Precompute for each point p, the list of triples containing p? No, 
        # we need "no triple contained in S".
        
        # Let's use a recursive backtracking or simple subset iteration with early exit.
        # Given the small size (16 points), iterating subsets is feasible if we 
        # are clever about ordering or using bit operations.
        
        # Alternative: Use the fact that if S is independent, its complement 
        # is a vertex cover. But finding min vertex cover is hard.
        
        # Let's stick to subset enumeration but optimize the inner loop.
        # We will check sizes from 16 down to 1.
        # To speed up, we can prune: if a subset S of size k is not independent, 
        # no superset of S is independent. But we are iterating by size, not 
        # building supersets.
        
        # Let's try a different angle: 
        # For each subset S, check independence.
        # We can precompute the "bad" masks? No.
        
        # Let's just make the inner loop as fast as possible.
        # Convert masks to a tuple for faster iteration?
        # Use a set of masks for O(1) lookup? No, we need subset check.
        
        # Optimization: 
        # If we have a list of triple masks, we can check:
        # for m in masks:
        #     if (mask & m) == m: return False
        
        # To make this faster, we can sort masks by popcount? No.
        
        # Let's implement a helper to check independence of a mask.
        def is_independent_mask(mask):
            for m in masks:
                if (mask & m) == m:
                    return False
            return True
            
        # Iterate k from 16 down to 1
        # We can stop early if we find one.
        # We know alpha is likely small (<=6) for good constructions.
        # So we can start checking from k=16 down to 1.
        # But C(16,16)=1, C(16,15)=16, C(16,14)=120, C(16,13)=560, C(16,12)=1820, 
        # C(16,11)=4368, C(16,10)=8008, C(16,9)=11440, C(16,8)=12870.
        # Total checks if alpha=6: 1+16+120+560+1820+4368+8008+11440+12870 = ~39k subsets.
        # Each check takes ~len(triples) bit ops.
        # If len(triples) ~ 50, that's ~2M ops. Python does ~10-50M ops/sec.
        # So ~0.05s per alpha calc. This is fast enough for local search!
        
        for k in range(16, 0, -1):
            # Generate combinations of points
            for subset in itertools.combinations(points, k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                if is_independent_mask(mask):
                    return k
        return 1

    # --- Helper: Validity Check for Linear Triple System ---
    def is_valid_addition(triple, existing_masks):
        a, b, c = triple
        new_mask = (1 << a) | (1 << b) | (1 << c)
        
        # Check intersection with all existing triples
        for m in existing_masks:
            inter = new_mask & m
            # If intersection has > 1 bit, invalid
            if inter and (inter & (inter - 1)):
                return False
        return True

    # --- Phase 1: Algebraic Construction ---
    # Construct a linear triple system on 16 points using AG(2,4) structure.
    # Points: (x, y) in F4^2.
    # F4 = {0, 1, w, w+1} where w^2 + w + 1 = 0.
    # We map these to 0..15.
    
    # Define F4 addition and multiplication tables
    # 0:0, 1:1, 2:w, 3:w+1
    add_table = [
        [0,1,2,3],
        [1,0,3,2],
        [2,3,0,1],
        [3,2,1,0]
    ]
    mul_table = [
        [0,0,0,0],
        [0,1,2,3],
        [0,2,3,1],
        [0,3,1,2]
    ]
    
    def f4_add(a, b): return add_table[a][b]
    def f4_mul(a, b): return mul_table[a][b]
    
    # Map F4^2 to 0..15
    # Point (x,y) -> 4*x + y
    # x in 0..3, y in 0..3
    
    # In AG(2,4), lines are defined by y = mx + c or x = c.
    # Each line has 4 points.
    # We need to partition these lines into triples? Or select triples from them?
    # A linear triple system requires any two triples share at most 1 point.
    # If we take lines, any two lines intersect in at most 1 point.
    # So any subset of lines is a valid linear hypergraph.
    # However, lines have 4 points. We need triples.
    # We can take any 3 points from a line to form a triple.
    # But if we take triple T1 = {a,b,c} from line L1, and T2 = {d,e,f} from line L2,
    # they intersect in at most 1 point (since L1 and L2 intersect in at most 1 point).
    # So we can form a system by selecting triples from the lines.
    
    # How many lines? 
    # 4 slopes * 4 intercepts = 16 lines of type y=mx+c.
    # 4 vertical lines x=c.
    # Total 20 lines.
    # Each point is on 5 lines (1 vertical + 4 non-vertical).
    
    # Strategy: 
    # 1. Identify all 20 lines (sets of 4 points).
    # 2. Randomly select a subset of these lines.
    # 3. From each selected line, randomly pick 3 points to form a triple.
    # 4. Check if the resulting set of triples is linear (it will be, by construction, 
    #    because any two lines intersect in <= 1 point, so any two triples from 
    #    different lines intersect in <= 1 point. Triples from the same line 
    #    are not allowed to coexist if they share 2 points, but we pick one triple 
    #    per line).
    
    # Wait, if we pick one triple per line, do we ensure linearity?
    # Yes. Let T1 be from L1, T2 from L2.
    # T1 subset L1, T2 subset L2.
    # T1 intersect T2 subset L1 intersect L2.
    # In AG(2,4), L1 intersect L2 has size 0 or 1.
    # So T1 intersect T2 has size 0 or 1.
    # Thus, the system is linear.
    
    # Now, what is the independence number?
    # We want to minimize the largest set of points containing no triple.
    # If we include many lines, we cover more "combinations".
    # Let's include all 20 lines?
    # If we include all 20 lines, and pick one triple from each, we have 20 triples.
    # Does this minimize alpha?
    # Likely yes, as it's a very dense structure.
    
    # Let's generate the lines.
    lines = []
    
    # Vertical lines: x = c
    for c in range(4):
        line = []
        for y in range(4):
            p = 4*c + y
            line.append(p)
        lines.append(line)
        
    # Non-vertical lines: y = mx + c
    for m in range(4):
        for c in range(4):
            line = []
            for x in range(4):
                y = f4_add(f4_mul(m, x), c)
                p = 4*x + y
                line.append(p)
            lines.append(line)
            
    # Now, construct a random linear triple system by picking one triple from each line.
    # To minimize alpha, we should pick triples that "break up" large independent sets.
    # A random selection is a good start.
    
    def generate_random_system():
        triples = []
        masks = []
        for line in lines:
            # Pick 3 random points from the line
            idxs = random.sample(range(4), 3)
            triple = [line[i] for i in idxs]
            # Shuffle the triple? Order doesn't matter for set, but for mask it does.
            triple.sort()
            triples.append(tuple(triple))
            masks.append((1 << triple[0]) | (1 << triple[1]) | (1 << triple[2]))
        return triples, masks
    
    # Initialize with a random instance from this family
    best_triples, best_masks = generate_random_system()
    best_alpha = compute_alpha(best_triples)
    
    # Local Search
    # We want to reduce alpha.
    # Move: Pick a line, change the triple selected from that line.
    # Since the structure is rigid (one triple per line), we can't add/remove lines 
    # without breaking the "one per line" property if we want to stay in this family.
    # But we can swap which 3 points we pick from a line.
    
    # There are 4 choices per line. So 4^20 possibilities.
    # We can do a local search where we flip the choice for one line at a time.
    
    current_triples, current_masks = best_triples, best_masks
    current_alpha = best_alpha
    
    # Precompute for each line the 4 possible triples (as tuples and masks)
    line_options = []
    for line in lines:
        options = []
        for i in range(4):
            # The triple is the line minus the i-th point
            t = [line[j] for j in range(4) if j != i]
            t.sort()
            t = tuple(t)
            m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            options.append((t, m))
        line_options.append(options)
    
    # Current indices for each line (0-3)
    # We need to recover which option was chosen in the initial random generation.
    # Or just re-initialize with a specific index vector.
    
    # Let's re-initialize with a vector of indices
    def system_from_indices(indices):
        triples = []
        masks = []
        for i, idx in enumerate(indices):
            t, m = line_options[i][idx]
            triples.append(t)
            masks.append(m)
        return triples, masks
    
    # Start with random indices
    current_indices = [random.randint(0, 3) for _ in range(20)]
    current_triples, current_masks = system_from_indices(current_indices)
    current_alpha = compute_alpha(current_triples)
    
    best_indices = current_indices[:]
    best_triples, best_masks = current_triples, current_masks
    best_alpha = current_alpha
    
    # Simulated Annealing / Hill Climbing on the index vector
    # Temperature starts high, cools down.
    T = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Pick a random line to change
        line_idx = random.randint(0, 19)
        old_idx = current_indices[line_idx]
        
        # Pick a new index (different from old)
        new_idx = random.randint(0, 3)
        if new_idx == old_idx:
            new_idx = (new_idx + 1) % 4
            
        # Apply change
        current_indices[line_idx] = new_idx
        new_triples, new_masks = system_from_indices(current_indices)
        new_alpha = compute_alpha(new_triples)
        
        # Acceptance criterion
        if new_alpha < current_alpha:
            # Always accept improvement
            current_alpha = new_alpha
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_indices = current_indices[:]
                best_triples, best_masks = new_triples, new_masks
        elif new_alpha == current_alpha:
            # Accept with some probability to explore
            if random.random() < 0.5:
                current_alpha = new_alpha
            else:
                # Revert
                current_indices[line_idx] = old_idx
        else:
            # Worse: Accept with probability exp(-delta/T)
            delta = new_alpha - current_alpha
            # delta is integer >= 1
            if random.random() < pow(2.718, -delta / T):
                current_alpha = new_alpha
            else:
                # Revert
                current_indices[line_idx] = old_idx
                
        T *= cooling
        if T < 0.01:
            T = 0.01 # Keep some randomness to escape local minima
            
    return best_triples

# Quick test to ensure it runs
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    print(f"Time: {time.time() - t0}, Triples: {len(res)}")
    # Verify validity
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            inter = set(res[i]) & set(res[j])
            if len(inter) > 1:
                print(f"Invalid pair: {res[i]} and {res[j]}")
    # Compute alpha
    # Note: compute_alpha is defined inside construct, so we can't call it here directly 
    # unless we move it or duplicate. For the test, we'll just trust the logic.
    # To verify alpha in test, we'd need to copy the function or call construct's internal.
    # Let's just print the result.
