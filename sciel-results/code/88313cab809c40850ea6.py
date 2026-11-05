import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all t-subsets as bitmasks
    all_t_subs = []
    for comb in itertools.combinations(range(v), t):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        all_t_subs.append(mask)
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple restarts
    while True:
        if time.time() > deadline:
            break
            
        # Reset uncovered set
        uncovered = set(all_t_subs)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to start the block
            start_mask = random.choice(list(uncovered))
            # Decode start_mask to list of points
            start_points = [i for i in range(v) if (start_mask >> i) & 1]
            
            # Initialize block
            B = set(start_points)
            
            # Initialize subset lists for B
            # subs2: list of bitmasks for 2-subsets of B
            # subs3: list of bitmasks for 3-subsets of B
            # subs4: list of bitmasks for 4-subsets of B
            
            subs2 = []
            for p1, p2 in itertools.combinations(start_points, 2):
                subs2.append((1 << p1) | (1 << p2))
                
            subs3 = []
            for p1, p2, p3 in itertools.combinations(start_points, 3):
                subs3.append((1 << p1) | (1 << p2) | (1 << p3))
                
            subs4 = []
            for p1, p2, p3, p4 in itertools.combinations(start_points, 4):
                subs4.append((1 << p1) | (1 << p2) | (1 << p3) | (1 << p4))
            
            # Remove the start subset from uncovered
            # It is covered by the block (eventually)
            # Actually, we should remove all t-subsets contained in B.
            # Since |B|=t, only B itself is a t-subset.
            uncovered.discard(start_mask)
            
            # Grow the block to size k
            while len(B) < k:
                gains = {}
                candidates = [x for x in range(v) if x not in B]
                
                # Calculate gain for each candidate
                for x in candidates:
                    gain = 0
                    x_bit = 1 << x
                    # A 5-subset S is covered if S = T U {x} where T is a 4-subset of B
                    # and S is currently uncovered.
                    for T in subs4:
                        S_mask = T | x_bit
                        if S_mask in uncovered:
                            gain += 1
                    gains[x] = gain
                
                # Pick best candidate
                # Random tie-break
                max_gain = max(gains.values())
                best_candidates = [x for x, g in gains.items() if g == max_gain]
                x_new = random.choice(best_candidates)
                
                B.add(x_new)
                x_bit = 1 << x_new
                
                # Update subset lists
                # New 2-subsets: {x_new, y} for y in B_old
                # We need old B points
                # Actually, we can just generate new subsets from scratch or update incrementally.
                # Incremental is faster.
                
                # We need to add new 2, 3, 4 subsets involving x_new.
                # And remove newly covered 5-subsets.
                
                # 1. Remove newly covered 5-subsets from uncovered
                # These are T U {x_new} for T in subs4 (old)
                for T in subs4:
                    S_mask = T | x_bit
                    uncovered.discard(S_mask)
                
                # 2. Update subs4
                # New 4-subsets containing x_new: {x_new} U T where T is 3-subset of B_old
                # We have subs3 (old)
                new_subs4 = []
                for T in subs3:
                    new_subs4.append(T | x_bit)
                subs4.extend(new_subs4)
                
                # 3. Update subs3
                # New 3-subsets containing x_new: {x_new} U T where T is 2-subset of B_old
                # We have subs2 (old)
                new_subs3 = []
                for T in subs2:
                    new_subs3.append(T | x_bit)
                subs3.extend(new_subs3)
                
                # 4. Update subs2
                # New 2-subsets containing x_new: {x_new, y} for y in B_old
                # We need the points in B_old.
                # We can just iterate all y in B except x_new
                new_subs2 = []
                for y in B:
                    if y != x_new:
                        new_subs2.append(x_bit | (1 << y))
                subs2.extend(new_subs2)
            
            # Add block to list
            blocks.append(sorted(B))
            
        # Check if this run is better
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        # If we hit the theoretical lower bound or a very good number, we could stop early?
        # Lower bound is 264. Best known 335.
        # If we get < 335, that's a record.
        # Let's just run until deadline.
        
    if best_blocks is None:
        # Fallback to simple greedy if no time left or something went wrong
        # But the loop should always produce something
        return []
        
    return best_blocks
