def _greedy_sidon_blocking(seed=None):
    """
    Greedy construction prioritizing elements that block the most other candidates.
    This aims to create a small inclusion-maximal set.
    """
    if seed is not None:
        random.seed(seed)
    
    S = []
    sums = set()
    candidates = set(range(1, N + 1))
    
    while candidates:
        # Find all candidates that can be added to S
        valid_candidates = []
        for x in candidates:
            new_sums = {2 * x} | {x + s for s in S}
            if not (new_sums & sums) and len(new_sums) == len(new_sums):
                valid_candidates.append(x)
        
        if not valid_candidates:
            break
            
        # Score each valid candidate by how many other valid candidates it blocks
        # We only consider blocking among the *currently valid* candidates
        # This is an approximation of the degree in the induced subgraph of valid candidates
        best_x = None
        best_score = -1
        
        for x in valid_candidates:
            new_sums_x = {2 * x} | {x + s for s in S}
            # How many other valid candidates y would become invalid?
            block_count = 0
            for y in valid_candidates:
                if y == x:
                    continue
                # Check if y is still valid if x is added
                # y's new sums relative to S U {x}
                # y must not conflict with S U {x}
                # y's sums: {2y} U {y+s for s in S U {x}}
                # = {2y, y+x} U {y+s for s in S}
                
                # If y was valid, {2y} U {y+s for s in S} was disjoint from sums and internally consistent.
                # Now we add x.
                # New sums for y: {2y, y+x} U {y+s for s in S}
                # We need to check if these collide with:
                # 1. Existing sums (S_old sums) -> Already checked for y validity? 
                #    y was valid against S. So {2y, y+s (s in S)} disjoint from sums.
                # 2. New sums from x: new_sums_x
                # 3. Internal consistency of y's new sums:
                #    {2y, y+x} U {y+s (s in S)} must be distinct.
                #    Previously {2y} U {y+s (s in S)} was distinct.
                #    So we just need to check y+x against {2y} U {y+s (s in S)} and new_sums_x.
                
                y_new_sum = y + x
                # Check collision with existing sums of y (excluding the new one)
                # Existing y-sums: {2y} U {y+s for s in S}
                # If y_new_sum is in that set, y becomes invalid.
                # Also if y_new_sum is in new_sums_x, y becomes invalid.
                
                # Optimization: Precompute y's "old" sums? 
                # For simplicity in this heuristic, we can just check the specific new interaction.
                
                # Condition 1: y+x collides with existing sums of y
                # y+x == 2y  => x == y (handled)
                # y+x == y+s => x == s (x in S? No, x is candidate not in S)
                # So y+x cannot collide with y+s for s in S.
                # Can y+x == 2y? Only if x=y.
                # So internal consistency of y's sums is preserved?
                # Wait. y+x is a new sum.
                # Does y+x collide with 2y? No (x!=y).
                # Does y+x collide with y+s? No (x not in S).
                # So y's internal sums remain distinct.
                
                # Condition 2: Collision between y's sums and x's sums (new_sums_x)
                # y's sums: {2y} U {y+s for s in S} U {y+x}
                # x's sums: {2x} U {x+s for s in S}
                
                # Check 2y against new_sums_x
                if 2 * y in new_sums_x:
                    block_count += 1
                    continue
                    
                # Check y+x against new_sums_x
                if y + x in new_sums_x:
                    block_count += 1
                    continue
                    
                # Check y+s (s in S) against new_sums_x
                # y+s == 2x => y == 2x - s
                # y+s == x+s' => y == x + s' - s
                # This loop is expensive. 
                
                # Let's use a simpler proxy for blocking:
                # Just check if y+x or 2y is in new_sums_x.
                # This catches the most direct conflicts.
                
            if block_count > best_score:
                best_score = block_count
                best_x = x
                
        # If tie, pick random or smallest
        if best_x is None:
            best_x = random.choice(valid_candidates)
            
        # Add best_x
        S.append(best_x)
        new_sums_best = {2 * best_x} | {best_x + s for s in S[:-1]} # S[:-1] because best_x just appended
        # Actually, S was updated. Let's be careful.
        # S currently has best_x at the end.
        # Previous S was S[:-1].
        # sums should be updated with sums involving best_x and previous S elements.
        # The code above calculated new_sums for x relative to S (before adding x).
        # So new_sums_x = {2x} U {x+s for s in S_old}.
        # We need to update `sums` with these.
        
        # Recalculate new_sums for best_x relative to S without best_x
        S_old = S[:-1]
        new_sums_best = {2 * best_x} | {best_x + s for s in S_old}
        sums.update(new_sums_best)
        candidates.remove(best_x)
        
    S.sort()
    return S
