def solve_scope(scope, seed, max_nodes=1000000):
    rng = random.Random(seed)
    used_bits = 0
    rows = []
    nodes = [0]
    
    def build_row(idx):
        nonlocal used_bits, nodes
        if nodes[0] > max_nodes:
            return False
        nodes[0] += 1
        
        if idx == n_rows:
            return True
        
        row = [0]
        row_diff_bits = 0
        
        # Symmetry breaking
        if idx == 0:
            min_start = 1
        else:
            min_start = rows[idx - 1][1] + 1
            
        # Pre-compute conflict mask for the current row state
        # conflict_mask bit x is 1 if x is invalid for the next element
        # Initially, row=[0]. blocked = used_bits.
        # Valid x must not have (x-0) in used_bits.
        # So x cannot be in used_bits.
        
        def rec(row, row_diff_bits, min_x, conflict_mask):
            nonlocal used_bits, nodes
            if nodes[0] > max_nodes:
                return False
            nodes[0] += 1
            
            if len(row) == k + 1:
                # Check global conflict
                if used_bits & row_diff_bits:
                    return False
                
                used_bits |= row_diff_bits
                rows.append(row)
                if build_row(idx + 1):
                    return True
                else:
                    rows.pop()
                    used_bits ^= row_diff_bits
                    return False

            remaining = (k + 1) - len(row)
            max_x = scope - (remaining - 1)
            
            if max_x < min_x:
                return False

            # Generate candidates
            # We need x such that for all a in row, (x - a) is not in (used_bits | row_diff_bits)
            # This is equivalent to: x is not in { a + d | a in row, d in (used_bits | row_diff_bits) }
            # We maintain conflict_mask which represents this set.
            
            # Find candidates in [min_x, max_x] where conflict_mask bit is 0
            candidates = []
            # Iterate through range
            # Optimization: if range is large, we can use bit operations to find set bits?
            # But we want 0 bits.
            # Let's just loop, it's fast enough for K=7.
            
            blocked = used_bits | row_diff_bits
            
            # Update conflict_mask? 
            # The current conflict_mask passed in is for the current row state.
            # We need to check candidates against it.
            
            # Actually, maintaining conflict_mask incrementally is better.
            # But passing it as a parameter is fine.
            
            # Let's just use the direct check for simplicity and correctness, 
            # but optimize the inner loop.
            
            # Heuristic: Pick x that minimizes options for next step?
            # Or just fail-first.
            
            # We'll collect a few candidates and pick one.
            # To speed up, we can stop collecting after N candidates and pick randomly?
            # No, we need to find a solution.
            
            # Let's try all candidates in order, but with a random shuffle if many.
            
            # Optimization: 
            # If we find a candidate, we can recurse immediately.
            # To allow backtracking, we must try others if it fails.
            
            # We will iterate x from min_x to max_x.
            # To add randomness, we can shuffle the range or pick a random start?
            # Shuffling 100s of items is expensive.
            # Instead, we can pick a random candidate from the valid ones.
            
            valid_xs = []
            for x in range(min_x, max_x + 1):
                # Check if x is valid
                # (x - a) not in blocked for all a in row
                ok = True
                for a in row:
                    d = x - a
                    if (blocked >> d) & 1:
                        ok = False
                        break
                if ok:
                    valid_xs.append(x)
                    if len(valid_xs) > 20: # Early exit for heuristic
                        break

            if not valid_xs:
                return False
            
            # Heuristic:
            # Try the one that leaves the most space?
            # Or just pick one.
            # To improve search, pick randomly from the first few?
            # Let's pick the one with the smallest value (greedy) or random.
            # Random is better for escaping local minima.
            
            # Let's try a few candidates in random order
            # Shuffle valid_xs? 
            # If len is small, shuffle.
            if len(valid_xs) < 10:
                rng.shuffle(valid_xs)
            else:
                # Pick a random subset to try?
                # Or just pick one randomly.
                # If we pick one and it fails, we restart the loop?
                # No, we must try all eventually.
                # But we can order them.
                # Let's try the middle one, then neighbors.
                mid = len(valid_xs) // 2
                order = [mid]
                for i in range(1, len(valid_xs)):
                    order.append(mid - i)
                    if mid + i < len(valid_xs):
                        order.append(mid + i)
                # Filter out of bounds
                valid_indices = [i for i in order if 0 <= i < len(valid_xs)]
                valid_xs = [valid_xs[i] for i in valid_indices]

            for x in valid_xs:
                new_diffs = 0
                for a in row:
                    d = x - a
                    new_diffs |= (1 << d)
                
                # Check if new_diffs conflicts with used_bits? 
                # This is checked at the end of the row, but we can check early?
                # If new_diffs & used_bits: return False (for this x)
                # This is a good pruning!
                if new_diffs & used_bits:
                    continue

                if rec(row + [x], row_diff_bits | new_diffs, x + 1, 0):
                    return True
            
            return False
        
        return rec(row, row_diff_bits, min_start, 0)
    
    success = build_row(0)
    if success:
        return rows
    return None
