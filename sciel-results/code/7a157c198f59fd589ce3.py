import time
import bisect

def construct():
    def get_block_products(seq):
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods

    def can_insert(seq, prods, y):
        # Find position
        pos = bisect.bisect_left(seq, y)
        
        # Calculate new products involving y
        # Blocks ending at y: y, seq[pos-1]*y, ...
        # Blocks starting at y: y, y*seq[pos+1], ...
        # Note: y itself is counted in both, but it's a set so it's fine.
        
        new_prods = set()
        
        # Suffixes ending at y (including y)
        p = 1
        for i in range(pos, -1, -1):
            p *= seq[i] if i < len(seq) else 1 # Wait, if pos == len(seq), i starts at len(seq) which is out of bounds for seq[i] in the multiplication loop if we aren't careful.
            # Let's rewrite carefully.
        
        # Correct logic for new products:
        # If inserted at pos:
        # New blocks are those that include the new element at index pos.
        # A block is defined by start i and end j.
        # New blocks have i <= pos <= j.
        # And the block did not exist before (which is true if it includes pos, since pos was a gap).
        
        # Products:
        # 1. y alone
        new_prods.add(y)
        
        # 2. y * (product of suffix of seq[:pos] ending at pos-1)
        p = y
        for i in range(pos - 1, -1, -1):
            p *= seq[i]
            new_prods.add(p)
            
        # 3. y * (product of prefix of seq[pos:] starting at pos)
        p = y
        for j in range(pos, len(seq)):
            p *= seq[j]
            new_prods.add(p)
            
        # 4. (product of suffix of seq[:pos]) * y * (product of prefix of seq[pos:])
        # This is covered by combining 2 and 3?
        # No. A block like seq[i] * ... * seq[pos-1] * y * seq[pos] * ... * seq[j]
        # is (suffix_prod) * y * (prefix_prod).
        # We need to check all combinations.
        
        # Let's just generate all products involving y in the new sequence.
        # New sequence is seq[:pos] + [y] + seq[pos:]
        # We can just compute products for the small segment around y?
        # No, products can span the whole sequence.
        
        # Efficient way:
        # Let L = seq[:pos], R = seq[pos:]
        # New products are:
        # - Products entirely in L (already in prods)
        # - Products entirely in R (already in prods)
        # - Products including y.
        
        # Products including y:
        # 1. y
        # 2. y * (suffix of L)
        # 3. y * (prefix of R)
        # 4. (suffix of L) * y * (prefix of R)
        
        # Let's compute these sets.
        
        # Suffixes of L (including empty product 1)
        suffixes_L = [1]
        p = 1
        for i in range(pos - 1, -1, -1):
            p *= seq[i]
            suffixes_L.append(p)
            
        # Prefixes of R (including empty product 1)
        prefixes_R = [1]
        p = 1
        for j in range(pos, len(seq)):
            p *= seq[j]
            prefixes_R.append(p)
            
        # New products:
        # s * y * r for s in suffixes_L, r in prefixes_R
        for s in suffixes_L:
            for r in prefixes_R:
                prod = s * y * r
                new_prods.add(prod)
                
        # Check intersection
        if new_prods & prods:
            return False
        return True

    def try_insert_two(seq, prods, unused):
        # Try to insert two elements a, b from unused into seq
        # Returns new_seq if successful, else None
        
        # To speed up, we can filter candidates a that can be inserted alone
        # But we need to be careful: a might not be insertable alone into seq,
        # but a and b together might be? 
        # No, if a is inserted, the sequence changes. 
        # However, if a cannot be inserted into seq because it conflicts with an existing product,
        # adding b won't remove that conflict.
        # So a MUST be insertable into seq?
        # Wait. If we insert a, we check against `prods` (products of seq).
        # If we then insert b, we check against `prods` U {products of a}.
        # So yes, a must be valid to insert into seq first.
        
        candidates_a = []
        for a in unused:
            if can_insert(seq, prods, a):
                candidates_a.append(a)
                
        if not candidates_a:
            return None
            
        # For each a, try to insert b
        for a in candidates_a:
            # Compute new prods with a
            # We need to add products of a to prods
            # We can reuse the logic from can_insert to get new_prods
            # Or just recompute?
            # Let's recompute new_prods for a to be safe, or modify prods temporarily?
            # Modifying prods is risky if we fail.
            # Let's just create a new set.
            
            # To avoid recomputing can_insert logic, let's make a helper
            # that returns the new products set if valid.
            
            def get_new_prods_if_valid(seq, prods, y):
                pos = bisect.bisect_left(seq, y)
                suffixes_L = [1]
                p = 1
                for i in range(pos - 1, -1, -1):
                    p *= seq[i]
                    suffixes_L.append(p)
                    
                prefixes_R = [1]
                p = 1
                for j in range(pos, len(seq)):
                    p *= seq[j]
                    prefixes_R.append(p)
                
                new_prods = set()
                for s in suffixes_L:
                    for r in prefixes_R:
                        new_prods.add(s * y * r)
                        
                if new_prods & prods:
                    return None
                return new_prods

            new_prods_a = get_new_prods_if_valid(seq, prods, a)
            if new_prods_a is None:
                continue
                
            # Now try to insert b into seq + [a]
            # We need the sequence with a inserted to find positions for b?
            # No, can_insert(seq_with_a, prods_with_a, b)
            # We can simulate seq_with_a without actually creating the list?
            # Or just create it. L is small.
            
            pos_a = bisect.bisect_left(seq, a)
            seq_with_a = seq[:pos_a] + [a] + seq[pos_a:]
            prods_with_a = prods.union(new_prods_a)
            
            for b in unused:
                if b == a:
                    continue
                if can_insert(seq_with_a, prods_with_a, b):
                    # Success
                    pos_b = bisect.bisect_left(seq_with_a, b)
                    new_seq = seq_with_a[:pos_b] + [b] + seq_with_a[pos_b:]
                    return new_seq
                    
        return None

    # 1. Greedy Initialization
    values = []
    products = set()
    for x in range(2, 129):
        tails = [x]
        prod = x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    
    best_seq = values
    best_len = len(values)
    current_seq = values
    current_prods = get_block_products(current_seq)
    unused = set(range(2, 129)) - set(current_seq)
    
    start_time = time.time()
    deadline = start_time + 50
    
    while time.time() < deadline:
        improved = False
        # Try removing each element
        for i in range(len(current_seq)):
            if time.time() > deadline:
                break
                
            x = current_seq[i]
            seq_no_x = current_seq[:i] + current_seq[i+1:]
            
            # If we remove x, the length decreases. We want to add 2.
            # So we look for a, b in unused (which now includes x? No, x is removed from seq, so x is in unused?
            # Wait, unused was defined as set(2..128) - set(current_seq).
            # If we remove x, x becomes available.
            # So candidates for a, b are (unused U {x})
            
            candidates = unused.union({x})
            
            new_seq = try_insert_two(seq_no_x, current_prods, candidates)
            
            if new_seq is not None:
                len_new = len(new_seq)
                if len_new > best_len:
                    best_seq = new_seq
                    best_len = len_new
                    # Update current state to the best one found so far to continue from there
                    current_seq = new_seq
                    current_prods = get_block_products(current_seq)
                    unused = set(range(2, 129)) - set(current_seq)
                    improved = True
                    # Break inner loop? Or continue?
                    # Continuing might find an even better one?
                    # But we updated current_seq, so the loop index i is now invalid/stale.
                    # Break to restart the scan with the new sequence.
                    break
        if improved:
            continue
        # If no improvement, we might be stuck.
        # We can try random restarts or just stop if we've done many scans?
        # Given the time limit, just stopping or continuing to scan is fine.
        # Actually, if we didn't improve, we can try to shuffle the order of removal?
        # Or just exit if we've scanned several times without improvement?
        # For now, let's just let the while loop run until deadline.
        # To avoid infinite loops of "no improvement", we can break if we've done, say, 10 scans without improvement.
        
    return best_seq
