import time
import random

def construct():
    start_time = time.time()
    # 60s limit, keep 5s margin
    deadline = start_time + 55.0 

    def get_products(seq):
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods

    def is_valid(seq):
        n = len(seq)
        if n == 0:
            return True
        prods = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    def try_insert_fast(seq, val, existing_prods):
        """
        Try to insert val into seq at its sorted position.
        existing_prods is the set of all products of consecutive blocks in seq.
        Returns (new_seq, new_prods) if valid, None otherwise.
        """
        n = len(seq)
        pos = 0
        while pos < n and seq[pos] < val:
            pos += 1
        if pos < n and seq[pos] == val:
            return None
        
        # Check if inserting val creates any duplicate products
        # New products are those that include val.
        
        # 1. val itself
        if val in existing_prods:
            return None
        
        # Precompute prefix products from pos-1 going left
        # left_prods[k] = product of seq[pos-1-k .. pos-1]
        left_prods = []
        p = 1
        for i in range(pos - 1, -1, -1):
            p *= seq[i]
            left_prods.append(p)
            
        # Precompute suffix products from pos going right
        # right_prods[k] = product of seq[pos .. pos+k]
        right_prods = []
        p = 1
        for j in range(pos, n):
            right_prods.append(p)
            p *= seq[j]
            
        new_prods_list = [val]
        
        # 2. Left extensions: left_prods[k] * val
        for lp in left_prods:
            prod = lp * val
            if prod in existing_prods:
                return None
            new_prods_list.append(prod)
            
        # 3. Right extensions: val * right_prods[k]
        for rp in right_prods:
            prod = val * rp
            if prod in existing_prods:
                return None
            new_prods_list.append(prod)
            
        # 4. Both sides: left_prods[i] * val * right_prods[j]
        for lp in left_prods:
            for rp in right_prods:
                prod = lp * val * rp
                if prod in existing_prods:
                    return None
                new_prods_list.append(prod)
                
        # Check for duplicates among new products
        # Since all new products include val and all old products do not,
        # we only need to check if any two new products are equal.
        # Note: val is unique. lp*val is unique for each lp because seq is strictly increasing 
        # and positive integers > 1. Similarly for others.
        # However, it's possible that lp1 * val * rp1 == lp2 * val * rp2?
        # lp1 * rp1 == lp2 * rp2. 
        # Since the sequence is strictly increasing, the products of distinct blocks are 
        # generally unique unless there's a specific multiplicative coincidence.
        # But we already checked against existing_prods.
        # We must check if new_prods_list has duplicates.
        
        if len(set(new_prods_list)) != len(new_prods_list):
            return None
            
        new_prods = existing_prods | set(new_prods_list)
        new_seq = seq[:pos] + [val] + seq[pos:]
        return (new_seq, new_prods)

    def greedy_construct(order=None):
        if order is None:
            order = list(range(2, 129))
            
        values = []
        prods = set()
        for x in order:
            result = try_insert_fast(values, x, prods)
            if result is not None:
                values, prods = result
        return values, prods

    # 1. Initial Greedy Construction (ascending)
    best_seq, best_prods = greedy_construct()
    best_len = len(best_seq)
    
    all_values = set(range(2, 129))
    current_seq = best_seq
    current_prods = best_prods
    current_excluded = all_values - set(current_seq)
    
    max_kicks = 50000
    kick_count = 0
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline and kick_count < max_kicks:
        kick_count += 1
        
        # Phase 1: Try to insert any excluded element
        improved = False
        excl_list = list(current_excluded)
        random.shuffle(excl_list)
        
        for e in excl_list:
            if time.time() > deadline:
                break
            result = try_insert_fast(current_seq, e, current_prods)
            if result is not None:
                current_seq, current_prods = result
                current_excluded.remove(e)
                improved = True
                no_improve_count = 0
                break
        
        if improved:
            if len(current_seq) > best_len:
                best_seq = current_seq
                best_prods = current_prods
                best_len = len(best_seq)
            continue
            
        # If no improvement, increment no_improve counter
        no_improve_count += 1
        
        # Phase 2: If stuck, perform a "kick"
        if no_improve_count > max_no_improve:
            no_improve_count = 0
            # Remove k random elements
            k = random.randint(2, 6)
            if len(current_seq) <= k:
                k = max(1, len(current_seq) // 2)
                
            if k > 0:
                to_remove = random.sample(current_seq, k)
                to_remove_set = set(to_remove)
                
                kicked_seq = [x for x in current_seq if x not in to_remove_set]
                kicked_prods = get_products(kicked_seq)
                
                # Greedily refill with random order
                candidates = current_excluded | to_remove_set
                cand_list = list(candidates)
                random.shuffle(cand_list)
                
                new_seq = kicked_seq
                new_prods = kicked_prods
                new_excluded = all_values - set(new_seq)
                
                for e in cand_list:
                    if time.time() > deadline:
                        break
                    if e in new_excluded:
                        result = try_insert_fast(new_seq, e, new_prods)
                        if result is not None:
                            new_seq, new_prods = result
                            new_excluded.remove(e)
                            
                # Accept if length is not significantly reduced
                # Usually we want to accept if it's the same or better, 
                # or if we are hoping to find a path to a longer sequence.
                # Let's accept if len >= len(current_seq) - 2
                if len(new_seq) >= len(current_seq) - 2:
                    current_seq = new_seq
                    current_prods = new_prods
                    current_excluded = new_excluded
                    
                    if len(current_seq) > best_len:
                        best_seq = current_seq
                        best_prods = current_prods
                        best_len = len(best_seq)
                        
        # Phase 3: Occasionally try a fresh random greedy restart if time permits
        remaining_time = deadline - time.time()
        if remaining_time > 10 and random.random() < 0.05:
            vals = list(range(2, 129))
            random.shuffle(vals)
            seq, prods = greedy_construct(vals)
            if len(seq) > best_len:
                best_seq = seq
                best_prods = prods
                best_len = len(seq)
                current_seq = seq
                current_prods = prods
                current_excluded = all_values - set(seq)
                no_improve_count = 0

    return best_seq
