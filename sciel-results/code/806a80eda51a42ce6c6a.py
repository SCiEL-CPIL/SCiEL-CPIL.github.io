import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    def block_products(seq):
        """Compute set of all nonempty consecutive block products."""
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods

    # Greedy initialization: add smallest numbers first
    seq = []
    prods = set()
    for x in range(2, 129):
        # Check if x can be appended to seq
        new_prods = set(prods)
        p = x
        if p in new_prods:
            continue
        new_prods.add(p)
        valid = True
        for a in reversed(seq):
            p *= a
            if p in new_prods:
                valid = False
                break
            new_prods.add(p)
        if valid:
            seq.append(x)
            prods = new_prods

    best_len = len(seq)
    missing = [x for x in range(2, 129) if x not in set(seq)]

    def try_insert_at(seq, prods, pos, x):
        """Try inserting x at position pos. Return (success, new_prods)."""
        new_prods = set(prods)
        n = len(seq)
        # Precompute left products: left[i] = prod(seq[i..pos-1]) for i in [0, pos]
        # Precompute right products: right[j] = prod(seq[pos..j]) for j in [pos-1, n-1]
        # For i in [0, pos], j in [pos-1, n-1]:
        #   block_prod = left[i] * x * right[j]
        
        # Compute left products
        left = [1] * (pos + 1)
        lp = 1
        for i in range(pos, -1, -1):
            left[i] = lp
            if i > 0:
                lp *= seq[i - 1]
        
        # Compute right products
        right = [1]  # j = pos-1 (empty right)
        rp = 1
        for j in range(pos, n):
            rp *= seq[j]
            right.append(rp)
        
        for i in range(pos + 1):
            for rp_val in right:
                bp = left[i] * x * rp_val
                if bp in new_prods:
                    return False, None
                new_prods.add(bp)
        
        return True, new_prods

    # Local search: iteratively try to insert missing numbers
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        improved = False
        for x in missing[:]:  # Copy to allow removal during iteration
            if time.time() > deadline:
                break
            found_pos = -1
            new_prods = None
            for pos in range(len(seq) + 1):
                ok, np = try_insert_at(seq, prods, pos, x)
                if ok:
                    found_pos = pos
                    new_prods = np
                    break
            
            if found_pos >= 0:
                seq = seq[:found_pos] + [x] + seq[found_pos:]
                prods = new_prods
                missing.remove(x)
                best_len = max(best_len, len(seq))
                improved = True
        
        if not improved:
            break

    return seq
