"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _compute_tails(values, x):
    """Compute all consecutive block products ending at x (i.e., x * suffix products)."""
    tails = [x]
    prod = x
    for a in reversed(values):
        prod *= a
        tails.append(prod)
    return tails

def _can_insert(values, products, x):
    """Check if x can be inserted into values (which is sorted) at its correct position."""
    # Find position where x would go
    lo, hi = 0, len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    pos = lo
    
    # x must not already be in values
    if pos < len(values) and values[pos] == x:
        return False
    
    # Compute all new products involving x
    new_products = set()
    # Block [x]
    new_products.add(x)
    # Blocks ending at x: [values[i]..values[pos-1], x]
    p = x
    for i in range(pos-1, -1, -1):
        p *= values[i]
        new_products.add(p)
    # Blocks starting at x: [x, values[pos]..values[j]]
    p = x
    for j in range(pos, len(values)):
        p *= values[j]
        new_products.add(p)
    # Blocks spanning x: [values[i]..values[pos-1], x, values[pos]..values[j]]
    if pos > 0 and pos < len(values):
        left = [1] * pos
        p = 1
        for i in range(pos-1, -1, -1):
            p *= values[i]
            left[i] = p
        right = [1] * (len(values) - pos)
        p = 1
        for j in range(pos, len(values)):
            p *= values[j]
            right[j - pos] = p
        for i in range(pos):
            for j in range(len(values) - pos):
                new_products.add(left[i] * x * right[j])
    
    # Check for conflicts with existing products
    if products.intersection(new_products):
        return False
    
    return True

def _recompute_products(values):
    """Recompute the set of all consecutive block products."""
    products = set()
    for i in range(len(values)):
        p = 1
        for j in range(i, len(values)):
            p *= values[j]
            products.add(p)
    return products

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s out of 90s limit
    
    # Phase 1: Greedy construction
    values, products = [], set()
    for x in range(2, 129):
        if time.time() > deadline:
            break
        tails = _compute_tails(values, x)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    
    base_len = len(values)
    
    # Phase 2: Try to insert excluded numbers
    excluded = [x for x in range(2, 129) if x not in set(values)]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in excluded[:]:
            if time.time() >= deadline:
                break
            if _can_insert(values, products, x):
                # Insert x into values
                lo, hi = 0, len(values)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if values[mid] < x:
                        lo = mid + 1
                    else:
                        hi = mid
                pos = lo
                values.insert(pos, x)
                excluded.remove(x)
                
                # Recompute products
                products = _recompute_products(values)
                improved = True
    
    # Phase 3: Beam search for further improvement
    # Use prime factor vectors for efficient product comparison
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    def get_vec(n):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        return tuple(v)
        
    vecs = {i: get_vec(i) for i in range(2, 129)}
    zero_vec = (0,) * num_primes

    # Beam search starting from current best
    beam_width = 100
    
    # State: (seq, prefix_vecs, prod_set)
    # seq: list of ints
    # prefix_vecs: list of tuples (V0, V1, ..., Vk)
    # prod_set: set of tuples representing all block products
    
    def compute_state(seq):
        """Compute prefix vectors and product set for a sequence."""
        prefix_vecs = [zero_vec]
        prod_set = set()
        for x in seq:
            last = prefix_vecs[-1]
            v_new = tuple(last[i] + vecs[x][i] for i in range(num_primes))
            prefix_vecs.append(v_new)
            # Add all suffix products ending at x
            for i in range(len(prefix_vecs) - 1):
                diff = tuple(v_new[j] - prefix_vecs[i][j] for j in range(num_primes))
                prod_set.add(diff)
        return prefix_vecs, prod_set
    
    # Initialize beam with current best
    initial_pvecs, initial_pset = compute_state(values)
    beam = [(values, initial_pvecs, initial_pset)]
    
    # Try to improve via beam search
    for n in range(2, 129):
        if time.time() > deadline:
            break
            
        new_beam = []
        v_n = vecs[n]
        
        for seq, pvecs, pset in beam:
            # Option 1: Exclude n
            new_beam.append((seq, pvecs, pset))
            
            # Option 2: Include n (if not already in seq)
            if n not in seq:
                last_pvec = pvecs[-1]
                v_new = tuple(last_pvec[i] + v_n[i] for i in range(num_primes))
                
                valid = True
                new_prod_ids = []
                
                # Check all suffixes ending at n
                for i in range(len(pvecs)):
                    diff = tuple(v_new[j] - pvecs[i][j] for j in range(num_primes))
                    if diff in pset:
                        valid = False
                        break
                    new_prod_ids.append(diff)
                
                if valid:
                    new_seq = seq + [n]
                    new_pvecs = pvecs + [v_new]
                    new_pset = pset.union(new_prod_ids)
                    new_beam.append((new_seq, new_pvecs, new_pset))
        
        # Select top beam_width states from new_beam
        # Sort by length of seq (descending)
        new_beam.sort(key=lambda x: len(x[0]), reverse=True)
        beam = new_beam[:beam_width]
        
        # Early termination
        if beam and len(beam[0][0]) == 127:
            return beam[0][0]
    
    # Return best found
    if beam:
        return beam[0][0]
    else:
        return values
# EVOLVE-BLOCK-END
