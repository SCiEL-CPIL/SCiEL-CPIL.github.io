import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute factor vectors as tuples
    def get_vec(n):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        return tuple(v)
        
    vecs = {i: get_vec(i) for i in range(2, 129)}
    
    best_seq = []
    best_len = 0
    
    # To speed up vector addition, we can precompute something?
    # No, just use tuples.
    
    def dfs(idx, current_seq, current_products):
        nonlocal best_seq, best_len, deadline
        
        if time.time() > deadline:
            return
            
        # Update best
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq[:]
            
        # Pruning
        if idx > 128:
            return
        # If even taking all remaining numbers doesn't beat best, prune
        if len(current_seq) + (128 - idx + 1) <= best_len:
            return
            
        # Try Include first (to maximize length quickly)
        # Check validity of adding idx
        valid = True
        new_products = set()
        
        # We need to check all suffixes of (current_seq + [idx])
        # Suffixes are: [idx], [last, idx], [second_last, idx], ...
        
        # Optimization: 
        # The new products are sums of vecs[idx] and suffix sums of current_seq.
        
        # Let's compute suffix sums of current_seq vectors.
        # Since current_seq is short, we can just iterate.
        
        # To avoid creating a new list of vectors for current_seq every time,
        # we can pass the list of vectors as well.
        
        # Let's refactor dfs to take `seq_vecs`
        
        # Wait, I'm already in the code. I'll just compute it.
        
        # Actually, let's pass `seq_vecs` to dfs to avoid recomputation.
        
        # I'll rewrite the dfs function signature in the final code.
        
        # For now, let's assume we have `seq_vecs`.
        
        # Since I'm thinking in the draft, I'll structure the final code properly.
        
        pass

    # Final Structure:
    def dfs(idx, current_seq, seq_vecs, current_products):
        nonlocal best_seq, best_len, deadline
        
        if time.time() > deadline:
            return
            
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq[:]
            
        if idx > 128:
            return
        if len(current_seq) + (128 - idx + 1) <= best_len:
            return
            
        # Try Include
        new_products = set()
        valid = True
        # Calculate new suffix products
        # Suffix sum of current_seq + idx
        # Let S_k be sum of last k vectors in seq_vecs
        # New suffix products are vecs[idx] + S_k for k=0..len(seq_vecs)
        # S_0 = 0
        
        # We can compute this by iterating backwards.
        
        # To be efficient, we can precompute suffix sums of seq_vecs?
        # No, seq_vecs changes.
        
        # Just iterate.
        
        # Note: current_products contains products of blocks in current_seq.
        # We need to check if any NEW product is in current_products.
        
        # Let's use a running sum.
        # sum_vec = 0
        # for v in reversed(seq_vecs):
        #    sum_vec += v
        #    new_prod = sum_vec + vecs[idx]
        #    check...
        # Also check vecs[idx] itself (k=0 case).
        
        v_idx = vecs[idx]
        
        # Check singleton
        if v_idx in current_products:
            valid = False
        else:
            new_products.add(v_idx)
            
            # Check longer suffixes
            sum_vec = [0] * num_primes
            for i in range(len(seq_vecs) - 1, -1, -1):
                v_i = seq_vecs[i]
                # Add v_i to sum_vec
                for j in range(num_primes):
                    sum_vec[j] += v_i[j]
                
                # New product vector is sum_vec + v_idx
                # We can create a tuple
                # To avoid creating a new list every time, we can compare directly?
                # No, we need to hash it.
                
                # Optimization: 
                # If we store current_products as a set of tuples, we need to create a tuple.
                
                new_v = [sum_vec[j] + v_idx[j] for j in range(num_primes)]
                new_v_tup = tuple(new_v)
                
                if new_v_tup in current_products or new_v_tup in new_products:
                    valid = False
                    break
                new_products.add(new_v_tup)
                
        if valid:
            # Recurse
            dfs(idx + 1, current_seq + [idx], seq_vecs + [v_idx], current_products.union(new_products))
            
        # Try Exclude
        dfs(idx + 1, current_seq, seq_vecs, current_products)
        
    dfs(2, [], [], set())
    
    return best_seq
