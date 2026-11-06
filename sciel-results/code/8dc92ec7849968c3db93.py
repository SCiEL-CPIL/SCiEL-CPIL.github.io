import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Primes up to 127
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute factor vectors as integers (encode each exponent in 4 bits, 31*4=124 bits)
    def get_vec_int(n):
        result = 0
        temp = n
        for i, p in enumerate(primes):
            exp = 0
            while temp % p == 0:
                exp += 1
                temp //= p
            result |= (exp << (i * 4))
        return result
        
    vecs = {i: get_vec_int(i) for i in range(2, 129)}
    
    # Beam Search with integer-encoded vectors
    # State: (seq, prefix_vecs, prod_set)
    # prefix_vecs[0] = 0, prefix_vecs[k] = vec(a1) + ... + vec(ak)
    
    beam = [([], [0], set())]
    beam_width = 50
    
    best_seq = []
    
    for n in range(2, 129):
        if time.time() > deadline:
            break
            
        new_beam = []
        v_n = vecs[n]
        
        for seq, pvecs, pset in beam:
            # Option 1: Exclude n
            new_beam.append((seq, pvecs, pset))
            
            # Option 2: Include n
            last_pvec = pvecs[-1]
            v_new = last_pvec + v_n
            
            valid = True
            new_prods = []
            
            # Check all suffixes ending at n
            for i in range(len(pvecs)):
                diff = v_new - pvecs[i]
                if diff in pset:
                    valid = False
                    break
                new_prods.append(diff)
            
            if valid:
                new_seq = seq + [n]
                new_pvecs = pvecs + [v_new]
                new_pset = pset.union(new_prods)
                new_beam.append((new_seq, new_pvecs, new_pset))
        
        # Select top beam_width states
        new_beam.sort(key=lambda x: len(x[0]), reverse=True)
        beam = new_beam[:beam_width]
        
        # Track best
        if beam and len(beam[0][0]) > len(best_seq):
            best_seq = beam[0][0]
        
        # Early termination
        if beam and len(beam[0][0]) >= 120:
            # We're close to maximum, can stop
            if time.time() > deadline - 5:
                break

    # Return best found
    if beam:
        return beam[0][0]
    else:
        return best_seq
