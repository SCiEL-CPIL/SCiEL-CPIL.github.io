import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    # Primes up to 127
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute factor vectors
    vecs = {}
    for n in range(2, 129):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
            if temp == 1:
                break
        vecs[n] = tuple(v)
    zero_vec = (0,) * num_primes

    # Beam Search
    # State: (seq_tuple, pref_list_tuple, diff_set)
    
    # Initial state: empty sequence
    beam = [((), (zero_vec,), frozenset())]
    beam_width = 100
    
    best_seq = ()
    
    for n in range(2, 129):
        if time.time() > deadline:
            break
            
        new_beam = []
        v_n = vecs[n]
        
        for state in beam:
            seq, pref_list, diff_set = state
            last_pref = pref_list[-1]
            
            # Option 1: Exclude n (carry over state)
            new_beam.append((seq, pref_list, diff_set))
            
            # Option 2: Include n
            new_pref = tuple(last_pref[i] + v_n[i] for i in range(num_primes))
            
            # Check collisions
            valid = True
            new_diffs = []
            
            for p in pref_list:
                diff = tuple(new_pref[i] - p[i] for i in range(num_primes))
                if diff in diff_set:
                    valid = False
                    break
                new_diffs.append(diff)
            
            if valid:
                new_pref_list = pref_list + (new_pref,)
                new_diff_set = diff_set.union(new_diffs)
                new_seq = seq + (n,)
                new_beam.append((new_seq, new_pref_list, new_diff_set))
        
        # Prune beam
        # Sort by length descending, then by last element ascending
        def score(state):
            seq, pl, ds = state
            l = len(seq)
            last_val = seq[-1] if seq else 0
            return (l, -last_val)
            
        new_beam.sort(key=score, reverse=True)
        beam = new_beam[:beam_width]
        
        # Update best
        if beam:
            best_len = len(beam[0][0])
            if best_len > len(best_seq):
                best_seq = beam[0][0]

    return list(best_seq)
