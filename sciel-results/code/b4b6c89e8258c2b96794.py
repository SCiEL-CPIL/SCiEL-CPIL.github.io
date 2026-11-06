import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    primes = [p for p in range(2, 129) if is_prime(p)]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    exp_vectors = {}
    for n in range(2, 129):
        vec = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            if p * p > temp:
                break
            while temp % p == 0:
                vec[i] += 1
                temp //= p
        if temp > 1:
            vec[prime_to_idx[temp]] += 1
        exp_vectors[n] = tuple(vec)
        
    # Beam Search
    # Key: (P_all, P_end)
    # Value: (length, seq_tuple)
    beam = { (frozenset(), frozenset()): (0, ()) }
    
    for x in range(2, 129):
        if time.time() > deadline:
            break
            
        e_x = exp_vectors[x]
        new_beam = {}
        
        for (p_all, p_end), (length, seq) in beam.items():
            # Option 1: Skip x
            key_skip = (p_all, p_end)
            if key_skip not in new_beam or length > new_beam[key_skip][0]:
                new_beam[key_skip] = (length, seq)
                
            # Option 2: Include x
            if length == 0:
                new_p_end = frozenset([e_x])
                new_p_all = frozenset([e_x])
                new_seq = (x,)
                key = (new_p_all, new_p_end)
                if key not in new_beam or 1 > new_beam[key][0]:
                    new_beam[key] = (1, new_seq)
            else:
                new_p_end_set = set()
                for vec in p_end:
                    new_vec = tuple(a + b for a, b in zip(vec, e_x))
                    new_p_end_set.add(new_vec)
                new_p_end_set.add(e_x)
                
                valid = True
                for vec in new_p_end_set:
                    if vec in p_all:
                        valid = False
                        break
                        
                if valid:
                    new_p_end = frozenset(new_p_end_set)
                    new_p_all = p_all.union(new_p_end)
                    new_seq = seq + (x,)
                    key = (new_p_all, new_p_end)
                    if key not in new_beam or length + 1 > new_beam[key][0]:
                        new_beam[key] = (length + 1, new_seq)
        
        # Prune beam
        beam_size = 1000
        if len(new_beam) > beam_size:
            # Sort by length desc, then len(p_all) asc
            sorted_items = sorted(new_beam.items(), key=lambda item: (-item[1][0], len(item[0][0])))
            new_beam = dict(sorted_items[:beam_size])
            
        beam = new_beam
        
    # Find best
    best_len = 0
    best_seq = ()
    for (p_all, p_end), (length, seq) in beam.items():
        if length > best_len:
            best_len = length
            best_seq = seq
            
    return list(best_seq)
