import time
import numpy as np
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin for 90s kill

    # Primes up to 127
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute factor vectors as numpy arrays
    def get_vec(n):
        v = np.zeros(num_primes, dtype=np.int8)
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
            if temp == 1:
                break
        return v
        
    vecs = {i: get_vec(i) for i in range(2, 129)}
    zero_vec = np.zeros(num_primes, dtype=np.int8)

    # Beam Search with improved state representation
    # State: (length, seq, prefix_array, diff_set)
    beam = [(0, [], np.array([[0]*num_primes], dtype=np.int8), frozenset())]
    
    for step, n in enumerate(range(2, 129)):
        if time.time() > deadline:
            break
            
        # Dynamic beam width
        if step < 30:
            current_beam_width = 300
        elif step < 60:
            current_beam_width = 200
        elif step < 90:
            current_beam_width = 120
        else:
            current_beam_width = 60
        
        new_beam = []
        v_n = vecs[n]
        
        for state in beam:
            length, seq, prefix_arr, diff_set = state
            last_pref = prefix_arr[-1]
            
            # Option 1: Exclude n
            new_beam.append((length, seq, prefix_arr, diff_set))
            
            # Option 2: Include n
            new_pref = last_pref + v_n
            
            # Check validity
            valid = True
            new_diffs = []
            diffs = new_pref - prefix_arr
            for i in range(len(prefix_arr)):
                diff_tup = tuple(diffs[i])
                if diff_tup in diff_set:
                    valid = False
                    break
                new_diffs.append(diff_tup)
            
            if valid:
                new_prefix_arr = np.vstack([prefix_arr, new_pref.reshape(1, -1)])
                new_diff_set = diff_set.union(new_diffs)
                new_seq = seq + [n]
                new_length = length + 1
                new_beam.append((new_length, new_seq, new_prefix_arr, new_diff_set))
        
        # Score: length, then forward potential
        def calc_score(state):
            l, seq, pl, ds = state
            potential = 0.0
            last_p = pl[-1]
            for i, m in enumerate(range(n+1, min(n+6, 129))):
                weight = np.exp(-0.4 * i)
                v_m = vecs[m]
                new_p = last_p + v_m
                ok = True
                diffs = new_p - pl
                for diff in diffs:
                    if tuple(diff) in ds:
                        ok = False
                        break
                if ok:
                    potential += weight
            return (l, potential)

        new_beam.sort(key=calc_score, reverse=True)
        beam = new_beam[:current_beam_width]
        
        if beam and beam[0][0] == 127:
            return beam[0][1]

    # Final: return best from beam
    if beam:
        return beam[0][1]
    return []
