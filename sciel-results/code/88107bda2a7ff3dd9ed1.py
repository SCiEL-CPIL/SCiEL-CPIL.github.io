import time
import numpy as np

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
        start_time = time.time()
        deadline = start_time + 85.0  # Safety margin

        lower = 1000
        upper = 1000000
        window_size = 24
        max_val = upper + window_size
        
        # Precompute small-prime parts
        nvals = max_val - lower + 1
        sp_part = np.ones(nvals, dtype=np.int64)
        
        for p in _SMALL_PRIMES:
            power = p
            while power <= max_val:
                # First multiple of power >= lower
                first_mult = ((lower + power - 1) // power) * power
                if first_mult > max_val:
                    break
                # Indices in sp_part
                # np.arange(start, stop, step)
                idxs = np.arange(first_mult, max_val + 1, power) - lower
                sp_part[idxs] *= p
                power *= p
                if power > max_val:
                    break
        
        if time.time() > deadline:
            return lower

        # Sliding window
        best_n = lower
        best_count = float('inf')
        
        # Convert to list for faster indexing in Python loop?
        # numpy array indexing is slower than list for single elements.
        sp_list = sp_part.tolist()
        
        freq = {}
        current_distinct = 0
        
        # Init window
        for i in range(1, window_size + 1):
            val = sp_list[i]
            if val in freq:
                freq[val] += 1
            else:
                freq[val] = 1
                current_distinct += 1
                
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = lower
            
        sp_len = len(sp_list)
        n = lower + 1
        upper_check = upper
        deadline_local = deadline
        time_check_interval = 100000
        check_counter = 0
        
        while n <= upper_check:
            check_counter += 1
            if check_counter >= time_check_interval:
                check_counter = 0
                if time.time() > deadline_local:
                    break
            
            out_idx = n - lower
            in_idx = out_idx + 24
            
            val_out = sp_list[out_idx]
            c_out = freq[val_out] - 1
            if c_out == 0:
                del freq[val_out]
                current_distinct -= 1
            else:
                freq[val_out] = c_out
                
            val_in = sp_list[in_idx]
            if val_in in freq:
                freq[val_in] += 1
            else:
                freq[val_in] = 1
                current_distinct += 1
                
            if current_distinct < best_count:
                best_count = current_distinct
                best_n = n
                
            n += 1

        return best_n
