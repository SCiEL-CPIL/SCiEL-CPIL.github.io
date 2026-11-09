def construct():
        # Precompute inverses
        inv = [0] * 11
        for i in range(1, 11):
            inv[i] = pow(i, -1, 101)
            
        # best_len[r] = minimum popcount found so far for residue r
        # best_mask[r] = mask corresponding to that popcount
        best_len = [100] * 101
        best_mask = [0] * 101
        
        # Iterate all non-empty subsets
        for mask in range(1, 1 << 10):
            # Calculate sum and popcount
            s = 0
            k = 0
            # Unroll or simple loop for speed? 1024 * 10 is tiny.
            for i in range(1, 11):
                if mask & (1 << (i - 1)):
                    s += inv[i]
                    k += 1
            s %= 101
            
            if k < best_len[s]:
                best_len[s] = k
                best_mask[s] = mask
        
        # Construct result
        result = []
        for r in range(101):
            mask = best_mask[r]
            terms = [i for i in range(1, 11) if mask & (1 << (i - 1))]
            result.append(terms)
            
        return result
