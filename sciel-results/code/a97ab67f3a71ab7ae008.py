            for x in range(v):
                if x in current_block_set:
                    continue
                
                gain = 0
                for sub in t_minus_1_subsets:
                    if t == 3:
                        a, b = sub
                        if x < a:
                            tset_tuple = (x, a, b)
                        elif x < b:
                            tset_tuple = (a, x, b)
                        else:
                            tset_tuple = (a, b, x)
                    else:
                        # Fallback for other t
                        tset_tuple = tuple(sorted(sub + (x,)))
                    
                    # Check uncovered
                    try:
                        bit_idx = t_set_to_bit[tset_tuple]
                    except KeyError:
                        continue # Should not happen if x in 0..v-1 and sub from B
                    
                    word_idx = bit_idx // 64
                    bit_in_word = bit_idx % 64
                    if (uncovered[word_idx] >> bit_in_word) & 1:
                        gain += 1
                            
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
