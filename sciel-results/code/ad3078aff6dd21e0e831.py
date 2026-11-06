    def local_search(S_set, S_bits):
        while True:
            # Phase 1: Add
            added = False
            for x_1_based in range(1, N + 1):
                if x_1_based not in S_set:
                    x = x_1_based - 1
                    if is_valid_add(x, S_bits):
                        S_bits |= (1 << x)
                        S_set.add(x_1_based)
                        added = True
                        break
            if not added:
                break
            
            # If we added, continue loop to add more
            continue
            
        # Wait, the above while loop structure is slightly off.
        # Let's restructure:
        
        # 1. Add all possible
        while True:
            added = False
            for x_1_based in range(1, N + 1):
                if x_1_based not in S_set:
                    x = x_1_based - 1
                    if is_valid_add(x, S_bits):
                        S_bits |= (1 << x)
                        S_set.add(x_1_based)
                        added = True
                        break
            if not added:
                break

        # 2. Swap until no more swaps
        improved = True
        while improved:
            improved = False
            S_list = list(S_set)
            for u_1_based in S_list:
                u = u_1_based - 1
                S_bits_temp = S_bits & ~(1 << u)
                S_set_temp = S_set.copy()
                S_set_temp.remove(u_1_based)
                
                for v_1_based in range(1, N + 1):
                    if v_1_based not in S_set_temp:
                        v = v_1_based - 1
                        if is_valid_add(v, S_bits_temp):
                            S_bits = S_bits_temp | (1 << v)
                            S_set = S_set_temp.copy()
                            S_set.add(v_1_based)
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break
        
        return S_set, S_bits
