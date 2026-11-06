<<<<<<< SEARCH
    def anneal(S_mask, time_budget):
        S_mask = S_mask
        T0 = 1.2
        T_end = 0.015
        cooling = 0.9995
        T = T0
        
        while time.time() - start_time < time_budget:
            if T < T_end:
                break
                
            # Pick random v in S
            verts = get_bits(S_mask)
            if not verts:
                break
            v = random.choice(verts)
            
            S_new = S_mask & ~(1 << v)
            
            # Find valid candidates
            cands_mask = adj[v] & ~S_new
            valid_mask = 0
            c = cands_mask
            while c:
                lsb = c & -c
                u = lsb.bit_length() - 1
                if (adj[u] & S_new) == 0:
                    valid_mask |= lsb
                c ^= lsb
            
            # For anneal, use a fast greedy for add_mask to keep it fast
            # Or use mis_small if valid_mask is very small
            if valid_mask.bit_count() < 20:
                add_mask = mis_small(valid_mask)
            else:
                # Greedy
                add_mask = 0
                rem = valid_mask
                while rem:
                    # Pick min degree in rem
                    min_v = -1
                    min_deg = float('inf')
                    r = rem
                    while r:
                        lsb = r & -r
                        u = lsb.bit_length() - 1
                        deg = (adj[u] & rem).bit_count()
                        if deg < min_deg:
                            min_deg = deg
                            min_v = u
                        r ^= lsb
                    if min_v == -1:
                        break
                    add_mask |= (1 << min_v)
                    rem &= ~(adj[min_v] | (1 << min_v))
            
            S_cand = S_new | add_mask
            d = S_cand.bit_count() - S_mask.bit_count()
            
            if d >= 0 or random.random() < math.exp(d / T):
                S_mask = S_cand
                
            T *= cooling
            
        return S_mask
=======
    def anneal(S_mask, time_budget):
        S_mask = S_mask
        T0 = 1.5
        T_end = 0.01
        cooling = 0.999
        T = T0
        no_improve_count = 0
        best_size = S_mask.bit_count()
        
        while time.time() - start_time < time_budget:
            if T < T_end:
                break
                
            # Pick random v in S
            verts = get_bits(S_mask)
            if not verts:
                break
            v = random.choice(verts)
            
            S_new = S_mask & ~(1 << v)
            
            # Find valid candidates
            cands_mask = adj[v] & ~S_new
            valid_mask = 0
            c = cands_mask
            while c:
                lsb = c & -c
                u = lsb.bit_length() - 1
                if (adj[u] & S_new) == 0:
                    valid_mask |= lsb
                c ^= lsb
            
            # For anneal, use a fast greedy for add_mask to keep it fast
            if valid_mask.bit_count() < 15:
                add_mask = mis_small(valid_mask)
            else:
                # Greedy
                add_mask = 0
                rem = valid_mask
                while rem:
                    min_v = -1
                    min_deg = float('inf')
                    r = rem
                    while r:
                        lsb = r & -r
                        u = lsb.bit_length() - 1
                        deg = (adj[u] & rem).bit_count()
                        if deg < min_deg:
                            min_deg = deg
                            min_v = u
                        r ^= lsb
                    if min_v == -1:
                        break
                    add_mask |= (1 << min_v)
                    rem &= ~(adj[min_v] | (1 << min_v))
            
            S_cand = S_new | add_mask
            d = S_cand.bit_count() - S_mask.bit_count()
            
            accept = False
            if d >= 0:
                accept = True
            else:
                # d is -1 or -2 typically
                if d == -1:
                    accept = random.random() < math.exp(1.0 / T)
                elif d == -2:
                    accept = random.random() < math.exp(2.0 / T)
            
            if accept:
                S_mask = S_cand
                if S_mask.bit_count() > best_size:
                    best_size = S_mask.bit_count()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            # Reset temperature if stuck
            if no_improve_count > 200:
                T = T0
                no_improve_count = 0
            
            T *= cooling
            
        return S_mask
>>>>>>> REPLACE

