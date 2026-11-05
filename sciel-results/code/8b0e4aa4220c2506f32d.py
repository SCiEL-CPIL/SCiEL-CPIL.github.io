    def local_search(S):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Try to add any element
            added = False
            candidates = [z for z in range(1, N+1) if z not in S]
            random.shuffle(candidates)
            for z in candidates:
                if can_add(z, S):
                    S.add(z)
                    added = True
                    improved = True
                    break 
            
            if added:
                continue
            
            # Phase 2: Try 1-2 swap
            S_list = list(S)
            random.shuffle(S_list)
            for x in S_list:
                if time.time() > deadline:
                    break
                
                S_temp = S - {x}
                candidates = [z for z in range(1, N+1) if z not in S_temp]
                random.shuffle(candidates)
                
                # Try all candidates if small, else subset
                # 512 is small enough to try all in most cases
                # Let's just try all
                found_swap = False
                for y in candidates:
                    if can_add(y, S_temp):
                        S = S_temp | {y}
                        found_swap = True
                        improved = True
                        break
                
                if found_swap:
                    break # Restart while loop
                
        return S
