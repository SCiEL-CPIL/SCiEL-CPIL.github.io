        # Phase 2: Try 1-2 swaps (remove 1, add k>=2)
        # To speed up, we can just check if any swap improves the set.
        # We iterate through y in S.
        for y in list(S):
            if time.time() > deadline - 5: # Safety check inside phase 2
                break
            S_temp = S - {y}
            gain = 0
            new_elements = []
            # Check all x not in S_temp
            for x in range(1, n + 1):
                if x not in S_temp and is_valid(x, S_temp):
                    gain += 1
                    new_elements.append(x)
            
            # We removed 1 element (y). We add 'gain' elements.
            # Net change = gain - 1.
            # We want net change > 0, so gain >= 2.
            if gain >= 2:
                S = S_temp | set(new_elements)
                improved = True
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
                break # Restart Phase 1 with new S
