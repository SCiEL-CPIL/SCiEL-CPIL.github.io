def construct():
        import time
        from math import gcd
        from fractions import Fraction
        from itertools import combinations

        start_time = time.time()
        deadline = start_time + 145.0

        # 1. Greedy Construction
        S = [1]
        for a in range(511, 1, -1):
            if all(gcd(a, s) == 1 for s in S):
                S.append(a)
        
        # Convert to set for fast lookup? No, list is fine for small N.
        # Use Fraction for exact comparison
        current_score = sum(Fraction(1, 512 - a) for a in S)

        # Helper to check coprimality with set
        def is_coprime_to_all(x, S_set):
            for s in S_set:
                if s == x: continue
                if gcd(x, s) != 1:
                    return False
            return True

        S_set = set(S)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(S)
            
            # 1-swap: remove i, add j
            for i in range(n):
                if time.time() > deadline: break
                a = S[i]
                S_remaining = [x for j, x in enumerate(S) if j != i]
                S_rem_set = set(S_remaining)
                
                best_add = None
                best_delta = Fraction(0, 1)
                
                # Only look for b > a to increase score
                for b in range(a + 1, 512):
                    if b in S_rem_set: continue
                    if not is_coprime_to_all(b, S_rem_set): continue
                    
                    delta = Fraction(1, 512 - b) - Fraction(1, 512 - a)
                    if delta > best_delta:
                        best_delta = delta
                        best_add = b
                
                if best_add is not None and best_delta > 0:
                    S[i] = best_add
                    S_set.remove(a)
                    S_set.add(best_add)
                    current_score += best_delta
                    improved = True
                    break # Restart loop with new state
            
            # 2-swap: remove i, add j, k
            if not improved and time.time() < deadline - 10:
                for i in range(n):
                    if time.time() > deadline: break
                    a = S[i]
                    S_remaining = [x for j, x in enumerate(S) if j != i]
                    S_rem_set = set(S_remaining)
                    
                    best_pair = None
                    best_delta = Fraction(0, 1)
                    
                    # Optimize: only consider b, c > a?
                    # Not necessarily, but likely.
                    # To save time, restrict b, c to range(a, 512)
                    # If b < a, w(b) < w(a). Unlikely to help unless c is huge.
                    # But we process descending, so huge numbers are already in or skipped.
                    # If skipped, they conflict with S_rem.
                    # So looking for b, c in S_rem is useless.
                    # Looking for b, c not in S_rem.
                    
                    for b in range(a + 1, 512):
                        if time.time() > deadline: break
                        if b in S_rem_set: continue
                        if not is_coprime_to_all(b, S_rem_set): continue
                        
                        for c in range(b + 1, 512):
                            if c in S_rem_set: continue
                            if not is_coprime_to_all(c, S_rem_set): continue
                            if gcd(b, c) != 1: continue
                            
                            delta = Fraction(1, 512 - b) + Fraction(1, 512 - c) - Fraction(1, 512 - a)
                            if delta > best_delta:
                                best_delta = delta
                                best_pair = (b, c)
                    
                    if best_pair is not None and best_delta > 0:
                        S[i] = best_pair[0]
                        S.append(best_pair[1])
                        S_set.remove(a)
                        S_set.add(best_pair[0])
                        S_set.add(best_pair[1])
                        current_score += best_delta
                        improved = True
                        break

        return S
