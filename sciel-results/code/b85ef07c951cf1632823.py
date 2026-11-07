def construct():
        N = 256
        current_set = []
        
        # Greedy: try adding numbers from largest to smallest
        # Larger numbers are less likely to divide the product of two others
        # because they are larger, so their multiples are sparser.
        for candidate in range(N, 1, -1):
            can_add = True
            # Check against all pairs in current_set
            # Condition: for distinct a,b,c in S, a !| b*c
            # If we add candidate 'c', we need to ensure:
            # 1. c does not divide u*v for any u,v in S
            # 2. u does not divide c*v for any u,v in S
            # 3. v does not divide c*u for any u,v in S
            
            for i in range(len(current_set)):
                u = current_set[i]
                for j in range(i + 1, len(current_set)):
                    v = current_set[j]
                    # Check 1: candidate | u*v
                    if (u * v) % candidate == 0:
                        can_add = False
                        break
                    # Check 2: u | candidate*v
                    if (candidate * v) % u == 0:
                        can_add = False
                        break
                    # Check 3: v | candidate*u
                    if (candidate * u) % v == 0:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                current_set.append(candidate)
                
        # Return as list of ints
        return current_set
