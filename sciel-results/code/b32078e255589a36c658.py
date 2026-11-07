def construct():
        import time
        start_time = time.time()
        deadline = 60.0 # seconds
        
        S = []
        # Try adding numbers from 256 down to 2
        for x in range(256, 1, -1):
            if time.time() - start_time > deadline - 5:
                break
            
            # Check if x can be added to S
            can_add = True
            
            # Condition 1: x does not divide b*c for any distinct b, c in S
            # b, c must be distinct from each other.
            # Since x is new, b, c are from S.
            # We need to check all pairs (b, c) in S with b != c.
            # Note: The problem statement says "three distinct members a, b, c".
            # So b != c is required.
            
            n = len(S)
            for i in range(n):
                b = S[i]
                for j in range(n):
                    if i == j: continue
                    c = S[j]
                    # Check x | b*c
                    if (b * c) % x == 0:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if not can_add:
                continue
            
            # Condition 2: For any b in S, b does not divide x*c for any c in S, c != b.
            # Also c != x (automatic).
            # So for each b in S, check all c in S, c != b.
            for i in range(n):
                b = S[i]
                for j in range(n):
                    if i == j: continue
                    c = S[j]
                    # Check b | x*c
                    if (x * c) % b == 0:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                S.append(x)
        
        # The problem requires distinct integers in 2..256.
        # The greedy approach ensures distinctness.
        # Return as list of ints.
        return S
