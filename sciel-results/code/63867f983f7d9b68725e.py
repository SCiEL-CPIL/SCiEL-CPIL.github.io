def construct():
        n = 256
        max_prod = n * (n - 1)
        divisors = [[] for _ in range(max_prod + 1)]
        for d in range(2, n + 1):
            for multiple in range(d, max_prod + 1, d):
                divisors[multiple].append(d)
        
        S = set(range(2, n + 1))
        
        # Greedy Removal
        while True:
            degrees = {x: 0 for x in S}
            has_conflict = False
            S_list = list(S)
            
            for i in range(len(S_list)):
                b = S_list[i]
                for j in range(i + 1, len(S_list)):
                    c = S_list[j]
                    prod = b * c
                    for a in divisors[prod]:
                        if a in S and a != b and a != c:
                            has_conflict = True
                            degrees[a] += 1
                            degrees[b] += 1
                            degrees[c] += 1
            
            if not has_conflict:
                break
            
            max_deg = -1
            best_candidate = None
            for x in S:
                d = degrees[x]
                if d > max_deg:
                    max_deg = d
                    best_candidate = x
                elif d == max_deg:
                    if best_candidate is None or x < best_candidate: # Remove smallest
                        best_candidate = x
            
            S.remove(best_candidate)
            
        # Add Back Phase
        # Try to add back any number from 2..256 not in S
        added = True
        while added:
            added = False
            S_list = list(S)
            for x in range(2, n + 1):
                if x in S:
                    continue
                
                # Check if x can be added
                compatible = True
                # 1. Check if x divides any a*b for a,b in S
                for i in range(len(S_list)):
                    a = S_list[i]
                    for j in range(i + 1, len(S_list)):
                        b = S_list[j]
                        if x != a and x != b: # distinctness is handled by a,b in S, x not in S
                            if (a * b) % x == 0:
                                compatible = False
                                break
                    if not compatible:
                        break
                
                if not compatible:
                    continue
                
                # 2. Check if any a in S divides x*b
                # a | x*b
                for a in S_list:
                    # We need to check for all b in S, b != a
                    # If a | x*b, conflict.
                    # This is equivalent to checking if there exists b in S \ {a} such that a | x*b
                    for b in S_list:
                        if b == a:
                            continue
                        if (x * b) % a == 0:
                            compatible = False
                            break
                    if not compatible:
                        break
                
                if compatible:
                    S.add(x)
                    added = True
                    # Restart the while loop to try adding more
                    # Actually, we can just continue checking other x's in this pass,
                    # but adding x might make other x's incompatible?
                    # No, adding x only increases constraints.
                    # So if y was compatible before, it might not be now.
                    # But we are iterating x.
                    # It's safer to restart the outer loop or just continue and check against updated S.
                    # Since we update S, the next candidate y will be checked against S + {x}.
                    # This is correct.
        
        return list(S)
