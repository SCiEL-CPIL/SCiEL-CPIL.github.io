import random

def construct():
    N = 128
    S = set(range(1, N + 1))
    
    def find_violation(S):
        in_S = [False] * (N + 1)
        for x in S:
            in_S[x] = True
            
        S_list = sorted(S)
        n = len(S_list)
        
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                s_ab = a + b
                if s_ab > N:
                    break
                if not in_S[s_ab]:
                    continue
                    
                for k in range(j + 1, n):
                    c = S_list[k]
                    s_bc = b + c
                    if s_bc > N:
                        break
                    if in_S[a + c] and in_S[s_bc]:
                        return (a, b, c)
        return None

    def is_valid_add(S, x):
        # Check if adding x creates a violation
        in_S = [False] * (N + 1)
        for val in S:
            in_S[val] = True
        # x is not in in_S yet, we treat S' = S U {x}
        
        # Helper to check if y is in S'
        def in_S_prime(y):
            return in_S[y] or (y == x)
            
        S_list = sorted(S)
        n = len(S_list)
        
        # Case A: x is one of a, b, c
        # We need to check triples from S' that include x.
        # This is equivalent to checking if there are a,b,c in S' distinct, with sums in S'.
        # If the triple does not include x, it was already valid in S.
        # So we only check triples involving x.
        
        # A1: x is c. a, b in S.
        for i in range(n):
            a = S_list[i]
            if a == x: continue # x not in S
            for j in range(i + 1, n):
                b = S_list[j]
                if b == x: continue
                # Check a, b, x
                # Sums: a+b, a+x, b+x
                if in_S_prime(a+b) and in_S_prime(a+x) and in_S_prime(b+x):
                    return False
                    
        # A2: x is b. a, c in S.
        for i in range(n):
            a = S_list[i]
            if a == x: continue
            for k in range(i + 1, n):
                c = S_list[k]
                if c == x: continue
                # Check a, x, c
                # Sums: a+x, x+c, a+c
                if in_S_prime(a+x) and in_S_prime(x+c) and in_S_prime(a+c):
                    return False
                    
        # A3: x is a. b, c in S.
        for j in range(n):
            b = S_list[j]
            if b == x: continue
            for k in range(j + 1, n):
                c = S_list[k]
                if c == x: continue
                # Check x, b, c
                # Sums: x+b, x+c, b+c
                if in_S_prime(x+b) and in_S_prime(x+c) and in_S_prime(b+c):
                    return False

        # Case B: x is one of the sums.
        # B1: x = a+b. a,b in S.
        # Need c in S, c!=a, c!=b, such that a+c in S' and b+c in S'.
        # Note a+c != x (implies c=b), b+c != x (implies c=a).
        # So a+c in S and b+c in S.
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                if a + b == x:
                    # Check for c
                    for k in range(n):
                        c = S_list[k]
                        if c == a or c == b:
                            continue
                        if in_S[a + c] and in_S[b + c]:
                            return False
                            
        # B2: x = a+c. Symmetric
        # B3: x = b+c. Symmetric
        # The loop above covers all pairs summing to x.
        
        return True

    best_S = set(S)
    best_size = len(S)
    
    for trial in range(10):
        current_S = set(S)
        
        # Removal Phase
        while True:
            viol = find_violation(current_S)
            if viol is None:
                break
            a, b, c = viol
            candidates = [a, b, c, a+b, a+c, b+c]
            valid_candidates = [x for x in candidates if x in current_S]
            rem = random.choice(valid_candidates)
            current_S.remove(rem)
            
        # Addition Phase
        candidates_to_add = [x for x in range(1, N + 1) if x not in current_S]
        random.shuffle(candidates_to_add)
        
        for x in candidates_to_add:
            if is_valid_add(current_S, x):
                current_S.add(x)
                
        if len(current_S) > best_size:
            best_size = len(current_S)
            best_S = set(current_S)
            
    return list(best_S)
