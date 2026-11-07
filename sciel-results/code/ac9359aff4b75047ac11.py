import time
import random

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    N = 128
    S = set(range(1, N + 1))
    
    def find_violations_fast(S):
        """
        Find violating triples and count how many times each element appears.
        Uses precomputed sums for speed.
        """
        S_list = sorted(S)
        n = len(S_list)
        counts = {}
        violation_count = 0
        
        # Precompute: for each pair (i,j), check if sum is in S
        # But we need triples, so let's be smart:
        # For each pair (a,b), if a+b in S, then for each c > b, check if a+c in S and b+c in S
        
        # Build a set for O(1) lookup
        S_set = S
        
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                sum_ab = a + b
                if sum_ab not in S_set:
                    continue
                # Now for each c > b, check if a+c and b+c are in S
                for k in range(j + 1, n):
                    c = S_list[k]
                    sum_ac = a + c
                    if sum_ac not in S_set:
                        continue
                    sum_bc = b + c
                    if sum_bc not in S_set:
                        continue
                    # Violation found
                    violation_count += 1
                    for elem in (a, b, c, sum_ab, sum_ac, sum_bc):
                        if elem in S_set:
                            counts[elem] = counts.get(elem, 0) + 1
                            
        return violation_count, counts

    def is_valid_addition_fast(S, x):
        """
        Check if S U {x} is valid. Optimized.
        """
        S_set = S
        
        # Case 1: x is one of a, b, c
        # Need distinct u, v in S such that x+u in S, x+v in S, u+v in S U {x}
        # Optimize: for each u in S where x+u in S, collect such u's
        valid_us = []
        for u in S:
            if x + u in S:
                valid_us.append(u)
        
        if len(valid_us) >= 2:
            # Check pairs of valid_us
            for i in range(len(valid_us)):
                u = valid_us[i]
                for j in range(i + 1, len(valid_us)):
                    v = valid_us[j]
                    sum_uv = u + v
                    if sum_uv in S or sum_uv == x:
                        return False
        
        # Case 2: x is one of the sums
        # a+b = x, a,b in S, a != b
        # Need c in S\{a,b} such that a+c in S and b+c in S
        for a in S:
            b = x - a
            if b not in S or a == b:
                continue
            # Find c in S such that c != a, c != b, a+c in S, b+c in S
            # a+c in S => c in {s-a : s in S}
            # b+c in S => c in {s-b : s in S}
            # Intersect with S, remove a and b
            candidates_c = set()
            for s in S:
                c = s - a
                if c in S and c != a and c != b:
                    candidates_c.add(c)
            for c in candidates_c:
                if (b + c) in S:
                    return False
        
        return True

    # Phase 1: Remove elements to make S valid
    # Use a more strategic removal: prefer removing elements that are sums of two other elements
    while True:
        if time.time() - start_time > timeout:
            break
            
        vcount, counts = find_violations_fast(S)
        if vcount == 0:
            break
            
        # Score each candidate: weight by count, but also penalize small numbers
        # (small numbers are more likely to be sums)
        best_elem = None
        best_score = -1
        for elem, count in counts.items():
            # Higher count is better, but prefer removing larger numbers? 
            # Actually, removing a sum breaks more "Case 2" violations.
            # Let's just use count with a small random tiebreaker
            score = count + random.random() * 0.1
            if score > best_score:
                best_score = score
                best_elem = elem
                
        S.remove(best_elem)
        
    # Phase 2: Try to add back elements
    complement = [x for x in range(1, N + 1) if x not in S]
    random.shuffle(complement)
    
    for x in complement:
        if time.time() - start_time > timeout:
            break
        if is_valid_addition_fast(S, x):
            S.add(x)
            
    # Phase 3: Local Search with multiple attempts per iteration
    last_improvement = time.time()
    no_improve_count = 0
    
    while time.time() - start_time < timeout - 2.0:
        improved = False
        
        # Try adding random elements
        for _ in range(5):
            if time.time() - start_time > timeout - 2.0:
                break
            x = random.randint(1, N)
            if x in S:
                continue
            if is_valid_addition_fast(S, x):
                S.add(x)
                improved = True
                break
        
        if improved:
            last_improvement = time.time()
            no_improve_count = 0
            continue
        
        # Try swaps
        S_list = list(S)
        for _ in range(5):
            if time.time() - start_time > timeout - 2.0:
                break
            x = random.randint(1, N)
            if x in S:
                continue
            y = random.choice(S_list)
            S.remove(y)
            if is_valid_addition_fast(S, x):
                S.add(x)
                improved = True
                break
            else:
                S.add(y)
        
        if improved:
            last_improvement = time.time()
            no_improve_count = 0
        else:
            no_improve_count += 1
            # If stuck for a while, try a more aggressive restart-like move
            if no_improve_count > 50:
                # Remove a random element and try to add back multiple
                if S:
                    y = random.choice(list(S))
                    S.remove(y)
                    # Try to add several elements
                    for _ in range(10):
                        if time.time() - start_time > timeout - 2.0:
                            break
                        x = random.randint(1, N)
                        if x in S:
                            continue
                        if is_valid_addition_fast(S, x):
                            S.add(x)
                    no_improve_count = 0
                
    # Ensure types are correct
    return sorted(list(S))

if __name__ == "__main__":
    # Self-test
    res = construct()
    print(len(res))
    # Verify
    S = set(res)
    valid = True
    res_list = list(res)
    for i, a in enumerate(res_list):
        for j, b in enumerate(res_list):
            if i >= j: continue
            for k, c in enumerate(res_list):
                if k <= j: continue
                if (a+b) in S and (a+c) in S and (b+c) in S:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    print("Valid:", valid)
