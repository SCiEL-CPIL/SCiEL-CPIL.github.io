import time
import random

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    N = 128
    S = set(range(1, N + 1))
    
    # Helper to check if a specific triple (a,b,c) causes a violation given the set S
    # We don't need this for the initial removal phase, we just find violations directly.
    
    def find_violations(S):
        """
        Returns a list of violating triples (a,b,c) where a,b,c in S distinct
        and a+b, a+c, b+c in S.
        Also returns a count of how many times each element appears in violations.
        """
        violations = []
        counts = {}
        S_list = list(S)
        n = len(S_list)
        
        # Optimization: If S is large, this is O(N^3). 128^3 is ~2M.
        # In Python, this might take a few seconds.
        # We can break early if we find many violations? No, we need counts to pick the best element to remove.
        # But we can stop if time is running out.
        
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                for k in range(j + 1, n):
                    c = S_list[k]
                    sum_ab = a + b
                    sum_ac = a + c
                    sum_bc = b + c
                    
                    if sum_ab in S and sum_ac in S and sum_bc in S:
                        violations.append((a, b, c))
                        # Increment counts for a, b, c and the sums if they are in S
                        for elem in [a, b, c, sum_ab, sum_ac, sum_bc]:
                            if elem in S:
                                counts[elem] = counts.get(elem, 0) + 1
                                
        return violations, counts

    # Phase 1: Remove elements to make S valid
    while True:
        if time.time() - start_time > timeout:
            break
            
        violations, counts = find_violations(S)
        if not violations:
            break
            
        # Find element with max count
        # If tie, pick random or smallest
        max_count = -1
        candidates = []
        for elem, count in counts.items():
            if count > max_count:
                max_count = count
                candidates = [elem]
            elif count == max_count:
                candidates.append(elem)
                
        # Remove one candidate
        # Prioritize removing elements that are sums? Or just random from candidates?
        # Removing an element that is a sum might break more "Case 2" violations.
        # Removing an element that is a,b,c breaks "Case 1" violations.
        # Let's just pick one.
        elem_to_remove = random.choice(candidates)
        S.remove(elem_to_remove)
        
    # Phase 2: Try to add back elements
    # Sort candidates by value? Or random?
    # Let's try adding elements from the complement.
    complement = [x for x in range(1, N + 1) if x not in S]
    random.shuffle(complement)
    
    def is_valid_addition(S, x):
        """
        Check if S U {x} is valid.
        """
        # Case 1: x is one of a, b, c
        # Need distinct u, v in S such that x+u in S, x+v in S, u+v in S U {x}
        S_list = list(S)
        n = len(S_list)
        
        # To speed up, we can precompute sums?
        # For now, O(N^2) check.
        for i in range(n):
            u = S_list[i]
            sum_xu = x + u
            if sum_xu not in S:
                continue
            for j in range(i + 1, n):
                v = S_list[j]
                sum_xv = x + v
                if sum_xv not in S:
                    continue
                sum_uv = u + v
                if sum_uv in S or sum_uv == x:
                    return False
                    
        # Case 2: x is one of the sums
        # Need distinct a, b, c in S such that one sum is x, others in S.
        # Subcase: a+b=x. Check if there is c in S\{a,b} s.t. a+c in S and b+c in S.
        # We can iterate pairs (a,b) summing to x.
        for a in S:
            b = x - a
            if b in S and a != b:
                # Check for c
                # a+c in S and b+c in S
                # c must be in S, c != a, c != b
                # We can iterate c in S
                # Optimization: 
                # a+c in S => c in S-a
                # b+c in S => c in S-b
                # So c must be in (S-a) intersection (S-b) intersection S
                # And c not in {a,b}
                
                # Simple loop for c
                for c in S:
                    if c == a or c == b:
                        continue
                    if (a + c) in S and (b + c) in S:
                        return False
                        
        return True

    for x in complement:
        if time.time() - start_time > timeout:
            break
        if is_valid_addition(S, x):
            S.add(x)
            
    # Phase 3: Local Search
    # Try swaps
    last_improvement = time.time()
    
    while time.time() - start_time < timeout - 2.0:
        if time.time() - last_improvement > 5.0:
            # Stagnation? Maybe restart or just continue
            # Let's just continue, maybe pick better candidates
            pass
            
        # Pick random x not in S
        x = random.randint(1, N)
        if x in S:
            continue
            
        if is_valid_addition(S, x):
            S.add(x)
            last_improvement = time.time()
        else:
            # Try swap
            # Pick random y in S
            y = random.choice(list(S))
            S.remove(y)
            if is_valid_addition(S, x):
                S.add(x)
                last_improvement = time.time()
            else:
                S.add(y)
                
    # Ensure types are correct
    return sorted(list(S))

if __name__ == "__main__":
    # Self-test
    res = construct()
    print(len(res))
    # Verify
    S = set(res)
    valid = True
    for i, a in enumerate(res):
        for j, b in enumerate(res):
            if i >= j: continue
            for k, c in enumerate(res):
                if k <= j: continue
                if (a+b) in S and (a+c) in S and (b+c) in S:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    print("Valid:", valid)
