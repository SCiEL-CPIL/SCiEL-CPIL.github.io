import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Initial candidate set: upper half
    # We know {171, ..., 256} is valid with size 86.
    # We start with a larger set and prune.
    S = set(range(129, 257))
    
    def check_violations(S):
        """
        Returns a list of tuples (a, b, c) such that a in S, b,c in S, a < b <= c, and a divides b+c.
        For efficiency, we only check a < b <= c.
        """
        violations = []
        S_list = sorted(S)
        # To speed up, we can just iterate
        for i, a in enumerate(S_list):
            # b, c must be > a
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        violations.append((a, b, c))
        return violations

    # Pruning loop
    while time.time() < deadline:
        violations = check_violations(S)
        if not violations:
            break
        
        # Resolve violations
        # Heuristic: Remove the divisor 'a' if it appears in many violations, 
        # or just remove 'a' from the first violation.
        # Better: Count frequency of each element in violations.
        # If we remove 'a', we kill all violations involving 'a' as divisor.
        # If we remove 'b', we kill violations involving 'b' as summand.
        
        # Simple strategy: Remove the element involved in the most violations.
        # We'll just remove 'a' from the first violation to start, 
        # but let's try to be smart.
        
        # Count occurrences
        counts = {}
        for a, b, c in violations:
            counts[a] = counts.get(a, 0) + 1
            counts[b] = counts.get(b, 0) + 1
            counts[c] = counts.get(c, 0) + 1
        
        # Remove the one with max count
        if counts:
            max_elem = max(counts, key=counts.get)
            S.remove(max_elem)
    
    # Try to add back elements from [1, 128] or removed elements
    # We removed some elements from [129, 256]. Let's try to re-add them.
    # Also try adding from [1, 128].
    
    # Create a list of candidates not in S
    candidates = [i for i in range(1, 257) if i not in S]
    # Sort by value descending? Smaller divisors are more restrictive.
    # So we should try adding larger elements first?
    # Or try to add elements that were removed?
    
    # Let's try adding candidates in random order or descending order.
    # Descending order: larger elements are less likely to be divisors of sums of larger elements?
    # Actually, if we add a large element, it's a summand.
    # If we add a small element, it's a divisor.
    
    # Let's try adding from 256 down to 1.
    for i in range(256, 0, -1):
        if time.time() > deadline:
            break
        if i in S:
            continue
        
        # Check if adding i causes violations
        # i can be a divisor or a summand.
        # 1. i as divisor: check if any b,c in S with b,c > i and (b+c) % i == 0
        # 2. i as summand: check if any a in S with a < i and (i+c) % a == 0 for some c in S, c >= i?
        #    Wait, condition is a < b <= c.
        #    If i is b: a < i <= c. Check (i+c) % a == 0.
        #    If i is c: a < b <= i. Check (b+i) % a == 0.
        
        valid = True
        
        # Check i as divisor
        for b in S:
            if b <= i:
                continue
            for c in S:
                if c < b:
                    continue
                if (b + c) % i == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Check i as summand
            # Case 1: i is b (middle). a < i <= c.
            for a in S:
                if a >= i:
                    continue
                for c in S:
                    if c < i:
                        continue
                    if (i + c) % a == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            # Case 2: i is c (largest). a < b <= i.
            if valid:
                for a in S:
                    if a >= i:
                        continue
                    for b in S:
                        if b > i:
                            continue
                        if b <= a:
                            continue
                        if (b + i) % a == 0:
                            valid = False
                            break
                    if not valid:
                        break
        
        if valid:
            S.add(i)
            
    return sorted(S)
