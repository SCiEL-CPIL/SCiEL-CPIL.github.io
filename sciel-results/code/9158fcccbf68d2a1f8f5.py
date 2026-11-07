import math
import time

def construct():
    """
    Returns a list of distinct integers in 2..256 such that for every three 
    distinct members a, b, c, a does not divide b*c.
    """
    deadline = time.time() + 80
    
    # Phase 1: Start with a large antichain [129, 256]
    S = list(range(129, 257))
    
    def count_violations(S):
        """
        Returns the number of violating triples in S.
        A triple (a,b,c) of distinct elements violates if a | b*c.
        """
        S_set = set(S)
        n = len(S)
        violations = 0
        
        for i, a in enumerate(S):
            for j, b in enumerate(S):
                if i == j:
                    continue
                g = math.gcd(a, b)
                req = a // g
                if req == 1:
                    # a divides b, so for any c distinct from a,b, violation
                    violations += max(0, n - 2)
                    continue
                # Count c in S, c != a, c != b, such that c % req == 0
                for c in S:
                    if c == a or c == b:
                        continue
                    if c % req == 0:
                        violations += 1
        
        return violations
    
    def get_conflict_counts(S):
        """
        Returns a dictionary of conflict counts for each element in S.
        An element's count is the number of violating triples it participates in.
        """
        counts = {x: 0 for x in S}
        n = len(S)
        
        for i, a in enumerate(S):
            for j, b in enumerate(S):
                if i == j:
                    continue
                g = math.gcd(a, b)
                req = a // g
                if req == 1:
                    # a | b, so any c distinct from a,b creates violation
                    # This triple involves a, b, and c
                    if n > 2:
                        counts[a] += 1
                        counts[b] += 1
                        # c is also involved, but we don't know which c specifically
                        # in a tight loop. For now, just count a and b.
                    continue
                
                # Find c in S, c != a, c != b, such that c % req == 0
                for c in S:
                    if c == a or c == b:
                        continue
                    if c % req == 0:
                        # Violation: a | b*c
                        counts[a] += 1
                        counts[b] += 1
                        counts[c] += 1
        
        return counts
    
    # Greedy Removal
    while time.time() < deadline:
        if not S:
            break
            
        conflicts = get_conflict_counts(S)
        max_conf = max(conflicts.values()) if conflicts else 0
        
        if max_conf == 0:
            break
        
        # Remove the element with the highest conflict count
        worst_elem = None
        max_val = -1
        for x in S:
            if conflicts[x] > max_val:
                max_val = conflicts[x]
                worst_elem = x
        
        if worst_elem is not None:
            S.remove(worst_elem)
        else:
            break
    
    # Phase 2: Greedy Addition
    # Try to add elements from 128 down to 2
    # Order: Larger numbers are "safer"
    S_set = set(S)
    
    for x in range(128, 1, -1):
        if time.time() > deadline:
            break
            
        if x in S_set:
            continue
            
        # Check if S U {x} is valid
        # Only need to check triples involving x.
        valid = True
        
        # Triples: (x, b, c), (b, x, c), (b, c, x) where b,c in S, b!=c
        # 1. x | b*c
        # 2. b | x*c
        # 3. c | x*b  (same as 2 with b,c swapped)
        
        for i, b in enumerate(S):
            for j, c in enumerate(S):
                if i == j:
                    continue
                
                # Check 1: x | b*c
                if (b * c) % x == 0:
                    valid = False
                    break
                
                # Check 2: b | x*c
                if (x * c) % b == 0:
                    valid = False
                    break
                
                # Check 3: c | x*b
                if (x * b) % c == 0:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            S.append(x)
            S_set.add(x)
    
    # Phase 3: Swap-based local search to escape local optima
    # Try swapping elements in S with elements not in S
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_set = set(S)
        
        # Try each element in S
        for i, a in enumerate(S):
            if time.time() > deadline:
                break
            
            # Try swapping with each element not in S
            for b in range(256, 1, -1):
                if b in S_set:
                    continue
                
                # Check if S - {a} + {b} is valid
                # Only need to check triples involving b (since S - {a} was valid)
                # and triples that involved a and now might be affected
                # Actually, removing a can only help, adding b can only hurt.
                # So we need to check all triples involving b.
                
                valid = True
                S_without_a = S[:i] + S[i+1:]
                
                for j, c in enumerate(S_without_a):
                    for k, d in enumerate(S_without_a):
                        if j == k:
                            continue
                        
                        # Check: b | c*d
                        if (c * d) % b == 0:
                            valid = False
                            break
                        
                        # Check: c | b*d
                        if (b * d) % c == 0:
                            valid = False
                            break
                        
                        # Check: d | b*c
                        if (b * c) % d == 0:
                            valid = False
                            break
                    
                    if not valid:
                        break
                
                if valid:
                    # Perform the swap
                    S[i] = b
                    S_set.remove(a)
                    S_set.add(b)
                    improved = True
                    break  # Restart from beginning after a swap
    
    # Final cleanup: ensure all are Python int
    S = [int(x) for x in S]
    
    return S

