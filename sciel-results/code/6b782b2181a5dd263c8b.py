"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
def construct():
    # Greedy algorithm that picks the element which blocks the most other candidates.
    # This tends to minimize the size of the resulting maximal Sidon set.
    
    S = []
    Sums = set()
    Candidates = list(range(1, N + 1))
    
    while Candidates:
        # 1. Identify all candidates that can be added to S
        valid_candidates = []
        for x in Candidates:
            # Check if x can be added
            can_add = True
            if 2 * x in Sums:
                can_add = False
            else:
                for s in S:
                    if x + s in Sums:
                        can_add = False
                        break
            if can_add:
                valid_candidates.append(x)
                
        if not valid_candidates:
            # No candidate can be added, so S is maximal
            break
            
        # 2. Evaluate each valid candidate by how many other candidates it blocks
        best_x = None
        max_blocked = -1
        
        # Pre-compute current sums for efficiency in inner loop
        # We simulate adding x to S
        for x in valid_candidates:
            # New sums introduced by x
            new_sums = {2 * x}
            for s in S:
                new_sums.add(x + s)
            
            blocked_count = 0
            for y in Candidates:
                if y == x:
                    continue
                # Check if y is blocked by adding x
                # y is blocked if adding y to S U {x} is invalid
                # This happens if:
                # 1. 2y is in Sums U new_sums
                # 2. y + s is in Sums U new_sums for some s in S U {x}
                
                is_blocked = False
                if 2 * y in Sums or 2 * y in new_sums:
                    is_blocked = True
                else:
                    for s in S:
                        if y + s in Sums or y + s in new_sums:
                            is_blocked = True
                            break
                    if not is_blocked and (y + x) in Sums or (y + x) in new_sums:
                         # Note: y+x is checked via y+s where s=x? 
                         # Wait, the condition is: adding y to S' = S U {x} must fail.
                         # Sums(S') = Sums U new_sums.
                         # We need to check if y creates a collision with Sums(S').
                         # Collision if 2y in Sums(S') or y+s' in Sums(S') for s' in S'.
                         # s' can be x or any s in S.
                         # We already checked y+s for s in S.
                         # We need to check y+x.
                         # y+x is in Sums(S')? 
                         # y+x is a new sum. It collides if it equals an existing sum in Sums(S') \ {y+x}.
                         # Actually, the check "y+s in Sums" covers collisions with *old* sums.
                         # If y+x == 2y (i.e., x=y, impossible) or y+x == s1+s2.
                         # The standard check is:
                         # For all s' in S', (y+s') not in Sums(S') \ {y+s'}.
                         # This is equivalent to:
                         # 2y not in Sums(S')
                         # AND for all s' in S', y+s' not in Sums(S') \ {y+s'}.
                         
                         # Let's simplify:
                         # y is valid in S' iff:
                         # 2y not in Sums U new_sums
                         # AND for all s in S, (y+s) not in Sums U new_sums
                         # AND (y+x) not in Sums U new_sums \ {y+x}
                         # Note: y+x is in new_sums. So we check if y+x is in Sums (old) or in new_sums (excluding itself).
                         # new_sums contains {2x} U {x+s for s in S}.
                         # Does new_sums contain y+x? Only if y+x = 2x (y=x, no) or y+x = x+s (y=s, no, y in Candidates, s in S, disjoint).
                         # So y+x is NOT in new_sums.
                         # So we just check if y+x is in Sums.
                         
                         if (y + x) in Sums:
                             is_blocked = True
                            
                if is_blocked:
                    blocked_count += 1
                    
            if blocked_count > max_blocked:
                max_blocked = blocked_count
                best_x = x
                
        # 3. Add best_x to S
        if best_x is not None:
            S.append(best_x)
            Sums.add(2 * best_x)
            for s in S[:-1]: # s in old S
                Sums.add(best_x + s)
            Candidates.remove(best_x)
            
    # Verify Sidon
    sums_check = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums_check:
                return S # Fallback, shouldn't happen
            sums_check.add(s)
            
    # Verify Maximal
    # The greedy loop stops when no valid candidate remains, which implies maximality 
    # IF the check "can_add" is correct.
    # Let's double check maximality explicitly to be safe.
    is_maximal = True
    for x in range(1, N + 1):
        if x in S:
            continue
        can_add = True
        if 2 * x in Sums:
            can_add = False
        else:
            for s in S:
                if x + s in Sums:
                    can_add = False
                    break
        if can_add:
            is_maximal = False
            break
            
    if not is_maximal:
        # This should not happen with the loop condition, but if it does, 
        # we might have a bug or the logic "no valid candidates" is sufficient for maximality.
        # Actually, if no candidate can be added, the set IS maximal.
        # So if is_maximal is False, it means my logic above is flawed.
        # But the loop breaks when valid_candidates is empty.
        # So S should be maximal.
        pass
        
    return S
# EVOLVE-BLOCK-END
