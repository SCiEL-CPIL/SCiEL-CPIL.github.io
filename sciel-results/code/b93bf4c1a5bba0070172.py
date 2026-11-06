def construct():
    N = 256
    S = set()
    
    for x in range(1, N + 1):
        # Check for conflicts
        conflict_elements = set()
        
        # x is the largest term: a, b, x => x = 2b - a => a = 2b - x
        # Since a, b in S and a < b < x
        for b in S:
            a = 2 * b - x
            if a > 0 and a < b and a in S:
                conflict_elements.add(a)
                conflict_elements.add(b)
                
        # x is the middle term: a, x, c => c = 2x - a
        # But c > x, so c is not in S yet. So no conflict here for current S.
        
        # x is the smallest term: x, b, c => b = (x+c)/2? No, c not in S.
        
        if not conflict_elements:
            S.add(x)
        else:
            # Find the element in conflict with the smallest reciprocal (largest value)
            # We want to remove the "cheapest" element to free up x
            # Actually, we should remove the element y such that 1/y is minimal among conflicts?
            # If we remove y, we lose 1/y. We gain 1/x.
            # We should swap if 1/x > 1/y.
            # To maximize gain, we should remove the y with the smallest 1/y (largest y).
            # But removing one y might not resolve all conflicts if x conflicts with multiple pairs.
            # Example: x=10. S={2,4,6,8}.
            # 2,6,10 (conflict 2,6)
            # 4,7,10 (7 not in S)
            # 8,9,10 (9 not in S)
            # Only one conflict.
            
            # If multiple conflicts, e.g., x=12, S={2,4,6,8,10}
            # 2,7,12 (7 no)
            # 4,8,12 (conflict 4,8)
            # 6,9,12 (9 no)
            # 10,11,12 (11 no)
            
            # Let's just find the max value in conflict_elements.
            max_conflict_val = max(conflict_elements)
            
            if 1.0 / x > 1.0 / max_conflict_val:
                # Remove max_conflict_val and add x
                # But what if removing max_conflict_val doesn't resolve all conflicts?
                # If x conflicts with (a1, b1) and (a2, b2), and max_conflict_val is b1.
                # Removing b1 resolves (a1, b1, x).
                # Does it resolve (a2, b2, x)? Only if b2 == b1 or a2 == b1.
                # If not, we still have a conflict.
                # In that case, we might need to remove more elements.
                # For simplicity and speed, let's just check if removing the single largest conflicting element
                # makes x valid.
                
                S.remove(max_conflict_val)
                
                # Verify if x is now valid
                is_valid = True
                for b in S:
                    a = 2 * b - x
                    if a > 0 and a < b and a in S:
                        is_valid = False
                        break
                    # Also check if x is middle? No, c > x not in S.
                    # Also check if x is smallest? No.
                
                if is_valid:
                    S.add(x)
                else:
                    # Revert
                    S.add(max_conflict_val)
                    # Don't add x
                    pass
            else:
                # Don't add x
                pass
                
    return list(S)
