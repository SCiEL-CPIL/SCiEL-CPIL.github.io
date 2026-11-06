def construct():
    n = 256
    S = []
    # We will add numbers from 256 down to 1
    for a in range(n, 0, -1):
        # Check if a can be added to S
        can_add = True
        
        # 1. Check a as divisor: a must not divide b+c for any b,c in S with a < b <= c
        # Since we add in descending order, all elements in S are > a.
        # So we need to check all pairs (b, c) in S.
        # Note: S contains elements > a.
        # Pairs (b, c) with b <= c.
        for i in range(len(S)):
            b = S[i]
            for j in range(i, len(S)):
                c = S[j]
                # b and c are > a because we iterate descending
                if (b + c) % a == 0:
                    can_add = False
                    break
            if not can_add:
                break
        
        if not can_add:
            continue
            
        # 2. Check existing elements x in S as divisors:
        # x must not divide (a + b) or (a + a) for any b in S with x < a <= b
        # Since S has elements > a, any b in S is > a.
        # So we check x in S (which are all > a? No, wait.)
        # We are adding a. S currently contains elements > a.
        # So x in S is > a.
        # The condition for x is: x < b <= c.
        # If x is in S, and we add a, the new summands are a.
        # The pairs involving a are (a, b) and (a, a) where b in S.
        # But the condition requires x < b <= c.
        # Here x is the divisor.
        # If x in S, x > a.
        # So x cannot be less than a.
        # So x cannot be a divisor for a sum involving a, because x > a.
        # The condition is a < b <= c.
        # So the divisor must be smaller than the summands.
        # Since we add a, and S has elements > a, a is the smallest element in the new set S U {a}.
        # So a is the only new potential divisor.
        # Existing elements x in S are larger than a, so they cannot be divisors for sums involving a (since a < x).
        # Wait, if x in S, x > a.
        # Can x divide a + b?
        # Condition: x < b <= c.
        # Here summands are a and b.
        # We need x < a <= b or x < b <= a.
        # Since x > a, x cannot be less than a.
        # So x cannot be a divisor for any sum involving a.
        # Thus, we only need to check condition 1!
        
        if can_add:
            S.append(a)
            
    return S
