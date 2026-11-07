def construct():
    S = set()
    # Try adding numbers from 128 down to 1
    for i in range(128, 0, -1):
        # Check if adding i creates a forbidden triple
        valid = True
        
        # Case 1: i is a member (a=i)
        # Need b, c in S distinct from i and each other
        # such that i+b in S U {i}, i+c in S U {i}, b+c in S U {i}
        # Note: i+b > i, so i+b cannot be i. So i+b must be in S.
        # Same for i+c.
        # b+c can be in S or be i? No, b,c > 0, so b+c > i is not guaranteed if b,c small?
        # But we are adding in decreasing order, so b,c < i? No, b,c are in S, which contains larger numbers.
        # So b,c > i? No, we added 128..i+1. So b,c > i.
        # So b+c > 2i > i. So b+c cannot be i.
        # So b+c must be in S.
        # So we need:
        # i+b in S
        # i+c in S
        # b+c in S
        
        # Iterate b, c in S
        # To speed up, we can check pairs
        # If |S| is large, this is O(|S|^2)
        
        # Optimization: 
        # If b+c not in S, skip.
        # Precompute sums in S?
        
        # Let's just do the check. 128 is small.
        
        # Collect elements in S
        elems = list(S)
        
        # Check Case 1: a=i
        for idx_b in range(len(elems)):
            b = elems[idx_b]
            if b == i: continue
            # i+b must be in S
            if (i + b) not in S:
                continue
            for idx_c in range(idx_b + 1, len(elems)):
                c = elems[idx_c]
                if c == i: continue
                if c == b: continue # distinct b,c
                # Check distinct a,b,c -> i, b, c distinct. 
                # b,c from S, i not in S yet. So i distinct from b,c.
                # b != c ensured by loop.
                
                # Check sums
                if (i + c) in S and (b + c) in S:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        # Case 2: i is a sum
        # Need a, b, c in S distinct
        # such that a+b, a+c, b+c are in S U {i}
        # And at least one sum is i.
        # Since S was valid, not all 3 were in S.
        # So exactly one or two were missing, and now covered by i?
        # If two sums are i, say a+b=i and a+c=i -> b=c, impossible.
        # So exactly one sum is i.
        # WLOG a+b=i.
        # Then a+c in S and b+c in S.
        
        # Iterate pairs a,b in S with a+b=i
        # Since we are adding decreasing, S contains numbers > i.
        # So a,b > i.
        # Then a+b > 2i.
        # If i < 64, 2i < 128.
        # If i >= 64, a+b > 128, so a+b cannot be i (since i <= 128).
        # So Case 2 only possible if i < 64.
        
        if i < 64:
            # Find pairs a,b in S with a+b=i
            # Since a,b in S and S has elements > i, this is impossible?
            # Wait. S contains elements added previously.
            # We are iterating 128 down to 1.
            # When we are at i, S contains {i+1, ..., 128} minus rejected ones.
            # So min element in S is > i.
            # So a > i and b > i.
            # a+b > 2i.
            # If i < 64, 2i < 128.
            # Can a+b=i? No, a+b > 2i > i.
            # So a+b cannot be i.
            # So Case 2 is IMPOSSIBLE in decreasing order?
            
            # Let's re-verify.
            # a,b,c are in S.
            # S = {x | x > i}.
            # So a,b,c > i.
            # Sums a+b, a+c, b+c > 2i.
            # We need one sum to be i.
            # But sum > 2i > i.
            # So no sum can be i.
            # So Case 2 never happens.
            
            pass

        if valid:
            S.add(i)
            
    return [int(x) for x in S]
