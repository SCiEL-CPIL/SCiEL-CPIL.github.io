    def is_valid_addition(S, x):
        """Check if adding x to S keeps S valid. S is a set of ints, x is an int not in S."""
        s_list = list(S)
        
        # Case 1: x is one of the three elements a, b, c.
        # The triple is {x, a, b} with a, b in S.
        # Pair sums: x+a, x+b, a+b.
        # We need x+a in S U {x}, x+b in S U {x}, a+b in S U {x}.
        # Since a,b >= 1, x+a > x and x+b > x, so x+a, x+b must be in S.
        # a+b can be in S or equal to x.
        
        # Find all a in S such that a+x is in S.
        K = [a for a in s_list if a + x <= n and (a + x) in S]
        
        if K:
            for i in range(len(K)):
                for j in range(i + 1, len(K)):
                    a, b = K[i], K[j]
                    ab = a + b
                    # Check if a+b is in the final set S U {x}
                    if ab == x or ab in S:
                        return False
        
        # Case 2: x is the sum of two elements a, b in S.
        # The triple is {a, b, x}.
        # Pair sums: a+b=x, a+x, b+x.
        # We need a+x in S U {x} and b+x in S U {x}.
        # Since a,b >= 1, a+x > x and b+x > x, so a+x, b+x must be in S.
        
        for a in s_list:
            b = x - a
            if b <= 0 or b > n or b == a or b not in S:
                continue
            # Check if a+x and b+x are in S
            if (a + x <= n and (a + x) in S) and (b + x <= n and (b + x) in S):
                return False
        
        return True
