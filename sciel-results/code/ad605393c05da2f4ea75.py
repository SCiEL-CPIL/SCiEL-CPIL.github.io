import random

def construct():
    # Greedy approach with decreasing order
    # Try to maximize set size
    # We can try a few random permutations if time permits, or just stick to one good heuristic.
    # Given the deterministic nature required for "seed" comparison, a fixed heuristic is safer.
    # Let's try decreasing order.
    
    S = set()
    # We iterate from 128 down to 1
    for x in range(128, 0, -1):
        if not can_add(x, S):
            continue
        S.add(x)
        
    return [int(x) for x in S]

def can_add(x, S):
    # Check if adding x to S creates a forbidden triple
    # Forbidden: a,b,c in S' distinct, a+b, a+c, b+c in S'
    # S' = S U {x}
    
    # Case 1: x is one of the bases (a, b, or c)
    # WLOG x = a.
    # Need b, c in S, b != c, b != x, c != x
    # s.t. x+b in S', x+c in S', b+c in S'
    # Since x not in S, x+b in S' => x+b in S (as x+b != x for b>0)
    # Similarly x+c in S, b+c in S.
    
    # Optimization: Precompute sums? No, S changes.
    # Just iterate pairs in S.
    # To avoid O(N^2) with high constant, we can break early.
    
    s_list = list(S)
    n = len(s_list)
    
    # Case 1: x is a base
    # We need b, c in S such that:
    # 1. b != c
    # 2. b != x, c != x (automatically true if S doesn't contain x yet, which it doesn't)
    # 3. x + b in S
    # 4. x + c in S
    # 5. b + c in S
    
    # Let's find candidates for b: b in S and x+b in S.
    # Let B = { b in S | x+b in S }
    # Then we need c in B, c != b, and b+c in S.
    
    B = []
    for b in s_list:
        if x + b in S:
            B.append(b)
            
    # Now check pairs in B
    # B size is likely small.
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            b = B[i]
            c = B[j]
            if b + c in S:
                return False
                
    # Case 2: x is a sum
    # WLOG x = a + b.
    # Need a, b in S, a != b.
    # And c in S, c not in {a,b,x}
    # s.t. a+c in S', b+c in S'
    # => a+c in S, b+c in S (since a+c != x, b+c != x for c distinct)
    
    # Find pairs (a,b) in S such that a+b == x.
    # Then for each such pair, check if there exists c in S \ {a,b} such that a+c in S and b+c in S.
    
    # To make this efficient:
    # For a given pair (a,b) summing to x, we need c in S such that:
    # 1. c != a, c != b
    # 2. a + c in S
    # 3. b + c in S
    
    # Let's iterate c in S? That's O(N) per pair.
    # Number of pairs summing to x is small.
    
    for i in range(len(s_list)):
        a = s_list[i]
        b = x - a
        if b in S and b > a: # Avoid duplicates, a < b
            # Check if there is a c
            found_c = False
            for c in s_list:
                if c == a or c == b:
                    continue
                # c must be distinct from a,b. Also c != x is implied if c in S and x not in S.
                if a + c in S and b + c in S:
                    return False
            
    return True
