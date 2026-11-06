"""Rewritten program with a stronger heuristic for the 'no element divides sum of 3 distinct others' task."""

def construct():
    """
    Returns a list of distinct positive integers <= 192, maximizing cardinality,
    such that no selected integer divides the sum of any 3 distinct OTHER selected integers.
    """
    n = 192
    # We want to maximize the number of selected elements.
    # The condition is: for all a in S, and for all distinct b, c, d in S \ {a},
    # a does not divide (b + c + d).
    
    # Strategy:
    # 1. Large numbers are safer because they are less likely to divide the sum of three other numbers
    #    unless the sum is a multiple of the large number. Since the max sum of three numbers <= 192 is
    #    192+191+190 = 573, any number > 573/2 = 286.5 would never divide a sum of 3 smaller numbers 
    #    (since the quotient would be 0 or 1, and if 1, sum==a, which implies b+c+d=a, possible but rare).
    #    However, our numbers are <= 192. 
    #    If a > 573, impossible. 
    #    If a is in (192, 573], the only possible multiple is a itself. So we just need b+c+d != a.
    #    If a <= 192, multiples can be a, 2a, 3a, ... up to 573.
    
    # Greedy approach:
    # Sort candidates in descending order. Larger numbers are generally "safer" to include first
    # because they impose stricter constraints on the sum being a multiple, but actually,
    # including large numbers makes it harder for SMALL numbers to divide their sums?
    # No, the constraint is on the SELECTED number 'a'.
    # If we pick 'a', we must ensure no 3 other selected numbers sum to a multiple of 'a'.
    # Small 'a' (like 2, 3) divide many sums. Large 'a' (like 192) divides few sums (only 192, 384, 576...).
    # So we should prefer LARGE numbers.
    
    # Let's try a greedy set construction in descending order.
    # For each candidate x from 192 down to 1:
    #   Check if adding x to the current set S violates the condition for ANY element in S U {x}.
    #   The condition for an existing element a in S:
    #     Previously, no 3 distinct elements in S summed to a multiple of a.
    #     Now, with x added, new triples involving x must not sum to a multiple of a.
    #     Triples involving x: (x, b, c) where b, c in S, b!=c, b!=x, c!=x.
    #     So we check: for all b, c in S with b<c, (x + b + c) % a != 0.
    #   The condition for the new element x:
    #     No 3 distinct elements in S (which are all other than x) sum to a multiple of x.
    #     So we check: for all distinct b, c, d in S, (b + c + d) % x != 0.
    
    # This greedy check is O(|S|^3) for the new element and O(|S|^2) for existing elements.
    # With |S| up to 48, this is 48^3 ~ 110k operations per candidate, times 192 candidates ~ 20M ops.
    # This is fast enough in Python if optimized slightly or just using simple loops.
    
    S = []
    
    for x in range(192, 0, -1):
        # Check if x can be added to S
        valid = True
        
        # 1. Check constraints imposed by x on the existing set S
        # No 3 distinct elements in S should sum to a multiple of x
        if len(S) >= 3:
            # Optimization: precompute sums? No, just iterate.
            # Iterate over combinations of 3 from S
            # Since S is small, we can use nested loops for speed
            nS = len(S)
            for i in range(nS):
                for j in range(i + 1, nS):
                    for k in range(j + 1, nS):
                        s = S[i] + S[j] + S[k]
                        if s % x == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
        
        if valid:
            # 2. Check constraints imposed by existing elements in S on the new triplets involving x
            # For each a in S, no pair (b, c) in S such that (x + b + c) % a == 0
            nS = len(S)
            for idx_a, a in enumerate(S):
                for i in range(nS):
                    if i == idx_a:
                        continue
                    for j in range(i + 1, nS):
                        if j == idx_a:
                            continue
                        # b = S[i], c = S[j]
                        # Ensure b and c are distinct from a (handled by index check)
                        # Ensure b != c (handled by i < j)
                        s = x + S[i] + S[j]
                        if s % a == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
        
        if valid:
            S.append(x)
            
    return S

# EVOLVE-BLOCK-START
# The construct function above is the main logic.
# We define it here to match the expected interface.
# EVOLVE-BLOCK-END
