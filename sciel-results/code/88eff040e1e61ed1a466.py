"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd

def construct():
    # S is the set of selected integers
    S = []
    
    # We iterate in descending order to prioritize larger numbers
    # which are less likely to divide products of other numbers.
    for a in range(256, 1, -1):
        can_add = True
        
        # Check compatibility with all pairs in current S
        # If S has fewer than 2 elements, there are no pairs (b,c) distinct from a
        # Wait, if S has 1 element, say {b}. Adding a gives {a, b}.
        # Triples require 3 distinct elements. So no triple exists.
        # Condition is vacuously true.
        # So we can always add if |S| < 2.
        
        if len(S) >= 2:
            n = len(S)
            for i in range(n):
                b = S[i]
                if not can_add:
                    break
                # Precompute gcd(a, b)
                g_ab = gcd(a, b)
                
                # Check if a divides b*c for any c in S, c != b
                # Condition: a | b*c  <=>  (a // gcd(a,b)) | c
                k_a = a // g_ab
                if k_a != 1:
                    # Need to check all c != b
                    for j in range(n):
                        if j == i:
                            continue
                        c = S[j]
                        if c % k_a == 0:
                            can_add = False
                            break
                    if not can_add:
                        break
                else:
                    # a | b, so a | b*c for any c.
                    # Since we have at least 2 elements, there is a c != b.
                    # So conflict.
                    can_add = False
                    break

                # Check if b divides a*c for any c in S, c != b
                # Condition: b | a*c  <=>  (b // gcd(a,b)) | c
                k_b = b // g_ab
                if k_b != 1:
                    for j in range(n):
                        if j == i:
                            continue
                        c = S[j]
                        if c % k_b == 0:
                            can_add = False
                            break
                    if not can_add:
                        break
                else:
                    # b | a, so b | a*c for any c.
                    # Conflict.
                    can_add = False
                    break

                # Check if c divides a*b for any c in S, c != b
                # This is symmetric to "b divides a*c" but with roles swapped?
                # No, we need to check for ALL c in S (c != b) if c | a*b.
                # Wait, the loop above checked "b | a*c" for all c.
                # Now we need to check "c | a*b" for all c.
                # This is a different condition.
                # For a specific c, does c divide a*b?
                # Condition: c | a*b  <=>  (c // gcd(c, a)) | b  (using c and a)
                # Or (c // gcd(c, b)) | a
                # Let's check for each c.
                
                for j in range(n):
                    if j == i:
                        continue
                    c = S[j]
                    g_ac = gcd(a, c)
                    k_c = c // g_ac
                    if b % k_c == 0:
                        can_add = False
                        break
                    if not can_add:
                        break
                if not can_add:
                    break
            
        if can_add:
            S.append(a)
            
    # Ensure return type is list of ints
    return [int(x) for x in S]

# EVOLVE-BLOCK-END
