"""Rewritten program using a greedy construction based on 2-adic valuation classes."""
import math

# EVOLVE-BLOCK-START
def construct():
    N = 256
    
    # We want a subset S of [1, 256] such that for all distinct a, b in S:
    # (a * b) % (a + b) != 0
    
    # Key insight:
    # If a and b are both odd, then a+b is even and a*b is odd.
    # An even number cannot divide an odd number (unless the divisor is 1, but a+b >= 1+3=4 or similar for distinct odds >=1).
    # Actually, if a=1, b=3, a+b=4, a*b=3, 3%4 != 0.
    # If a=1, b=1 is not distinct.
    # So any set of distinct odd numbers is safe internally.
    
    # What about even numbers?
    # If a and b are both even, let a=2x, b=2y.
    # a+b = 2(x+y), a*b = 4xy.
    # We require 2(x+y) does not divide 4xy.
    # i.e., x+y does not divide 2xy.
    # This is not automatically safe. For example, a=2, b=4.
    # a+b=6, a*b=8. 8%6 = 2 != 0. Safe.
    # a=2, b=6. a+b=8, a*b=12. 12%8 = 4 != 0. Safe.
    # a=4, b=8. a+b=12, a*b=32. 32%12 = 8 != 0. Safe.
    # a=6, b=12. a+b=18, a*b=72. 72%18 = 0. Conflict!
    # So even numbers can conflict with each other.
    
    # If a is odd and b is even:
    # Let a=2x+1, b=2y.
    # a+b is odd. a*b is even.
    # An odd number can divide an even number.
    # Example: a=1, b=2. a+b=3, a*b=2. 2%3 != 0.
    # a=1, b=6. a+b=7, a*b=6. 6%7 != 0.
    # a=3, b=6. a+b=9, a*b=18. 18%9 = 0. Conflict!
    # a=1, b=24. a+b=25, a*b=24. 24%25 != 0.
    # a=5, b=10. a+b=15, a*b=50. 50%15 = 5 != 0.
    # a=3, b=12. a+b=15, a*b=36. 36%15 = 6 != 0.
    # a=9, b=18. a+b=27, a*b=162. 162%27 = 0. Conflict!
    # Pattern: If b = 2a, then a+b = 3a, a*b = 2a^2.
    # 2a^2 % 3a = a(2a) % 3a.
    # If a is multiple of 3, say a=3k, b=6k.
    # a+b = 9k, a*b = 18k^2.
    # 18k^2 % 9k = 0.
    # So if a is a multiple of 3 and b=2a, they conflict.
    # Also if a=1, b=2? 2%3 != 0.
    # If a=2, b=4 (both even), we checked safe.
    # If a=4, b=8 (both even), safe.
    # If a=6, b=12 (both even), conflict.
    # Note 6 is even, 12 is even. 6=2*3, 12=4*3.
    
    # Strategy:
    # 1. Include all odd numbers. (Size 128)
    # 2. Add even numbers that do not conflict with any odd number in the set.
    # 3. Among the added even numbers, ensure they don't conflict with each other.
    
    odds = [i for i in range(1, N+1) if i % 2 != 0]
    
    def conflicts(a, b):
        if a == b:
            return False
        return (a * b) % (a + b) == 0
    
    # Candidate evens
    evens = [i for i in range(2, N+1, 2)]
    
    safe_evens = []
    for e in evens:
        ok = True
        for o in odds:
            if conflicts(e, o):
                ok = False
                break
        if ok:
            safe_evens.append(e)
    
    # Now, we have a set of 'safe_evens' that are compatible with all odds.
    # But they might conflict with each other.
    # We need to select a maximum subset of safe_evens such that no two conflict.
    # This is a Maximum Independent Set problem on the graph of conflicts among safe_evens.
    # The number of safe_evens is likely small enough for a greedy approach or even brute force if it's very small.
    # Let's check the size first.
    
    # Greedy: sort by degree (number of conflicts with other safe_evens) and pick lowest degree first?
    # Or just iterate and add if no conflict with already added ones.
    
    selected_evens = []
    for e in sorted(safe_evens):
        ok = True
        for se in selected_evens:
            if conflicts(e, se):
                ok = False
                break
        if ok:
            selected_evens.append(e)
    
    final_set = odds + selected_evens
    
    # Verify
    for i in range(len(final_set)):
        for j in range(i+1, len(final_set)):
            a = final_set[i]
            b = final_set[j]
            if (a * b) % (a + b) == 0:
                # Fallback if greedy failed
                # This should not happen if logic is correct
                pass
                
    return final_set
# EVOLVE-BLOCK-END
