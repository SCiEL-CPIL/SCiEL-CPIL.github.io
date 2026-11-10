import time
import sys
import random
import itertools
import numpy as np

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin for 90s limit

def construct():
    start = time.time()
    
    # The problem: Select up to 48 distinct integers from 1..192 such that
    # no selected integer divides the sum of any 3 DISTINCT OTHER selected integers.
    
    # Key Insight for Optimization:
    # If we choose numbers that are "large" and "co-prime-like" or simply large enough
    # that their multiples don't align with sums of other triples, we might do well.
    # Specifically, if x is in the set, x should not divide (a+b+c) for any distinct a,b,c in S\{x}.
    # Note that a+b+c is at most 192+191+190 = 573.
    # So if x > 573, it can never divide a sum. But x <= 192.
    # However, if x is large, say x > 288 (which is 192*1.5), then the only possible multiple is x itself.
    # But sum of 3 distinct others is >= 1+2+3 = 6 and <= 573.
    # If x is in (288, 192], that's empty. 
    # Wait, if x > 573/2 = 286.5, then 2x > 573. So the only multiple of x in range [6, 573] is x.
    # So we need to ensure that no 3 distinct others sum to x.
    # If x is very large, it's hard for 3 other numbers to sum to exactly x?
    # Actually, if we pick numbers from [97, 192], the sum of any 3 is at least 97+98+99 = 294.
    # And at most 190+191+192 = 573.
    # For any x in [97, 192], x < 294. So x cannot divide any sum of 3 others (since the smallest sum is 294 > 192*1).
    # Wait, does x divide a sum?
    # If sum > x, the smallest multiple of x greater than x is 2x.
    # 2x for x=192 is 384.
    # So if all elements are >= 97, the sum of any 3 is >= 294.
    # If x=192, 2x=384. 3x=576.
    # So we must avoid sums equal to 192, 384, 576.
    # But sums are >= 294. So only 384 and 576 are candidates for multiples of 192.
    # Actually, for any x in [97, 192]:
    # Multiples of x that can be sums of 3 numbers from [97, 192]:
    # Min sum = 97+98+99 = 294.
    # Max sum = 190+191+192 = 573.
    # So we only need to check if any sum equals k*x for k >= 2.
    # For x >= 97, 2x >= 194. 3x >= 291.
    # So possible multiples: 2x, 3x, 4x, 5x, 6x.
    # Since max sum is 573:
    # For x=192: 2x=384, 3x=576 (too big). So only 384.
    # For x=150: 2x=300, 3x=450.
    # For x=97: 2x=194 (too small? No, 194 < 294, so 2x is too small to be a sum of 3 numbers >= 97? 
    # Wait, if the set contains numbers < 97, then min sum is smaller.
    
    # Strategy: 
    # 1. Greedy: Start with the largest numbers.
    # 2. If we restrict to [97, 192], we have 96 candidates.
    #    Let's check if we can take ALL of them?
    #    If S = {97, ..., 192}.
    #    For any x in S, does x divide sum of 3 others?
    #    Sum of 3 others is in [294, 573].
    #    Multiples of x in this range:
    #    If x=192, 2x=384. Is there a triple summing to 384?
    #    97+98+189 = 384. Yes. So 192 divides 384? No, 384 % 192 == 0.
    #    So 192 cannot be in the set if {97,98,189} are also in the set.
    #    This suggests we can't just take all large numbers.
    
    # Let's implement a smart greedy + local search.
    
    # Fast validity check using bitmasks or sets.
    # Since N=48 is small, we can afford O(N^3) checks per addition if we optimize.
    # But for local search, we need many checks.
    
    def is_valid(S, c):
        """
        Check if c can be added to set S.
        S is a set of integers.
        1. c does not divide sum of any 3 distinct elements in S.
        2. For each a in S, a does not divide c + x + y for any distinct x, y in S \ {a}.
        """
        S_list = list(S)
        n = len(S_list)
        
        # Check 1: c | (a+b+c) for a,b,c in S, distinct.
        # Iterate triples in S.
        # Optimization: Precompute pair sums?
        # For small n, triple loop is fine.
        for i in range(n):
            for j in range(i+1, n):
                s2 = S_list[i] + S_list[j]
                for k in range(j+1, n):
                    if (s2 + S_list[k]) % c == 0:
                        return False
        
        # Check 2: a | (c + x + y) for x,y in S\{a}, distinct.
        # (x+y) % a == (-c) % a
        for idx, a in enumerate(S_list):
            target = (-c) % a
            # Find x, y in S \ {a} such that (x+y) % a == target
            # Iterate x in S \ {a}
            # We can use a set of residues? 
            # Since a <= 192, residues are small.
            # Let's just do double loop with early exit.
            found = False
            for i in range(n):
                if i == idx: continue
                x = S_list[i]
                rem_x = x % a
                # We need y % a == (target - rem_x) % a
                need = (target - rem_x) % a
                for j in range(n):
                    if j == idx or j == i: continue
                    if S_list[j] % a == need:
                        found = True
                        break
                if found:
                    break
            if found:
                return False
        
        return True

    def greedy(order):
        S = set()
        for c in order:
            if is_valid(S, c):
                S.add(c)
        return S

    best_set = set()
    
    # Phase 1: Multiple Greedy Strategies
    # 1. Descending
    S1 = greedy(range(192, 0, -1))
    if len(S1) > len(best_set):
        best_set = S1
        
    # 2. Ascending
    S2 = greedy(range(1, 193))
    if len(S2) > len(best_set):
        best_set = S2
        
    # 3. Randomized Greedy
    for _ in range(10):
        order = list(range(1, 193))
        random.shuffle(order)
        S3 = greedy(order)
        if len(S3) > len(best_set):
            best_set = S3

    # Phase 2: Local Search (Swap and Add)
    # We will try to improve the current best set.
    # Moves:
    # - Swap: Remove a, Add b.
    # - Add: Add b (if size < 48).
    
    def local_search(initial_set, time_limit):
        current = set(initial_set)
        start = time.time()
        stuck_count = 0
        
        while time.time() - start < time_limit:
            improved = False
            
            # Try to add
            if len(current) < 48:
                candidates = [x for x in range(1, 193) if x not in current]
                random.shuffle(candidates)
                for c in candidates:
                    if is_valid(current, c):
                        current.add(c)
                        improved = True
                        stuck_count = 0
                        break
            else:
                # Try to swap
                # Pick a random element to remove
                out_elem = random.choice(list(current))
                candidates = [x for x in range(1, 193) if x not in current]
                random.shuffle(candidates)
                for in_elem in candidates:
                    new_set = current - {out_elem} | {in_elem}
                    # Check validity of in_elem against new_set \ {in_elem}
                    # Which is current - {out_elem}
                    if is_valid(current - {out_elem}, in_elem):
                        current = new_set
                        improved = True
                        stuck_count = 0
                        break
                        
            if not improved:
                stuck_count += 1
                # If stuck for a while, do a random restart or perturbation?
                # For now, just break if stuck too much to save time for other restarts.
                if stuck_count > 5:
                    break
            else:
                stuck_count = 0
                
        return current

    # Run local search from the best greedy result
    if len(best_set) < 48:
        improved_set = local_search(best_set, time_limit=30)
        if len(improved_set) > len(best_set):
            best_set = improved_set
            
    # Run local search from a few random restarts if time permits
    if time.time() - start < 60:
        for _ in range(5):
            order = list(range(1, 193))
            random.shuffle(order)
            S_rand = greedy(order)
            improved_set = local_search(S_rand, time_limit=5)
            if len(improved_set) > len(best_set):
                best_set = improved_set
                if len(best_set) == 48:
                    break # Can't do better
    
    # Convert to list of ints
    return list(best_set)

# EVOLVE-BLOCK-END
