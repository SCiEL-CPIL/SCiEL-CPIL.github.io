"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin within 90s limit
    
    n = 256
    # We want to maximize |S| where S ⊆ [1,n] and for all a < b <= c in S, a ∤ (b+c)
    
    # Strategy: Greedy from top down with local improvement
    # Start with a large set and try to add elements while maintaining validity
    
    def is_valid_set(S):
        """Check if set S satisfies the property."""
        S = list(S)
        S.sort()
        for i, a in enumerate(S):
            for j in range(i+1, len(S)):
                b = S[j]
                for k in range(j, len(S)):
                    c = S[k]
                    if a < b <= c and (b + c) % a == 0:
                        return False
        return True
    
    def check_pairwise_validity(S, new_elem):
        """Check if adding new_elem to S maintains validity."""
        S = list(S) + [new_elem]
        S.sort()
        # Only need to check triples involving new_elem
        for i, a in enumerate(S):
            if a == new_elem:
                # new_elem is 'a', need b > a, c >= b
                for j in range(i+1, len(S)):
                    b = S[j]
                    for k in range(j, len(S)):
                        c = S[k]
                        if (b + c) % a == 0:
                            return False
            elif a < new_elem:
                # new_elem is b or c
                for j in range(i+1, len(S)):
                    b = S[j]
                    if b == new_elem:
                        # new_elem is b, c >= b
                        for k in range(j, len(S)):
                            c = S[k]
                            if (b + c) % a == 0:
                                return False
                    elif b > new_elem:
                        # new_elem could be c (since c >= b > new_elem is false, so new_elem < b)
                        # Actually if a < new_elem < b, then new_elem can't be b or c in a < b <= c
                        # Wait, we need a < b <= c. If new_elem < b, it's not b or c.
                        pass
                # Also check cases where new_elem is c: a < b <= new_elem
                for j in range(i+1, len(S)):
                    b = S[j]
                    if b <= new_elem:
                        if (b + new_elem) % a == 0:
                            return False
                    else:
                        break
        return True
    
    # Better approach: Use a different strategy
    # The condition is: for all a < b <= c in S, a does not divide (b+c)
    
    # Let's think about this differently. If we take numbers from the upper half,
    # say [n/2+1, n], then for a < b <= c, we have a >= n/2+1, b >= n/2+2, c >= n/2+2
    # b+c >= 2*(n/2+2) = n+4 > n >= a, so b+c > a. But a could still divide b+c if b+c = 2a, 3a, etc.
    
    # For a in [n/2+1, n], 2a > n, so b+c < 2n, and 2a could be in range if b+c = 2a.
    # Actually b+c can be up to 2n, and 2a is in [n+2, 2n], so it's possible.
    
    # Let me try a different approach: start with the top half and greedily add elements.
    
    # Actually, let's use a smarter initialization.
    # Take all odd numbers from 1 to n? No, that doesn't work either.
    
    # Let's try: take numbers from [floor(n/2)+1, n] = [129, 256]
    # That gives 128 elements.
    
    # Can we do better? Let's try to include some smaller numbers.
    
    # Key insight: if a is small, then b+c must not be a multiple of a.
    # For a=1, b+c must not be divisible by 1, which is impossible. So 1 cannot be in S.
    # For a=2, b+c must be odd, so b and c must have different parity.
    
    # Let's try a greedy approach: start empty, add numbers from n down to 1 if possible.
    
    S = []
    
    for candidate in range(n, 0, -1):
        if time.time() > deadline:
            break
        # Check if we can add candidate to S
        can_add = True
        for a in S:
            if a < candidate:
                # candidate could be b or c
                # Case 1: candidate is b, so need c >= candidate in S
                for c in S:
                    if c >= candidate and (candidate + c) % a == 0:
                        can_add = False
                        break
                if not can_add:
                    break
                # Case 2: candidate is c, so need b with a < b <= candidate
                for b in S:
                    if a < b <= candidate and (b + candidate) % a == 0:
                        can_add = False
                        break
                if not can_add:
                    break
            elif candidate < a:
                # a > candidate, so candidate could be b or c with a as the larger? No, a < b <= c
                # If candidate < a, then candidate can't be b or c since we need a < b <= c
                # So no constraint here
                pass
            # If a == candidate, skip (set has unique elements)
        
        if can_add:
            S.append(candidate)
    
    # Now try local improvement: swap elements or add/remove
    # Try to add more elements by removing some
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding each missing element
        for candidate in range(n, 0, -1):
            if candidate in S:
                continue
            if time.time() > deadline:
                break
            # Check if candidate can be added
            can_add = True
            for a in S:
                if a < candidate:
                    for c in S:
                        if c >= candidate and (candidate + c) % a == 0:
                            can_add = False
                            break
                    if not can_add:
                        break
                    for b in S:
                        if a < b <= candidate and (b + candidate) % a == 0:
                            can_add = False
                            break
                    if not can_add:
                        break
            if can_add:
                S.append(candidate)
                improved = True
                break
    
    return S

# EVOLVE-BLOCK-END
