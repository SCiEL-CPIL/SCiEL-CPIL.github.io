"""Optimized solution for the "no element divides sum of 3 others" problem."""

import time
import random
import itertools

def construct():
    """
    Construct a set of distinct positive integers <= 192 (max 48) such that
    no selected integer divides the sum of any 3 distinct OTHER selected integers.
    Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    N = 192
    MAX_SIZE = 48
    
    # Precompute valid triples for each potential divisor x (1..192)
    # forbidden_triples[x] = set of (a,b,c) with a<b<c, all in 1..192, 
    #                       a!=x, b!=x, c!=x, and (a+b+c) % x == 0
    # We actually need to check if any triple from S (excluding x) is in this set.
    
    # Optimization: Instead of storing triples, for each x, store the set of 
    # forbidden pair sums? No, we need triples.
    # Let's store for each x, a set of triples as frozensets or sorted tuples.
    # But iterating all triples of 192 is C(192,3) = ~1.2M, times 192 x's = 230M. Too slow.
    
    # Alternative approach for validation:
    # To check if a candidate x can be added to set S:
    # 1. Check if x divides sum of any 3 elements in S.
    #    Iterate all triples in S. If any sum % x == 0, invalid.
    # 2. Check if any a in S divides (x + b + c) for distinct b,c in S\{a}.
    #    For each a in S, check if there exist b,c in S\{a} such that (x+b+c)%a==0.
    #    This is equivalent to: for each a in S, check if there exist b,c in S\{a}
    #    such that b+c ≡ -x (mod a).
    
    # Let's build a fast validator.
    
    def is_valid_set(S):
        """Check if set S satisfies the condition."""
        if len(S) < 4:
            return True
        
        S_list = sorted(S)
        n = len(S_list)
        
        for i in range(n):
            x = S_list[i]
            # Check 1: x does not divide sum of any 3 other elements
            others = [S_list[j] for j in range(n) if j != i]
            if len(others) >= 3:
                for a_idx in range(len(others)):
                    for b_idx in range(a_idx+1, len(others)):
                        for c_idx in range(b_idx+1, len(others)):
                            if (others[a_idx] + others[b_idx] + others[c_idx]) % x == 0:
                                return False
        
        return True
    
    def can_add(x, S):
        """Check if x can be added to S without violating constraints."""
        if x in S:
            return False
        
        S_list = sorted(S)
        n = len(S_list)
        
        # Check 1: x does not divide sum of any 3 elements in S
        if n >= 3:
            for a_idx in range(n):
                for b_idx in range(a_idx+1, n):
                    for c_idx in range(b_idx+1, n):
                        if (S_list[a_idx] + S_list[b_idx] + S_list[c_idx]) % x == 0:
                            return False
        
        # Check 2: For each a in S, a does not divide (x + b + c) for any distinct b,c in S\{a}
        for a in S_list:
            # Need b,c in S, b != a, c != a, b != c, such that (x + b + c) % a == 0
            # i.e., b + c ≡ -x (mod a)
            # Iterate pairs (b,c) from S\{a}
            S_without_a = [v for v in S_list if v != a]
            m = len(S_without_a)
            if m < 2:
                continue
            
            # For efficiency, we can check if there exist b,c in S_without_a 
            # such that (b + c) % a == (-x) % a
            target = (-x) % a
            
            found = False
            for i in range(m):
                for j in range(i+1, m):
                    if (S_without_a[i] + S_without_a[j]) % a == target:
                        found = True
                        break
                if found:
                    break
            
            if found:
                return False
        
        return True
    
    # Greedy construction starting from large numbers
    S = []
    candidates = list(range(N, 0, -1))
    
    for x in candidates:
        if time.time() > deadline:
            break
        if can_add(x, S):
            S.append(x)
    
    # Local search to try to increase size
    # Try adding any missing number
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v in range(1, N + 1):
            if time.time() > deadline:
                break
            if v not in S and can_add(v, S):
                S.append(v)
                improved = True
                break
    
    # Swap-based local search
    # Try to find swaps that allow adding more elements later
    # For now, just try random swaps to explore the space
    attempts = 0
    max_attempts = 5000
    
    while attempts < max_attempts and time.time() < deadline:
        attempts += 1
        
        if not S:
            break
        
        u_idx = random.randint(0, len(S) - 1)
        u = S[u_idx]
        
        v = random.randint(1, N)
        if v in S:
            continue
        
        S_temp = S[:]
        S_temp.pop(u_idx)
        
        if can_add(v, S_temp):
            # Check if this swap allows adding more elements
            # Try to add elements after the swap
            S_after_swap = S_temp[:]
            S_after_swap.append(v)
            
            added = 0
            for w in range(1, N + 1):
                if w not in S_after_swap and can_add(w, S_after_swap):
                    S_after_swap.append(w)
                    added += 1
            
            if len(S_after_swap) > len(S):
                S = S_after_swap
            else:
                # Keep the swap if it doesn't reduce size, might help later
                # Actually, let's only accept if it increases or maintains size
                # and we haven't filled the max
                if len(S_after_swap) >= len(S) and len(S) < MAX_SIZE:
                    S = S_after_swap
    
    # Final cleanup: remove any invalid elements (shouldn't happen but just in case)
    # Actually, our construction ensures validity, so we just return S
    
    # Ensure we don't exceed MAX_SIZE
    if len(S) > MAX_SIZE:
        S = S[:MAX_SIZE]
    
    return S
