"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    n = 256
    
    # Precompute conflict pairs
    # conf_pairs[x] is a list of pairs (u, v) such that {x, u, v} forms an invalid triple.
    # Specifically, if x is the smallest, (u, v) are the larger two.
    # If x is the middle, (u, v) are the smallest and largest.
    # If x is the largest, (u, v) are the two smallest.
    # The condition is: for sorted a < b <= c, a does NOT divide b+c.
    # So a triple {x, u, v} is invalid if, when sorted as a<b<=c, a | (b+c).
    
    conf_pairs = [[] for _ in range(n + 1)]
    
    # Iterate through all triples a < b <= c
    # This is O(N^3) which is ~16M for N=256.
    # We can optimize by iterating a, then b, then c.
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            # b+c must be a multiple of a
            # min b+c = b + b = 2b
            # max b+c = n + n = 2n
            # k*a must be in [2b, 2n]
            # k >= ceil(2b/a)
            # k <= floor(2n/a)
            
            k_min = (2 * b + a - 1) // a
            k_max = (2 * n) // a
            
            for k in range(k_min, k_max + 1):
                s = k * a
                c = s - b
                if c < b:
                    continue
                if c > n:
                    break
                # We have an invalid triple (a, b, c)
                # Add to conf_pairs for a, b, c
                conf_pairs[a].append((b, c))
                conf_pairs[b].append((a, c))
                conf_pairs[c].append((a, b))
                
    # Note: This might add duplicates if there are multiple k's?
    # No, for fixed a, b, c, the sum b+c is fixed, so k is fixed.
    # But different (a,b,c) triples are distinct.
    # However, for a specific x, the pair (u,v) might appear multiple times?
    # No, {x,u,v} is a set of 3 distinct numbers (since a<b<=c implies distinct a,b,c? 
    # Wait, b<=c allows b=c. 
    # If b=c, then a<b=c. Triple is (a,b,b).
    # My loop: b in a+1..n, c in b..n.
    # If b=c, it is included.
    # Are the numbers distinct? a < b. So a, b, c are distinct if b<c.
    # If b=c, a, b, b. The set is {a, b}. 
    # The condition is "subset of [1,n]". Sets don't have duplicates.
    # So the triple is effectively defined by the set of elements.
    # If b=c, the element b is just one element in the set.
    # So the "pair" for a is (b, b)? No, the pair is the other two elements in the set.
    # If the set is {a, b}, and b=c, then the triple is a, b, b.
    # But b is only in the set once.
    # The condition "for every a<b<=c in the set" implies we pick 3 elements from the set.
    # If the set contains a and b, and we pick a, b, b, is that allowed?
    # "a<b<=c in the set". Usually this means we choose indices or values.
    # If the set is a set of values, we can't pick b twice.
    # However, the problem statement says "Repeated larger summands b=c are included".
    # This implies that we consider the multiset of values? Or just that the condition $b \le c$ allows $b=c$.
    # If the set is $S$, and we pick $a, b, c \in S$ such that $a < b \le c$.
    # If $S$ is a set of unique integers, we can only pick $b=c$ if $b$ is in $S$ and we use it twice?
    # No, "in the set" usually means the values are present.
    # If I have $\{1, 2\}$, can I form $1, 2, 2$?
    # If the condition is on the *values* chosen from the set, and we can choose the same value twice if it's in the set?
    # "subset of [1,n]". A subset contains unique elements.
    # "a<b<=c in the set". This phrasing is ambiguous.
    # If it means $a, b, c$ are elements of the set, then $b=c$ implies $b$ is an element.
    # But a set cannot have duplicate elements.
    # Unless "in the set" refers to the *values* being available, and we can pick them with replacement?
    # "Repeated larger summands b=c are included" strongly suggests that we CAN have $b=c$.
    # This implies the "set" is actually a multiset, or the condition allows using the same element twice?
    # If it allows using the same element twice, then for any $b \in S$, the pair $(b,b)$ is a valid candidate for sum.
    # So if $a \in S$ and $b \in S$ with $a < b$, we must check if $a | (2b)$.
    # This matches my previous analysis: "Sub-case: c = x: a < x, c=x. Check a | (x+x) = 2x."
    
    # So, for the conflict graph:
    # If $b=c$, the "pair" for $a$ is $(b,b)$.
    # In my `conf_pairs` list, I should store `(b, b)` if $b=c$.
    # And for $b$, the pair is $(a, b)$.
    # And for $c$ (which is $b$), the pair is $(a, b)$.
    # So `conf_pairs[b]` gets `(a, b)` twice?
    # Or just once?
    # If I check `count_active_pairs(b, S)`, I iterate `conf_pairs[b]`.
    # If `(a, b)` is in the list, I check if $a \in S$ and $b \in S$.
    # Since $b \in S$ is true (we are checking conflict for $b$ which is in S? No, we check when adding/removing).
    # If I am adding $x$, I check pairs in `conf_pairs[x]`.
    # If $x=b$, and the pair is $(a, b)$, I check if $a \in S$ and $b \in S$.
    # But $b$ is the element being added. It is not in $S$ yet.
    # So the check "b in S" would be false.
    # This is a bug in the logic if we use "both in S".
    
    # Correct Logic for Adding x:
    # We want to know if adding x creates a conflict.
    # A conflict exists if there is a triple involving x and two other elements in S (or x itself if b=c).
    # Case 1: x is smallest (a=x). Pairs (b,c) with b,c in S, x < b <= c.
    #   Check if (b,c) in conf_pairs[x] AND b in S AND c in S.
    # Case 2: x is middle (b=x). Pairs (a,c) with a,c in S, a < x <= c.
    #   If c=x, pair is (a,x). Check if a in S AND x in S? No, x is being added.
    #   Wait, if c=x, the triple is a, x, x.
    #   This requires a in S. The "pair" is (a, x).
    #   But x is not in S yet.
    #   So we should check if a in S.
    #   My `conf_pairs[x]` for the case where x is the middle element (b=x, c=x) would contain (a, x).
    #   If I check "a in S and x in S", it fails because x not in S.
    #   So I need to handle the case where one of the pair elements is x itself.
    
    # Better approach:
    # Precompute `conflict_with_x[x]`: a set of "requirements".
    # Actually, let's just stick to the O(N^2) check in `check_add` but optimize it with NumPy.
    # Or, fix the SA logic.
    
    # Let's restart with a simpler, robust Local Search.
    
    def check_add(x, S):
        """
        Returns the set of elements in S that conflict with x.
        If x is added to S, all these elements must be removed to maintain validity.
        """
        S_set = S # Assume S is a set
        conflicting = set()
        
        # 1. x is smallest: x < b <= c, b,c in S, x | (b+c)
        # b+c is multiple of x.
        # min b+c = (x+1) + (x+1) = 2x+2
        # max b+c = 2n
        for k in range(3, 2 * n // x + 1): # k >= 3 because b,c > x => b+c > 2x
            s = k * x
            if s > 2 * n:
                break
            # Find b, c in S such that b+c=s, x < b <= c
            # b must be in [x+1, s//2]
            # c = s - b
            # We can iterate b in S? Or iterate b in range?
            # S is a set.
            # For efficiency, iterate b in range(x+1, s//2 + 1):
            #   if b in S and s-b in S and s-b >= b:
            #       conflicting.add(b); conflicting.add(s-b)
            # This is O(N) per k. Total O(N^2/x).
            for b in range(x + 1, s // 2 + 1):
                c = s - b
                if c in S_set and c >= b:
                    conflicting.add(b)
                    conflicting.add(c)
                    # We can break early? No, need all conflicts.
                    
        # 2. x is middle or largest
        # a < x <= c, a,c in S, a | (x+c)
        # Sub-case 2a: c = x. a | (2x). a in S, a < x.
        for a in S_set:
            if a < x and (2 * x) % a == 0:
                conflicting.add(a)
                
        # Sub-case 2b: c > x. a < x < c. a | (x+c).
        # For each a in S, a < x:
        #   x+c is multiple of a.
        #   c = k*a - x.
        #   c > x => k*a > 2x => k > 2x/a.
        #   c <= n.
        for a in S_set:
            if a >= x:
                continue
            k_min = (2 * x) // a + 1
            for k in range(k_min, n // a + 1):
                c = k * a - x
                if c > n:
                    break
                if c > x and c in S_set:
                    conflicting.add(a)
                    conflicting.add(c)
                    
        # 3. x is largest: a < b < x. a | (b+x).
        # For each a in S, a < x:
        #   b+x is multiple of a.
        #   b = k*a - x.
        #   a < b < x.
        for a in S_set:
            if a >= x:
                continue
            # b < x => k*a < 2x => k < 2x/a
            # b > a => k*a > 2a => k > 2
            k_min = 3
            k_max = (2 * x) // a
            for k in range(k_min, k_max + 1):
                b = k * a - x
                if b > x:
                    break
                if b > a and b in S_set:
                    conflicting.add(a)
                    conflicting.add(b)
                    
        return conflicting

    # Initial set: Upper tail
    # Find largest L such that [L, 256] is valid.
    # We know 171 works.
    best_set = set(range(171, 257))
    best_size = len(best_set)
    
    # Local Search
    # Try to add elements from 1 to 170
    # If adding x causes conflicts C, and len(C) == 0, add x.
    # If len(C) > 0, we don't add it (net loss).
    # BUT, we can try to remove elements from the tail to make room.
    # This is a 1-for-k swap.
    
    # Let's do a more sophisticated search:
    # 1. Start with [171, 256].
    # 2. Iterate x from 1 to 170.
    # 3. Try adding x.
    # 4. If conflicts, try to find a minimal set of conflicts to remove?
    #    Actually, if we remove ALL conflicts, size decreases.
    #    So we only add if conflicts are empty.
    
    # To improve, we can try to REMOVE some tail elements first, then add x.
    # This is complex.
    
    # Alternative: Random Restart Greedy
    # 1. Randomize order of [1, 256].
    # 2. Greedily add if valid.
    
    for restart in range(100):
        if time.time() > deadline - 5:
            break
        random.seed(restart)
        order = list(range(1, 257))
        random.shuffle(order)
        current_set = set()
        for x in order:
            conf = check_add(x, current_set)
            if not conf:
                current_set.add(x)
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(best_set)
            
    # Local Search on best_set
    # Try to add any x not in best_set
    improved = True
    while improved:
        improved = False
        for x in range(1, 257):
            if x in best_set:
                continue
            conf = check_add(x, best_set)
            if not conf:
                best_set.add(x)
                improved = True
                best_size += 1
    
    return list(best_set)
# EVOLVE-BLOCK-END
