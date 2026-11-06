import time
import random

def construct():
    start_time = time.time()
    time_limit = 115
    
    N = 1000
    max_sum = 2 * N
    
    in_set = [False] * (N + 1)
    counts = [0] * (max_sum + 1)
    S_list = []
    
    def add_element(x):
        in_set[x] = True
        S_list.append(x)
        # Update counts
        # We need to iterate over existing elements y <= x
        # S_list is not sorted.
        # We can just iterate all y in S_list
        for y in S_list:
            if y <= x:
                counts[x + y] += 1
            else:
                # y > x, so x + y is same as y + x
                # But we only added pairs (a,b) with a<=b.
                # When we added y (which was larger), did we add x+y?
                # No, because x was not in S.
                # So we must add x+y now.
                counts[x + y] += 1
        # Wait, if we iterate all y in S_list, we add x+y for all y.
        # This is correct because the pair is {x, y}.
        # The representation count is for the sum, regardless of order.
        # The constraint is on the number of pairs {a,b} with a<=b.
        # The pair is unique.
        # So adding x+y for all y in S is correct.
        
    def remove_element(x):
        in_set[x] = False
        S_list.remove(x)
        for y in S_list:
            counts[x + y] -= 1
        # Note: S_list no longer contains x.
        # So we don't double count 2x.
        # But when we added x, we added x+y for all y in S_list (which did not contain x).
        # So we added x+y for y != x.
        # Did we add 2x?
        # No, because x was not in S_list when we iterated.
        # So we never added 2x in the loop!
        # We must handle 2x separately.
        
    # Correction:
    # When adding x, the new pairs are {x, y} for all y in S.
    # This includes y=x? No, x is not in S yet.
    # So pairs are {x, y} for y in S.
    # If y=x, it's not in S.
    # So 2x is NOT formed by x and another element in S.
    # Wait, the representation is a+b with a<=b.
    # If we add x, can x be part of a pair with itself?
    # Yes, 2x.
    # But is 2x a new representation?
    # Yes, if x is in S.
    # But the pair is (x, x).
    # This pair exists only if x is in S.
    # When we add x, the pair (x, x) becomes available.
    # So we must increment counts[2x] by 1.
    # The loop `for y in S_list` does NOT include x.
    # So we must do `counts[2*x] += 1` explicitly.
    
    # Let's rewrite add/remove carefully.
    
    def add_element_correct(x):
        in_set[x] = True
        S_list.append(x)
        # Add pairs (x, y) for y in S (old S)
        for y in S_list[:-1]: # Exclude x
            counts[x + y] += 1
        # Add pair (x, x)
        counts[2 * x] += 1
        
    def remove_element_correct(x):
        in_set[x] = False
        S_list.remove(x)
        # Remove pairs (x, y) for y in S (new S, which is old S - x)
        for y in S_list:
            counts[x + y] -= 1
        # Remove pair (x, x)
        counts[2 * x] -= 1

    # This is correct.
    
    def is_valid(x):
        if in_set[x]:
            return False
        # Check 2x
        if counts[2 * x] >= 2:
            return False
        # Check x + y for y in S
        for y in S_list:
            if counts[x + y] >= 2:
                return False
        return True

    # Greedy
    for x in range(1, N + 1):
        if is_valid(x):
            add_element_correct(x)
            
    # Local Search
    while time.time() - start_time < time_limit:
        improved = False
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if not in_set[x] and is_valid(x):
                add_element_correct(x)
                improved = True
                break
        if improved:
            continue
            
        if not S_list:
            break
            
        u = random.choice(S_list)
        remove_element_correct(u)
        
        swap_improved = False
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for v in candidates:
            if not in_set[v] and is_valid(v):
                add_element_correct(v)
                swap_improved = True
                break
        if not swap_improved:
            add_element_correct(u)
            
    return [x for x in range(1, N + 1) if in_set[x]]
