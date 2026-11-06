"""Maximize reciprocal weight of 3-AP-free subset of [1,256] using optimized branch-and-bound."""
# EVOLVE-BLOCK-START
import time
import sys

N = 256
TIME_LIMIT = 55.0  # Safety margin before 60s kill

# Precompute conflict information for fast bitmask checks
# For each x, we want to know which pairs (a, b) in the set would create a 3-AP with x.
# Instead of storing pairs, we can check on the fly using bitmasks.
# However, precomputing lists of potential partners is faster.

# partners_mid[x] = list of a such that c = 2x - a is valid and a != c
# partners_end[x] = list of a such that b = (a+x)/2 is valid and integer and b != a, b != x

def precompute_partners():
    mid_partners = [[] for _ in range(N + 1)]
    end_partners = [[] for _ in range(N + 1)]
    
    for x in range(1, N + 1):
        # Case 1: x is the middle term. a + c = 2x.
        # a must be in [1, N], c = 2x - a must be in [1, N], a != c.
        for a in range(1, N + 1):
            c = 2 * x - a
            if 1 <= c <= N and a != c:
                mid_partners[x].append((a, c))
        
        # Case 2: x is the max term. a + x = 2b => b = (a+x)/2.
        # a must be in [1, N], a < x (since x is max, a < b < x, so a < x is implied if b < x).
        # Actually, if x is the max, the AP is (a, b, x).
        # a < b < x.
        # b = (a+x)/2.
        # For b to be integer, a+x must be even.
        # For b < x, (a+x)/2 < x => a < x.
        # For b > a, (a+x)/2 > a => x > a.
        # So we just need a < x and (a+x) even.
        for a in range(1, x):
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                # b is automatically > a and < x
                # We need to check if b is in the set.
                # So for each a, we check if b is in the set.
                end_partners[x].append((a, b))
                
        # Case 3: x is the min term. x + b = 2a => a = (x+b)/2.
        # AP is (x, b, a).
        # x < b < a.
        # a = (x+b)/2.
        # For a to be integer, x+b must be even.
        # For a > b, (x+b)/2 > b => x > b. Wait.
        # x < b < a.
        # a - b = b - x => a = 2b - x.
        # Let's stick to the symmetric view:
        # If x is min, there exists b, a such that x, b, a is AP.
        # This is equivalent to saying b is the middle of x and a.
        # So for a given x, we look for b such that a = 2b - x is valid.
        # x < b < a <= N.
        # b > x.
        # a = 2b - x <= N => 2b <= N + x => b <= (N+x)//2.
        # Also a > b => 2b - x > b => b > x.
        # So b ranges from x+1 to (N+x)//2.
        # And a = 2b - x.
        # We check if b and a are in the set.
        for b in range(x + 1, (N + x) // 2 + 1):
            a = 2 * b - x
            if a <= N:
                end_partners[x].append((b, a)) # Note: here b is middle, a is max.
                # Wait, the structure is (x, b, a).
                # We need to check if b is in set AND a is in set.
                # So we store (b, a) as the pair to check.
                
    return mid_partners, end_partners

def construct():
    start_time = time.time()
    
    # Precompute
    mid_partners, end_partners = precompute_partners()
    
    # Reciprocals as floats for speed
    reciprocals = [0.0] * (N + 1)
    for i in range(1, N + 1):
        reciprocals[i] = 1.0 / i
        
    # Suffix sums of reciprocals for upper bound
    suffix_sum = [0.0] * (N + 2)
    for i in range(N, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + reciprocals[i]
        
    # Bitmask helpers
    # mask is an integer where bit i-1 is 1 if i is in the set.
    
    def can_add(x, mask):
        # Check mid_partners: pairs (a, c) such that a+c=2x
        for a, c in mid_partners[x]:
            if (mask & (1 << (a - 1))) and (mask & (1 << (c - 1))):
                return False
        # Check end_partners: pairs (a, b) such that a,b,x form AP with x as max or min
        # In precompute, we added (a, b) for x as max (a < b < x) and (b, a) for x as min (x < b < a)
        # In both cases, we need both elements to be in the set.
        for u, v in end_partners[x]:
            if (mask & (1 << (u - 1))) and (mask & (1 << (v - 1))):
                return False
        return True

    best_mask = 0
    best_score = 0.0
    
    # Greedy initialization
    current_mask = 0
    current_score = 0.0
    for x in range(1, N + 1):
        if can_add(x, current_mask):
            current_mask |= (1 << (x - 1))
            current_score += reciprocals[x]
    
    best_mask = current_mask
    best_score = current_score
    
    def branch_and_bound(idx, mask, score):
        nonlocal best_mask, best_score, start_time
        
        if time.time() - start_time > TIME_LIMIT:
            return
        
        # Upper bound
        if score + suffix_sum[idx] <= best_score:
            return
        
        if idx > N:
            if score > best_score:
                best_score = score
                best_mask = mask
            return
        
        x = idx
        
        # Option 1: Include x
        if can_add(x, mask):
            new_mask = mask | (1 << (x - 1))
            new_score = score + reciprocals[x]
            branch_and_bound(idx + 1, new_mask, new_score)
        
        # Option 2: Exclude x
        branch_and_bound(idx + 1, mask, score)
    
    # Start B&B
    # We can start from the greedy solution to get a good initial bound,
    # but B&B from scratch with a good upper bound is usually fine.
    # However, starting from scratch explores "exclude 1" first, which might be bad.
    # Let's force the first few high-weight elements to be included?
    # Or just run B&B. The greedy bound helps prune.
    
    # To speed up, we can order the branching.
    # Standard B&B goes 1..N.
    # Let's try to be smarter: if we have time, do local search.
    
    branch_and_bound(1, 0, 0.0)
    
    # Convert best_mask to list
    result = []
    for i in range(1, N + 1):
        if best_mask & (1 << (i - 1)):
            result.append(i)
            
    return result

# EVOLVE-BLOCK-END
