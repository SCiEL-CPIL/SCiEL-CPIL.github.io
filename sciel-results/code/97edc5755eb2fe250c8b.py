import time
from math import lcm, gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    # Initial set: 2 and all odd numbers in [3, 256]
    # This is the result of the seed program's greedy logic
    S = [2] + [x for x in range(3, 257) if x % 2 != 0]
    
    def is_valid_set(S):
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if lcm(S[i], S[j]) <= 256:
                    return False
        return True

    def get_score(S):
        return sum(1.0/x for x in S)

    # Verify initial set
    if not is_valid_set(S):
        # Fallback to simple greedy if initial fails (should not happen)
        S = []
        for x in range(2, 257):
            if all(lcm(x,y) > 256 for y in S):
                S.append(x)
    
    best_S = S
    best_score = get_score(S)
    
    # Local Search: Single swaps
    # Try to swap an element b in S with an element a not in S
    # To speed up, we can precompute compatibility
    
    # Optimization: 
    # If we remove b, the set is S\{b}. 
    # We can add a if a is compatible with all x in S\{b}.
    
    # Let's iterate over a not in S, and find best b to remove?
    # Or iterate over b in S, and find best a to add?
    
    # Since S is large (~129 elements), and candidates are ~127, 
    # 129 * 127 checks is ~16k. Each check is O(|S|) = 129.
    # 16k * 129 = 2M operations. Fast.
    
    improved = True
    while improved:
        improved = False
        for b in list(S):
            # Try removing b
            S_minus_b = [x for x in S if x != b]
            # Try adding a
            for a in range(2, 257):
                if a in S_minus_b:
                    continue
                # Check if a is compatible with all in S_minus_b
                compatible = True
                for x in S_minus_b:
                    if lcm(a, x) <= 256:
                        compatible = False
                        break
                if compatible:
                    new_S = S_minus_b + [a]
                    new_score = get_score(new_S)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_S = new_S
                        S = new_S
                        improved = True
                        # Restart loop with new S
                        break
            if improved:
                break
        
        if time.time() > deadline:
            break

    # Try adding elements (if S is not maximal)
    # The seed set is maximal? 
    # With 2, no evens can be added. 
    # Can any odd be added? 
    # All odds > 128 are in S. 
    # Odds < 129? 
    # If we add odd a < 129, we must check compatibility with 2 and other odds.
    # lcm(2, a) = 2a. Need 2a > 256 => a > 128. 
    # So no odd < 129 can be added with 2.
    # So S is maximal with respect to additions if 2 is in S.
    
    # What if we remove 2?
    # Then we can add evens.
    # This is a bigger move. 
    # Let's try swapping 2 with a set of evens?
    # Or just doing the swap search above handles it if we allow removing 2.
    # The loop above iterates b in S. If b=2, it tries to add a.
    # If a is even, it checks compatibility with S\{2} (which are odds).
    # lcm(even, odd) is even*odd.
    # If even=4, odd=129. lcm=516 > 256.
    # So we can add 4 if we remove 2.
    # But we might need to remove multiple odds to add 4?
    # The single swap only adds 1 element.
    # If adding 4 requires removing multiple odds, single swap won't find it.
    
    # Let's add a heuristic: 
    # Try removing 2, and then greedily adding best evens?
    
    # Alternative: 
    # Run the local search, but also try starting from other configurations.
    
    # Configuration 2: No 2. 
    # Start with [129, 256]. 
    # Then local search.
    
    S2 = list(range(129, 257))
    if is_valid_set(S2):
        score2 = get_score(S2)
        if score2 > best_score:
            best_score = score2
            best_S = S2
        
        # Local search on S2
        improved = True
        while improved:
            improved = False
            for b in list(S2):
                S_minus_b = [x for x in S2 if x != b]
                for a in range(2, 257):
                    if a in S_minus_b:
                        continue
                    compatible = True
                    for x in S_minus_b:
                        if lcm(a, x) <= 256:
                            compatible = False
                            break
                    if compatible:
                        new_S = S_minus_b + [a]
                        new_score = get_score(new_S)
                        if new_score > best_score + 1e-10:
                            best_score = new_score
                            best_S = new_S
                            S2 = new_S
                            improved = True
                            break
                if improved:
                    break
            if time.time() > deadline:
                break

    return best_S
