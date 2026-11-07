import time
import random

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    N = 128
    # Start with the upper half, which is always valid
    # Sums of two elements > 64 are > 128, so not in S.
    S = set(range(65, N + 1))
    
    def is_valid_addition(S, x):
        """
        Check if S U {x} is valid.
        We only need to check triples involving x.
        Triples (x, a, b) with a, b in S, a != b.
        Condition for violation: (x+a in S) and (x+b in S) and (a+b in S).
        Note: x+a cannot be x, so x+a must be in S.
        """
        S_list = list(S)
        n = len(S_list)
        
        # Optimization: Precompute which elements in S can pair with x
        # x+a in S => a in S-x
        # x+b in S => b in S-x
        # So a and b must be in (S-x) intersection S
        # And a+b must be in S
        
        candidates = []
        for a in S_list:
            if x + a in S:
                candidates.append(a)
                
        # Now check pairs in candidates
        # If there is a pair (a, b) in candidates such that a+b in S, violation.
        # Note a != b.
        for i in range(len(candidates)):
            a = candidates[i]
            for j in range(i + 1, len(candidates)):
                b = candidates[j]
                if a + b in S:
                    return False
                    
        return True

    def is_valid_swap(S, x, y):
        """
        Check if (S - {y}) U {x} is valid.
        y is in S, x is not in S.
        New set S' = (S - {y}) U {x}.
        We need to check triples involving x in S'.
        Triples (x, a, b) with a, b in S - {y}, a != b.
        Condition: (x+a in S') and (x+b in S') and (a+b in S').
        
        x+a in S' means:
          - x+a == x (implies a=0, impossible)
          - x+a in S - {y} => x+a in S and x+a != y
        
        a+b in S' means:
          - a+b == x
          - a+b in S - {y} => a+b in S and a+b != y
        """
        # Filter candidates for a, b
        # a must be in S - {y}
        # x+a must be in S' => x+a in S and x+a != y
        
        candidates = []
        for a in S:
            if a == y:
                continue
            # Check if x+a is in S'
            sum_xa = x + a
            if sum_xa in S and sum_xa != y:
                candidates.append(a)
                
        # Now check pairs (a, b) in candidates
        for i in range(len(candidates)):
            a = candidates[i]
            for j in range(i + 1, len(candidates)):
                b = candidates[j]
                # Check if a+b is in S'
                sum_ab = a + b
                # sum_ab must be in S'
                # sum_ab == x?
                # sum_ab in S - {y}?
                if sum_ab == x:
                    return False
                elif sum_ab in S and sum_ab != y:
                    return False
                    
        return True

    # Phase 1: Greedy Addition
    # Try to add elements from 1..64
    # We'll do multiple passes to ensure we don't miss any due to ordering
    added_any = True
    while added_any:
        added_any = False
        # Randomize order to avoid local minima in greedy choice
        candidates = [x for x in range(1, 65) if x not in S]
        random.shuffle(candidates)
        for x in candidates:
            if time.time() - start_time > timeout:
                return sorted(list(S))
            if is_valid_addition(S, x):
                S.add(x)
                added_any = True
                
    # Phase 2: Local Search with Swaps
    # Try to improve the set structure to allow more additions
    # We track the best size found
    
    best_S = set(S)
    best_size = len(S)
    
    # List of elements not in S
    # We will try swaps: remove y from S, add x to S
    # If valid, we accept the swap. This keeps size constant but changes composition.
    # We hope this opens up room for future additions.
    
    while time.time() - start_time < timeout - 2.0:
        # Try to add
        candidates = [x for x in range(1, 65) if x not in S]
        random.shuffle(candidates)
        for x in candidates:
            if time.time() - start_time > timeout:
                return sorted(list(best_S))
            if is_valid_addition(S, x):
                S.add(x)
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
                # After adding, restart the addition loop to check for more
                # But for now, just continue to next candidate
                # Actually, breaking and restarting the outer loop might be better
                # to re-evaluate all candidates
                break
        
        # If we added something, continue to next iteration to try more additions
        # If not, try a swap
        
        # Pick a random x not in S
        x = random.choice(candidates) if candidates else None
        if x is None:
            break # All 1..128 are in S, which is unlikely to be valid but possible?
            
        # Pick a random y in S
        y = random.choice(list(S))
        
        if is_valid_swap(S, x, y):
            S.remove(y)
            S.add(x)
            
        # Check if current S is better than best_S (size is same, so no improvement in size)
        # But maybe we can add more now?
        # The loop continues, so we will try to add again.
        
    # Final check for any remaining additions
    candidates = [x for x in range(1, 65) if x not in S]
    random.shuffle(candidates)
    for x in candidates:
        if time.time() - start_time > timeout:
            break
        if is_valid_addition(S, x):
            S.add(x)
            
    if len(S) > best_size:
        best_S = set(S)
        
    return sorted(list(best_S))

if __name__ == "__main__":
    res = construct()
    print(len(res))
    S = set(res)
    valid = True
    res_list = res
    for i, a in enumerate(res_list):
        for j, b in enumerate(res_list):
            if i >= j: continue
            for k, c in enumerate(res_list):
                if k <= j: continue
                if (a+b) in S and (a+c) in S and (b+c) in S:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    print("Valid:", valid)
