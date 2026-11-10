import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    N = 1000
    max_sum = 2 * N
    
    def is_valid_addition(S, cnt, x):
        # Check if adding x to S keeps all sums <= 2
        # Sums involving x: x + y for y in S, y != x (but y < x if we maintain sorted or just distinct)
        # Since S is a set of distinct ints, y != x.
        # We need to check counts for x+y and 2x.
        # Note: The problem says a <= b. 
        # If we add x, new pairs are (x, y) for y in S, y < x? 
        # Actually, the representation count is for the sum value.
        # The pair (a,b) with a<=b.
        # If we add x, the new pairs are (x, y) for all y in S.
        # If y < x, pair is (y, x). Sum x+y.
        # If y > x, pair is (x, y). Sum x+y.
        # If y == x, not possible since distinct.
        # So we just check counts for x+y for all y in S, and 2*x.
        
        # Optimization: 
        # We need to ensure that for all y in S, cnt[x+y] < 2, and cnt[2*x] < 2.
        # Actually, if cnt[x+y] is currently 2, adding x makes it 3. Invalid.
        # So we need cnt[x+y] <= 1 for all y in S, and cnt[2*x] <= 1.
        
        # Wait, if cnt[s] is 1, adding another representation makes it 2, which is allowed.
        # So we need cnt[x+y] < 2 and cnt[2*x] < 2.
        
        # Let's check.
        # Current state is valid (all counts <= 2).
        # If we add x, new counts are cnt[s] + 1 for s in {x+y | y in S} U {2x}.
        # We need new counts <= 2.
        # So we need current cnt[s] <= 1 for all affected s.
        
        # Check 2*x
        if cnt[2*x] > 1:
            return False
            
        # Check x+y
        # We can break early if any violation found
        for y in S:
            if cnt[x + y] > 1:
                return False
        return True

    def perform_add(S, cnt, x):
        S.add(x)
        cnt[2*x] += 1
        for y in S:
            if y != x: # y is from old S, so y != x is true
                cnt[x + y] += 1

    def perform_remove(S, cnt, x):
        S.remove(x)
        cnt[2*x] -= 1
        for y in S:
            # y is from new S (after removal), so y != x
            cnt[x + y] -= 1

    def greedy_random(seed):
        random.seed(seed)
        S = set()
        cnt = [0] * (max_sum + 1)
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_addition(S, cnt, x):
                perform_add(S, cnt, x)
        
        return S, cnt

    def local_search(S, cnt, time_left):
        # S and cnt are current state
        # We try to increase size or maintain size while exploring
        best_size = len(S)
        best_S = set(S)
        best_cnt = cnt[:]
        
        no_improve_count = 0
        max_no_improve = 5000 # Reset/shake if stuck
        
        while time.time() - start_time < time_limit:
            if time.time() - start_time > time_limit - 5:
                break
                
            # Try to add a random element
            added = False
            # Sample a few candidates to add
            for _ in range(10):
                if not S: break
                v = random.randint(1, N)
                if v in S:
                    continue
                if is_valid_addition(S, cnt, v):
                    perform_add(S, cnt, v)
                    added = True
                    break
            
            if added:
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                    best_cnt = cnt[:]
                no_improve_count = 0
            else:
                # Try a swap
                swapped = False
                for _ in range(10):
                    if not S: break
                    u = random.choice(list(S))
                    v = random.randint(1, N)
                    if v in S or v == u:
                        continue
                    
                    # Check swap u -> v
                    # Temporarily remove u
                    perform_remove(S, cnt, u)
                    
                    if is_valid_addition(S, cnt, v):
                        perform_add(S, cnt, v)
                        swapped = True
                        break
                    else:
                        # Revert removal
                        perform_add(S, cnt, u)
                
                if swapped:
                    # Size is same, but landscape changed
                    # Check if we can add now? (Loop will try next iter)
                    # For now, just accept swap
                    no_improve_count += 1
                else:
                    no_improve_count += 1
                
                if no_improve_count > max_no_improve:
                    # Shake: remove a few elements to break deadlock
                    # Remove 2-3 elements with high conflict? Or random?
                    # Let's remove 2 random elements
                    to_remove = random.sample(list(S), min(2, len(S)))
                    for u in to_remove:
                        perform_remove(S, cnt, u)
                    no_improve_count = 0

        return best_S, best_cnt

    # Phase 1: Multiple Randomized Greedy
    best_S = set()
    best_cnt = [0] * (max_sum + 1)
    best_size = 0
    
    # Run a few greedy passes
    for i in range(20):
        S, cnt = greedy_random(i)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            best_cnt = cnt
            
    # Phase 2: Local Search on the best greedy result
    if len(best_S) > 0:
        final_S, final_cnt = local_search(best_S, best_cnt, time_limit)
        if len(final_S) > best_size:
            best_S = final_S
            best_size = len(final_S)
            
    # Phase 3: Try to improve further with more local search if time permits
    # We can run local search again on the result of phase 2
    if time.time() - start_time < time_limit - 10:
        final_S2, final_cnt2 = local_search(best_S, best_cnt, time_limit)
        if len(final_S2) > best_size:
            best_S = final_S2
            best_size = len(final_S2)

    return list(best_S)

# EVOLVE-BLOCK-END
