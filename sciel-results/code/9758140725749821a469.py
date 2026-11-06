import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 192
    max_val = n
    
    def is_valid_addition(S, x):
        """
        Check if adding x to valid set S keeps it valid.
        S is a list of distinct integers. x is not in S.
        Condition: No element in S U {x} divides the sum of any 3 distinct OTHER elements in S U {x}.
        """
        S_list = list(S)
        
        # 1. Check if x divides sum of any 3 distinct elements in S
        # If S has < 3 elements, this is vacuously true
        if len(S_list) >= 3:
            # Optimization: Precompute sums? No, just iterate.
            # We can break early if found
            for a, b, c in combinations(S_list, 3):
                if (a + b + c) % x == 0:
                    return False
        
        # 2. Check for each a in S: does a divide sum of any 3 distinct from (S U {x}) \ {a}?
        # The new triples involving x are of the form {x, b, c} where b, c in S \ {a} and b != c.
        # Note: b and c must be distinct from a and each other.
        # So we iterate over pairs (b, c) from S \ {a}.
        for a in S_list:
            # Get candidates for b and c: elements in S that are not a
            # We need distinct b, c.
            # If len(S) < 2, we can't form a pair from S\{a} if a is in S?
            # If S = {a}, S\{a} is empty.
            # If S = {a, y}, S\{a} = {y}. Can't form pair.
            # If S = {a, y, z}, S\{a} = {y, z}. Pair (y, z).
            
            others = [v for v in S_list if v != a]
            if len(others) < 2:
                continue
                
            # Check all pairs in 'others'
            # We can optimize by precomputing remainders?
            # (b + c + x) % a == 0  => (b + c) % a == (-x) % a
            target_rem = (-x) % a
            
            # Iterate pairs
            for i in range(len(others)):
                b = others[i]
                for j in range(i + 1, len(others)):
                    c = others[j]
                    if (b + c) % a == target_rem:
                        return False
                        
        return True

    # Greedy initialization
    # Strategy: Try adding numbers from high to low.
    # High numbers are less likely to divide sums of smaller numbers?
    # Actually, small numbers divide many things. Large numbers divide few.
    # So we want to keep large numbers?
    # If we keep large numbers, the sums of 3 large numbers are large.
    # Does a large number divide a sum of 3 large numbers?
    # e.g., 100 divides 101+102+103 = 306? No.
    # 100 divides 100+... no, distinct others.
    
    S = []
    # Try adding from n down to 1
    for x in range(n, 0, -1):
        if is_valid_addition(S, x):
            S.append(x)
    
    best_set = list(S)
    best_size = len(best_set)
    
    # Local Search
    # We will try to improve the set size.
    # Moves:
    # 1. Add an element.
    # 2. Swap: Remove one, add one.
    # 3. Remove one (to allow future adds? usually not helpful for max size unless we are stuck in a dead end where no add is possible and no swap increases size).
    
    # To escape local optima, we can use a randomized approach.
    # However, simple hill climbing on size often works well for these small N.
    
    # Let's try a more aggressive search:
    # Random restarts with different greedy orders + local search.
    
    # Since we have ~85 seconds, we can run multiple restarts.
    # Each restart:
    # 1. Shuffle or use different heuristic for greedy.
    # 2. Local search to local optimum.
    # 3. Update global best.
    
    # Heuristic 1: Descending
    # Heuristic 2: Ascending
    # Heuristic 3: Random
    # Heuristic 4: "Prime-heavy" or other?
    
    # Let's implement a single strong local search with randomization of candidate order.
    
    # First, let's polish the current S.
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Try to add any element
        # Randomize order to avoid bias? Or try all?
        # Trying all is better for finding *any* add.
        candidates = [x for x in range(1, n + 1) if x not in S]
        random.shuffle(candidates)
        
        added = False
        for x in candidates:
            if is_valid_addition(S, x):
                S.append(x)
                improved = True
                added = True
                break
        
        if added:
            if len(S) > best_size:
                best_set = list(S)
                best_size = len(best_set)
            continue
        
        # If no add possible, try swaps
        # For each element a in S, try removing it and adding some x.
        # We want to maintain or increase size.
        # Since we couldn't add, size is fixed. Swaps keep size same.
        # Why swap? To reach a state where we *can* add.
        
        # Let's try random swaps to explore the landscape.
        # Pick random a in S, random x not in S.
        # If S - {a} + {x} is valid, accept it (even if size same).
        # This allows us to move to a different local optimum.
        
        # To avoid cycling, we can keep track of visited states or use simulated annealing.
        # Given time constraints, simple random walk with occasional restarts is good.
        
        # Let's do a limited random walk phase if stuck.
        # If we are stuck (no adds), we do 100 random valid swaps.
        # Then try adds again.
        
        # Actually, a better strategy:
        # If stuck, try "2-for-1" or "1-for-2"? No, we want max size.
        # So we only accept swaps that keep size same, hoping to unlock an add.
        
        # Let's try to find a swap that allows an immediate add?
        # That's complex.
        
        # Simpler: Random restarts.
        # If we are stuck for a while, restart.
        
        # Let's just do a few random restarts of the whole process.
        # The current S is from Descending Greedy.
        # Let's try Ascending Greedy.
        # Let's try Random Greedy.
        
        # Break out of while loop to do next restart strategy if we are stuck.
        # How to detect stuck? 'improved' is False.
        # We can just break and try a different initialization.
        break

    # Restart 2: Ascending Greedy
    S2 = []
    for x in range(1, n + 1):
        if is_valid_addition(S2, x):
            S2.append(x)
    
    # Polish S2
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        candidates = [x for x in range(1, n + 1) if x not in S2]
        random.shuffle(candidates)
        for x in candidates:
            if is_valid_addition(S2, x):
                S2.append(x)
                improved = True
                break
        if len(S2) > best_size:
            best_set = list(S2)
            best_size = len(best_set)
        if not improved:
            break

    # Restart 3: Random Greedy (Multiple times if time permits)
    # And local search with swaps.
    
    # Let's implement a more sophisticated local search on the best_set found so far.
    # We will use the best_set as the starting point for a randomized local search.
    
    current_set = list(best_set)
    current_size = len(current_set)
    
    # Simulated Annealing-like approach
    # Temperature starts high, cools down.
    # Objective: Maximize size.
    # Move: Add, Remove, Swap.
    # Acceptance: Always accept if size increases.
    # If size same, accept with probability p (to explore).
    # If size decreases, accept with very low probability (only if stuck).
    
    # Given the small N, "Always accept size increase" + "Random swaps when stuck" is robust.
    
    # Let's try a few random restarts with random greedy initialization.
    
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        S_r = []
        # Random order
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_addition(S_r, x):
                S_r.append(x)
        
        # Local search polish
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adds
            cands = [x for x in range(1, n + 1) if x not in S_r]
            random.shuffle(cands)
            for x in cands:
                if is_valid_addition(S_r, x):
                    S_r.append(x)
                    improved = True
                    break
            
            if len(S_r) > best_size:
                best_set = list(S_r)
                best_size = len(best_set)
            
            if not improved:
                # Try some random swaps to escape local optimum
                # Do 50 random swaps
                for _ in range(50):
                    if time.time() > deadline - 5:
                        break
                    if len(S_r) == 0:
                        break
                    # Pick random element to remove
                    idx = random.randint(0, len(S_r) - 1)
                    a = S_r[idx]
                    # Pick random element to add
                    cands_swap = [x for x in range(1, n + 1) if x not in S_r]
                    if not cands_swap:
                        break
                    x = random.choice(cands_swap)
                    
                    # Check if swap is valid
                    # S_r - {a} + {x}
                    temp_set = [v for i, v in enumerate(S_r) if i != idx]
                    if is_valid_addition(temp_set, x):
                        # Accept swap (size remains same)
                        # This might help us find a state where we can add more
                        S_r = temp_set + [x]
                        # Note: We don't mark improved=True here because size didn't change
                        # But we break out of the "try adds" loop to try adds again?
                        # Actually, if we swap, we should re-evaluate adds.
                        # So we set a flag to restart the add-checking loop.
                        # But the 'while improved' loop condition is 'improved'.
                        # If we swap, we haven't improved size.
                        # So we need to structure the loop differently.
                        # Let's just break out of the swap loop and go back to add-checking.
                        break
                
                # If we did swaps, we should try adds again.
                # But the 'while improved' loop will exit if 'improved' is False.
                # So we need to force it to continue if we did a swap.
                # Let's refactor:
                # The loop above breaks if no add found.
                # If we do swaps, we should continue the outer while loop?
                # No, the 'while improved' loop exits.
                # So the random swaps inside 'if not improved' happen once, then the loop exits.
                # This means we only do one round of swaps per "stuck" state.
                # That might not be enough.
                
                # Let's change the structure:
                # Use a time-based loop instead of 'improved' based.
                
                # Let's restart the local search logic for S_r in a time-boxed manner.
                break

    # Final polish on best_set
    # Try to add any element to best_set
    S_final = list(best_set)
    cands = [x for x in range(1, n + 1) if x not in S_final]
    random.shuffle(cands)
    for x in cands:
        if is_valid_addition(S_final, x):
            S_final.append(x)
    
    if len(S_final) > best_size:
        best_set = S_final
        best_size = len(best_set)

    return best_set

# Run a quick test to ensure it doesn't crash
if __name__ == "__main__":
    res = construct()
    print(len(res))
