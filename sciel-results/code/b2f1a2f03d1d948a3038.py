import time
from math import lcm
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # LCM of 1 to 24
    L = 1
    for i in range(1, 25):
        L = lcm(L, i)
        
    # V[i] = L / i
    V = [0] * 25
    for i in range(1, 25):
        V[i] = L // i
        
    # The condition for a set S to be valid is:
    # For all x in S, V[x] is NOT a subset sum of {V[y] for y in S, y != x}.
    
    # Key Insight for Greedy/Backtracking:
    # If we maintain `sums_S` (the set of all subset sums of the current set S),
    # and we want to add a new element x (where x not in S), the new set is S' = S U {x}.
    # We need to check:
    # 1. V[x] is not a subset sum of S. (i.e., V[x] not in sums_S)
    # 2. For all y in S, V[y] is not a subset sum of S' \ {y} = (S \ {y}) U {x}.
    #    A subset of S' \ {y} either contains x or not.
    #    Case A: Subset does not contain x. Then it is a subset of S \ {y}.
    #            Since S is valid, V[y] is not a subset sum of S \ {y}. So this case is safe.
    #    Case B: Subset contains x. Let the subset be {x} U T, where T is a subset of S \ {y}.
    #            Then sum = V[x] + sum(T) = V[y].
    #            => sum(T) = V[y] - V[x].
    #            We need to check if V[y] - V[x] is a subset sum of S \ {y}.
    #            Note: If V[y] - V[x] < 0, it's impossible (sums are positive).
    #            If V[y] - V[x] >= 0:
    #            Is V[y] - V[x] a subset sum of S \ {y}?
    #            We only have `sums_S` (subset sums of S).
    #            If V[y] - V[x] is in `sums_S`, is it necessarily in `sums_{S\{y}}`?
    #            Let T be a subset of S such that sum(T) = V[y] - V[x].
    #            If y in T, then sum(T) >= V[y].
    #            But V[y] - V[x] < V[y] (since V[x] > 0).
    #            Contradiction. So y cannot be in T.
    #            Therefore, any subset of S summing to V[y] - V[x] is actually a subset of S \ {y}.
    #            So checking `V[y] - V[x] in sums_S` is sufficient and necessary.
    
    # This allows a very fast incremental check.
    
    def try_add(x, S, sums_S):
        """
        Returns (success, new_S, new_sums_S) if x can be added, else (False, S, sums_S).
        """
        if V[x] in sums_S:
            return False, S, sums_S
            
        for y in S:
            target = V[y] - V[x]
            if target > 0 and target in sums_S:
                return False, S, sums_S
            # if target == 0, it implies V[y] == V[x], which is impossible for distinct y,x in [1,24]
            # if target < 0, no subset sum can be negative.
            
        # Safe to add
        new_S = S + [x]
        new_sums_S = set(sums_S)
        # Optimization: if sums_S is large, this is expensive.
        # But S size is small (max 24).
        for s in sums_S:
            new_sums_S.add(s + V[x])
            
        return True, new_S, new_sums_S

    # We want to find the largest valid subset.
    # Strategy:
    # 1. Greedy search with different orders to find a good initial set.
    # 2. Backtracking search to try to improve it.
    
    best_set = []
    
    # Greedy Search
    def greedy_search(order):
        S = []
        sums_S = {0}
        for x in order:
            success, new_S, new_sums_S = try_add(x, S, sums_S)
            if success:
                S = new_S
                sums_S = new_sums_S
        return S

    # Try a few deterministic orders
    orders = [
        list(range(1, 25)),      # Ascending
        list(range(24, 0, -1)),  # Descending
    ]
    
    for order in orders:
        S = greedy_search(order)
        if len(S) > len(best_set):
            best_set = S
            
    # Randomized Greedy Search
    import random
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        order = list(range(1, 25))
        random.shuffle(order)
        S = greedy_search(order)
        if len(S) > len(best_set):
            best_set = S
            
    # Backtracking Search to improve best_set
    # We can try to remove elements from best_set and add others?
    # Or do a full backtracking search with pruning, starting from an empty set,
    # but using the size of best_set as a lower bound to prune.
    
    # A full backtracking search over 24 elements is 2^24 = 16M states.
    # With pruning, it might be fast enough if we order candidates well.
    # Let's try a DFS backtracking.
    
    def backtrack(remaining_candidates, current_S, current_sums_S, depth):
        nonlocal best_set
        
        if time.time() > deadline - 2:
            return
            
        if len(current_S) + len(remaining_candidates) <= len(best_set):
            return
            
        if not remaining_candidates:
            if len(current_S) > len(best_set):
                best_set = current_S[:]
            return
            
        # Heuristic: try adding candidates that are "safe" first?
        # Or just iterate.
        # To maximize depth, we should try to add elements that allow adding more elements later.
        # Usually, smaller numbers (larger reciprocals) are more "dangerous" to add later?
        # Actually, adding large numbers (small reciprocals) is safer.
        # So maybe process candidates in descending order of value (ascending index)?
        # Or just use the order provided.
        
        # We iterate through remaining candidates.
        # We can choose to include or exclude each candidate.
        # To find the *maximum* size, we should try to include as many as possible.
        # So we try to include first. If it fails, we exclude.
        
        # However, a simple DFS that tries to include everything might get stuck.
        # We can use the greedy result as a guide.
        
        # Let's try a simple recursive step.
        # Pick the first candidate in remaining_candidates.
        x = remaining_candidates[0]
        rest = remaining_candidates[1:]
        
        # Option 1: Include x
        success, new_S, new_sums_S = try_add(x, current_S, current_sums_S)
        if success:
            backtrack(rest, new_S, new_sums_S, depth + 1)
            
        # Option 2: Exclude x
        # Only explore if we can still potentially beat the best set
        if len(current_S) + len(rest) > len(best_set):
            backtrack(rest, current_S, current_sums_S, depth)

    # To make backtracking effective, we should order the candidates.
    # Candidates that are "hard" to add should be considered later?
    # Or candidates that are "safe" should be considered first to build up the set?
    # Let's try sorting candidates by V[x] descending (small x first) or ascending (large x first).
    # Actually, the greedy search already found a good set.
    # We can use the greedy set as a starting point for a local search?
    # Or just run the backtracking on the full set of candidates with a good ordering.
    
    # Let's try backtracking with candidates sorted by descending index (24, 23, ..., 1).
    # This adds "safe" elements first.
    candidates = list(range(24, 0, -1))
    backtrack(candidates, [], {0}, 0)
    
    # Also try ascending order
    candidates = list(range(1, 25))
    backtrack(candidates, [], {0}, 0)
    
    if not best_set:
        return [1]
        
    return best_set
