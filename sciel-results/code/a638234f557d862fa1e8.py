import time
import random
import sys

def _count_conflicts(S):
    """
    Counts the total number of subset-mean violations in S.
    A violation is (a, subset) where a is in S, subset is a subset of S\{a} with size >= 2,
    and mean(subset) == a.
    We use bitset DP for each element a.
    """
    n = len(S)
    if n < 3:
        return 0
    
    s_set = set(S)
    total_violations = 0
    
    for i, a in enumerate(S):
        others = [x for j, x in enumerate(S) if j != i]
        m = len(others)
        
        # dp[k] is a bitset where bit s is 1 if sum s is achievable with k elements
        # Python integers act as arbitrary precision bitsets
        dp = [0] * (m + 1)
        dp[0] = 1
        
        for x in others:
            for k in range(m - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= (dp[k] << x)
        
        # Check for violations: sum = k * a for k >= 2
        for k in range(2, m + 1):
            target_sum = k * a
            if target_sum <= 512 * k: # Max possible sum
                if (dp[k] >> target_sum) & 1:
                    total_violations += 1
                    
    return total_violations

def _is_valid(S):
    """
    Returns True if S has no subset-mean violations.
    """
    return _count_conflicts(S) == 0

def _has_3ap(S):
    """
    Fast check for 3-term AP (subset size 2).
    """
    s = set(S)
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            a = S[i]
            b = S[j]
            c = 2 * b - a
            if c in s and c != a and c != b:
                return True
    return False

def _conflict_scores():
    """
    Pre-compute conflict scores for each number 1..512.
    Score(x) = number of pairs (u, v) in [1, 512] \ {x} such that x = (u+v)/2.
    This is the number of 3-term APs centered at x.
    """
    scores = {}
    for x in range(1, 513):
        count = 0
        # u < v, u+v = 2x
        # u ranges from max(1, 2x-512) to x-1
        # v = 2x - u
        # u < x => v > x
        # u >= 1, v <= 512 => 2x-u <= 512 => u >= 2x-512
        start_u = max(1, 2*x - 512)
        end_u = x - 1
        if start_u <= end_u:
            count = end_u - start_u + 1
        scores[x] = count
    return scores

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    rng = random.Random(42)
    
    best_set = [1]
    best_score = 0 # Maximize cardinality
    
    # Initial set of all numbers
    all_nums = list(range(1, 513))
    
    # Pre-compute conflict scores
    scores = _conflict_scores()
    
    # Strategy: 
    # 1. Use conflict scores to guide greedy initialization.
    # 2. Use local search with swaps to improve.
    # 3. Use randomized restarts.
    
    # Heuristic 1: Sort by conflict score ascending (safest first)
    order_safe = sorted(all_nums, key=lambda x: scores[x])
    
    # Heuristic 2: Sort by conflict score descending (most constrained first)
    order_unsafe = sorted(all_nums, key=lambda x: -scores[x])
    
    # Heuristic 3: Random order
    order_random = all_nums[:]
    rng.shuffle(order_random)
    
    # Try each heuristic for initialization
    for order in [order_safe, order_unsafe, order_random]:
        if time.time() > deadline:
            break
            
        current_S = []
        for x in order:
            if time.time() > deadline:
                break
            new_S = current_S + [x]
            # Fast 3AP check
            if _has_3ap(new_S):
                continue
            # Full check
            if _is_valid(new_S):
                current_S = new_S
            if len(current_S) >= 48:
                break
                
        if len(current_S) > len(best_set):
            best_set = current_S[:]
            best_score = len(current_S)
            
        if best_score >= 48:
            break
            
    # Now perform local search swaps
    # We want to increase the size of the set.
    # Current size is K.
    # We can try to replace an element x in current_S with y not in current_S.
    # If the new set is valid, we keep it.
    # If it's not valid, we might still keep it if it has fewer conflicts?
    
    # To speed up, we maintain a set of candidates for addition.
    # And a set of elements to remove.
    
    S_set = set(best_set)
    complement = [x for x in all_nums if x not in S_set]
    
    no_improve_count = 0
    current_conflicts = _count_conflicts(best_set)
    
    for step in range(5000): # Max steps per restart
        if time.time() > deadline:
            break
            
        if len(best_set) >= 48:
            break
            
        # 1. Try to add
        if complement:
            y = rng.choice(complement)
            new_S = best_set + [y]
            if _has_3ap(new_S):
                # Fast reject
                pass
            elif _is_valid(new_S):
                # Accept
                best_set = new_S
                S_set.add(y)
                complement.remove(y)
                no_improve_count = 0
                if len(best_set) > len(best_set):
                    best_score = len(best_set)
                continue
            else:
                # Invalid, try swap?
                pass
        
        # 2. Try swap
        if not best_set or not complement:
            break
            
        x = rng.choice(best_set)
        y = rng.choice(complement)
        
        if x == y:
            continue
            
        new_S = [z for z in best_set if z != x] + [y]
        
        # Check validity
        if _has_3ap(new_S):
            # Fast reject for 3AP
            # We can check conflict count roughly?
            # Or just skip if 3AP exists?
            # 3AP is a type of conflict.
            # Let's compute full conflict count to guide the search.
            pass
        
        new_conflicts = _count_conflicts(new_S)
        
        # Accept if:
        # a) It is valid (conflicts == 0)
        # b) It has fewer conflicts than current (hill climbing on conflicts)
        # c) Random restart/probability?
        
        if new_conflicts == 0:
            # Valid swap
            best_set = new_S
            S_set.remove(x)
            S_set.add(y)
            complement.remove(y)
            complement.append(x)
            no_improve_count = 0
            current_conflicts = 0
            # Size is unchanged, but we might be in a better position to add later
        elif new_conflicts < current_conflicts:
            # Move towards validity
            best_set = new_S
            S_set.remove(x)
            S_set.add(y)
            complement.remove(y)
            complement.append(x)
            current_conflicts = new_conflicts
            no_improve_count = 0
        else:
            # No improvement in conflicts
            no_improve_count += 1
            if no_improve_count > 100:
                # Break out of this local minimum, restart search phase for this restart
                break

    return best_set

if __name__ == "__main__":
    s = construct()
    print(f"Found set of size {len(s)}")
    print(s)
    if len(s) > 0:
        print(f"Valid: {_is_valid(s)}")
