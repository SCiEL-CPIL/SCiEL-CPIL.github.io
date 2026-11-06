"""Combined program: Conflict-graph / 3-term-AP aware greedy with full DP verification."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """
    Constructs a set of up to 48 distinct positive integers <= 512.
    No element is the mean of any subset of at least two other elements.
    """
    N = 512
    MAX_SIZE = 48
    START_TIME = time.time()
    TIME_LIMIT = 55.0  # Safety margin before 60s hard limit

    def is_full_valid(S):
        """
        Verifies S is valid using DP with bitsets.
        """
        S_list = list(S)
        m = len(S_list)
        if m == 0:
            return True

        # Sort for deterministic behavior and potential early breaks
        S_list.sort()
        max_sum = sum(S_list)

        for a in S_list:
            others = [x for x in S_list if x != a]
            m_others = len(others)
            if m_others < 2:
                continue

            # DP for others: dp[k] is a bitset where bit s is 1 if sum s can be formed by k elements
            dp = [0] * (m_others + 1)
            dp[0] = 1  # Sum 0 with 0 elements

            for x in others:
                # Update dp in reverse order of k to avoid using x multiple times in one step
                for k in range(m_others, 0, -1):
                    dp[k] |= (dp[k - 1] << x)

            # Check if any k*a is in dp[k] for k >= 2
            valid = True
            for k in range(2, m_others + 1):
                target = k * a
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    valid = False
                    break
            if not valid:
                return False
        return True

    def has_3ap(x, S_set):
        """
        Checks if x forms a 3-term arithmetic progression with any two other elements in S_set.
        This is a necessary condition for validity (if a 3-AP exists, the middle element is the mean of the other two).
        It is a fast heuristic filter.
        """
        # Case 1: x is the middle: a, x, b => 2x = a + b
        # Case 2: x is the end: x, a, b => 2a = x + b (x is smaller end)
        #           a, b, x => 2b = a + x (x is larger end)
        
        # We can just check if there exist a, b in S_set such that:
        # 1) a + b == 2*x
        # 2) a + x == 2*b  => b = (a+x)/2
        # 3) b + x == 2*a  => a = (b+x)/2
        
        # Simpler: iterate pairs in S_set? No, S_set size grows.
        # Iterate a in S_set:
        for a in S_set:
            if a == x:
                continue
            # Check if x is middle: need b = 2*x - a
            b = 2 * x - a
            if b != a and b != x and b in S_set:
                return True
            # Check if a is middle: need b such that 2*a = x + b => b = 2*a - x
            b = 2 * a - x
            if b != a and b != x and b in S_set:
                return True
            # Check if b is middle: need b such that 2*b = a + x => b = (a+x)/2
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if b != a and b != x and b in S_set:
                    return True
        return False

    def is_fast_valid_candidate(x, S_set):
        """
        Fast check: 
        1. x doesn't form a 3-AP with any pair in S_set.
        2. x doesn't create a 3-AP involving two elements from S_set and x (covered above).
        Actually, the condition "no element is mean of subset of >=2 others" implies no 3-APs.
        But it's stronger. For a fast filter during greedy growth, checking 3-APs is a good first step.
        However, to be safer and align with the "conflict graph" idea, we should check if adding x creates
        any 3-AP. The function has_3ap(x, S_set) does exactly that: it checks if x is part of any 3-AP 
        with two elements from S_set.
        
        Wait, if S_set is valid (no 3-APs), and we add x, the only new 3-APs must involve x.
        So checking if x forms a 3-AP with any pair in S_set is sufficient to maintain the "no 3-AP" property.
        """
        return not has_3ap(x, S_set)

    best_set = set()
    
    # Strategy: Multiple restarts with random ordering, biased towards sparse structures if possible.
    # We will try to greedily add elements that don't create 3-APs.
    # If we reach MAX_SIZE, we verify with full DP.
    # If not, we keep the best valid set found so far.
    
    candidates = list(range(1, N + 1))
    
    # Pre-compute some properties? Not really needed for this simple check.
    
    for restart in range(2000):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        random.shuffle(candidates)
        
        S = set()
        
        for x in candidates:
            if len(S) >= MAX_SIZE:
                break
            
            # Fast check: does x create a 3-AP with existing elements?
            if is_fast_valid_candidate(x, S):
                S.add(x)
        
        # If we reached max size, verify
        if len(S) == MAX_SIZE:
            if is_full_valid(S):
                return list(S)
        
        # If we didn't reach max size, check if current S is valid (it should be, as we only added non-3AP elements, 
        # but 3-AP is a subset of the condition. It's possible we missed a higher-order mean violation?
        # No, if there's a mean violation, there might not be a 3-AP. 
        # Example: 1, 2, 4, 7. Mean of 1,2,4 is 7/3 (no). Mean of 1,4,7 is 4. 4 is in set. 
        # 1,4,7 is a 3-AP? 1, 4, 7. 4-1=3, 7-4=3. Yes.
        # Example: 1, 3, 5, 9. Mean of 1,3,5 is 3. 3 is in set. 1,3,5 is 3-AP.
        # Is it possible to have a mean violation without a 3-AP?
        # Mean of subset T is a. Sum(T) = k*a.
        # If k=2, T={u,v}, u+v=2a => u,a,v is 3-AP.
        # If k=3, T={u,v,w}, u+v+w=3a.
        # If u,v,w are distinct and not a.
        # Does this imply a 3-AP? Not necessarily.
        # Example: 1, 2, 5, 6. Mean of 1,2,5 is 8/3. Mean of 1,2,6 is 9/3=3 (not in set).
        # Mean of 1,5,6 is 12/3=4 (not in set). Mean of 2,5,6 is 13/3.
        # Let's try to find a counter-example where k=3 mean violation exists but no 3-AP.
        # Set: {1, 4, 5, 8}. 
        # 3-APs: 1,4,7(no), 1,5,9(no), 1,8,15(no), 4,5,6(no), 4,8,12(no), 5,8,11(no).
        # Means:
        # a=1: others {4,5,8}. Sums: 4,5,8,9,12,13,17,22. k=2: 2*1=2 (no). k=3: 3*1=3 (no).
        # a=4: others {1,5,8}. Sums: 1,5,8,6,9,13,14,18,22. k=2: 8 (no, 4+4=8 but only one 4? No, distinct others. 1+5=6, 1+8=9, 5+8=13). k=3: 12 (no).
        # a=5: others {1,4,8}. Sums: 1,4,8,5,9,12,13,17,22. k=2: 10 (no). k=3: 15 (no).
        # a=8: others {1,4,5}. Sums: 1,4,5,5,6,9,10,14. k=2: 16 (no). k=3: 24 (no).
        # So {1,4,5,8} is valid and has no 3-APs.
        
        # What about a set with a 3-AP violation? {1,2,3}. 2 is mean of 1,3.
        
        # It is known that "Sidon sets" or "B2 sets" avoid sums, but here we avoid means.
        # The condition "no element is the mean of any subset" is stronger than "no 3-AP".
        # So the fast filter (no 3-AP) might let through invalid sets.
        # Therefore, we MUST run is_full_valid on the final candidate.
        
        if len(S) > len(best_set):
            # Check validity of S before updating best
            if is_full_valid(S):
                best_set = S.copy()
        
        # Optional: Local search / repair?
        # If we are close to MAX_SIZE, we could try swapping elements.
        # But for now, let's rely on restarts.
        
        # If we found a set of size MAX_SIZE that is valid, we returned.
        # If we found a set of size MAX_SIZE that is invalid, we didn't update best (unless it was better than previous best AND valid, which it isn't).
        # Wait, the code above:
        # if len(S) == MAX_SIZE:
        #     if is_full_valid(S): return ...
        # if len(S) > len(best_set):
        #     if is_full_valid(S): best_set = S.copy()
        
        # This means if S is size 48 but invalid, we don't store it.
        # If S is size 47 and valid, we store it.
        
        # We can also try to prune invalid large sets to see if they contain a large valid subset?
        # That's expensive.
        
    if best_set:
        # Final verification just in case
        if is_full_valid(best_set):
            return list(best_set)
    
    # Fallback: Return a known valid set
    # Powers of 2 are valid because no sum of distinct powers of 2 equals another power of 2 times k (unless k=1, but subset size >=2).
    # Actually, sum of distinct powers of 2 is unique. 
    # Can a power of 2 be the mean of a subset of others?
    # 2^j = (sum of k distinct powers of 2 not including 2^j) / k
    # k * 2^j = sum
    # LHS has bit j set (and possibly higher if k>1). RHS is sum of bits <j or >j.
    # If all bits <j, max sum is 2^j - 1. So k*2^j >= 2^j > 2^j - 1. Impossible.
    # If some bits >j, sum > 2^j.
    # Example: k=2. 2*2^j = 2^(j+1). Need two distinct powers of 2 summing to 2^(j+1).
    # 2^a + 2^b = 2^(j+1). If a < b, 2^a (1 + 2^(b-a)) = 2^(j+1).
    # 1 + 2^(b-a) must be a power of 2. Only if b-a=1 and 1+2=3 (no) or ... wait.
    # 1 + 2^m is a power of 2 only if m=0 (1+1=2) or m=1 (1+2=3 no).
    # So 2^a + 2^(a+1) = 3*2^a. Not a power of 2.
    # So no 2-term subset works.
    # k=3. 3*2^j = sum of 3 distinct powers.
    # It is a known result that powers of 2 form a set with no element as the mean of any subset of others.
    
    fallback = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    return fallback

# EVOLVE-BLOCK-END
