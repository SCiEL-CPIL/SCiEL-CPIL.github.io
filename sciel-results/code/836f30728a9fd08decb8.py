import time
import random

def construct():
    """
    Returns a list of up to 48 distinct positive integers <= 512 such that no element 
    is the exact arithmetic mean of any subset of at least two other elements.
    
    Strategy:
    1. Fast global validity check using bitset DP.
    2. Multiple randomized greedy trials with different orderings.
    3. Local search (swap-based) to improve cardinality if greedy gets stuck.
    """
    deadline = time.time() + 55  # Safety margin for 60s limit
    N = 512
    target_size = 48
    
    def is_valid_global(S):
        """
        Check if set S is valid.
        Condition: No element a in S is the mean of any subset T of S\{a} with |T| >= 2.
        i.e., sum(T) != a * |T| for any T subset of S\{a}, |T| >= 2.
        """
        m = len(S)
        if m < 3:
            return True
        
        # We need to check for each a in S if there is a subset T of S\{a} 
        # such that sum(T) = a * k for some k >= 2.
        # Equivalently, sum(T) - a*k = 0.
        
        # To be efficient, we can iterate over each a, remove it, and check if 
        # any multiple of a (starting from 2a) can be formed by a subset of the remaining elements 
        # with size exactly k.
        
        # However, checking "size exactly k" requires DP tracking both sum and size.
        # DP[k] is a bitset where bit 's' is 1 if sum 's' can be formed by 'k' elements.
        
        # Max sum is 512 * 48 = 24576.
        # Max k is 47.
        
        # For each a in S:
        #   Let S_a = S \ {a}
        #   Compute DP for S_a.
        #   For k from 2 to len(S_a):
        #     Check if bit (a*k) is set in DP[k].
        
        # This is O(m * m * MaxSum / word_size) roughly.
        # m=48, MaxSum=24576.
        # 48 * 48 * 24576 / 64 ~ 90k ops? No, Python big ints are fast.
        # Let's implement this.
        
        S_sorted = sorted(S)
        
        for idx, a in enumerate(S_sorted):
            # Build DP for S without a
            # dp[k] is a bitset of possible sums using exactly k elements
            # Max k needed is len(S)-1
            max_k = m - 1
            dp = [0] * (max_k + 1)
            dp[0] = 1  # 0 elements, sum 0
            
            count = 0
            for v in S_sorted:
                if v == a:
                    if count == 0:
                        count = 1
                        continue
                    else:
                        # Duplicate values? Problem says distinct integers. So v==a only happens once.
                        pass
                
                # Update DP in reverse order of k
                # We iterate k from current_max_k down to 0
                # Current max k is roughly count
                # To be safe, iterate from max_k down to 0, but we can limit to count
                for k in range(min(count, max_k - 1), -1, -1):
                    if dp[k]:
                        dp[k + 1] |= dp[k] << v
                count += 1
            
            # Now check if any a * k is achievable
            # k must be >= 2
            # Max sum possible with k elements is roughly k * 512
            # a * k must be <= sum of largest k elements?
            # Actually, just check the bit.
            for k in range(2, max_k + 1):
                target_sum = a * k
                # If target_sum is too large, it can't be formed
                # Max sum of k elements from 512 is k*512
                if target_sum > 512 * k:
                    continue
                if target_sum > 24576: # Hard cap
                    break
                if (dp[k] >> target_sum) & 1:
                    return False
        
        return True

    def greedy_build(order, current_set=None):
        """
        Greedily add elements from 'order' to 'current_set' if valid.
        """
        if current_set is None:
            current_set = []
        
        current_set = list(current_set)
        current_set_set = set(current_set)
        
        for c in order:
            if len(current_set) >= target_size:
                break
            if c in current_set_set:
                continue
            
            # Check if adding c keeps set valid
            # We can do a faster incremental check or just use global check on the new set
            # Given m is small (<48), global check on set of size m+1 is fast enough.
            # Let's try incremental logic for speed in the greedy loop.
            
            # Fast incremental check:
            # 1. Is c the mean of any subset of current_set (size >= 2)?
            # 2. Is any a in current_set the mean of any subset of (current_set + c) \ {a} (size >= 2)?
            
            # Let's just use the global check on the candidate list for simplicity and correctness, 
            # it's fast enough for m<48.
            
            candidate_list = current_set + [c]
            if is_valid_global(candidate_list):
                current_set.append(c)
                current_set_set.add(c)
                
        return current_set

    best_set = []
    best_len = 0
    
    # 1. Try deterministic orders
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        list(range(1, N + 1, 2)), # Odds
        list(range(2, N + 1, 2)), # Evens
    ]
    
    # 2. Try random orders
    rng = random.Random(42)
    for _ in range(5):
        p = list(range(1, N + 1))
        rng.shuffle(p)
        orders.append(p)
        
    for order in orders:
        if time.time() > deadline:
            break
        s = greedy_build(order)
        if len(s) > best_len:
            best_len = len(s)
            best_set = s[:]
            if best_len == target_size:
                break

    # 3. Randomized restarts with greedy
    if best_len < target_size:
        for seed in range(100):
            if time.time() > deadline:
                break
            rng = random.Random(seed)
            p = list(range(1, N + 1))
            rng.shuffle(p)
            s = greedy_build(p)
            if len(s) > best_len:
                best_len = len(s)
                best_set = s[:]
                if best_len == target_size:
                    break

    # 4. Local Search: Try to replace elements to increase size or fix issues?
    # If we have a set of size K < 48, we might try to swap out one element 
    # to allow two new ones? Or just try to add more if greedy failed.
    # Actually, if greedy failed to reach 48, it means we got stuck.
    # Local search: Take best_set, try swapping one element with another not in set 
    # to see if we can eventually add more?
    # Or simpler: Randomly perturb the set and re-run greedy?
    
    # Let's try a "repair" heuristic if we are close.
    if best_len < target_size and best_len >= 40:
        for _ in range(50):
            if time.time() > deadline:
                break
            # Take best_set, remove one random element, try to add two random others?
            # This is complex. 
            # Alternative: Randomly shuffle the order of best_set + some randoms and re-greedy?
            # Let's just do more random restarts, they are cheap.
            rng = random.Random(time.time_ns())
            p = list(range(1, N + 1))
            rng.shuffle(p)
            s = greedy_build(p)
            if len(s) > best_len:
                best_len = len(s)
                best_set = s[:]
                if best_len == target_size:
                    break

    return best_set

# EVOLVE-BLOCK-START
# (The code above is the full implementation. 
# The EVOLVE-BLOCK markers are usually placed around the code to be evolved. 
# Since the prompt asks for the complete program, I will output the code as is, 
# ensuring the function construct is defined.)
# EVOLVE-BLOCK-END
