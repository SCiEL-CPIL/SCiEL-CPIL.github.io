"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def _is_valid(S):
    """Return True iff S (list of distinct ints in 1..256) satisfies the constraint:
    no two distinct nonempty subset sums s, t with s != t have s | t or t | s.
    """
    if not S:
        return True
    total = sum(S)
    n = len(S)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += S[i]
        sums.add(s)
    sums_list = sorted(sums)
    for s in sums_list:
        if s == 0:
            continue
        multiple = 2 * s
        while multiple <= total:
            if multiple in sums:
                return False
            multiple += s
    return True

def _fast_is_valid(S):
    """Optimized validity check for the greedy/local search phases."""
    if not S:
        return True
    n = len(S)
    total = sum(S)
    sums = set()
    # Incremental subset sum generation is faster for small n, but for n up to 15-16,
    # bitmask is fine. Let's use a list of sums and update it to avoid recomputation if called repeatedly?
    # Actually, for a one-shot check, bitmask is okay.
    # But we can be smarter:
    sums.add(0)
    for x in S:
        new_sums = []
        for s in sums:
            new_sums.append(s + x)
        sums.update(new_sums)
    sums.discard(0)
    
    sums_list = sorted(sums)
    for s in sums_list:
        if s == 0:
            continue
        multiple = 2 * s
        while multiple <= total:
            if multiple in sums:
                return False
            multiple += s
    return True

def _greedy_build(pool, deadline, seed=None):
    """Fast greedy build to get a good initial feasible solution for the ILP warm start."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    current = []
    available = pool[:]
    rng.shuffle(available)
    
    while time.time() < deadline:
        if not available:
            break
        
        added = False
        # Try to add a random element from the front of the list (shuffled)
        # To speed up, we can just iterate and check
        for i in range(len(available)):
            x = available[i]
            test_set = current + [x]
            if _fast_is_valid(test_set):
                current.append(x)
                available.pop(i)
                added = True
                break
        
        if not added:
            # If we didn't add anything in this pass, try a few random swaps?
            # Or just break to save time for ILP
            # Let's do a quick local search step
            if len(current) > 0 and available:
                # Try 1-for-1 swap
                for i in range(len(current)):
                    if time.time() > deadline:
                        break
                    removed = current[i]
                    temp = current[:i] + current[i+1:]
                    for j in range(len(available)):
                        if time.time() > deadline:
                            break
                        x = available[j]
                        test_set = temp + [x]
                        if _fast_is_valid(test_set):
                            current = test_set
                            available[j] = removed
                            added = True
                            break
                    if added:
                        break
            if not added:
                break
                
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # 1. Get a good initial solution via heuristic
    pool = list(range(1, 257))
    # Prefer numbers that are "hard" to divide (primes, large numbers)
    # Simple heuristic: sort by number of divisors in 1..256
    def div_score(x):
        cnt = 0
        for d in range(1, x + 1):
            if x % d == 0:
                cnt += 1
        for m in range(2 * x, 257, x):
            cnt += 1
        return cnt
    
    # Randomize pool slightly but keep some structure
    # Actually, let's just run a few random greedy attempts to find a decent starting set
    best_heuristic_set = []
    for i in range(5):
        if time.time() > deadline - 20: # Leave 20s for ILP
            break
        S = _greedy_build(pool, deadline - 15, seed=i)
        if len(S) > len(best_heuristic_set):
            best_heuristic_set = S
            
    # 2. Formulate ILP
    # Variables:
    # z_i for i in 1..256 (binary): 1 if i is in the set.
    # w_m for each possible sum m in 1..256*256 (binary): 1 if sum m is achievable.
    #
    # Objective: Maximize sum(z_i)
    #
    # Constraints:
    # 1. For each sum m:
    #    sum_{i in S, i <= m} z_i * (subset_sum_indicator[i, m]) ... wait, this is non-linear (product of z and subset structure).
    #    
    #    Standard formulation:
    #    Let S be the set of elements.
    #    A sum m is achievable if there exists a subset T of S such that sum(T) = m.
    #    This is hard to linearize directly with just z_i.
    #    
    #    Alternative approach:
    #    We can iterate over possible sets? No, 2^256 is too big.
    #    
    #    Let's use the "subset sum" variables.
    #    For each m in 1..max_sum, let w_m be 1 if m is a subset sum.
    #    
    #    Constraint: If w_m = 1, then there must be a subset.
    #    This is an OR of subsets.
    #    w_m <= sum_{T: sum(T)=m} prod_{i in T} z_i  (Non-linear)
    #    
    #    Linearization:
    #    w_m <= sum_{i} z_i (trivial, not helpful)
    #    
    #    Actually, we can define variables y_{m, T}? No, too many.
    #    
    #    Let's rethink. The constraint is:
    #    For any two distinct achievable sums s, t, s does not divide t.
    #    This means: if w_s = 1 and w_t = 1 and s | t and s != t, then contradiction.
    #    So: w_s + w_t <= 1 for all pairs (s, t) with s | t, s != t.
    #    
    #    Now we just need to correctly model w_m.
    #    w_m = 1 if and only if m is a subset sum of {i : z_i = 1}.
    #    
    #    This is the "Subset Sum" problem formulation.
    #    We can use the standard linearization for "w_m is 1 iff subset sum is m".
    #    
    #    Actually, we don't need "iff". We need:
    #    If w_m = 1, then m IS a subset sum. (Implication: w_m => Exists subset)
    #    If m IS a subset sum, w_m can be 0 or 1? No, if we set w_m=0 when m is achievable, we might violate the "w_s+w_t<=1" constraint incorrectly?
    #    If m is achievable, and we set w_m=0, we are "hiding" the sum. This is SAFE for the divisibility constraint (we are saying "I don't have this sum" which is a lie, but it doesn't create a conflict).
    #    Wait, if we hide a sum, we might be able to include MORE elements in the set.
    #    Example: Set {2, 3}. Sums: 2, 3, 5.
    #    If we hide 5 (w_5=0), we don't care about 5 dividing anything.
    #    But if we include 10, sums are 2,3,5,10...
    #    If we hide 5, we might think 10 is okay (since 5 is hidden, 5 doesn't divide 10 in our model).
    #    But 5 IS a sum, so 5 divides 10. The set is INVALID.
    #    So we MUST have w_m = 1 if m is achievable.
    #    So we need: (Exists subset summing to m) => w_m = 1.
    #    
    #    How to linearize "Exists subset"?
    #    This is generally hard.
    #    
    #    Alternative: Iterate over the set size K.
    #    For a fixed K, is there a set of size K?
    #    
    #    Let's try a different ILP formulation.
    #    Variables z_i (binary).
    #    We can't easily express the subset sum condition in linear constraints without exponential variables.
    #    
    #    However, we can use the following trick:
    #    For each possible sum m, define w_m.
    #    Constraint: w_m <= sum_{i=1}^{256} z_i * (1 if i can be part of a sum m) ... no.
    #    
    #    Let's look at the "Knapsack" style formulation.
    #    Let x_{i, m} be 1 if element i is used to form sum m? No, elements are reused in different sums.
    #    
    #    Okay, standard "Subset Sum" ILP:
    #    Maximize w_target
    #    Subject to: sum z_i * a_i = target
    #    
    #    Here we have ALL targets.
    #    
    #    Maybe we can restrict the search space.
    #    The maximum sum is 256*256 = 65536.
    #    Number of sums is ~65k.
    #    Number of elements is 256.
    #    
    #    If we define w_m for all m, the constraint "m is a subset sum" is:
    #    w_m <= 1
    #    w_m >= ... ?
    #    
    #    Actually, we can use the following:
    #    For each m, w_m = 1 if and only if there exists a vector y in {0,1}^256 such that y <= z and sum y_i * i = m.
    #    
    #    This is an existential quantifier.
    #    
    #    Let's try a heuristic approach that is faster than the current one, but maybe not ILP.
    #    
    #    Revised Strategy:
    #    The current program is slow because _is_valid is O(2^n * n) or similar.
    #    We can speed up the heuristic significantly.
    #    
    #    1. Pre-compute all subset sums for the current set incrementally.
    #    2. Use a bitset for sums.
    #    3. Check divisibility in O(S / log S) where S is max sum.
    #    
    #    Let's implement a highly optimized heuristic.
    
    # 3. Optimized Heuristic Search
    #    Use bitsets for subset sums.
    #    Python integers are arbitrary precision bitsets.
    #    sum_bits = 1 (bit 0 is set)
    #    For each x in S: sum_bits |= (sum_bits << x)
    #    Then check divisibility.
    
    def _fast_check(S):
        if not S:
            return True
        sum_bits = 1
        for x in S:
            sum_bits |= (sum_bits << x)
        
        # Extract sums
        # max_sum = sum(S)
        # We need to check if any two set bits s, t have s|t.
        # Iterate through set bits.
        # To get set bits:
        # while sum_bits:
        #    lsb = sum_bits & -sum_bits
        #    s = lsb.bit_length() - 1
        #    ...
        #    sum_bits ^= lsb
        
        # This is O(number_of_sums * log(max_sum)).
        # Number of sums can be up to 2^n, but for n=15, 32k sums.
        # Divisibility check: for each s, check multiples.
        
        sums = []
        temp = sum_bits
        while temp:
            lsb = temp & -temp
            s = lsb.bit_length() - 1
            if s > 0: # Exclude 0
                sums.append(s)
            temp ^= lsb
        
        sums.sort()
        for s in sums:
            if s == 0:
                continue
            multiple = 2 * s
            # Check if multiple is in sums
            # Using a set for O(1) lookup
            # But sums is sorted, we can use binary search or just a set.
            pass
        
        # Let's use a set for O(1) lookup for the divisibility check
        sum_set = set(sums)
        for s in sums:
            if s == 0:
                continue
            multiple = 2 * s
            while multiple <= sum(S):
                if multiple in sum_set:
                    return False
                multiple += s
        return True

    # This bitset approach is MUCH faster for n > 10.
    # Let's rewrite the greedy using this.
    
    def _greedy_build_fast(pool, deadline, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
            
        current = []
        available = pool[:]
        rng.shuffle(available)
        
        while time.time() < deadline:
            if not available:
                break
            
            added = False
            # Try to add an element
            # To be efficient, we should try elements that are "compatible"
            # But we don't know compatibility without checking.
            # Just iterate.
            
            # Optimization: Sort available by "divisibility risk"?
            # Or just random.
            
            for i in range(len(available)):
                x = available[i]
                test_set = current + [x]
                if _fast_check(test_set):
                    current.append(x)
                    available.pop(i)
                    added = True
                    break
            
            if not added:
                # Local search: 1-for-1 swap
                if len(current) > 0 and available:
                    for i in range(len(current)):
                        if time.time() > deadline:
                            break
                        removed = current[i]
                        temp = current[:i] + current[i+1:]
                        for j in range(len(available)):
                            if time.time() > deadline:
                                break
                            x = available[j]
                            test_set = temp + [x]
                            if _fast_check(test_set):
                                current = test_set
                                available[j] = removed
                                added = True
                                break
                        if added:
                            break
                
                # Local search: 2-for-1 swap (remove 2, add 1) - reduces size, skip.
                # Local search: 1-for-2 swap (remove 1, add 2) - increases size.
                if not added and len(current) >= 1 and len(available) >= 2:
                    for i in range(len(current)):
                        if time.time() > deadline:
                            break
                        removed = current[i]
                        temp = current[:i] + current[i+1:]
                        
                        # Try to add two
                        # Sample a few pairs
                        for _ in range(10):
                            if time.time() > deadline:
                                break
                            a_idx = rng.randint(0, len(available)-1)
                            b_idx = rng.randint(0, len(available)-1)
                            if a_idx == b_idx:
                                continue
                            a = available[a_idx]
                            b = available[b_idx]
                            test_set = temp + [a, b]
                            if _fast_check(test_set):
                                current = test_set
                                # Update available
                                # Remove a and b, add removed
                                # Careful with indices
                                if a_idx < b_idx:
                                    available.pop(b_idx)
                                    available.pop(a_idx)
                                else:
                                    available.pop(a_idx)
                                    available.pop(b_idx)
                                available.append(removed)
                                added = True
                                break
                        if added:
                            break
                
                if not added:
                    break
                    
        return current

    # Run the fast heuristic
    best_set = []
    for i in range(20): # More attempts since it's faster
        if time.time() > deadline - 5:
            break
        S = _greedy_build_fast(pool, deadline - 5, seed=i)
        if len(S) > len(best_set):
            best_set = S
            # If we find a large set, we can stop early?
            # Max possible is likely around 15-20.
            if len(S) > 18:
                break

    # Final verification
    if best_set and not _is_valid(best_set):
        # Fallback to a known small set if verification fails (should not happen)
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        # Verify fallback
        if not _is_valid(best_set):
             best_set = [2, 3, 5, 7, 11, 13, 17, 19]

    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
