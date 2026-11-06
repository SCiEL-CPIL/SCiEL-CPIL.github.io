"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # Strategy:
    # 1. Pre-compute a "conflict score" for each number in [1, 512].
    #    A number 'a' is "dangerous" if it can easily be the mean of other numbers.
    #    Specifically, if 'a' is small, it's harder to be a mean of large numbers, 
    #    but if 'a' is in the middle, it's a mean of many pairs (a-d, a+d).
    #    However, the condition is "No element may be the exact arithmetic mean of any subset of at least two OTHER elements".
    #    This means for any x in S, there is no subset T of S\{x} (|T|>=2) such that sum(T)/|T| = x.
    #    This is equivalent to: for any x in S, there is no k>=2 and subset T of S\{x} of size k such that sum(T) = k*x.
    
    # 2. Greedy Construction:
    #    We build the set S incrementally.
    #    Candidate pool: [1, 512].
    #    We try to add candidates in a specific order.
    #    Order heuristic: 
    #       - Numbers that are "structurally independent" are better.
    #       - Often, taking numbers that are far apart or have specific modular properties helps.
    #       - A simple heuristic is to sort candidates by their "density" of potential mean-partners.
    #       - Alternatively, we can just try random orders or sorted orders and pick the best result within time.
    
    #    Check function:
    #    When considering adding 'x' to current set 'S':
    #    Condition 1: 'x' is not the mean of any subset of S.
    #       This requires checking if for any k>=2, sum(T) = k*x for some T subset of S, |T|=k.
    #       This is a subset-sum problem on S with target k*x.
    #       Since |S| is at most 48, we can use bitset DP for subset sums of S.
    #       We need to check targets 2x, 3x, ..., |S|*x.
    #       
    #    Condition 2: No element y in S is the mean of any subset of (S U {x}).
    #       This is harder. Adding x might create a new subset T' of (S U {x})\{y} that has mean y.
    #       T' can include x or not.
    #       If T' does not include x, it was already a subset of S\{y}, so it was already checked when y was added (or when S was built).
    #       Wait, the condition is "No element may be the mean...". If we build S incrementally, when we add x, we must ensure:
    #       a) x is not the mean of any subset of S.
    #       b) For every y in S, y is not the mean of any subset of (S U {x}) \ {y}.
    #          This is equivalent to: y is not the mean of any subset of S\{y} (already true) AND
    #          y is not the mean of any subset of (S\{y}) U {x}.
    #          Let T be a subset of (S\{y}) U {x}.
    #          If x not in T, T is subset of S\{y}. Sum(T) = k*y was already false.
    #          If x in T, let T = {x} U T_s where T_s is subset of S\{y}.
    #          Then sum(T) = x + sum(T_s).
    #          We need x + sum(T_s) = k*y for some k >= 2.
    #          Note: The size of T is |T_s| + 1. So k = |T_s| + 1.
    #          Thus we need: sum(T_s) = (|T_s| + 1)*y - x.
    #          This must hold for some subset T_s of S\{y} with size m = |T_s| >= 1 (since k>=2 implies m>=1).
    #          Actually, the subset T must have size k>=2. If T={x, y1, ..., ym}, size is m+1.
    #          So we need to check if there exists a subset T_s of S\{y} of size m>=1 such that sum(T_s) = (m+1)y - x.
    
    #    This check (b) is expensive if done naively for every y in S.
    #    However, we can optimize.
    #    Let's refine the check for adding x to S:
    #    1. Check if x is mean of S:
    #       Compute bitset DP for subset sums of S, grouped by size.
    #       dp_s[k] is bitset of sums achievable with k elements from S.
    #       Check if (2x) in dp_s[2], (3x) in dp_s[3], etc.
    #       
    #    2. Check if any y in S becomes a mean involving x:
    #       For each y in S:
    #       We need to check if there is a subset T_s of S\{y} with sum(T_s) + x = k*y.
    #       Let m = |T_s|. Then k = m+1.
    #       sum(T_s) = (m+1)y - x.
    #       We can precompute for the current S the subset sums of S\{y}? That's too slow.
    #       
    #    Alternative approach: 
    #    Instead of checking "does adding x break y", we can just check the validity of the new set S' = S U {x} from scratch?
    #    No, that's O(|S'|^2 * bitops).
    #    
    #    Let's look at the constraint again.
    #    "No element may be the exact arithmetic mean of any subset of at least two OTHER elements."
    #    
    #    If we use a "conflict graph" approach where we pre-identify pairs (or tuples) that cannot coexist, we might simplify.
    #    But the mean condition involves subsets of arbitrary size, so it's not just pairwise.
    #    
    #    Let's stick to the incremental check but optimize it.
    #    
    #    Optimization for Condition 2:
    #    When adding x to S, we only need to check if x creates a new mean for some y.
    #    The new subsets involving x are of the form {x} U T_s where T_s subset S\{y}.
    #    For a fixed y, we need to know if (m+1)y - x is a subset sum of S\{y} with exactly m elements.
    #    This is still complex.
    
    #    Let's try a different heuristic: 
    #    Randomized Local Search / Simulated Annealing on the set?
    #    Or, simpler: 
    #    Generate many random sets, and for each, run a "repair" phase.
    #    Repair: 
    #       While invalid:
    #          Find a violation (y, T).
    #          Remove one element from T or y. 
    #          Heuristic: Remove the element that is involved in the most violations.
    
    #    Let's implement the "Repair" strategy with a good violation finder.
    #    Violation finder:
    #       For each y in S:
    #          Compute subset sums of S\{y} by size.
    #          Check if k*y is in dp[k] for k>=2.
    #          If yes, return (y, k).
    #    
    #    To speed up:
    #       We can cache the DP for S\{y}? No, S changes.
    #       But S is small (<=48).
    #       DP for 47 elements: 48 * 47 bitset shifts.
    #       Bitset size: sum(S) <= 48 * 512 = 24576.
    #       Python ints handle this fast.
    #       48 * 47 = 2256 shifts per y.
    #       48 y's -> ~100k shifts per full check.
    #       If we do 1000 checks, 100M shifts. Might be too slow in Python.
    
    #    Let's try to be smarter about the check.
    #    We only need to check k such that k*y <= sum(S\{y}).
    #    Also, k >= 2.
    
    #    Let's try the "Greedy Construction" with a simpler check:
    #    When adding x to S, check if x is a mean of S. (Fast)
    #    Then, check if any y in S is a mean of S U {x}.
    #    We can approximate this: If x is large, it's unlikely to be part of a small subset summing to small k*y.
    
    #    Actually, let's look at the "Inspirations".
    #    Inspiration 1 (1.1538) uses a greedy approach iterating 1 to 128? No, "iterates through numbers 1 to 128".
    #    Wait, the range is 1 to 512.
    
    #    Let's try a "Conflict-Aware Greedy".
    #    Precompute for each pair (a, b) if they form an arithmetic progression with some c?
    #    No, the subset size is >= 2.
    #    If subset size is 2: mean of a, b is (a+b)/2. If (a+b)/2 is in S, violation.
    #    So, if we ensure that S is "sum-free" in the sense that no element is the average of two others, we handle k=2.
    #    For k=3: mean of a, b, c is (a+b+c)/3.
    
    #    A strong heuristic: 
    #    Select numbers that are "sparse". 
    #    E.g., powers of 2? 1, 2, 4, ..., 256. Size 9. Too small.
    #    We need 48 numbers.
    
    #    Let's go with the **Repair** strategy, but optimize the violation detection.
    #    We will maintain the set S.
    #    We will check validity.
    #    If invalid, we identify a violating element y and a subset size k.
    #    We then remove an element from S. Which one?
    #    Option A: Remove y.
    #    Option B: Remove an element from the subset T that sums to k*y.
    #    We don't have T explicitly, but we know a sum exists.
    #    We can reconstruct T if we store the DP choices.
    #    
    #    Let's implement a DP that stores the parent pointer to reconstruct the subset.
    #    This allows us to remove a "bad" element from T, potentially preserving y.
    #    Or we can just remove y, which is simpler.
    #    Removing y is safe.
    
    #    To maximize size, we want to remove the element that "causes" the most future conflicts.
    #    A simple proxy: Remove the element that is the mean in the most violations?
    #    Or just random removal?
    
    #    Let's try: 
    #    1. Start with a random set of size 60.
    #    2. While invalid:
    #       Find one violation (y, T).
    #       Remove a random element from T U {y}.
    #    3. Repeat with different random seeds/orders.
    
    #    To make it faster, we can use a "soft" constraint or just optimize the check.
    
    #    Let's write the code for the Repair strategy with reconstruction.
    
    def _find_violation_with_reconstruction(S):
        """
        Returns (y, subset_list) if violation exists, else (None, None).
        S is a list of distinct ints.
        """
        n = len(S)
        if n < 3:
            return None, None
        
        # Sort S for determinism? Not needed.
        
        for i, y in enumerate(S):
            T = S[:i] + S[i+1:]
            n_t = len(T)
            
            # DP: dp[k] is a dict or bitset?
            # To reconstruct, we need to know which element was added.
            # We can use a list of dicts: dp[k][sum] = last_element_added
            # Or we can just use bitsets and if we find a hit, we backtrack.
            # Backtracking with bitsets is hard because we don't store the path.
            
            # Let's use a list of sets of sums, and a parent map.
            # parent[k][s] = (prev_sum, element)
            # This uses too much memory.
            
            # Alternative: 
            # Just find IF a violation exists.
            # If it does, we don't need the exact subset to remove y.
            # Removing y is a valid move.
            # So we don't need reconstruction.
            
            # Optimization: 
            # We can break early if we find a violation.
            
            # Let's stick to the bitset check for existence.
            dp = [0] * (n_t + 1)
            dp[0] = 1
            
            for x in T:
                for k in range(n_t, 0, -1):
                    if dp[k-1]:
                        dp[k] |= (dp[k-1] << x)
                        
            max_sum_t = sum(T)
            for k in range(2, n_t + 1):
                target = k * y
                if target > max_sum_t:
                    break
                if (dp[k] >> target) & 1:
                    return y, None
                    
        return None, None

    def _repair(S):
        """
        Iteratively removes elements until S is valid.
        Returns the valid set.
        """
        S = list(S)
        while True:
            if time.time() > deadline:
                return S
            y, _ = _find_violation_with_reconstruction(S)
            if y is None:
                return S
            # Remove y
            S.remove(y)
            if len(S) < 2:
                return S

    best_set = []
    
    # Try multiple random initializations
    # Also try sorted initializations?
    
    num_attempts = 500
    for i in range(num_attempts):
        if time.time() > deadline:
            break
            
        # Random size between 50 and 70
        candidate_size = random.randint(55, 70)
        candidates = random.sample(range(1, 513), candidate_size)
        
        # Shuffle to randomize removal order?
        random.shuffle(candidates)
        
        valid_set = _repair(candidates)
        
        if len(valid_set) > len(best_set):
            best_set = valid_set
            if len(best_set) == 48:
                break
                
    # If we haven't found a good set, try a greedy construction from scratch
    # Greedy: Add elements one by one, checking validity of the new set.
    # This is slower per step but might find a larger core if random repair fails.
    
    if len(best_set) < 45:
        greedy_set = []
        # Try adding in random order
        pool = list(range(1, 513))
        random.shuffle(pool)
        
        for x in pool:
            if time.time() > deadline:
                break
            if len(greedy_set) >= 48:
                break
            new_set = greedy_set + [x]
            # Check if new_set is valid
            # We only need to check if x is a mean of greedy_set
            # AND if any y in greedy_set is a mean of new_set\{y}
            
            # Fast check for x being mean of greedy_set:
            # Use DP on greedy_set.
            # If greedy_set is small, this is fast.
            
            # Let's just use the full check for simplicity, as greedy_set grows slowly.
            y, _ = _find_violation_with_reconstruction(new_set)
            if y is None:
                greedy_set.append(x)
                
        if len(greedy_set) > len(best_set):
            best_set = greedy_set

    return best_set

# EVOLVE-BLOCK-END
