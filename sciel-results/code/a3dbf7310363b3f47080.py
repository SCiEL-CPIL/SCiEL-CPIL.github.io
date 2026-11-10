import time
import itertools

# EVOLVE-BLOCK-START

def _is_dissociated(S):
    """
    Check if all subset sums of reciprocals of elements in S are distinct.
    S is a list of integers.
    We check sums of 1/x for x in S.
    """
    if not S:
        return True
    
    # Use exact arithmetic to avoid floating point errors
    # Sum of reciprocals: sum(1/x for x in subset)
    # To be distinct, no two different subsets can have the same sum.
    # This is equivalent to checking if the set {sum(1/x for x in A) | A subset of S} has size 2^|S|.
    
    # For efficiency, we can check collisions incrementally or just compute all sums if |S| is small.
    # Max |S| we might reach is likely around 10-12. 2^12 = 4096 sums. This is fast.
    
    sums = set()
    n = len(S)
    # Iterate over all subsets using bitmasks
    for mask in range(1 << n):
        s = 0.0
        # Using float might be risky for large n, but for n<=12 and values 1..22, 
        # precision should be sufficient to distinguish distinct rational sums?
        # Actually, 1/2 + 1/3 = 5/6. 1/6 = 0.1666... 
        # Floating point errors can be problematic. Let's use Fraction or a scaled integer approach.
        # Since denominators are <= 22, LCM is manageable? LCM(1..22) is large.
        # Let's stick to Fraction for correctness, or just careful float comparison if we assume 
        # the sums are well-separated. 
        # However, the task says "distinct". 
        # Let's use a set of tuples of (numerator, denominator) reduced? 
        # Or simply use the `fractions.Fraction` class.
        
        # Optimization: If we use floats, we might get false positives (distinct sums looking same) 
        # or false negatives (same sums looking different). 
        # Given the small size, Fraction is safe.
        
        subset_sum = 0
        for i in range(n):
            if mask & (1 << i):
                subset_sum += 1.0 / S[i]
        
        # To handle precision, we can round to a certain number of digits?
        # Or use Fraction.
        # Let's try a hybrid: use Fraction for the check.
        
        # Actually, let's just use a set of Fractions.
        # But creating Fractions for every subset in a tight loop might be slow.
        # Let's try a different approach: 
        # The condition is that the map A -> sum(1/x_i) is injective.
        # This is equivalent to the vectors v_i = (1/x_i) being linearly independent over Q? No, that's for linear combinations with integer coefficients.
        # Here coefficients are 0 or 1. This is "dissociated" in the additive combinatorics sense.
        
        # Let's use Fraction for robustness.
        
        # Re-doing the loop with Fraction
        # We'll compute the sum using Fraction for the current mask.
        
        pass # Placeholder, logic below

def _check_valid(S):
    from fractions import Fraction
    sums = set()
    n = len(S)
    for mask in range(1 << n):
        s = Fraction(0, 1)
        for i in range(n):
            if mask & (1 << i):
                s += Fraction(1, S[i])
        if s in sums:
            return False
        sums.add(s)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # We want to maximize cardinality.
    # Strategy: Greedy addition.
    # Order candidates. Which order?
    # Smaller numbers have larger reciprocals. 
    # Larger numbers have smaller reciprocals.
    # Collisions are more likely when sums get close.
    # It is generally easier to keep a set dissociated if the elements are "spread out" or if we add smaller reciprocals (larger indices) later?
    # Actually, if we start with 1, 1/1 = 1. 
    # If we add 2, sums: 0, 1, 0.5, 1.5. Distinct.
    # If we add 3, sums: 0, 1, 0.5, 1.5, 0.33, 1.33, 0.83, 1.83. Distinct.
    
    # Let's try adding elements in increasing order of value (1, 2, 3...) or decreasing (22, 21...)?
    # If we add 1 first, the "scale" is large. 
    # If we add 22 first, the "scale" is small.
    # Adding large numbers (small reciprocals) allows for more elements to be packed into the range [0, sum] before collisions occur?
    # Actually, the range of sums is [0, Sum(1/x)]. 
    # If we pick large x, the total sum is small, so sums are dense? 
    # No, if x is large, 1/x is small. 
    # Consider the set {22, 21, ...}. 
    # Let's try a heuristic: Start with the largest numbers (smallest reciprocals) and try to add smaller numbers?
    # Or start with smallest numbers (largest reciprocals)?
    
    # The current seed [1, 2, 3, 5, 7, 11, 13, 17, 19] has size 9.
    # Let's try to find a set of size 10, 11, etc.
    
    # We will try multiple greedy orders and keep the best result.
    # Orders to try:
    # 1. Ascending: 1, 2, ..., 22
    # 2. Descending: 22, 21, ..., 1
    # 3. Randomized restarts (if time permits)
    
    best_set = []
    
    # Candidate pool: 1 to 22
    candidates = list(range(1, 23))
    
    # Strategy 1: Greedy Ascending
    S = []
    for x in candidates:
        if time.time() > deadline:
            break
        new_S = S + [x]
        if _check_valid(new_S):
            S = new_S
    if len(S) > len(best_set):
        best_set = S
        
    # Strategy 2: Greedy Descending
    S = []
    for x in reversed(candidates):
        if time.time() > deadline:
            break
        new_S = S + [x]
        if _check_valid(new_S):
            S = new_S
    if len(S) > len(best_set):
        best_set = S
        
    # Strategy 3: Greedy with "Gap" heuristic?
    # Maybe skip 1? 1 is very large (1.0). 
    # If we include 1, the sum range is [0, 1+...]. 
    # If we exclude 1, we can potentially fit more elements?
    # Let's try greedy on [2..22], [3..22], etc.
    
    for start_val in range(1, 10):
        if time.time() > deadline:
            break
        pool = [x for x in candidates if x >= start_val]
        S = []
        for x in pool:
            if time.time() > deadline:
                break
            new_S = S + [x]
            if _check_valid(new_S):
                S = new_S
        if len(S) > len(best_set):
            best_set = S
            
    # Strategy 4: Randomized Greedy
    import random
    random.seed(42)
    for _ in range(100): # 100 restarts
        if time.time() > deadline:
            break
        pool = candidates[:]
        random.shuffle(pool)
        S = []
        for x in pool:
            if time.time() > deadline:
                break
            new_S = S + [x]
            # Optimization: Only check if adding x doesn't immediately cause obvious issues?
            # _check_valid is O(2^k * k). For k=10, 2^10=1024. 1024*10 ops = 10k ops. Very fast.
            if _check_valid(new_S):
                S = new_S
        if len(S) > len(best_set):
            best_set = S
            # If we hit 12 or 13, we might be near optimal, but keep going if time allows
            
    return best_set

# EVOLVE-BLOCK-END
