import time
import random
from math import gcd
from functools import reduce

# Deadline guard
_START_TIME = time.time()
DEADLINE = 85.0  # seconds, safety margin before 90s kill

# Primes for modular checking
# We need primes large enough to distinguish subset sums.
# If |S| = k, there are 2^k sums. We need P > 2^k to avoid wrap-around collisions 
# that look like real collisions. 
# However, even if P < 2^k, a collision mod P implies a collision in integers 
# only if the difference is a multiple of P. 
# Since the max sum is sum(1/s) < 1 + 1/2 + ... < 4, the integer values 
# L/s are bounded. 
# Actually, the modular check is: 
# Does there exist distinct subsets A, B such that sum_{a in A} 1/a == sum_{b in B} 1/b?
# This is equivalent to sum_{a in A} P/a - sum_{b in B} P/b == 0 mod P.
# Let w_i = P * inv(s_i) mod P.
# We check if any subset sum of w_i is 0 (excluding empty set) or if any two subset sums are equal.
# Checking if any subset sum is 0 mod P is sufficient to detect collisions 
# if we consider the difference set. 
# Specifically, if sum(A) == sum(B), then sum(A \ B) - sum(B \ A) = 0.
# So we just need to check if there is a non-empty subset of {w_i} with sum 0 mod P.
# Wait, this is for the difference. 
# Let's stick to checking for duplicate subset sums. It's safer.
# But checking for a zero-sum subset is equivalent to checking for collisions 
# if we allow coefficients -1, 0, 1.
# Yes, dissociation is exactly that no non-trivial combination of -1,0,1 sums to 0.
# So we can check: for a given set of weights w, is there a subset with sum 0 mod P?
# This is the "Subset Sum" problem. 
# For k=15, 2^15 checks.
# We can do this for multiple primes.

def _is_dissociated_fast(S, primes=(1000000007, 1000000009)):
    """
    Check if S is dissociated using modular arithmetic.
    Returns True if likely dissociated.
    """
    n = len(S)
    if n == 0:
        return True
    if n > 20:
        # For very large n, we can't do 2^n. 
        # But max n is 22. 2^22 is 4 million. 
        # 4 million * 2 primes = 8 million ops. Might be slow in Python.
        # We'll try up to 18 for fast checks, and use a slower heuristic for >18.
        # Actually, let's just try up to 18. If we find a set of size 18, we are done.
        # If not, we might miss size 19+.
        # Let's optimize the loop.
        pass

    for p in primes:
        # Check if p divides any element
        valid = True
        for s in S:
            if s % p == 0:
                valid = False
                break
        if not valid:
            continue
        
        # Compute modular weights: w_i = P * inv(s_i) mod P
        # Note: 1/s mod P is inv(s). We want to check sum(1/s) collisions.
        # Collision: sum_{a in A} 1/a = sum_{b in B} 1/b
        # => sum_{a in A} P/a - sum_{b in B} P/b = 0 mod P
        # Let w_i = P * inv(s_i) mod P.
        # We need to check if there exist disjoint subsets A, B (not both empty) 
        # such that sum_{a in A} w_a == sum_{b in B} w_b mod P.
        # This is equivalent to checking if there is a non-empty subset C 
        # such that sum_{c in C} w_c == 0 mod P, where C is the symmetric difference?
        # No. 
        # If sum(A) == sum(B), then sum(A \ B) - sum(B \ A) = 0.
        # Let D = A \ B and E = B \ A. D and E are disjoint.
        # sum_{d in D} w_d - sum_{e in E} w_e = 0 mod P.
        # This is a signed subset sum problem.
        # We can check this by generating all subset sums of w, and checking 
        # if any sum appears twice? No, that's for positive sums.
        # For signed sums, we can just check if any subset sum is 0?
        # No, because we have negative signs.
        # However, if we consider the set of all subset sums of w, 
        # a collision sum(A) == sum(B) means sum(A) - sum(B) = 0.
        # This difference is a sum of elements with coefficients -1, 0, 1.
        # So we need to check if there is a non-empty subset of indices I 
        # and signs s_i in {-1, 1} such that sum s_i w_i = 0 mod P.
        
        # Efficient way:
        # Generate all 2^n subset sums of w.
        # If any sum (excluding 0) appears, it doesn't mean collision.
        # Collision means sum(A) == sum(B).
        # This is equivalent to: does the set of subset sums contain a duplicate?
        # Yes! 
        # If sum(A) == sum(B) mod P, and A != B, then we have a collision.
        # So we just need to check if the 2^n subset sums are distinct mod P.
        
        if n > 18:
            # Too slow for 2 primes if n is large.
            # We'll just do one prime for n > 18.
            if n > 20:
                # Fallback: assume valid if it passed smaller checks? 
                # Or just skip?
                # Let's try to be safe. If n > 20, we are in deep search.
                # We'll do a single prime check.
                pass

        seen = set()
        seen.add(0)
        # Iterate over all 2^n subsets
        # Optimization: use iterative DP or simple loop
        sums = [0]
        for w in [pow(s, p - 2, p) for s in S]:
            # Add w to all existing sums
            new_sums = [(s + w) % p for s in sums]
            sums.extend(new_sums)
            # Check for duplicates
            # If len(sums) != len(set(sums)), collision found
            # But set(sums) is O(N log N) or O(N).
            # For n=18, len(sums) = 2^18 = 262k.
            # Checking duplicates at each step is expensive.
            # Better to generate all, then check.
            
        # Actually, generating all sums and then checking for duplicates is O(2^n).
        # Checking for duplicates is O(2^n).
        # Total O(2^n).
        
        # Let's do it properly.
        all_sums = [0]
        for s in S:
            w = pow(s, p - 2, p)
            current_len = len(all_sums)
            for i in range(current_len):
                all_sums.append((all_sums[i] + w) % p)
        
        # Check for duplicates
        if len(all_sums) != len(set(all_sums)):
            return False
            
    return True

def construct():
    """
    Return a nonempty subset of [1,22] such that ALL reciprocal subset sums,
    including the empty sum, are distinct. Maximize cardinality.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    best_set = []
    best_size = 0
    rng = random.Random(42)
    
    # 1. Greedy Initialization
    # Try adding from largest to smallest
    S = []
    for x in range(22, 0, -1):
        if _is_dissociated_fast(S + [x]):
            S.append(x)
    if len(S) > best_size:
        best_set = S[:]
        best_size = len(S)
        
    # 2. Local Search
    current_S = best_set[:]
    
    while time.time() < DEADLINE:
        improved = False
        
        # Try to add
        if len(current_S) < 22:
            candidates = [x for x in range(1, 23) if x not in current_S]
            rng.shuffle(candidates)
            for c in candidates:
                if _is_dissociated_fast(current_S + [c]):
                    current_S.append(c)
                    improved = True
                    if len(current_S) > best_size:
                        best_size = len(current_S)
                        best_set = current_S[:]
                    break
        
        # Try to swap
        if not improved and len(current_S) >= 1:
            # Try removing one and adding another
            # To save time, only try a few random removals
            for _ in range(10):
                i = rng.randint(0, len(current_S) - 1)
                removed = current_S[i]
                S_new = current_S[:i] + current_S[i+1:]
                
                candidates = [x for x in range(1, 23) if x not in S_new and x != removed]
                rng.shuffle(candidates)
                for c in candidates[:10]: # Try a few candidates
                    if _is_dissociated_fast(S_new + [c]):
                        # Check if this swap actually improves size? 
                        # No, swap keeps size same. 
                        # But it might allow adding more later.
                        # So we accept it if it's valid.
                        current_S = S_new + [c]
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # Stuck. Perturb.
            # Remove 2 random elements
            if len(current_S) > 2:
                i1 = rng.randint(0, len(current_S) - 1)
                i2 = rng.randint(0, len(current_S) - 1)
                if i1 == i2:
                    continue
                # Remove both
                if i1 > i2:
                    i1, i2 = i2, i1
                current_S.pop(i1)
                current_S.pop(i2)
                # Try to add 2 random elements
                candidates = [x for x in range(1, 23) if x not in current_S]
                rng.shuffle(candidates)
                for c in candidates[:5]:
                    if _is_dissociated_fast(current_S + [c]):
                        current_S.append(c)
                for c in candidates[:5]:
                    if c not in current_S and _is_dissociated_fast(current_S + [c]):
                        current_S.append(c)
                
                if len(current_S) > best_size:
                    best_size = len(current_S)
                    best_set = current_S[:]
            
            # If we are stuck at a small size, restart with a random set
            if best_size < 10 and rng.random() < 0.1:
                size = rng.randint(5, 8)
                current_S = rng.sample(range(1, 23), size)
                if not _is_dissociated_fast(current_S):
                    # Fix it
                    while current_S and not _is_dissociated_fast(current_S):
                        current_S.pop(rng.randint(0, len(current_S)-1))

    # Final verification
    if not _is_dissociated_fast(best_set):
        # Fallback to a known good set or greedy
        best_set = _greedy_build_fallback()
        
    return sorted(best_set)

def _greedy_build_fallback():
    S = []
    for x in range(22, 0, -1):
        if _is_dissociated_fast(S + [x]):
            S.append(x)
    return S

if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}: {result}")
