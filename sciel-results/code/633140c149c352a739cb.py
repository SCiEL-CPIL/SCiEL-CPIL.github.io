import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # L is the set of numbers in [129, 255]
    L = list(range(129, 256))
    
    # Precompute conflict sets B_k for k in [2, 128]
    # B_k contains y in L such that lcm(k, y) <= 256
    B = {}
    for k in range(2, 129):
        B[k] = set()
        for y in L:
            if lcm(k, y) <= 256:
                B[k].add(y)
                
    # Precompute reciprocal sums for subsets of L? No, too big.
    # We will calculate the cost of a set K by iterating through union of B_k.
    
    # We need to find K subset of [2, 128] such that:
    # 1. For all distinct a, b in K, lcm(a, b) > 256
    # 2. Maximize sum(1/k for k in K) - sum(1/y for y in union(B_k for k in K))
    
    # Base score from L
    score_L = sum(1.0 / y for y in L)
    
    best_score = 0.0
    best_K = []
    
    # List of candidates for K
    candidates = list(range(2, 129))
    
    # Backtracking search
    # Sort candidates by 1/k descending (small k first) to find good solutions early?
    # Or just iterate.
    
    def is_valid_pair(a, b):
        return lcm(a, b) > 256
    
    def calculate_score(K):
        if not K:
            return 0.0
        sum_k = sum(1.0 / k for k in K)
        # Union of B_k
        union_B = set()
        for k in K:
            union_B.update(B[k])
        # Cost is sum of 1/y for y in union_B
        cost = sum(1.0 / y for y in union_B)
        return sum_k - cost
    
    # Recursive search
    # To keep it fast, we can prune.
    # If current sum_k + max_possible_gain < best_score, prune.
    # Max possible gain from adding more elements is sum of remaining 1/k.
    
    def search(current_K, current_score, start_idx, remaining_sum):
        nonlocal best_score, best_K, deadline
        
        if time.time() > deadline:
            return
            
        if current_score > best_score:
            best_score = current_score
            best_K = list(current_K)
            
        # Try adding more elements
        for i in range(start_idx, len(candidates)):
            k = candidates[i]
            
            # Check if k can be added to current_K
            valid = True
            for existing in current_K:
                if not is_valid_pair(existing, k):
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Calculate new score
            # New score = current_score + 1/k - (cost of new B_k \ union_old_B_k)
            # This is complex to update incrementally without storing union.
            # Given small size of K, just recalculate?
            # K size is likely small (< 10).
            
            new_K = current_K + [k]
            new_score = calculate_score(new_K)
            
            # Heuristic bound:
            # new_score + sum(1/j for j in candidates[i+1:])
            if new_score + remaining_sum - (1.0/k) > best_score:
                search(new_K, new_score, i + 1, remaining_sum - (1.0/k))
            else:
                # Still check if new_score itself is better?
                if new_score > best_score:
                    best_score = new_score
                    best_K = new_K
            
            if time.time() > deadline:
                return

    # Initial remaining sum
    total_recip = sum(1.0 / k for k in candidates)
    search([], 0.0, 0, total_recip)
    
    # Construct final set
    # S = best_K + [y in L such that y not in union(B_k for k in best_K)] + [256]
    
    if not best_K:
        # Fallback: just L + 256
        final_set = L + [256]
    else:
        union_B = set()
        for k in best_K:
            union_B.update(B[k])
        
        # Filter L
        filtered_L = [y for y in L if y not in union_B]
        
        final_set = best_K + filtered_L + [256]
        
    return final_set

# Quick check:
# If best_K is empty, we return L + [256].
# Is L + [256] valid?
# Pairs in L: valid.
# Pairs in [256] and L:
# y in L (129..255).
# lcm(256, y) > 256?
# y does not divide 256 (checked in B? No, B is for k in 2..128).
# We need to ensure y in L does not divide 256.
# Divisors of 256 in L: None (max divisor < 256 is 128).
# So lcm(256, y) = 256*y/gcd(256,y).
# Since y does not divide 256, gcd(256,y) < y?
# Actually, if y does not divide 256, then 256/y is not integer.
# lcm(256, y) = 256 * (y / gcd(256, y)).
# y / gcd(256, y) is an integer >= 1.
# If it is 1, then y divides 256.
# Since no y in L divides 256, the factor is >= 2.
# So lcm >= 512 > 256.
# So L + [256] is always valid.

# What if best_K is not empty?
# We need to ensure 256 is compatible with best_K.
# lcm(256, k) > 256?
# k in [2, 128].
# If k divides 256, lcm = 256 (fail).
# So we must ensure no k in best_K divides 256.
# My search did NOT enforce this!
# I must add this constraint.

# Revised Plan:
# 1. Filter candidates: k must NOT divide 256.
# 2. Run search.
# 3. Construct set.

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    L = list(range(129, 256))
    
    # Candidates for K: [2, 128] that do not divide 256
    candidates = [k for k in range(2, 129) if 256 % k != 0]
    
    B = {}
    for k in candidates:
        B[k] = set()
        for y in L:
            if lcm(k, y) <= 256:
                B[k].add(y)
                
    score_L = sum(1.0 / y for y in L)
    
    best_score = 0.0
    best_K = []
    
    def is_valid_pair(a, b):
        return lcm(a, b) > 256
    
    def calculate_score(K):
        if not K:
            return 0.0
        sum_k = sum(1.0 / k for k in K)
        union_B = set()
        for k in K:
            union_B.update(B[k])
        cost = sum(1.0 / y for y in union_B)
        return sum_k - cost
    
    def search(current_K, current_score, start_idx, remaining_sum):
        nonlocal best_score, best_K, deadline
        
        if time.time() > deadline:
            return
            
        if current_score > best_score:
            best_score = current_score
            best_K = list(current_K)
            
        for i in range(start_idx, len(candidates)):
            k = candidates[i]
            
            valid = True
            for existing in current_K:
                if not is_valid_pair(existing, k):
                    valid = False
                    break
            
            if not valid:
                continue
            
            new_K = current_K + [k]
            new_score = calculate_score(new_K)
            
            if new_score + remaining_sum - (1.0/k) > best_score:
                search(new_K, new_score, i + 1, remaining_sum - (1.0/k))
            else:
                if new_score > best_score:
                    best_score = new_score
                    best_K = new_K
            
            if time.time() > deadline:
                return

    total_recip = sum(1.0 / k for k in candidates)
    search([], 0.0, 0, total_recip)
    
    if not best_K:
        final_set = L + [256]
    else:
        union_B = set()
        for k in best_K:
            union_B.update(B[k])
        
        filtered_L = [y for y in L if y not in union_B]
        
        # Ensure 256 is compatible with best_K (already enforced by candidate filter)
        # Ensure 256 is compatible with filtered_L (already proven)
        # Ensure best_K is compatible with filtered_L (enforced by B definition)
        
        final_set = best_K + filtered_L + [256]
        
    return final_set
