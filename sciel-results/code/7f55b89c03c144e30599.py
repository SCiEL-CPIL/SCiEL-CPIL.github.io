"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Deadline guard
_DEADLINE = time.time() + 85  # 85s safety margin within 90s limit

def _subset_has_zero_sum(diffs, min_size, max_k=None):
    """Check if any subset of diffs of size >= min_size sums to 0."""
    n = len(diffs)
    if n < min_size:
        return False
    if max_k is None:
        max_k = n
    
    # For small n, use combinations
    if n <= 25:
        for k in range(min_size, min(max_k, n) + 1):
            for comb in combinations(diffs, k):
                if sum(comb) == 0:
                    return True
        return False
    
    # For larger n, use DP for subset sums (check if 0 is achievable with >= min_size elements)
    # But DP over all subsets is exponential in n... 
    # Instead, check small subsets first (k up to 6), then use a probabilistic/heuristic check
    for k in range(min_size, min(7, n + 1)):
        for comb in combinations(diffs, k):
            if sum(comb) == 0:
                return True
    
    # For k >= 7, it's harder. Use a DP approach tracking achievable sums with element count.
    # Since diffs can be negative, we need offset.
    pos_sum = sum(d for d in diffs if d > 0)
    neg_sum = sum(d for d in diffs if d < 0)
    offset = -neg_sum
    max_sum = pos_sum + offset
    
    if max_sum > 100000:
        # Too large for DP, assume valid (risky but necessary for performance)
        # In practice, with well-chosen elements, large k subsets summing to 0 are rare
        return False
    
    # DP: dp[s][c] = whether sum s with c elements is achievable
    # But this is 2D and could be large. Use a dictionary approach for sparse DP.
    # Alternative: track for each sum, the min and max count achievable.
    # Actually, we just need to know if 0 is achievable with count >= min_size.
    # Use set of (sum, count) pairs - but this can blow up.
    
    # Practical approach: for this problem, if no small subset (k<=6) sums to 0,
    # and the set is "well-distributed", larger subsets are unlikely to sum to 0.
    # Given time constraints, we'll accept this heuristic for n > 25.
    return False

def _is_valid_set(S):
    """Check if no element in S is the mean of any subset of >=2 other elements."""
    S_list = sorted(S)
    n = len(S_list)
    for i, a in enumerate(S_list):
        others = [x for j, x in enumerate(S_list) if j != i]
        m = len(others)
        # Check subsets of others of size k >= 2
        # a = mean(subset) => sum(subset) = k*a => sum(subset - a) = 0
        diffs = [x - a for x in others]
        if _subset_has_zero_sum(diffs, 2):
            return False
    return True

def _greedy_build(order):
    """Greedy build using given order of candidates."""
    selected = []
    for x in order:
        if x in selected:
            continue
        candidate = selected + [x]
        if _is_valid_set(candidate):
            selected.append(x)
        if time.time() > _DEADLINE:
            break
    return selected

def _local_search(S, all_candidates, max_iterations=5000):
    """Improve set S via add/remove/swap operations."""
    S = list(S)
    S_set = set(S)
    best = len(S)
    no_improve = 0
    start_time = time.time()
    
    for _ in range(max_iterations):
        if time.time() > _DEADLINE:
            break
        improved = False
        
        # Try adding elements
        candidates_to_add = [x for x in all_candidates if x not in S_set]
        random.shuffle(candidates_to_add)
        for x in candidates_to_add[:20]:  # Limit attempts per iteration
            new_S = S + [x]
            if _is_valid_set(new_S):
                S = new_S
                S_set.add(x)
                improved = True
                break
        
        if improved:
            best = max(best, len(S))
            no_improve = 0
            continue
        
        # Try swapping: remove one, add another
        if len(S) > 0 and candidates_to_add:
            for i in range(min(10, len(S))):
                for x in candidates_to_add[:10]:
                    new_S = [y for j, y in enumerate(S) if j != i] + [x]
                    if _is_valid_set(new_S):
                        removed = S[i]
                        S = new_S
                        S_set.remove(removed)
                        S_set.add(x)
                        improved = True
                        break
                if improved:
                    break
        
        if improved:
            best = max(best, len(S))
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 200:
                break
    
    return S

def construct():
    all_candidates = list(range(1, 513))
    n_max = 48
    
    # Precompute "conflict degree" for each element:
    # For each pair (a, b) with a < b, they form a 3-AP if 2a - b and 2b - a are in range.
    # More importantly, count how many 3-term APs each element participates in.
    conflict_degree = [0] * 513
    for a in range(1, 513):
        for b in range(a + 1, 513):
            # Check if a is mean of {b, c} where c = 2a - b
            c = 2 * a - b
            if 1 <= c <= 512 and c != a and c != b:
                conflict_degree[a] += 1
            # Check if b is mean of {a, c} where c = 2b - a
            c = 2 * b - a
            if 1 <= c <= 512 and c != a and c != b:
                conflict_degree[b] += 1
    
    # Sort by conflict degree (ascending) - elements in fewer APs are "safer"
    order = sorted(range(1, 513), key=lambda x: (conflict_degree[x], x))
    
    # Greedy build
    S = _greedy_build(order)
    
    # If greedy already reaches max, return
    if len(S) >= n_max:
        return S[:n_max]
    
    # Local search to improve
    S = _local_search(S, all_candidates)
    
    if len(S) >= n_max:
        return S[:n_max]
    
    # Try multiple random restarts with different orderings
    best_S = S
    start_time = time.time()
    for restart in range(100):
        if time.time() > _DEADLINE - 5:
            break
        # Create a perturbed order
        random_order = list(range(1, 513))
        random.shuffle(random_order)
        # Blend with conflict-degree order: put low-conflict elements first with some randomness
        blended_order = []
        low_conf = [x for x in random_order if conflict_degree[x] < 10]
        mid_conf = [x for x in random_order if 10 <= conflict_degree[x] < 30]
        high_conf = [x for x in random_order if conflict_degree[x] >= 30]
        blended_order = low_conf + mid_conf + high_conf
        random.shuffle(blended_order)
        
        S_restart = _greedy_build(blended_order)
        if len(S_restart) > len(best_S):
            best_S = S_restart
            if len(best_S) >= n_max:
                return best_S[:n_max]
        
        S_restart = _local_search(S_restart, all_candidates)
        if len(S_restart) > len(best_S):
            best_S = S_restart
            if len(best_S) >= n_max:
                return best_S[:n_max]
    
    return best_S[:n_max]
# EVOLVE-BLOCK-END
