"""Improved program for sumset/difference-set exponent optimization."""
import time
import math
from collections import defaultdict

# Deadline guard: evaluator kills after 150s wall-clock; keep safety margin.
_DEADLINE = time.time() + 130.0  # 20s safety margin

def _compute_score(U):
    """Compute the lower bound score for set U."""
    if not U or 0 not in U:
        return 0.0
    U = list(set(U))
    n = len(U)
    max_u = max(U)
    if max_u == 0:
        return 0.0
    # Compute |U+U| and |U-U|
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    n_sum = len(sum_set)
    n_diff = len(diff_set)
    if n_sum == 0:
        return 0.0
    ratio = n_diff / n_sum
    if ratio <= 0:
        return 0.0
    log_arg = 2 * max_u + 1
    if log_arg <= 1:
        return 0.0
    score = 1.0 + math.log(ratio) / math.log(log_arg)
    return score


def _greedy_construct(target_size, max_val, start_set=None):
    """
    Greedy construction: start from a small set and iteratively add the element
    that maximizes the marginal improvement in the score.
    """
    if start_set is None:
        U = [0, 1, 3]
    else:
        U = list(set(start_set))
    
    if len(U) >= target_size:
        return U
    
    # Precompute for efficiency
    def _eval_candidate(U, cand):
        """Compute score if we add cand to U."""
        U_new = U + [cand]
        # Only compute sum and diff sets involving cand for efficiency
        # Actually for small sets, full recomputation is fine
        n = len(U_new)
        max_u = max(U_new)
        if max_u == 0:
            return 0.0
        
        sum_set = set()
        diff_set = set()
        for a in U_new:
            for b in U_new:
                sum_set.add(a + b)
                diff_set.add(a - b)
        
        n_sum = len(sum_set)
        n_diff = len(diff_set)
        if n_sum == 0:
            return 0.0
        ratio = n_diff / n_sum
        if ratio <= 0:
            return 0.0
        log_arg = 2 * max_u + 1
        if log_arg <= 1:
            return 0.0
        return 1.0 + math.log(ratio) / math.log(log_arg)
    
    current_score = _eval_candidate(U, 0)  # dummy, just to get current
    
    # Actually compute current score properly
    n = len(U)
    max_u = max(U)
    if max_u == 0:
        return U
    
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    n_sum = len(sum_set)
    n_diff = len(diff_set)
    if n_sum == 0:
        return U
    ratio = n_diff / n_sum
    if ratio <= 0:
        return U
    log_arg = 2 * max_u + 1
    if log_arg <= 1:
        return U
    current_score = 1.0 + math.log(ratio) / math.log(log_arg)
    
    U_set = set(U)
    
    for _ in range(target_size - len(U)):
        if time.time() > _DEADLINE:
            break
        
        best_cand = None
        best_score = current_score
        best_gain = 0.0
        
        # Candidate range: from 0 to max_val
        # To speed up, focus on candidates near existing elements or at strategic positions
        # Try a subset of candidates
        candidates = set()
        
        # Always consider small values
        for c in range(0, min(max_val + 1, max_u + 50)):
            if c not in U_set:
                candidates.add(c)
        
        # Consider values near existing elements
        for u in U:
            for delta in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]:
                c = u + delta
                if 0 <= c <= max_val and c not in U_set:
                    candidates.add(c)
                c = u - delta
                if 0 <= c <= max_val and c not in U_set:
                    candidates.add(c)
        
        # Consider some larger values
        for c in range(max_u, max_val + 1, max(1, (max_val - max_u) // 50)):
            if c not in U_set:
                candidates.add(c)
        
        # Limit candidates for speed
        if len(candidates) > 200:
            # Sample
            import random
            candidates = random.sample(list(candidates), 200)
        
        for cand in candidates:
            if time.time() > _DEADLINE:
                break
            s = _eval_candidate(U, cand)
            gain = s - current_score
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_cand = cand
                best_score = s
        
        if best_cand is not None:
            U.append(best_cand)
            U_set.add(best_cand)
            current_score = best_score
        else:
            break
    
    return U


def _local_search(U, max_val, iterations=50):
    """
    Local search: try swapping elements or adding/removing to improve score.
    """
    U = list(set(U))
    if not U:
        return U
    
    def _score(U):
        if not U or 0 not in U:
            return 0.0
        n = len(U)
        max_u = max(U)
        if max_u == 0:
            return 0.0
        sum_set = set()
        diff_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
                diff_set.add(a - b)
        n_sum = len(sum_set)
        n_diff = len(diff_set)
        if n_sum == 0:
            return 0.0
        ratio = n_diff / n_sum
        if ratio <= 0:
            return 0.0
        log_arg = 2 * max_u + 1
        if log_arg <= 1:
            return 0.0
        return 1.0 + math.log(ratio) / math.log(log_arg)
    
    current_score = _score(U)
    U_set = set(U)
    
    for _ in range(iterations):
        if time.time() > _DEADLINE:
            break
        
        improved = False
        
        # Try adding a new element
        best_add = None
        best_add_score = current_score
        best_add_gain = 0.0
        
        # Generate candidates
        candidates = set()
        for u in U:
            for delta in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]:
                c = u + delta
                if 0 <= c <= max_val and c not in U_set:
                    candidates.add(c)
                c = u - delta
                if 0 <= c <= max_val and c not in U_set:
                    candidates.add(c)
        
        # Also try removing an element
        best_remove = None
        best_remove_score = current_score
        best_remove_gain = 0.0
        
        for u in U:
            if u == 0:
                continue  # Must keep 0
            if time.time() > _DEADLINE:
                break
            U_test = [x for x in U if x != u]
            s = _score(U_test)
            gain = s - current_score
            if gain > best_remove_gain + 1e-12:
                best_remove_gain = gain
                best_remove = u
                best_remove_score = s
        
        # Evaluate add candidates (limit for speed)
        if len(candidates) > 100:
            import random
            candidates = random.sample(list(candidates), 100)
        
        for cand in candidates:
            if time.time() > _DEADLINE:
                break
            U_test = U + [cand]
            s = _score(U_test)
            gain = s - current_score
            if gain > best_add_gain + 1e-12:
                best_add_gain = gain
                best_add = cand
                best_add_score = s
        
        # Apply best move
        if best_add is not None and best_add_gain >= best_remove_gain:
            U.append(best_add)
            U_set.add(best_add)
            current_score = best_add_score
            improved = True
        elif best_remove is not None:
            U.remove(best_remove)
            U_set.discard(best_remove)
            current_score = best_remove_score
            improved = True
        
        if not improved:
            break
    
    return U


def _structured_construct(size):
    """
    Construct a set with good sumset/difference-set properties using
    a structured approach based on known good constructions.
    """
    # Use a base that grows slowly to keep |U+U| small while |U-U| grows
    # Strategy: use elements that are spread out but with controlled density
    
    if size <= 3:
        return [0, 1, 3]
    
    # Start with a good small set
    U = [0, 1, 3, 7, 15]  # Powers of 2 minus 1
    
    # Extend greedily
    max_val = 2 * (2 ** (size - 1)) + 100  # generous bound
    
    U = _greedy_construct(size, max_val, start_set=U)
    U = _local_search(U, max_val, iterations=20)
    
    return sorted(set(U))


def construct():
    """
    Construct a finite set U of nonnegative integers containing 0 to maximize
    the sumset/difference-set exponent lower bound.
    
    Returns: list of nonnegative integers containing 0.
    """
    start_time = time.time()
    
    # Multiple strategies with different parameters
    best_U = [0, 1, 3]
    best_score = _compute_score(best_U)
    
    # Strategy 1: Greedy from different starting points
    starts = [
        [0, 1, 3],
        [0, 1, 4],
        [0, 2, 5],
        [0, 1, 2, 5],
        [0, 3, 7, 15],
    ]
    
    target_sizes = [10, 20, 30, 50, 80, 120]
    
    for start in starts:
        if time.time() > _DEADLINE:
            break
        
        for ts in target_sizes:
            if time.time() > _DEADLINE:
                break
            
            max_val = 2 * (2 ** ts) + 500
            U = _greedy_construct(ts, max_val, start_set=list(start))
            if not U or 0 not in U:
                continue
            
            # Local search polish
            U = _local_search(U, max_val, iterations=10)
            if not U or 0 not in U:
                continue
            
            s = _compute_score(U)
            if s > best_score + 1e-12:
                best_score = s
                best_U = list(U)
    
    # Strategy 2: Structured construction
    if time.time() < _DEADLINE:
        for ts in [20, 40, 60]:
            if time.time() > _DEADLINE:
                break
            U = _structured_construct(ts)
            if not U or 0 not in U:
                continue
            s = _compute_score(U)
            if s > best_score + 1e-12:
                best_score = s
                best_U = list(U)
    
    # Final local search on best
    if time.time() < _DEADLINE - 5:
        max_val = max(best_U) * 2 + 100
        best_U = _local_search(best_U, max_val, iterations=30)
        if best_U and 0 in best_U:
            s = _compute_score(best_U)
            if s > best_score + 1e-12:
                best_score = s
    
    return sorted(set(best_U))
