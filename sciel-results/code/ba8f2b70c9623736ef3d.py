import time
import random
from math import log
import numpy as np
from collections import Counter

def fast_score(U):
    """
    Calculate the score for a set U.
    Score = 1 + log(|U-U| / |U+U|) / log(2 * max(U) + 1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    U_arr = np.array(U)
    
    # Compute all pairwise sums (i <= j)
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    num_sums = len(np.unique(sums))
    
    # Compute all pairwise differences (i < j)
    i_idx_d, j_idx_d = np.triu_indices(n, k=1)
    diffs = U_arr[j_idx_d] - U_arr[i_idx_d]
    num_pos_diffs = len(np.unique(diffs))
    num_diffs = 2 * num_pos_diffs + 1
    
    if num_sums == 0:
        return -1
        
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)


def _compute_sum_diff_counts(U):
    """Return (sum_counter, diff_counter, num_sums, num_diffs) for a sorted set U."""
    U_arr = np.array(U)
    n = len(U)
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    sum_counts = Counter(sums.tolist())
    num_sums = len(sum_counts)
    
    i_idx_d, j_idx_d = np.triu_indices(n, k=1)
    diffs = U_arr[j_idx_d] - U_arr[i_idx_d]
    diff_counts = Counter(diffs.tolist())
    num_pos_diffs = len(diff_counts)
    num_diffs = 2 * num_pos_diffs + 1
    
    return sum_counts, diff_counts, num_sums, num_diffs


def _marginal_score_impact(U, new_val):
    """
    Estimate the marginal change in score if we add new_val to U.
    Returns a heuristic score: positive means adding new_val likely helps.
    We approximate by counting how many new unique sums/diffs would be created.
    """
    if new_val in U:
        return -999
    
    U_sorted = sorted(U)
    n = len(U_sorted)
    U_arr = np.array(U_sorted)
    
    # New sums: new_val + u for all u in U, plus new_val + new_val
    new_sums = U_arr + new_val
    new_sums = np.append(new_sums, 2 * new_val)
    
    # New positive diffs: |new_val - u| for all u in U where new_val != u
    new_diffs = np.abs(new_val - U_arr)
    new_diffs = new_diffs[new_diffs > 0]
    
    # Count how many of these are already in existing sums/diffs
    existing_sums = set((U_arr[np.triu_indices(n)[0]] + U_arr[np.triu_indices(n)[1]]).tolist())
    existing_pos_diffs = set((U_arr[np.triu_indices(n, k=1)[1]] - U_arr[np.triu_indices(n, k=1)[0]]).tolist())
    
    new_unique_sums = sum(1 for s in new_sums if s not in existing_sums)
    new_unique_diffs = sum(1 for d in new_diffs if d not in existing_pos_diffs)
    
    # Heuristic: prefer values that create more new diffs than new sums
    # and keep max(U) small
    max_u = max(U_sorted)
    new_max = max(max_u, new_val)
    D_old = 2 * max_u + 1
    D_new = 2 * new_max + 1
    
    # Approximate current ratio
    total_sums = len(existing_sums)
    total_diffs = 2 * len(existing_pos_diffs) + 1
    
    if total_sums == 0:
        return 0
    
    old_ratio = total_diffs / total_sums
    new_total_sums = total_sums + new_unique_sums
    new_total_diffs = total_diffs + 2 * new_unique_diffs
    if new_total_sums == 0:
        return -999
    new_ratio = new_total_diffs / new_total_sums
    
    # Score based on ratio improvement and denominator change
    if D_new <= 1:
        return -999
    
    old_log_score = log(old_ratio) / log(D_old) if old_ratio > 1 else -1
    new_log_score = log(new_ratio) / log(D_new) if new_ratio > 1 else -1
    
    return new_log_score - old_log_score

def local_search(U, max_iter=1000, deadline=None):
    """
    Local search with dynamic candidate generation.
    Uses structurally relevant neighbors (sums/diffs of existing elements)
    to escape local minima.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
        
    best = current
    best_score = current_score
    
    start_time = time.time()
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        n = len(current)
        i = random.randint(1, n - 1)
        old_val = current[i]
        
        # Generate dynamic candidates from sums and diffs
        candidates = set()
        
        # Add values near old_val
        for delta in range(-3, 4):
            v = old_val + delta
            if v >= 1:
                candidates.add(v)
        
        # Add values derived from pairwise sums/diffs (structural neighbors)
        if n >= 2:
            for a in current:
                for b in current:
                    if a == b:
                        continue
                    v = a + b
                    if v >= 1 and v <= 2 * max(current) + 5:
                        candidates.add(v)
                    v = abs(a - b)
                    if v >= 1:
                        candidates.add(v)
        
        # Also try smaller values to reduce max(U)
        if old_val > 1:
            candidates.add(random.randint(1, max(1, old_val - 1)))
        
        if not candidates:
            candidates = {max(1, old_val - 1), max(1, old_val + 1)}
        
        new_val = random.choice(list(candidates))
        
        new_set = current[:]
        new_set[i] = new_val
        new_set = sorted(set(new_set))
        
        if len(new_set) != n:
            continue
            
        new_score = fast_score(new_set)
        
        if new_score > best_score:
            best_score = new_score
            best = new_set
            
        if new_score >= current_score:
            current = new_set
            current_score = new_score
        else:
            if random.random() < 0.1:
                current = new_set
                current_score = new_score
                
    return best, best_score


def greedy_construct(target_size, deadline):
    """
    Greedy construction using marginal score impact to select elements.
    Starts with {0} and iteratively adds the element that maximizes
    the approximate marginal score improvement.
    """
    U = [0]
    max_val = 0
    
    while len(U) < target_size:
        if time.time() > deadline:
            break
        
        # Generate candidate values
        candidates = set()
        
        # Small range near existing elements
        for u in U:
            for delta in range(-2, 3):
                v = u + delta
                if v >= 1:
                    candidates.add(v)
        
        # Structural candidates from sums and diffs
        for a in U:
            for b in U:
                if a == b:
                    continue
                v = a + b
                if 1 <= v <= 3 * max(U) + 10:
                    candidates.add(v)
                v = abs(a - b)
                if v >= 1:
                    candidates.add(v)
        
        # Some random exploration
        for _ in range(10):
            candidates.add(random.randint(1, max(2, 2 * max(U))))
        
        if not candidates:
            candidates = {1, 2, 3}
        
        # Score each candidate by marginal impact
        best_candidate = None
        best_impact = -999
        
        for cand in candidates:
            if cand in U:
                continue
            impact = _marginal_score_impact(U, cand)
            if impact > best_impact:
                best_impact = impact
                best_candidate = cand
        
        if best_candidate is None:
            # Fallback: add next integer
            best_candidate = max(U) + 1
        
        U.append(best_candidate)
        U = sorted(U)
    
    return U

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Known optimal or near-optimal Golomb rulers
    golomb_rulers = [
        [0, 1],
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29, 37, 41],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 64],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 64, 72],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 64, 72, 78],
    ]
    
    best_set = None
    best_score = -1
    
    # Evaluate all hardcoded rulers
    for U in golomb_rulers:
        s = fast_score(U)
        if s > best_score:
            best_score = s
            best_set = U
            
    # Greedy construction with different target sizes
    for target in [8, 10, 12, 14, 16, 20]:
        if time.time() > deadline - 30:
            break
        greedy_set = greedy_construct(target, deadline - 30)
        s = fast_score(greedy_set)
        if s > best_score:
            best_score = s
            best_set = greedy_set
    
    # Local search on the best few candidates (hardcoded + greedy)
    all_candidates = list(golomb_rulers)
    # Add greedy results if they were generated
    for target in [8, 10, 12, 14, 16, 20]:
        if time.time() > deadline - 20:
            break
        g = greedy_construct(target, deadline - 20)
        all_candidates.append(g)
    
    # Sort by score and refine top candidates
    scored_candidates = [(U, fast_score(U)) for U in all_candidates]
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    for U, _ in scored_candidates[:8]:
        if time.time() > deadline - 5:
            break
        refined, refined_score = local_search(U, max_iter=1500, deadline=deadline - 2)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
            
    # Final fallback
    if best_set is None:
        best_set = [0, 1, 4, 9, 11]
        
    return best_set

if __name__ == "__main__":
    print(f"Best set: {construct()}")
