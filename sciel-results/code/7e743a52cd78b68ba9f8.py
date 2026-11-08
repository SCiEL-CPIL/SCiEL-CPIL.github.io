"""Rewritten program for sumset/difference-set exponent optimization."""
import time
import random
import math
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize
    the lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1).
    
    Returns a list of nonnegative integers containing 0.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    def _compute_score(U):
        """Compute the score for a given set U using vectorized operations."""
        if not U or 0 not in U:
            return 0.0
        U_arr = np.array(sorted(set(U)), dtype=np.int64)
        n = len(U_arr)
        if n < 2:
            return 0.0
        max_u = int(U_arr[-1])
        if max_u == 0:
            return 0.0
        
        # Vectorized sumset and diffset
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        
        num_sum = len(np.unique(sums))
        num_diff = len(np.unique(diffs))
        
        if num_sum == 0:
            return 0.0
        
        ratio = num_diff / num_sum
        log_arg = 2 * max_u + 1
        if log_arg <= 1:
            return 0.0
        
        score = 1.0 + math.log(ratio) / math.log(log_arg)
        return score

    best_set = [0, 1, 3]
    best_score = _compute_score(best_set)

    # Exhaustive search for small sets (n=3,4,5) to find strong seeds
    def exhaustive_small(max_n=5, max_val=20):
        """Try all subsets of {0,...,max_val} of size max_n containing 0."""
        best_local = best_set[:]
        best_local_score = best_score
        candidates_pool = list(range(1, max_val + 1))
        
        for n in range(3, max_n + 1):
            for combo in combinations(candidates_pool, n - 1):
                if time.time() > deadline:
                    return best_local, best_local_score
                U = [0] + list(combo)
                s = _compute_score(U)
                if s > best_local_score:
                    best_local_score = s
                    best_local = U[:]
        
        return best_local, best_local_score

    # Run exhaustive search for small sets
    ex_set, ex_score = exhaustive_small(max_n=5, max_val=18)
    if ex_score > best_score:
        best_score = ex_score
        best_set = list(ex_set)

    # Initial candidate sets
    candidates = [
        [0, 1, 3],
        [0, 1, 2, 4],
        [0, 1, 3, 4],
        [0, 1, 2, 5],
        [0, 1, 3, 7],
        [0, 1, 2, 3, 5],
        [0, 1, 2, 4, 5],
        [0, 1, 3, 4, 6],
        [0, 1, 2, 3, 7],
        [0, 1, 3, 5, 7],
        [0, 1, 2, 4, 7],
        [0, 1, 3, 4, 8],
        [0, 1, 2, 5, 6],
        [0, 1, 3, 6, 7],
        [0, 1, 2, 3, 4, 8],
        [0, 1, 2, 3, 5, 7],
        [0, 1, 2, 4, 5, 8],
        [0, 1, 3, 4, 6, 8],
        [0, 1, 2, 3, 7, 9],
        [0, 1, 3, 5, 7, 9],
    ]

    for cand in candidates:
        if time.time() > deadline:
            break
        score = _compute_score(cand)
        if score > best_score:
            best_score = score
            best_set = list(cand)

    # Vectorized local search: try all single add/remove/swap moves
    def local_search(U, max_iter=200, expand_range=15):
        """Local search with add, remove, and swap moves."""
        U = sorted(set(U))
        current_score = _compute_score(U)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            max_u = max(U) if U else 0
            U_set = set(U)
            
            # Try adding an element
            for e in range(1, max_u + expand_range + 1):
                if e in U_set:
                    continue
                new_U = U + [e]
                s = _compute_score(new_U)
                if s > current_score + 1e-12:
                    U = new_U
                    current_score = s
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try removing an element (keep 0)
            for i in range(len(U)):
                if U[i] == 0:
                    continue
                new_U = [x for j, x in enumerate(U) if j != i]
                if len(new_U) < 2:
                    continue
                s = _compute_score(new_U)
                if s > current_score + 1e-12:
                    U = new_U
                    current_score = s
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: remove one non-zero, add another
            for i in range(len(U)):
                if U[i] == 0:
                    continue
                for e in range(1, max_u + expand_range + 1):
                    if e in U_set or e == U[i]:
                        continue
                    new_U = [x for j, x in enumerate(U) if j != i] + [e]
                    new_U = sorted(set(new_U))
                    if len(new_U) < 2:
                        continue
                    s = _compute_score(new_U)
                    if s > current_score + 1e-12:
                        U = new_U
                        current_score = s
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return U, current_score

    # Run local search on best set found so far
    if time.time() < deadline:
        ls_set, ls_score = local_search(best_set, max_iter=150, expand_range=20)
        if ls_score > best_score:
            best_score = ls_score
            best_set = list(ls_set)

    # Greedy construction with wider search
    def greedy_construct(max_size=14, max_val=60):
        U = [0]
        for _ in range(max_size - 1):
            best_elem = None
            best_s = _compute_score(U)
            U_set = set(U)
            for e in range(1, max_val + 1):
                if e in U_set:
                    continue
                new_U = U + [e]
                s = _compute_score(new_U)
                if s > best_s:
                    best_s = s
                    best_elem = e
            if best_elem is None:
                break
            U.append(best_elem)
            if time.time() > deadline:
                break
        return U, _compute_score(U)

    # Try greedy from different starting points
    for start_set in [[0], [0, 1], [0, 1, 3]]:
        if time.time() > deadline:
            break
        g_set, g_score = greedy_construct(max_size=12, max_val=50)
        if g_score > best_score:
            best_score = g_score
            best_set = list(g_set)

    # Random perturbations + local search
    if time.time() < deadline:
        random.seed(42)
        for trial in range(80):
            if time.time() > deadline:
                break
            # Random set construction
            size = random.randint(3, 12)
            max_val = random.randint(10, 80)
            U = [0]
            for _ in range(size - 1):
                e = random.randint(1, max_val)
                if e not in U:
                    U.append(e)
            U.sort()

            # Local search
            ls_set, ls_score = local_search(U, max_iter=60, expand_range=15)
            if ls_score > best_score:
                best_score = ls_score
                best_set = list(ls_set)

    # Final polish: tighter local search on the best set
    if time.time() < deadline - 10:
        final_set, final_score = local_search(best_set, max_iter=200, expand_range=25)
        if final_score > best_score:
            best_score = final_score
            best_set = list(final_set)

    return best_set
# EVOLVE-BLOCK-END
