"""Rewritten program for sumset/difference-set exponent optimization."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize
    the lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1).
    
    Returns a list of nonnegative integers containing 0.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_set = [0, 1, 3]
    best_score = _compute_score(best_set)
    
    def _compute_score(U):
        """Compute the score for a given set U."""
        if not U or 0 not in U:
            return 0.0
        U = list(set(U))
        n = len(U)
        if n < 2:
            return 0.0
        max_u = max(U)
        if max_u == 0:
            return 0.0
        # Compute sumset U+U
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        # Compute difference set U-U
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        
        num_sum = len(sumset)
        num_diff = len(diffset)
        
        if num_sum == 0:
            return 0.0
        
        ratio = num_diff / num_sum
        log_arg = 2 * max_u + 1
        if log_arg <= 1:
            return 0.0
        
        import math
        score = 1.0 + math.log(ratio) / math.log(log_arg)
        return score
    
    # Initial candidate sets to try
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
    
    # Greedy construction: build a set by adding elements that maximize score
    # Start with [0] and greedily add elements
    def greedy_construct(max_size=15, max_val=50):
        U = [0]
        for _ in range(max_size - 1):
            best_elem = None
            best_s = best_score
            for e in range(1, max_val + 1):
                if e in U:
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
        return U, best_s
    
    # Try greedy from different starting points
    for start in [[0], [0, 1], [0, 1, 3]]:
        if time.time() > deadline:
            break
        g_set, g_score = greedy_construct(max_size=12, max_val=40)
        if g_score > best_score:
            best_score = g_score
            best_set = list(g_set)
    
    # Local search: try swapping or adding/removing elements
    def local_search(U, max_iter=200):
        U = list(U)
        current_score = _compute_score(U)
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            # Try adding an element
            max_u = max(U) if U else 0
            for e in range(1, max_u + 20):
                if e in U:
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
            if not improved:
                break
        return U, current_score
    
    # Run local search on best set found so far
    if time.time() < deadline:
        ls_set, ls_score = local_search(best_set, max_iter=100)
        if ls_score > best_score:
            best_score = ls_score
            best_set = list(ls_set)
    
    # Try random perturbations + local search
    if time.time() < deadline:
        random.seed(42)
        for trial in range(50):
            if time.time() > deadline:
                break
            # Random set construction
            size = random.randint(3, 10)
            max_val = random.randint(10, 60)
            U = [0]
            for _ in range(size - 1):
                e = random.randint(1, max_val)
                if e not in U:
                    U.append(e)
            U.sort()
            
            # Local search
            ls_set, ls_score = local_search(U, max_iter=50)
            if ls_score > best_score:
                best_score = ls_score
                best_set = list(ls_set)
    
    # Final check
    if time.time() > deadline - 5:
        return best_set
    
    return best_set
# EVOLVE-BLOCK-END
