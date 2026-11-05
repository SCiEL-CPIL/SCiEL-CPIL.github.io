"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict

def construct():
    random.seed(42)
    np.random.seed(42)
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 128
    TARGET_LO = 128
    TARGET_HI = 256
    NUM_TARGETS = TARGET_HI - TARGET_LO + 1  # 129 targets
    
    def count_pairs(A):
        """Count representations of each target sum using unordered pairs with equal summands allowed."""
        counts = np.zeros(NUM_TARGETS, dtype=np.int32)
        A_sorted = sorted(A)
        n = len(A_sorted)
        for i in range(n):
            for j in range(i, n):
                s = A_sorted[i] + A_sorted[j]
                if TARGET_LO <= s <= TARGET_HI:
                    counts[s - TARGET_LO] += 1
        return counts
    
    def max_count(counts):
        return int(np.max(counts))
    
    def is_valid(counts):
        return np.all(counts > 0)
    
    def score(counts):
        if not is_valid(counts):
            return float('inf')
        return max_count(counts)
    
    # Greedy construction: build A to cover all targets with low multiplicity
    def greedy_construct(max_mult=3):
        """Greedily construct A such that every target is covered and no target has > max_mult representations."""
        A = set()
        counts = np.zeros(NUM_TARGETS, dtype=np.int32)
        
        # Precompute all possible pairs and their sums
        # For each element x in [0, 128], compute what pairs it forms with existing A
        # We want to add elements that help cover uncovered targets without exceeding max_mult
        
        # Start with some base elements
        # Key insight: to cover sum=128, we need pairs like (0,128), (1,127), ..., (64,64)
        # To cover sum=256, we need (128,128)
        
        # Strategy: add elements that maximize coverage of currently uncovered targets
        # while respecting the max_mult constraint
        
        for step in range(200):
            if time.time() > deadline:
                break
            
            # Check if all targets are covered
            if np.all(counts > 0):
                break
            
            # Find uncovered targets
            uncovered = np.where(counts == 0)[0]
            if len(uncovered) == 0:
                break
            
            # Try each candidate element not in A
            best_elem = None
            best_improvement = -1
            best_new_counts = None
            
            for elem in range(N + 1):
                if elem in A:
                    continue
                
                # Compute delta if we add elem
                delta = np.zeros(NUM_TARGETS, dtype=np.int32)
                for a in A:
                    s = elem + a
                    if TARGET_LO <= s <= TARGET_HI:
                        idx = s - TARGET_LO
                        delta[idx] += 1
                s = elem + elem
                if TARGET_LO <= s <= TARGET_HI:
                    idx = s - TARGET_LO
                    delta[idx] += 1
                
                new_counts = counts + delta
                
                # Check if this violates max_mult
                if np.any(new_counts > max_mult):
                    continue
                
                # Score: number of newly covered targets
                newly_covered = np.sum((new_counts > 0) & (counts == 0))
                # Penalize if we increase max count
                new_max = int(np.max(new_counts))
                improvement = newly_covered * 10 - new_max
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_elem = elem
                    best_new_counts = new_counts
            
            if best_elem is not None:
                A.add(best_elem)
                counts = best_new_counts
            else:
                # No valid move, break
                break
        
        return A, counts
    
    # CSP-based backtracking search for a given max multiplicity K
    def csp_search(K, max_time=10):
        """
        Try to find A such that every target in [128,256] is covered and no target has > K representations.
        Uses backtracking with forward checking.
        """
        csp_start = time.time()
        
        # Represent A as a set
        A = set()
        counts = np.zeros(NUM_TARGETS, dtype=np.int32)
        
        # Order elements by how "useful" they are for covering targets
        # Elements near 64 are most useful (can pair with many other elements to cover range)
        element_order = sorted(range(N + 1), key=lambda x: -min(abs(x - 64), 128 - abs(x - 64)))
        
        def backtrack(idx, A_set, counts_arr):
            if time.time() > csp_start + max_time:
                return None
            
            # Check if all targets are covered
            if np.all(counts_arr > 0):
                return set(A_set)
            
            # If we've tried all elements, fail
            if idx >= len(element_order):
                return None
            
            elem = element_order[idx]
            
            # Option 1: don't include elem
            result = backtrack(idx + 1, A_set, counts_arr)
            if result is not None:
                return result
            
            # Option 2: include elem
            delta = np.zeros(NUM_TARGETS, dtype=np.int32)
            for a in A_set:
                s = elem + a
                if TARGET_LO <= s <= TARGET_HI:
                    idx_s = s - TARGET_LO
                    delta[idx_s] += 1
            s = elem + elem
            if TARGET_LO <= s <= TARGET_HI:
                idx_s = s - TARGET_LO
                delta[idx_s] += 1
            
            new_counts = counts_arr + delta
            
            # Forward checking: ensure no target exceeds K
            if np.any(new_counts > K):
                return None
            
            A_set.add(elem)
            result = backtrack(idx + 1, A_set, new_counts)
            A_set.remove(elem)
            
            if result is not None:
                return result
            
            return None
        
        return backtrack(0, set(), counts)
    
    # Local search to improve a solution
    def local_search(A_init, max_iters=5000):
        """Improve a solution using local search (add/remove elements)."""
        A = set(A_init)
        counts = count_pairs(A)
        current_score = score(counts)
        best_score = current_score
        best_A = set(A)
        
        no_improve = 0
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if no_improve > 100:
                break
            
            improved = False
            # Try removing elements that are least useful
            if len(A) > 1:
                for elem in list(A):
                    if time.time() > deadline:
                        break
                    # Compute what happens if we remove elem
                    delta = np.zeros(NUM_TARGETS, dtype=np.int32)
                    for a in A:
                        if a != elem:
                            s = elem + a
                            if TARGET_LO <= s <= TARGET_HI:
                                idx_s = s - TARGET_LO
                                delta[idx_s] += 1
                    s = elem + elem
                    if TARGET_LO <= s <= TARGET_HI:
                        idx_s = s - TARGET_LO
                        delta[idx_s] += 1
                    
                    new_counts = counts - delta
                    new_score = score(new_counts)
                    
                    if new_score < current_score:
                        A.remove(elem)
                        counts = new_counts
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_A = set(A)
                        break
            
            # Try adding elements that help
            if not improved:
                best_add = None
                best_add_score = current_score
                for elem in range(N + 1):
                    if elem in A:
                        continue
                    delta = np.zeros(NUM_TARGETS, dtype=np.int32)
                    for a in A:
                        s = elem + a
                        if TARGET_LO <= s <= TARGET_HI:
                            idx_s = s - TARGET_LO
                            delta[idx_s] += 1
                    s = elem + elem
                    if TARGET_LO <= s <= TARGET_HI:
                        idx_s = s - TARGET_LO
                        delta[idx_s] += 1
                    
                    new_counts = counts + delta
                    new_score = score(new_counts)
                    
                    if new_score < best_add_score:
                        best_add_score = new_score
                        best_add = elem
                
                if best_add is not None and best_add_score < current_score:
                    A.add(best_add)
                    counts = count_pairs(A)
                    current_score = score(counts)
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_A = set(A)
            
            if improved:
                no_improve = 0
            else:
                no_improve += 1
        
        return best_A, best_score
    
    # Phase 1: Try greedy construction with different max_mult values
    best_overall_A = None
    best_overall_score = float('inf')
    
    for max_mult in [2, 3, 4, 5]:
        if time.time() > deadline - 20:
            break
        A_greedy, counts_greedy = greedy_construct(max_mult)
        if is_valid(counts_greedy):
            sc = score(counts_greedy)
            if sc < best_overall_score:
                best_overall_score = sc
                best_overall_A = set(A_greedy)
    
    # Phase 2: Try CSP search for low K values
    for K in [2, 3, 4]:
        if time.time() > deadline - 15:
            break
        A_csp = csp_search(K, max_time=5)
        if A_csp is not None and len(A_csp) > 0:
            counts_csp = count_pairs(A_csp)
            if is_valid(counts_csp):
                sc = score(counts_csp)
                if sc < best_overall_score:
                    best_overall_score = sc
                    best_overall_A = set(A_csp)
    
    # Phase 3: Local search on best solution found
    if best_overall_A is not None and time.time() < deadline - 5:
        A_ls, score_ls = local_search(best_overall_A)
        if score_ls < best_overall_score:
            best_overall_score = score_ls
            best_overall_A = A_ls
    
    # Phase 4: Fallback - use a known good construction if needed
    if best_overall_A is None or best_overall_score == float('inf'):
        # Use a structured construction
        # Include elements that help cover the full range
        A_fallback = set()
        # Add elements to cover low sums
        for x in range(0, 65, 4):
            A_fallback.add(x)
        # Add elements to cover high sums
        for x in range(64, 129, 4):
            A_fallback.add(x)
        # Add some middle elements
        A_fallback.add(64)
        A_fallback.add(32)
        A_fallback.add(96)
        
        counts_fallback = count_pairs(A_fallback)
        if is_valid(counts_fallback):
            best_overall_A = A_fallback
            best_overall_score = score(counts_fallback)
        else:
            # Last resort: use the original seed
            best_overall_A = set(range(64, 129))
            best_overall_score = score(count_pairs(best_overall_A))
    
    return sorted(best_overall_A)
# EVOLVE-BLOCK-END
