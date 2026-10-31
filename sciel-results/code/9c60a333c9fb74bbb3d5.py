"""Dual-view construction: define U via its difference set D, minimize additive energy of U."""
import math
import random
from itertools import combinations

def construct():
    """
    Dual-view approach:
    1. Generate candidate difference sets D (Sidon-like structures)
    2. For each D, find U such that U-U = D and |U+U| is minimized
    3. Optimize via local search on U
    
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
    """
    
    def compute_score(U):
        if not U or 0 not in U:
            return 0.0
        U_set = set(U)
        max_u = max(U_set)
        if max_u == 0:
            return 0.0
        sum_set = set()
        diff_set = set()
        for a in U_set:
            for b in U_set:
                sum_set.add(a + b)
                diff_set.add(a - b)
        n_diff = len(diff_set)
        n_sum = len(sum_set)
        if n_sum == 0:
            return 0.0
        log_denom = math.log(2 * max_u + 1)
        if log_denom <= 0:
            return 0.0
        return 1.0 + math.log(n_diff / n_sum) / log_denom
    
    def get_diff_set(U):
        U_set = set(U)
        diff_set = set()
        for a in U_set:
            for b in U_set:
                diff_set.add(a - b)
        return diff_set
    
    def get_sum_set(U):
        U_set = set(U)
        sum_set = set()
        for a in U_set:
            for b in U_set:
                sum_set.add(a + b)
        return sum_set
    
    def find_U_from_diff_set(D, max_val=300, max_size=15):
        """
        Given a target difference set D, find U ⊆ [0, max_val] containing 0
        such that U-U ⊇ D (ideally = D) and |U+U| is minimized.
        
        Strategy: Greedy construction - add elements that create required differences
        while minimizing sum collisions.
        """
        if 0 not in D:
            D = D | {0}
        
        D_set = set(D)
        U = [0]
        U_set = {0}
        
        # Track which differences are still needed
        remaining_diffs = D_set - {0}
        
        # For each step, try adding an element that maximally reduces remaining differences
        # while minimizing new sum collisions
        for step in range(max_size - 1):
            best_candidate = None
            best_score = float('-inf')
            
            # Try candidates in [1, max_val]
            for candidate in range(1, max_val + 1):
                if candidate in U_set:
                    continue
                
                new_U = U + [candidate]
                new_U_set = U_set | {candidate}
                
                # Compute new differences
                new_diffs = set()
                for a in new_U_set:
                    for b in new_U_set:
                        new_diffs.add(a - b)
                
                # How many required differences are now covered?
                covered = new_diffs & remaining_diffs
                newly_covered = len(covered)
                
                # Penalize if we create too many extra differences (beyond D)
                extra_diffs = new_diffs - D_set
                extra_penalty = len(extra_diffs) * 0.5
                
                # Compute sum collisions
                new_sums = set()
                for a in new_U_set:
                    for b in new_U_set:
                        new_sums.add(a + b)
                sum_size = len(new_sums)
                
                # Score: prefer covering more required diffs, fewer extra diffs, smaller sum set
                # Also prefer smaller candidate values (to keep max(U) small)
                score = newly_covered * 10 - extra_penalty * 5 - sum_size * 0.1 - candidate * 0.05
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            
            if best_candidate is None:
                break
            
            U.append(best_candidate)
            U_set.add(best_candidate)
            
            # Update remaining differences
            current_diffs = set()
            for a in U_set:
                for b in U_set:
                    current_diffs.add(a - b)
            remaining_diffs = D_set - current_diffs
            
            if not remaining_diffs:
                break
        
        return sorted(U)
    
    def local_search(U, max_iters=200, max_val=400):
        """Local search to improve score."""
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        
        best_U = list(U)
        best_score = compute_score(U)
        
        for iteration in range(max_iters):
            improved = False
            U_set = set(best_U)
            n = len(best_U)
            
            # Try moving each element (except 0)
            for i in range(1, n):
                elem = best_U[i]
                for delta in range(-15, 16):
                    new_val = elem + delta
                    if new_val < 0 or new_val in U_set or new_val > max_val:
                        continue
                    new_U = list(best_U)
                    new_U[i] = new_val
                    new_U = sorted(set(new_U))
                    if 0 not in new_U or len(new_U) < 2:
                        continue
                    new_score = compute_score(new_U)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_U = new_U
                        improved = True
                        break
                if improved:
                    break
            
            # Try adding an element
            if not improved:
                U_set = set(best_U)
                # Sample candidates
                candidates = [x for x in range(1, min(max_val + 1, 300)) if x not in U_set]
                random.shuffle(candidates)
                for candidate in candidates[:30]:
                    new_U = sorted(best_U + [candidate])
                    new_score = compute_score(new_U)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_U = new_U
                        improved = True
                        break
            
            # Try removing an element (except 0)
            if not improved:
                for i in range(1, n):
                    new_U = [x for j, x in enumerate(best_U) if j != i]
                    if len(new_U) < 2:
                        continue
                    new_score = compute_score(new_U)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_U = new_U
                        improved = True
                        break
            
            if not improved:
                break
        
        return best_U, best_score
    
    # Step 1: Generate diverse candidate difference sets
    random.seed(123)
    candidate_diff_sets = []
    
    # A) Sidon-type difference sets: differences are all distinct in magnitude
    # For a Sidon set of size n, |U-U| ≈ n(2n-1)/2 + 1
    # Generate by creating sets with distinct pairwise differences
    
    # B) Structured difference sets from arithmetic progressions with perturbations
    for n in range(6, 14):
        # Base: arithmetic progression differences
        base_diffs = set()
        for i in range(n):
            for j in range(n):
                base_diffs.add(i - j)
        
        # Perturb: add some structure
        if random.random() < 0.5:
            # Add gaps
            perturbed = set()
            for d in base_diffs:
                if random.random() > 0.1:
                    perturbed.add(d)
        else:
            perturbed = base_diffs
        
        candidate_diff_sets.append(perturbed)
    
    # C) Random Sidon-like difference sets
    for trial in range(30):
        n = random.randint(5, 12)
        max_diff = random.randint(50, 200)
        D = {0}
        # Generate differences that are "Sidon-like" - few collisions
        for _ in range(n * n):
            d = random.randint(-max_diff, max_diff)
            D.add(d)
        # Ensure symmetry
        D = D | {-d for d in D}
        candidate_diff_sets.append(D)
    
    # D) Difference sets from known good constructions
    # Mian-Chowla type
    mc_sets = [
        [0, 1, 4, 10, 18, 23, 31, 40, 50, 61, 73, 86, 100, 115],
        [0, 1, 3, 7, 12, 20, 30, 44, 60, 79, 101],
        [0, 1, 2, 5, 10, 20, 40, 80, 160],
    ]
    for mc in mc_sets:
        D = get_diff_set(mc)
        candidate_diff_sets.append(D)
    
    # Step 2: For each difference set, find optimal U
    best_U = [0, 1]
    best_score = 0.0
    seen_U = set()
    
    for D in candidate_diff_sets:
        if not D:
            continue
        
        # Try to find U from this difference set
        U = find_U_from_diff_set(D, max_val=300, max_size=14)
        
        if len(U) < 2 or 0 not in U:
            continue
        
        U = sorted(set(U))
        U_tuple = tuple(U)
        if U_tuple in seen_U:
            continue
        seen_U.add(U_tuple)
        
        # Local search to improve
        searched_U, searched_score = local_search(U, max_iters=150, max_val=350)
        
        if searched_score > best_score:
            best_score = searched_score
            best_U = searched_U
    
    # Step 3: Also try direct greedy construction as a baseline
    random.seed(456)
    for trial in range(15):
        max_val = random.randint(150, 350)
        U = [0]
        U_set = {0}
        
        for _ in range(12):
            best_candidate = None
            best_ratio = float('-inf')
            
            for candidate in range(1, max_val + 1, 2):
                if candidate in U_set:
                    continue
                new_U = sorted(U + [candidate])
                new_score = compute_score(new_U)
                if new_score > best_ratio:
                    best_ratio = new_score
                    best_candidate = candidate
            
            if best_candidate is not None:
                U.append(best_candidate)
                U_set.add(best_candidate)
            else:
                break
        
        U = sorted(U)
        searched_U, searched_score = local_search(U, max_iters=100, max_val=300)
        
        if searched_score > best_score:
            best_score = searched_score
            best_U = searched_U
    
    return best_U

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
