import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit (evaluator kills at 90s, but we want safety)

    MIN_SUM = 128
    MAX_SUM = 256
    NUM_SUMS = MAX_SUM - MIN_SUM + 1  # 129

    def calc_mult(A_list):
        """A_list must be sorted. Returns multiplicity array for sums in [MIN_SUM, MAX_SUM]."""
        M = np.zeros(NUM_SUMS, dtype=np.int32)
        n = len(A_list)
        for i in range(n):
            a = A_list[i]
            s = 2 * a
            if MIN_SUM <= s <= MAX_SUM:
                M[s - MIN_SUM] += 1
            for j in range(i + 1, n):
                b = A_list[j]
                s = a + b
                if s > MAX_SUM:
                    break
                if s >= MIN_SUM:
                    M[s - MIN_SUM] += 1
        return M

    def calc_score(A):
        A = sorted(A)
        M = calc_mult(A)
        gaps = int(np.sum(M == 0))
        if gaps > 0:
            return (gaps, float('inf'))
        peak = int(np.max(M))
        return (0, peak)

    def get_elements_contributing_to_sum(A, target_sum):
        """Return list of indices in A that contribute to target_sum."""
        contributors = []
        A_set = set(A)
        for i, a in enumerate(A):
            b = target_sum - a
            if b in A_set and b >= a:
                contributors.append(i)
        return contributors

    def targeted_local_search(A, M, deadline, max_iters=500):
        """Local search that specifically targets high-multiplicity sums."""
        A = sorted(list(A))
        current_score = calc_score(A)
        
        if current_score == (0, 1):
            return current_score, A, M
        
        improved = True
        while improved and time.time() < deadline and max_iters > 0:
            improved = False
            max_iters -= 1
            
            # Find the highest multiplicity targets
            max_mult = int(np.max(M))
            if max_mult <= 1:
                break
            
            # Get targets with max multiplicity
            high_mult_targets = np.where(M == max_mult)[0] + MIN_SUM
            
            # For each high-mult target, try to reduce its multiplicity
            for target in high_mult_targets:
                if time.time() > deadline:
                    break
                
                contributors = get_elements_contributing_to_sum(A, target)
                if len(contributors) < 2:
                    continue
                
                # Try removing or swapping each contributor
                A_set = set(A)
                available = [x for x in range(129) if x not in A_set]
                
                for idx in contributors:
                    if time.time() > deadline:
                        break
                    a = A[idx]
                    
                    # Try removing this element
                    new_A = A[:idx] + A[idx+1:]
                    if new_A:
                        new_M = calc_mult(new_A)
                        if np.all(new_M >= 1):
                            new_max = int(np.max(new_M))
                            new_score = (0, new_max)
                            if new_score < current_score:
                                current_score = new_score
                                A = new_A
                                M = new_M
                                improved = True
                                break
                    
                    # Try swapping with a random available element
                    if not improved and available:
                        n_swaps = min(5, len(available))
                        for b in random.sample(available, n_swaps):
                            new_A = A.copy()
                            new_A[idx] = b
                            new_A.sort()
                            new_M = calc_mult(new_A)
                            gaps = int(np.sum(new_M == 0))
                            if gaps > 0:
                                continue
                            new_max = int(np.max(new_M))
                            new_score = (0, new_max)
                            if new_score < current_score:
                                current_score = new_score
                                A = new_A
                                M = new_M
                                improved = True
                                break
                    
                    if improved:
                        break
                
                if improved:
                    break
        
        return current_score, A, M

    def greedy_removal(A, M, deadline, max_rounds=2000):
        """Try to remove elements while maintaining validity and not increasing peak."""
        A = list(A)
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            if len(A) <= 1:
                break
            current_max = int(np.max(M))
            improved = False
            # Try removing each element, prefer those that reduce max
            for idx in range(len(A)):
                new_A = A[:idx] + A[idx+1:]
                if not new_A:
                    continue
                new_M = calc_mult(new_A)
                if np.all(new_M >= 1):
                    new_max = int(np.max(new_M))
                    if new_max < current_max:
                        A = new_A
                        M = new_M
                        improved = True
                        break
                    elif new_max == current_max and random.random() < 0.1:
                        A = new_A
                        M = new_M
                        improved = True
                        break
            if not improved:
                break
        return A, M

    def local_search(A, deadline):
        A = sorted(list(A))
        M = calc_mult(A)
        current_score = calc_score(A)
        
        if current_score == (0, 1):
            return current_score, A, M
        
        improved = True
        max_iters = 300
        
        while improved and time.time() < deadline and max_iters > 0:
            improved = False
            max_iters -= 1
            
            best_neighbor_score = current_score
            best_neighbor_A = None
            best_neighbor_M = None
            
            A_set = set(A)
            available = [x for x in range(129) if x not in A_set]
            
            for i in range(len(A)):
                if time.time() > deadline:
                    break
                a = A[i]
                # Try random swaps
                n_candidates = min(8, len(available))
                if n_candidates == 0:
                    break
                candidates = random.sample(available, n_candidates)
                for b in candidates:
                    new_A = A.copy()
                    new_A[i] = b
                    new_A.sort()
                    new_M = calc_mult(new_A)
                    gaps = int(np.sum(new_M == 0))
                    if gaps > 0:
                        new_score = (gaps, float('inf'))
                    else:
                        new_score = (0, int(np.max(new_M)))
                    if new_score < best_neighbor_score:
                        best_neighbor_score = new_score
                        best_neighbor_A = new_A
                        best_neighbor_M = new_M
                        
            if best_neighbor_score < current_score and best_neighbor_A is not None:
                current_score = best_neighbor_score
                A = best_neighbor_A
                M = best_neighbor_M
                improved = True
                if current_score == (0, 1):
                    break
                    
        return current_score, A, M

    def polish_pass(A, M, deadline):
        """High-precision polish: targeted removal/swap on peak sums, then greedy shrink."""
        A = sorted(list(A))
        M = calc_mult(A)
        score = calc_score(A)
        if score[0] > 0:
            return score, A, M
        
        # Iterative targeted reduction
        for _ in range(50):
            if time.time() > deadline:
                break
            score, A, M = targeted_local_search(A, M, deadline, max_iters=50)
            if score == (0, 1):
                break
        
        # Final greedy shrink
        A, M = greedy_removal(A, M, deadline, max_rounds=2000)
        final_score = calc_score(A)
        return final_score, A, M

    best_score = (float('inf'), float('inf'))
    best_A = None
    best_M = None

    # Phase 1: Multi-restart local search with greedy removal
    # Start with dense sets for reliability, then try sparser
    for k in range(16, 28):
        if time.time() > deadline:
            break
        n_restarts = max(5, int(30 * (1 - (k - 16) / 12)))
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            A = sorted(random.sample(range(129), k))
            
            # Local search
            current_score, A, M = local_search(A, deadline)
            
            # Targeted local search to reduce peak multiplicity
            current_score, A, M = targeted_local_search(A, M, deadline, max_iters=200)
            
            # Greedy removal to shrink
            A, M = greedy_removal(A, M, deadline, max_rounds=500)
            current_score = calc_score(A)
            
            if current_score < best_score:
                best_score = current_score
                best_A = list(A)
                best_M = M.copy()
                
            if best_score == (0, 1):
                return sorted(best_A)

    # Phase 2: Start from the seed [64, 128] and optimize
    seed_A = list(range(64, 129))
    seed_M = calc_mult(seed_A)
    seed_score = calc_score(seed_A)
    
    if seed_score < best_score:
        best_score = seed_score
        best_A = list(seed_A)
        best_M = seed_M.copy()

    # Phase 3: Polish the best solution found
    if best_A is not None and best_score[0] == 0:
        # Apply polish pass
        polish_score, polish_A, polish_M = polish_pass(best_A, best_M, deadline)
        if polish_score < best_score:
            best_score = polish_score
            best_A = polish_A
            best_M = polish_M

    # Phase 4: If we haven't found a great solution, try more aggressive search from best
    if best_score[1] > 3 and time.time() < deadline:
        # Try to improve further with more targeted search
        for _ in range(20):
            if time.time() > deadline:
                break
            # Perturb best solution
            A_pert = list(best_A)
            if len(A_pert) > 1:
                idx = random.randint(0, len(A_pert)-1)
                A_pert.pop(idx)
            if len(A_pert) < 129:
                A_set = set(A_pert)
                avail = [x for x in range(129) if x not in A_set]
                if avail:
                    A_pert.append(random.choice(avail))
                    A_pert.sort()
            
            A_pert = sorted(A_pert)
            M_pert = calc_mult(A_pert)
            score_pert, A_pert, M_pert = local_search(A_pert, deadline)
            score_pert, A_pert, M_pert = targeted_local_search(A_pert, M_pert, deadline, max_iters=100)
            A_pert, M_pert = greedy_removal(A_pert, M_pert, deadline, max_rounds=200)
            score_pert = calc_score(A_pert)
            
            if score_pert < best_score:
                best_score = score_pert
                best_A = A_pert
                best_M = M_pert
            
            if best_score == (0, 1):
                break

    # Final polish on the absolute best
    if best_A is not None and best_score[0] == 0 and time.time() < deadline:
        polish_score, polish_A, polish_M = polish_pass(best_A, best_M, deadline)
        if polish_score < best_score:
            best_score = polish_score
            best_A = polish_A
            best_M = polish_M

    if best_A is None:
        # Fallback
        return list(range(64, 129))
        
    return sorted(best_A)
