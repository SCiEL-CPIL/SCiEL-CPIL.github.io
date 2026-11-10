import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
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
    
    best_score = (float('inf'), float('inf'))
    best_A = None
    best_M = None
    
    # Phase 1: Multi-restart local search with greedy removal
    for k in range(14, 30):
        if time.time() > deadline:
            break
        n_restarts = max(3, int(60 * (1 - (k - 14) / 16)))
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            A = sorted(random.sample(range(129), k))
            
            # Local search
            current_score, A, M = local_search(A, deadline)
            
            # Greedy removal to shrink
            A, M = greedy_removal(A, M, deadline, max_rounds=500)
            current_score = calc_score(A)
            
            if current_score < best_score:
                best_score = current_score
                best_A = A
                best_M = M
                
            if best_score == (0, 1):
                return sorted(best_A)
    
    # Phase 2: Final greedy removal on best solution
    if best_A is not None:
        best_A, best_M = greedy_removal(best_A, best_M, deadline, max_rounds=1000)
        final_score = calc_score(best_A)
        if final_score < best_score:
            best_score = final_score
    
    if best_A is None:
        # Fallback
        return list(range(64, 129))
        
    return sorted(best_A)
