import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    best_score = -1
    best_set = [0, 1, 3]

    def compute_score_fast(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        if len(U) < 2:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 0
        
        # Create indicator vector
        size = 2 * max_u + 1
        A = np.zeros(size, dtype=np.float64)
        for x in U:
            A[x] = 1.0
            
        # Sumset: Convolution of A with A
        # A + A will have indices from 0 to 2*max_u
        # np.convolve(A, A) will have length 2*size - 1
        conv_sum = np.convolve(A, A, mode='full')
        n_sum = np.count_nonzero(conv_sum)
        
        # Diffset: Convolution of A with reversed A
        # A - A will have indices from -max_u to max_u
        # Shift A so that 0 is in the middle? 
        # Alternatively, convolve A with A_flipped.
        A_flipped = A[::-1]
        conv_diff = np.convolve(A, A_flipped, mode='full')
        # The result of conv_diff corresponds to differences.
        # The center of conv_diff corresponds to difference 0.
        # We just need the count of non-zero elements.
        n_diff = np.count_nonzero(conv_diff)
        
        if n_sum == 0:
            return -1
            
        base = 2 * max_u + 1
        if base <= 1:
            return 0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
            
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if len(cand) < 2:
            return
        s = compute_score_fast(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Initial candidates: Known Golomb Rulers
    golomb_rulers = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 26, 29, 34, 41],
        [0, 1, 4, 10, 18, 26, 29, 34, 41, 49, 54]
    ]
    
    for gr in golomb_rulers:
        if time.time() > deadline - 10:
            break
        try_set(gr)

    # Local Search
    if time.time() < deadline - 5:
        current = list(best_set)
        no_improve_count = 0
        max_iterations = 50000
        
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # 1. Try adding an element
            if len(current) < 15:
                max_cur = max(current)
                # Search range: up to max_cur + 20
                search_limit = max_cur + 20
                # To make it faster, we can sample or limit the range
                # But adding an element increases N, which might increase ratio
                # We should focus on extending the ruler
                for new_el in range(1, search_limit + 1):
                    if new_el in current:
                        continue
                    cand = sorted(current + [new_el])
                    s = compute_score_fast(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        no_improve_count = 0
                        break
                    elif s > best_score - 0.001:
                        # Accept if close? No, strict improvement for now to avoid getting stuck in local minima with slightly worse scores
                        pass
            
            # 2. Try replacing an element
            if len(current) >= 2:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    max_cur = max(current)
                    # Search for a replacement in a wider range
                    # Sometimes moving an element further out can help
                    search_limit = max_cur + 10
                    for new_el in range(1, search_limit + 1):
                        if new_el in current:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand or len(cand) < 2:
                            continue
                        s = compute_score_fast(cand)
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                            current = list(cand)
                            improved = True
                            no_improve_count = 0
                            break
                    if improved:
                        break
            
            # 3. Try removing an element
            if len(current) > 2:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    cand = [x for j, x in enumerate(current) if j != i]
                    s = compute_score_fast(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        no_improve_count = 0
                        break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 100:
                    # Random restart or perturbation
                    if random.random() < 0.2:
                        # Jump to a random Golomb-like set
                        n = random.randint(3, 10)
                        max_m = random.randint(10, 60)
                        # Generate a random set with distinct differences (approx)
                        # Just random sample
                        inner = random.sample(range(1, max_m + 1), min(n-1, max_m))
                        cand = sorted([0] + inner)
                        if len(cand) >= 2:
                            s = compute_score_fast(cand)
                            if s > best_score:
                                best_score = s
                                best_set = list(cand)
                                current = list(cand)
                                improved = True
                                no_improve_count = 0
                    else:
                        # Small perturbation
                        if len(current) > 2:
                            idx = random.randint(1, len(current)-1)
                            delta = random.randint(-5, 5)
                            new_val = current[idx] + delta
                            if new_val > 0 and new_val not in current:
                                cand = list(current)
                                cand[idx] = new_val
                                cand = sorted(set(cand))
                                if 0 in cand and len(cand) >= 2:
                                    s = compute_score_fast(cand)
                                    if s > best_score:
                                        best_score = s
                                        best_set = list(cand)
                                        current = list(cand)
                                        improved = True
                                        no_improve_count = 0

    return list(best_set)
