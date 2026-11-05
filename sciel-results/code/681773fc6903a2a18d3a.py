import random
import time
import numpy as np

N = 256
MAX_SUM = 2 * N

def construct():
    start_time = time.time()
    deadline = start_time + 25.0

    def is_sidon(A):
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def is_maximal(A):
        A_set = set(A)
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums.add(A[i] + A[j])
        for x in range(1, N + 1):
            if x in A_set:
                continue
            compatible = True
            if 2 * x in sums:
                compatible = False
            else:
                for y in A:
                    if x + y in sums:
                        compatible = False
                        break
            if compatible:
                return False
        return True

    def greedy_np(init_A=None, randomize_ties=False, seed=None, score_mode="block"):
        if seed is not None:
            random.seed(seed)
        
        A = list(init_A) if init_A else []
        A_set = set(A)
        
        sum_arr = np.zeros(MAX_SUM + 1, dtype=bool)
        for i in range(len(A)):
            for j in range(i, len(A)):
                sum_arr[A[i] + A[j]] = True
        
        unblocked = [x for x in range(1, N + 1) if x not in A_set]
        
        while unblocked:
            best_x = None
            best_score = -1
            best_new_sum_arr = None
            
            A_np = np.array(A) if A else np.array([], dtype=int)
            
            for x in unblocked:
                if sum_arr[2 * x]:
                    continue
                if len(A) > 0 and np.any(sum_arr[x + A_np]):
                    continue
                
                new_sums_list = [2 * x]
                if A:
                    new_sums_list.extend(x + y for y in A)
                new_sums_arr = np.unique(np.array(new_sums_list))
                
                new_sum_arr = sum_arr.copy()
                new_sum_arr[new_sums_arr] = True
                
                z_candidates = np.array([z for z in unblocked if z != x])
                if len(z_candidates) == 0:
                    score = 0
                else:
                    blocked_diag = new_sum_arr[2 * z_candidates]
                    A_new = np.append(A_np, x) if len(A) > 0 else np.array([x])
                    z_plus_y = z_candidates[:, None] + A_new[None, :]
                    blocked_off = np.any(new_sum_arr[z_plus_y], axis=1)
                    blocked_mask = blocked_diag | blocked_off
                    
                    if score_mode == "block":
                        score = float(np.sum(blocked_mask))
                    elif score_mode == "weighted":
                        # Weight by how "central" the blocked elements are
                        weights = 1.0 / (1.0 + np.abs(z_candidates - N/2) / (N/2))
                        score = float(np.sum(blocked_mask * weights))
                    else:
                        score = float(np.sum(blocked_mask))
                
                if randomize_ties:
                    score += random.uniform(0, 0.3)
                
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_new_sum_arr = new_sum_arr

            if best_x is None:
                break
            
            A.append(best_x)
            A_set.add(best_x)
            sum_arr = best_new_sum_arr
            unblocked.remove(best_x)
            
        return A

    def try_remove_and_resaturate(A, remove_idx, deadline):
        """Remove element at remove_idx and try to re-saturate to smaller set."""
        if time.time() > deadline:
            return None
        A_minus = A[:remove_idx] + A[remove_idx+1:]
        if not A_minus:
            return None
        if not is_sidon(A_minus):
            return None
        saturated = greedy_np(init_A=A_minus)
        return saturated

    def local_search_improve(A, deadline):
        A = list(A)
        improved = True
        while improved:
            improved = False
            current_size = len(A)
            
            # Try single removals
            for i in range(len(A)):
                if time.time() > deadline:
                    return A
                candidate = A[:i] + A[i+1:]
                if not candidate:
                    continue
                if not is_sidon(candidate):
                    continue
                saturated = greedy_np(init_A=candidate)
                if len(saturated) < current_size:
                    A = saturated
                    current_size = len(A)
                    improved = True
                    break
            
            if improved:
                continue

            # Try double removals (more aggressive)
            for i in range(len(A)):
                if time.time() > deadline:
                    return A
                for j in range(i+1, len(A)):
                    if time.time() > deadline:
                        return A
                    candidate = A[:i] + A[i+1:j] + A[j+1:]
                    if len(candidate) < 2:
                        continue
                    if not is_sidon(candidate):
                        continue
                    saturated = greedy_np(init_A=candidate)
                    if len(saturated) < current_size:
                        A = saturated
                        current_size = len(A)
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue

            # Try swaps
            for i in range(len(A)):
                if time.time() > deadline:
                    return A
                a = A[i]
                A_minus = A[:i] + A[i+1:]
                
                for _ in range(8):
                    if time.time() > deadline:
                        return A
                    b = random.randint(1, N)
                    if b in A_minus:
                        continue
                    candidate_swap = A_minus + [b]
                    if not is_sidon(candidate_swap):
                        continue
                    saturated = greedy_np(init_A=candidate_swap)
                    if len(saturated) < current_size:
                        A = saturated
                        current_size = len(A)
                        improved = True
                        break
                if improved:
                    break
            
        return A

    # Initial deterministic run with different score modes
    best_set = greedy_np(score_mode="block")
    best_size = len(best_set)
    
    # Try weighted scoring
    weighted_set = greedy_np(score_mode="weighted")
    if len(weighted_set) < best_size:
        best_size = len(weighted_set)
        best_set = weighted_set

    # Apply local search
    improved = local_search_improve(best_set, deadline)
    if len(improved) < best_size:
        best_size = len(improved)
        best_set = improved

    # Randomized restarts
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        r = random.random()
        
        if r < 0.3:
            current_set = greedy_np(randomize_ties=True, seed=attempt, score_mode="block")
        elif r < 0.5:
            current_set = greedy_np(randomize_ties=True, seed=attempt, score_mode="weighted")
        elif r < 0.7:
            init_x = random.randint(1, N)
            current_set = greedy_np(init_A=[init_x], randomize_ties=True, seed=attempt, score_mode="block")
        elif r < 0.85:
            x1 = random.randint(1, N)
            x2 = random.randint(1, N)
            attempts = 0
            while (x2 == x1 or x1 + x2 == 2*x1 or x1 + x2 == 2*x2) and attempts < 50:
                x2 = random.randint(1, N)
                attempts += 1
            init = [x1, x2]
            if is_sidon(init):
                current_set = greedy_np(init_A=init, randomize_ties=True, seed=attempt, score_mode="weighted")
            else:
                current_set = greedy_np(randomize_ties=True, seed=attempt, score_mode="block")
        else:
            # Start with 3 random compatible elements
            init = []
            for _ in range(3):
                x = random.randint(1, N)
                if x not in init:
                    test = init + [x]
                    if is_sidon(test):
                        init.append(x)
            if len(init) >= 2:
                current_set = greedy_np(init_A=init, randomize_ties=True, seed=attempt, score_mode="block")
            else:
                current_set = greedy_np(randomize_ties=True, seed=attempt, score_mode="block")
        
        # Apply local search if promising
        if len(current_set) <= best_size + 3:
            improved = local_search_improve(current_set, deadline)
            if len(improved) < best_size:
                best_size = len(improved)
                best_set = improved
    
    return best_set

if __name__ == "__main__":
    s = construct()
    print(f"Size: {len(s)}")
    assert is_sidon(s), "Not Sidon"
    assert is_maximal(s), "Not Maximal"
