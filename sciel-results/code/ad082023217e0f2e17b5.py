import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = range(min_sum, max_sum + 1)
    
    def get_counts(A):
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        A_list = sorted(A)
        k = len(A_list)
        if k == 0:
            return counts
        A_arr = np.array(A_list, dtype=np.int32)
        i_idx, j_idx = np.triu_indices(k)
        sums = A_arr[i_idx] + A_arr[j_idx]
        valid_sums = sums[sums <= max_sum]
        if len(valid_sums) > 0:
            bin_counts = np.bincount(valid_sums, minlength=max_sum + 1)
            counts += bin_counts
        return counts

    def get_cost(A):
        counts = get_counts(A)
        target_counts = counts[min_sum:max_sum+1]
        if np.any(target_counts == 0):
            return 1000 + int(np.sum(target_counts == 0))
        return int(np.max(target_counts))

    def local_search(A, time_limit):
        current_cost = get_cost(A)
        best_cost = current_cost
        best_A = set(A)
        
        while time.time() < time_limit:
            # Try swap
            if len(A) > 0:
                x_in = random.choice(list(A))
                x_out = random.randint(0, N)
                if x_out == x_in or x_out in A:
                    continue
                A.remove(x_in)
                A.add(x_out)
                new_cost = get_cost(A)
                if new_cost <= current_cost:
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                else:
                    A.remove(x_out)
                    A.add(x_in)
            else:
                break
                
            # Occasionally try add/remove if size is small/large
            if random.random() < 0.1:
                if len(A) < 30:
                    x = random.randint(0, N)
                    if x not in A:
                        A.add(x)
                        new_cost = get_cost(A)
                        if new_cost <= current_cost:
                            current_cost = new_cost
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_A = set(A)
                        else:
                            A.remove(x)
                elif len(A) > 15:
                    x = random.choice(list(A))
                    A.remove(x)
                    new_cost = get_cost(A)
                    if new_cost <= current_cost:
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_A = set(A)
                    else:
                        A.add(x)
                        
            if best_cost <= 2:
                break
        return best_A, best_cost

    # Phase 1: Generate initial candidates
    candidates = []
    
    # Candidate 1: Random
    random.seed(42)
    A1 = set(random.sample(range(0, N + 1), 25))
    candidates.append(A1)
    
    # Candidate 2: Greedy Add
    A2 = set()
    # Start with some random elements to break symmetry
    A2.update(random.sample(range(0, N + 1), 5))
    for _ in range(20):
        counts = get_counts(A2)
        target_counts = counts[min_sum:max_sum+1]
        uncovered = [s for s in target_sums if target_counts[s - min_sum] == 0]
        if not uncovered:
            break
        best_x = None
        best_gain = -1
        for x in range(0, N + 1):
            if x in A2: continue
            gain = 0
            for y in A2:
                s = x + y
                if min_sum <= s <= max_sum and target_counts[s - min_sum] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x
        if best_x is not None and best_gain > 0:
            A2.add(best_x)
        else:
            break
    candidates.append(A2)
    
    # Candidate 3: Sidon-like (Powers of 2 scaled?)
    # Let's just use another random seed
    random.seed(123)
    A3 = set(random.sample(range(0, N + 1), 20))
    candidates.append(A3)

    # Phase 2: Run local search on best candidate
    # We have 85s total. Use 5s to evaluate candidates, 80s for search.
    eval_deadline = start_time + 5.0
    
    best_candidate = None
    best_initial_cost = float('inf')
    
    for A in candidates:
        if time.time() > eval_deadline: break
        c = get_cost(A)
        if c < best_initial_cost:
            best_initial_cost = c
            best_candidate = A

    if best_candidate is None:
        best_candidate = candidates[0]

    # Run local search
    final_A, final_cost = local_search(best_candidate, deadline)
    
    return list(final_A)
