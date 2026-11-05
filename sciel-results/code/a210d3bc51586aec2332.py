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
    num_targets = max_sum - min_sum + 1

    # Precompute pair sums for all i, j in [0, 128]
    # sums[i, j] = i + j
    # We need to track counts for each sum s in [0, 256]
    # counts[s] = number of pairs (i, j) with i <= j, i in A, j in A, i+j=s
    
    # To make incremental updates fast, we precompute for each element x,
    # the list of sums it contributes to, and the other elements involved.
    # Actually, simpler:
    # When adding x to A:
    #   For each y in A:
    #       s = x + y
    #       counts[s] += 1
    #   s = 2*x
    #   counts[s] += 1
    # When removing x from A:
    #   For each y in A (excluding x):
    #       s = x + y
    #       counts[s] -= 1
    #   s = 2*x
    #       counts[s] -= 1

    def get_counts(A):
        counts = np.zeros(max_sum + 1, dtype=np.int64)
        A_list = sorted(A)
        for i in range(len(A_list)):
            xi = A_list[i]
            s = 2 * xi
            if s <= max_sum:
                counts[s] += 1
            for j in range(i + 1, len(A_list)):
                xj = A_list[j]
                s = xi + xj
                if s <= max_sum:
                    counts[s] += 1
        return counts

    def get_cost(A, counts):
        target_counts = counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            return 1000 + uncovered
        return int(np.max(target_counts))

    best_A = None
    best_cost = float('inf')

    # Generate diverse initial candidates
    candidates = []

    # 1. Greedy Construction
    # Start with empty A. Add element that covers most uncovered sums.
    # This might be slow if done naively, but N=128 is small.
    # Uncovered sums: initially all [128, 256].
    # For each candidate x in [0, 128], calculate how many uncovered sums it covers.
    # A sum s is covered by x if there exists y in A such that x+y=s.
    # If A is empty, x covers no sums (needs y=x, so 2x).
    # Actually, if A is empty, adding x covers 2x.
    # Let's just use random and structured sets for initialization, 
    # and let SA do the heavy lifting.

    # Random sets
    for size in range(15, 25):
        for _ in range(10):
            A_init = set(random.sample(range(129), size))
            candidates.append(A_init)

    # Structured: Arithmetic Progressions
    for start in range(0, 10):
        for step in range(5, 15):
            A_init = set(range(start, 129, step))
            if 0 < len(A_init) < 30:
                candidates.append(A_init)

    # Structured: Two clusters
    for k in range(10, 20):
        A_init = set(range(0, k)) | set(range(128 - k + 1, 129))
        candidates.append(A_init)

    # Evaluate candidates
    for A in candidates:
        counts = get_counts(A)
        cost = get_cost(A, counts)
        if cost < best_cost:
            best_cost = cost
            best_A = set(A)

    # Simulated Annealing with Tabu
    if best_A is not None:
        A = set(best_A)
        counts = get_counts(A)
        current_cost = get_cost(A, counts)

        # SA Parameters
        T_init = 50.0
        T_final = 0.1
        cooling_rate = 0.999
        T = T_init
        tabu_list = []
        tabu_len = 20
        rng = random.Random(42)

        outside = list(set(range(129)) - A)
        A_list = list(A)

        steps = 0
        max_steps = 200000
        
        while T > T_final and steps < max_steps and time.time() < deadline - 5.0:
            steps += 1
            move_type = rng.random()
            
            if move_type < 0.4 and len(outside) > 0:
                # Swap
                idx_in = rng.randrange(len(A_list))
                idx_out = rng.randrange(len(outside))
                x_in = A_list[idx_in]
                x_out = outside[idx_out]
                
                # Check tabu
                if (x_in, x_out) in tabu_list or (x_out, x_in) in tabu_list:
                    continue
                    
                # Calculate new cost
                # Remove x_in, Add x_out
                new_counts = counts.copy()
                
                # Remove x_in
                s = 2 * x_in
                if s <= max_sum: new_counts[s] -= 1
                for y in A_list:
                    if y == x_in: continue
                    s = x_in + y
                    if s <= max_sum: new_counts[s] -= 1
                    
                # Add x_out
                s = 2 * x_out
                if s <= max_sum: new_counts[s] += 1
                for y in A_list:
                    if y == x_in: continue # y is in A, x_in is being removed, so y != x_in is true for all y in A_list except x_in itself
                    # Wait, A_list contains x_in. We are iterating over A.
                    # The pairs involving x_out are (x_out, y) for y in A.
                    # Since x_in is removed, A_new = A - {x_in} + {x_out}.
                    # Pairs are (x_out, x_out) and (x_out, y) for y in A_new, y != x_out.
                    # y in A_new means y in A and y != x_in.
                    # So we iterate y in A_list. If y == x_in, skip.
                    s = x_out + y
                    if s <= max_sum: new_counts[s] += 1

                new_cost = get_cost(A, new_counts) # A is not used in get_cost, only counts
                delta = new_cost - current_cost
                
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    # Accept
                    A.remove(x_in)
                    A.add(x_out)
                    A_list[idx_in] = x_out
                    outside[idx_out] = x_in
                    counts = new_counts
                    current_cost = new_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                        
                    tabu_list.append((x_in, x_out))
                    if len(tabu_list) > tabu_len:
                        tabu_list.pop(0)

            elif move_type < 0.7 and len(A_list) > 10:
                # Remove
                idx_in = rng.randrange(len(A_list))
                x_rem = A_list[idx_in]
                
                if x_rem in [t[0] for t in tabu_list] or x_rem in [t[1] for t in tabu_list]:
                    continue

                new_counts = counts.copy()
                s = 2 * x_rem
                if s <= max_sum: new_counts[s] -= 1
                for y in A_list:
                    if y == x_rem: continue
                    s = x_rem + y
                    if s <= max_sum: new_counts[s] -= 1
                    
                new_cost = get_cost(A, new_counts)
                delta = new_cost - current_cost
                
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    A.remove(x_rem)
                    A_list.pop(idx_in)
                    outside.append(x_rem)
                    counts = new_counts
                    current_cost = new_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                        
                    tabu_list.append((x_rem, -1)) # Marker for removal
                    if len(tabu_list) > tabu_len:
                        tabu_list.pop(0)

            elif move_type < 1.0 and len(outside) > 0 and len(A_list) < 30:
                # Add
                idx_out = rng.randrange(len(outside))
                x_add = outside[idx_out]
                
                if x_add in [t[0] for t in tabu_list] or x_add in [t[1] for t in tabu_list]:
                    continue

                new_counts = counts.copy()
                s = 2 * x_add
                if s <= max_sum: new_counts[s] += 1
                for y in A_list:
                    s = x_add + y
                    if s <= max_sum: new_counts[s] += 1
                    
                new_cost = get_cost(A, new_counts)
                delta = new_cost - current_cost
                
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    A.add(x_add)
                    A_list.append(x_add)
                    outside.pop(idx_out)
                    counts = new_counts
                    current_cost = new_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                        
                    tabu_list.append((x_add, -1))
                    if len(tabu_list) > tabu_len:
                        tabu_list.pop(0)
                        
            T *= cooling_rate

    # Final Polish: Local search without SA (greedy) to ensure we are at a local minimum
    if best_A is not None:
        A = set(best_A)
        counts = get_counts(A)
        current_cost = get_cost(A, counts)
        A_list = list(A)
        outside = list(set(range(129)) - A)
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try all swaps
            for i in range(len(A_list)):
                for j in range(len(outside)):
                    x_in = A_list[i]
                    x_out = outside[j]
                    
                    new_counts = counts.copy()
                    # Remove x_in
                    s = 2 * x_in
                    if s <= max_sum: new_counts[s] -= 1
                    for y in A_list:
                        if y == x_in: continue
                        s = x_in + y
                        if s <= max_sum: new_counts[s] -= 1
                    # Add x_out
                    s = 2 * x_out
                    if s <= max_sum: new_counts[s] += 1
                    for y in A_list:
                        if y == x_in: continue
                        s = x_out + y
                        if s <= max_sum: new_counts[s] += 1
                        
                    new_cost = get_cost(A, new_counts)
                    if new_cost < current_cost:
                        A.remove(x_in)
                        A.add(x_out)
                        A_list[i] = x_out
                        outside[j] = x_in
                        counts = new_counts
                        current_cost = new_cost
                        best_cost = current_cost
                        best_A = set(A)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            
            # Try all removals
            for i in range(len(A_list)):
                x_rem = A_list[i]
                new_counts = counts.copy()
                s = 2 * x_rem
                if s <= max_sum: new_counts[s] -= 1
                for y in A_list:
                    if y == x_rem: continue
                    s = x_rem + y
                    if s <= max_sum: new_counts[s] -= 1
                new_cost = get_cost(A, new_counts)
                if new_cost < current_cost:
                    A.remove(x_rem)
                    A_list.pop(i)
                    outside.append(x_rem)
                    counts = new_counts
                    current_cost = new_cost
                    best_cost = current_cost
                    best_A = set(A)
                    improved = True
                    break
            if improved:
                continue

            # Try all additions
            for j in range(len(outside)):
                x_add = outside[j]
                new_counts = counts.copy()
                s = 2 * x_add
                if s <= max_sum: new_counts[s] += 1
                for y in A_list:
                    s = x_add + y
                    if s <= max_sum: new_counts[s] += 1
                new_cost = get_cost(A, new_counts)
                if new_cost < current_cost:
                    A.add(x_add)
                    A_list.append(x_add)
                    outside.pop(j)
                    counts = new_counts
                    current_cost = new_cost
                    best_cost = current_cost
                    best_A = set(A)
                    improved = True
                    break

    return list(best_A)
