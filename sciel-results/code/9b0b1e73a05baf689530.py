import time
import bisect
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    L = 1
    for d in denominators:
        L = lcm(L, d)
    
    weights = [L // d for d in denominators]
    
    half = n // 2
    left_size = half
    right_size = n - half
    left_weights = weights[:left_size]
    right_weights = weights[left_size:]
    
    # Generate all subset sums for left half
    left_sums = [0] * (1 << left_size)
    for i in range(left_size):
        bit = 1 << i
        w = left_weights[i]
        for mask in range(1 << i):
            left_sums[bit | mask] = left_sums[mask] + w
    
    # Generate all subset sums for right half
    right_sums = [0] * (1 << right_size)
    for i in range(right_size):
        bit = 1 << i
        w = right_weights[i]
        for mask in range(1 << i):
            right_sums[bit | mask] = right_sums[mask] + w
    
    # Sort right sums with their masks
    right_pairs = [(right_sums[mask], mask) for mask in range(1 << right_size)]
    right_pairs.sort()
    right_sum_values = [p[0] for p in right_pairs]
    
    best_deficit = L
    best_left_mask = 0
    best_right_mask = 0
    
    for mask in range(1 << left_size):
        if time.time() > deadline:
            break
        ls = left_sums[mask]
        if ls >= L:
            continue
        limit = L - ls
        idx = bisect.bisect_left(right_sum_values, limit) - 1
        if idx >= 0:
            rs = right_sum_values[idx]
            total = ls + rs
            deficit = L - total
            if deficit < best_deficit:
                best_deficit = deficit
                best_left_mask = mask
                best_right_mask = right_pairs[idx][1]
    
    # Build the best subset found
    in_subset = [False] * n
    for i in range(left_size):
        if best_left_mask & (1 << i):
            in_subset[i] = True
    for i in range(right_size):
        if best_right_mask & (1 << i):
            in_subset[left_size + i] = True
    
    current_sum = sum(weights[i] for i in range(n) if in_subset[i])
    
    # Efficient local search with multiple move types
    # Precompute sorted weights for faster candidate evaluation
    
    def local_search(in_subset, current_sum, best_deficit, deadline):
        """Perform iterative local search with 1-swap, 1-for-2, 2-for-1, 2-for-2 moves."""
        n = len(in_subset)
        
        while time.time() < deadline:
            improved = False
            in_indices = [i for i in range(n) if in_subset[i]]
            out_indices = [i for i in range(n) if not in_subset[i]]
            
            if not in_indices:
                break
            
            # 1-swap: remove one in, add one out
            # Sort by how much each swap improves the deficit
            candidates = []
            for i in in_indices:
                wi = weights[i]
                for j in out_indices:
                    wj = weights[j]
                    delta = wj - wi  # change in sum
                    new_sum = current_sum + delta
                    if 0 < new_sum < L:
                        new_deficit = L - new_sum
                        if new_deficit < best_deficit:
                            candidates.append((new_deficit, i, j))
            
            if candidates:
                candidates.sort()
                new_deficit, i, j = candidates[0]
                in_subset[i] = False
                in_subset[j] = True
                current_sum += weights[j] - weights[i]
                best_deficit = new_deficit
                improved = True
            
            if improved:
                continue
            
            # 1-for-2: remove one in, add two out
            candidates = []
            for i in in_indices:
                wi = weights[i]
                for j in out_indices:
                    wj = weights[j]
                    for k in out_indices:
                        if k == j:
                            continue
                        wk = weights[k]
                        delta = wj + wk - wi
                        new_sum = current_sum + delta
                        if 0 < new_sum < L:
                            new_deficit = L - new_sum
                            if new_deficit < best_deficit:
                                candidates.append((new_deficit, i, j, k))
            
            if candidates:
                candidates.sort()
                new_deficit, i, j, k = candidates[0]
                in_subset[i] = False
                in_subset[j] = True
                in_subset[k] = True
                current_sum += weights[j] + weights[k] - weights[i]
                best_deficit = new_deficit
                improved = True
            
            if improved:
                continue
            
            # 2-for-1: remove two in, add one out
            candidates = []
            for a in range(len(in_indices)):
                ia = in_indices[a]
                wia = weights[ia]
                for b in range(a + 1, len(in_indices)):
                    ib = in_indices[b]
                    wib = weights[ib]
                    for j in out_indices:
                        wj = weights[j]
                        delta = wj - wia - wib
                        new_sum = current_sum + delta
                        if 0 < new_sum < L:
                            new_deficit = L - new_sum
                            if new_deficit < best_deficit:
                                candidates.append((new_deficit, ia, ib, j))
            
            if candidates:
                candidates.sort()
                new_deficit, ia, ib, j = candidates[0]
                in_subset[ia] = False
                in_subset[ib] = False
                in_subset[j] = True
                current_sum += weights[j] - weights[ia] - weights[ib]
                best_deficit = new_deficit
                improved = True
            
            if improved:
                continue
            
            # 2-for-2: remove two in, add two out
            candidates = []
            for a in range(len(in_indices)):
                ia = in_indices[a]
                wia = weights[ia]
                for b in range(a + 1, len(in_indices)):
                    ib = in_indices[b]
                    wib = weights[ib]
                    for j in out_indices:
                        wj = weights[j]
                        for k in out_indices:
                            if k == j:
                                continue
                            wk = weights[k]
                            delta = wj + wk - wia - wib
                            new_sum = current_sum + delta
                            if 0 < new_sum < L:
                                new_deficit = L - new_sum
                                if new_deficit < best_deficit:
                                    candidates.append((new_deficit, ia, ib, j, k))
            
            if candidates:
                candidates.sort()
                new_deficit, ia, ib, j, k = candidates[0]
                in_subset[ia] = False
                in_subset[ib] = False
                in_subset[j] = True
                in_subset[k] = True
                current_sum += weights[j] + weights[k] - weights[ia] - weights[ib]
                best_deficit = new_deficit
                improved = True
            
            if not improved:
                break
        
        return in_subset, current_sum, best_deficit
    
    # Run local search
    in_subset, current_sum, best_deficit = local_search(
        in_subset, current_sum, best_deficit, deadline
    )
    
    # Try a few perturbation restarts if time remains
    if time.time() < deadline - 10:
        import random
        rng = random.Random(42)
        best_in_subset = in_subset[:]
        best_cs = current_sum
        best_bd = best_deficit
        
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            # Perturb: flip 3-5 random bits
            perturbed = best_in_subset[:]
            num_flips = rng.randint(3, 5)
            flipped = set()
            while len(flipped) < num_flips:
                idx = rng.randint(0, n - 1)
                if idx not in flipped:
                    perturbed[idx] = not perturbed[idx]
                    flipped.add(idx)
            
            # Ensure non-empty and sum < L
            if not any(perturbed):
                perturbed[rng.randint(0, n - 1)] = True
            
            ps = sum(weights[i] for i in range(n) if perturbed[i])
            if ps >= L or ps == 0:
                continue
            
            # Local search from perturbed state
            p_in, p_sum, p_def = local_search(
                perturbed, ps, min(best_bd, L - ps), deadline
            )
            
            if p_def < best_bd:
                best_in_subset = p_in[:]
                best_cs = p_sum
                best_bd = p_def
        
        in_subset = best_in_subset
        current_sum = best_cs
        best_deficit = best_bd
    
    result_indices = [i for i in range(n) if in_subset[i]]
    
    if not result_indices:
        result_indices = [0]
    
    return result_indices
