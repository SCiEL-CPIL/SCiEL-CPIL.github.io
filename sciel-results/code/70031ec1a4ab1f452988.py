import time
from math import gcd
from fractions import Fraction
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit

    N = 256
    values = list(range(2, N + 1))
    size = len(values)  # 255

    # Precompute pairwise conflict: i and j are incompatible if lcm(values[i], values[j]) <= 256
    # conflict_mask[i] is a bitmask where bit j is 1 if values[i] and values[j] conflict
    conflict_mask = [0] * size
    for i in range(size):
        vi = values[i]
        for j in range(i + 1, size):
            vj = values[j]
            g = gcd(vi, vj)
            if vi * vj // g <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    # Precompute weights as floats for fast comparison
    weights = [1.0 / values[i] for i in range(size)]
    
    # Sort indices by weight descending (smaller values first)
    weight_sorted_indices = sorted(range(size), key=lambda i: weights[i], reverse=True)
    
    def greedy_construct(order):
        S_mask = 0
        for i in order:
            if not (conflict_mask[i] & S_mask):
                S_mask |= (1 << i)
        return S_mask

    def local_search(S_mask):
        current_score = sum(weights[i] for i in range(size) if S_mask & (1 << i))
        in_set = [False] * size
        for i in range(size):
            if S_mask & (1 << i):
                in_set[i] = True
        
        improved = True
        while improved:
            if time.time() - start_time > deadline:
                break
            improved = False
            
            # 1. Try Additions
            # Iterate in weight order to find best addition quickly
            for v in weight_sorted_indices:
                if in_set[v]:
                    continue
                if not (conflict_mask[v] & S_mask):
                    S_mask |= (1 << v)
                    in_set[v] = True
                    current_score += weights[v]
                    improved = True
                    # Don't break, keep adding if possible
            
            # 2. Try Swaps
            # For each u in set, try to replace with v not in set where weight[v] > weight[u]
            # and v is compatible with S \ {u}
            for u in range(size):
                if not in_set[u]:
                    continue
                S_without_u = S_mask & ~(1 << u)
                # Find best v
                best_v = -1
                best_gain = 0.0
                # Iterate v in weight order
                for v in weight_sorted_indices:
                    if in_set[v]:
                        continue
                    if weights[v] <= weights[u]:
                        break # No further v can improve
                    if not (conflict_mask[v] & S_without_u):
                        best_v = v
                        best_gain = weights[v] - weights[u]
                        break # First compatible v in weight order is best for this u
                
                if best_v != -1:
                    S_mask = S_without_u | (1 << best_v)
                    in_set[u] = False
                    in_set[best_v] = True
                    current_score += best_gain
                    improved = True
        
        return S_mask, current_score

    # Generate diverse initial solutions
    initial_solutions = []
    
    # 1. Greedy by weight (standard)
    initial_solutions.append(greedy_construct(weight_sorted_indices))
    
    # 2. Greedy by conflict count (least constrained first)
    conflict_counts = [bin(cm).count('1') for cm in conflict_mask]
    order_by_conflict = sorted(range(size), key=lambda x: (conflict_counts[x], -weights[x]))
    initial_solutions.append(greedy_construct(order_by_conflict))
    
    # 3. Randomized Greedy starts
    random.seed(42)
    num_random_starts = 50
    for _ in range(num_random_starts):
        # Shuffle with bias towards high weights? Or just shuffle?
        # Let's do a simple shuffle
        order = list(range(size))
        random.shuffle(order)
        initial_solutions.append(greedy_construct(order))

    best_mask = initial_solutions[0]
    best_score = -1.0
    
    # Local search on each initial solution
    for i, S in enumerate(initial_solutions):
        if time.time() - start_time > deadline - 5:
            break
        S_opt, val = local_search(S)
        if val > best_score:
            best_score = val
            best_mask = S_opt

    # Iterated Local Search (ILS) for further improvement
    # Use remaining time
    if time.time() - start_time < deadline - 10:
        current_mask = best_mask
        current_score = best_score
        in_set = [False] * size
        for i in range(size):
            if current_mask & (1 << i):
                in_set[i] = True
        
        # Perturbation and re-optimization
        while time.time() - start_time < deadline - 2:
            # Perturb: remove k random elements
            k = random.randint(1, 5)
            S_perturbed = current_mask
            for _ in range(k):
                # Pick a random element from current set
                set_indices = [i for i in range(size) if in_set[i]]
                if not set_indices:
                    break
                rem_idx = random.choice(set_indices)
                S_perturbed &= ~(1 << rem_idx)
                in_set[rem_idx] = False
            
            # Local search on perturbed
            S_opt, val = local_search(S_perturbed)
            
            if val > current_score:
                current_score = val
                current_mask = S_opt
                # Update in_set
                in_set = [False] * size
                for i in range(size):
                    if current_mask & (1 << i):
                        in_set[i] = True
                
                if val > best_score:
                    best_score = val
                    best_mask = current_mask
            else:
                # Accept with some probability? Or just reject.
                # Simple rejection is fine for ILS if we have many iterations.
                pass

    # Final verification and conversion
    result = [values[i] for i in range(size) if best_mask & (1 << i)]
    
    # Fallback if empty (should not happen)
    if not result:
        result = [2]
        
    return result

# EVOLVE-BLOCK-END
