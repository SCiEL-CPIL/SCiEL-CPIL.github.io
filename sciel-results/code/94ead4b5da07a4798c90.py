import time
import random
from math import log, log2
import numpy as np
from itertools import combinations

def fast_score(U):
    """
    Score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all pairwise sums (i <= j)
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    num_sums = len(np.unique(sums))
    
    # Compute all pairwise differences (i < j)
    i_idx_d, j_idx_d = np.triu_indices(n, k=1)
    diffs = U_arr[j_idx_d] - U_arr[i_idx_d]
    num_pos_diffs = len(np.unique(diffs))
    num_diffs = 2 * num_pos_diffs + 1
    
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def base_expansion_set(base, max_digits, seed_val=None):
    """
    Generate a set using base-expansion: u_k = sum_{j=0}^{k} b_j * base^j, b_j in {0,1}
    This structurally limits sumset collisions to local digit-wise interactions
    while difference sets expand with number of active digits.
    """
    if seed_val is None:
        seed_val = random.randint(1, max_digits * base)
    
    rng = random.Random(seed_val)
    digits = []
    for j in range(max_digits):
        # Probabilistic digit selection - favor patterns that maximize diff/sum ratio
        if j == 0:
            digits.append(1)  # Always include base^0 to get small elements
        else:
            # Higher probability for certain digit patterns
            p = 0.4 + 0.2 * (j % 2)  # Alternating bias
            if rng.random() < p:
                digits.append(1)
            else:
                digits.append(0)
    
    U = [0]
    for j, b in enumerate(digits):
        if b == 1:
            val = base ** j
            if val not in U:
                U.append(val)
    
    # Also add some combinations of active digits
    active_digits = [j for j, b in enumerate(digits) if b == 1]
    for i in range(len(active_digits)):
        for j in range(i+1, len(active_digits)):
            val = base ** active_digits[i] + base ** active_digits[j]
            if val not in U and val < base ** max_digits:
                U.append(val)
    
    U = sorted(set(U))
    if 0 not in U:
        U = [0] + U
    return U

def geometric_set(base, ratio, n):
    """Generate a geometric-like set starting from base with given ratio."""
    if n <= 1:
        return [0]
    U = [0]
    val = base
    for _ in range(n - 1):
        U.append(val)
        val = int(val * ratio + 0.5)
    return U

def base_expansion_seeds():
    """Generate multiple base-expansion seeds with varying parameters."""
    seeds = []
    for base in [3, 4, 5, 7, 10]:
        for max_digits in [4, 5, 6, 7, 8]:
            for seed_val in range(5):
                G = base_expansion_set(base, max_digits, seed_val)
                if len(G) >= 3:
                    seeds.append(G)
    return seeds

def generate_candidates(U, max_val, n_cands=100):
    """Generate candidate values for adding to set U."""
    candidates = set()
    max_u = max(U) if U else 0
    
    # Random candidates
    for _ in range(n_cands):
        c = random.randint(1, max_val)
        candidates.add(c)
    
    # Candidates near existing elements
    for u in U:
        if u == 0:
            continue
        for delta in [-2, -1, 1, 2, 3, 5, 8, 13]:
            c = u + delta
            if 1 <= c <= max_val and c not in U:
                candidates.add(c)
    
    # Candidates that are sums/differences of existing elements
    for i in range(len(U)):
        for j in range(i + 1, len(U)):
            s = U[i] + U[j]
            d = abs(U[i] - U[j])
            if 1 <= s <= max_val and s not in U:
                candidates.add(s)
            if 1 <= d <= max_val and d not in U:
                candidates.add(d)
    
    return list(candidates)

def local_search_dynamic(U, deadline, max_iter=20000):
    """
    Dynamic local search with adaptive step sizes and simulated annealing.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current[:]
    best_score_local = current_score
    
    # Adaptive parameters
    temp = 0.3
    cooling = 0.999
    step_scale = 1.0
    no_improve_count = 0
    max_no_improve = 500
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        n = len(current)
        if n < 2:
            break
        
        max_val = max(current)
        
        # Adaptive step size - smaller steps for finer search
        step_size = max(1, int(max_val * 0.05 * step_scale))
        
        # Generate multiple candidates per iteration
        num_candidates = 4 + random.randint(0, 3)
        candidates = []
        
        current_set = set(current)
        
        for _ in range(num_candidates):
            i = random.randint(1, n - 1)  # Don't remove 0
            old_val = current[i]
            
            r = random.random()
            if r < 0.5:
                # Local perturbation
                delta = random.randint(-step_size, step_size)
                new_val = old_val + delta
            elif r < 0.8:
                # Medium jump
                new_val = random.randint(1, max_val)
            else:
                # Larger jump
                new_val = random.randint(1, max_val * 2)
            
            if new_val <= 0 or new_val in current_set:
                continue
            
            new_set = current[:]
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            
            if len(new_set) != n:
                continue
            
            candidates.append(new_set)
        
        if not candidates:
            continue
        
        # Evaluate candidates - find best
        best_candidate = None
        best_candidate_score = -1
        
        for new_set in candidates:
            s = fast_score(new_set)
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = new_set
        
        if best_candidate is None:
            continue
        
        # Accept or reject
        if best_candidate_score > current_score:
            current = best_candidate
            current_score = best_candidate_score
            step_scale = min(1.5, step_scale * 1.05)  # Expand search slightly
            no_improve_count = 0
            if current_score > best_score_local:
                best_score_local = current_score
                best = current[:]
        else:
            # Metropolis acceptance
            delta_s = current_score - best_candidate_score
            if delta_s < temp:
                accept_prob = 1.0 / (1.0 + delta_s / max(0.01, temp))
                if random.random() < accept_prob:
                    current = best_candidate
                    current_score = best_candidate_score
                    step_scale = max(0.3, step_scale * 0.95)
            
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Restart with larger step size
                step_scale = min(2.0, step_scale * 1.5)
                no_improve_count = 0
        
        temp *= cooling
    
    return best, best_score_local

def final_polish(U, deadline, max_iter=5000):
    """
    High-precision polish pass with very small steps.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current[:]
    best_score_local = current_score
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        n = len(current)
        if n < 2:
            break
        
        max_val = max(current)
        current_set = set(current)
        
        # Try small perturbations on each element
        for i in range(1, n):
            if time.time() > deadline:
                break
            
            old_val = current[i]
            # Try small deltas
            for delta in [-2, -1, 1, 2]:
                new_val = old_val + delta
                if new_val <= 0 or new_val in current_set:
                    continue
                
                new_set = current[:]
                new_set[i] = new_val
                new_set = sorted(set(new_set))
                
                if len(new_set) != n:
                    continue
                
                s = fast_score(new_set)
                if s > current_score:
                    current = new_set
                    current_score = s
                    current_set = set(current)
                    if current_score > best_score_local:
                        best_score_local = current_score
                        best = current[:]
                    break  # Move on to next element after improvement
    
    return best, best_score_local

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin: evaluator kills at 150s
    
    random.seed(12345)
    
    # Multiple seed strategies
    seeds = []
    
    # Basic seeds
    seeds.append([0, 1, 4, 9, 11])
    seeds.append([0, 1, 2, 4, 8, 16])
    seeds.append([0, 1, 3, 7, 15, 31])
    
    # Geometric seeds with various parameters
    for base in [1, 2, 3, 5]:
        for ratio in [1.5, 2.0, 2.5, 3.0]:
            for n in [5, 6, 7, 8]:
                G = geometric_set(base, ratio, n)
                if 0 in G and len(set(G)) == len(G):
                    seeds.append(G)
    
    # Base-expansion seeds (structural approach)
    be_seeds = base_expansion_seeds()
    seeds.extend(be_seeds)
    
    # Random seeds
    for _ in range(20):
        n = random.randint(4, 8)
        max_v = random.randint(50, 300)
        R = [0] + sorted(random.sample(range(1, max_v), n - 1))
        seeds.append(R)
    
    best_set = seeds[0]
    best_score = fast_score(best_set)
    
    # Evaluate all seeds and keep the best
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        s = fast_score(seed)
        if s > best_score:
            best_score = s
            best_set = seed
    
    # Sort seeds by score to prioritize promising ones
    scored_seeds = []
    for seed in seeds:
        s = fast_score(seed)
        scored_seeds.append((s, seed))
    scored_seeds.sort(key=lambda x: -x[0])
    
    # Local search from best seed
    if time.time() < deadline - 20:
        refined, refined_score = local_search_dynamic(best_set, deadline - 15, max_iter=15000)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    # Local search from other promising seeds (top 5 by score)
    for score, seed in scored_seeds[:5]:
        if time.time() > deadline - 10:
            break
        if seed == best_set:
            continue
        if score > best_score * 0.9:  # Only refine seeds that are close to best
            refined, refined_score = local_search_dynamic(seed, deadline - 5, max_iter=5000)
            if refined_score > best_score:
                best_score = refined_score
                best_set = refined
    
    # Final polish: high-precision pass on the best set
    if time.time() < deadline - 3:
        refined, refined_score = final_polish(best_set, deadline - 1, max_iter=3000)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    # One more local search with remaining time
    if time.time() < deadline - 2:
        remaining_iter = max(1000, int((deadline - time.time()) * 500))
        refined, refined_score = local_search_dynamic(best_set, deadline - 0.5, max_iter=remaining_iter)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    return best_set

if __name__ == "__main__":
    print(f"Best set: {construct()}")
