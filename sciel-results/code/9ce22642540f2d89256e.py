"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd

def construct():
    _t0 = time.time()
    _DEADLINE = 115.0  # safety margin before 150s wall-clock kill

    def _remaining():
        return _DEADLINE - time.time()

    N = 512
    candidates = list(range(1, N))
    weights = {a: 1.0 / (N - a) for a in candidates}
    
    # Sort candidates by weight descending (largest a first)
    sorted_cands = sorted(candidates, key=lambda x: -weights[x])
    
    # Precompute prime factors for each candidate as a bitmask
    primes = []
    is_prime = [True] * (N)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, N):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, N, i):
                is_prime[j] = False
                
    prime_to_bit = {p: 1 << i for i, p in enumerate(primes)}
    
    cand_masks = [0] * N
    for a in range(1, N):
        mask = 0
        temp = a
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= prime_to_bit[p]
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            mask |= prime_to_bit[temp]
        cand_masks[a] = mask

    # Multiple greedy initializations with different orderings
    def _greedy(ordering):
        chosen = []
        used_mask = 0
        for a in ordering:
            if (cand_masks[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= cand_masks[a]
        return chosen, sum(weights[a] for a in chosen)
    
    # Try several orderings
    orderings = [
        sorted_cands,  # by weight desc
        sorted(candidates, key=lambda x: -math.log(weights[x])),  # log-weight
        sorted(candidates, key=lambda x: -weights[x] / (x + 1)),  # weight/size
    ]
    
    best_sol, best_score = _greedy(sorted_cands)
    for ord_ in orderings[1:]:
        sol, sc = _greedy(ord_)
        if sc > best_score:
            best_sol, best_score = sol, sc
    
    # Improved local search with better swap/add logic
    def _local_search(initial, max_iter=200):
        chosen = list(initial)
        chosen_set = set(chosen)
        used_mask = 0
        for a in chosen:
            used_mask |= cand_masks[a]
        score = sum(weights[a] for a in chosen)
        
        iter_count = 0
        improved = True
        while improved and _remaining() > 5.0 and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Try removing each element and adding the best compatible one
            for i in range(len(chosen)):
                a = chosen[i]
                trial_list = chosen[:i] + chosen[i+1:]
                trial_set = set(trial_list)
                trial_mask = 0
                for b in trial_list:
                    trial_mask |= cand_masks[b]
                
                best_add = -1
                best_add_w = weights[a]
                for cand in sorted_cands:
                    if cand in trial_set:
                        continue
                    if weights[cand] <= best_add_w:
                        continue
                    if (cand_masks[cand] & trial_mask) == 0:
                        best_add = cand
                        best_add_w = weights[cand]
                        
                if best_add != -1:
                    chosen = trial_list + [best_add]
                    chosen_set = set(chosen)
                    used_mask = 0
                    for b in chosen:
                        used_mask |= cand_masks[b]
                    score = sum(weights[a] for a in chosen)
                    improved = True
                    break
            
            # Try adding any compatible element without removal
            if not improved:
                best_add = -1
                best_add_w = 0.0
                for cand in sorted_cands:
                    if cand in chosen_set:
                        continue
                    if (cand_masks[cand] & used_mask) == 0:
                        if weights[cand] > best_add_w:
                            best_add = cand
                            best_add_w = weights[cand]
                if best_add != -1:
                    chosen.append(best_add)
                    chosen_set.add(best_add)
                    used_mask |= cand_masks[best_add]
                    score += weights[best_add]
                    improved = True
        
        return chosen, score

    best_sol, best_score = _local_search(best_sol)
    
    # Branch and Bound with tighter bounds
    n_cands = len(sorted_cands)
    cand_list = sorted_cands
    cand_weights = [weights[a] for a in cand_list]
    cand_m_list = [cand_masks[a] for a in cand_list]
    
    # Precompute suffix sums for tighter bounds
    suffix_weight_sum = [0.0] * (n_cands + 1)
    for i in range(n_cands - 1, -1, -1):
        suffix_weight_sum[i] = suffix_weight_sum[i + 1] + cand_weights[i]
    
    def _bb(idx, used_mask, current_score, current_list):
        nonlocal best_score, best_sol
        
        if time.time() > _DEADLINE:
            return
        
        # Quick upper bound: current + remaining weights
        if current_score + suffix_weight_sum[idx] <= best_score:
            return
            
        if idx == n_cands:
            if current_score > best_score:
                best_score = current_score
                best_sol = current_list[:]
            return

        # Tighter bound: only add weights of compatible candidates
        bound = current_score
        for j in range(idx, n_cands):
            if (cand_m_list[j] & used_mask) == 0:
                bound += cand_weights[j]
                if bound > best_score:
                    break
        
        if bound <= best_score:
            return

        a = cand_list[idx]
        mask = cand_m_list[idx]
        
        # Option 1: Take a (if compatible)
        if (mask & used_mask) == 0:
            current_list.append(a)
            _bb(idx + 1, used_mask | mask, current_score + cand_weights[idx], current_list)
            current_list.pop()
            
            # After taking, try to add more compatible elements greedily
            if _remaining() > 10.0:
                extra_mask = used_mask | mask
                extra_score = current_score + cand_weights[idx]
                extra_list = current_list[:]
                for j in range(idx + 1, n_cands):
                    if (cand_m_list[j] & extra_mask) == 0:
                        extra_list.append(cand_list[j])
                        extra_mask |= cand_m_list[j]
                        extra_score += cand_weights[j]
                if extra_score > best_score:
                    best_score = extra_score
                    best_sol = extra_list[:]
            
        # Option 2: Skip a
        _bb(idx + 1, used_mask, current_score, current_list)

    _bb(0, 0, 0.0, [])
    
    # Final polish: one more local search pass
    if _remaining() > 10.0:
        best_sol, best_score = _local_search(best_sol, max_iter=50)
    
    return best_sol
# EVOLVE-BLOCK-END
