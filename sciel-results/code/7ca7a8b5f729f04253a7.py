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

    # Greedy initial solution for good lower bound
    def _greedy():
        chosen = []
        used_mask = 0
        for a in sorted_cands:
            if (cand_masks[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= cand_masks[a]
        return chosen, sum(weights[a] for a in chosen)

    best_sol, best_score = _greedy()
    
    # Local search to improve initial lower bound
    def _local_search(initial):
        chosen = list(initial)
        chosen_set = set(chosen)
        used_mask = 0
        for a in chosen:
            used_mask |= cand_masks[a]
        score = sum(weights[a] for a in chosen)
        
        improved = True
        while improved and _remaining() > 5.0:
            improved = False
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
        return chosen, score

    best_sol, best_score = _local_search(best_sol)
    
    # Branch and Bound
    n_cands = len(sorted_cands)
    cand_list = sorted_cands
    cand_weights = [weights[a] for a in cand_list]
    cand_m_list = [cand_masks[a] for a in cand_list]
    
    def _bb(idx, used_mask, current_score, current_list):
        nonlocal best_score, best_sol
        
        if time.time() > _DEADLINE:
            return
            
        if idx == n_cands:
            if current_score > best_score:
                best_score = current_score
                best_sol = current_list[:]
            return

        # Calculate upper bound
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
        
        # Option 1: Take a
        if (mask & used_mask) == 0:
            current_list.append(a)
            _bb(idx + 1, used_mask | mask, current_score + cand_weights[idx], current_list)
            current_list.pop()
            
        # Option 2: Skip a
        _bb(idx + 1, used_mask, current_score, current_list)

    _bb(0, 0, 0.0, [])
    
    return best_sol
# EVOLVE-BLOCK-END
