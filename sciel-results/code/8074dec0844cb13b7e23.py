"""Improved program: greedy + local search with deadline guard."""
import time
from math import gcd
from fractions import Fraction

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    N = 512
    
    # Precompute weights: weight of vertex a is 1/(512 - a)
    weights_float = [0.0] * N
    for a in range(1, N):
        weights_float[a] = 1.0 / (N - a)
    
    # Precompute prime factors for each number 1..511
    pfactors = [set() for _ in range(N)]
    for a in range(2, N):
        n = a
        d = 2
        while d * d <= n:
            while n % d == 0:
                pfactors[a].add(d)
                n //= d
            d += 1
        if n > 1:
            pfactors[a].add(n)
    
    def is_coprime(a, b):
        return len(pfactors[a] & pfactors[b]) == 0
    
    # Build conflict graph: conflict[a] = set of b != a where gcd(a,b) > 1
    conflict = [set() for _ in range(N)]
    for a in range(2, N):
        for b in range(a + 1, N):
            if not is_coprime(a, b):
                conflict[a].add(b)
                conflict[b].add(a)
    
    # Step 1: Greedy initialization - pick highest weight elements that are coprime
    greedy_set = []
    for a in range(N - 1, 0, -1):  # descending weight
        if time.time() > deadline:
            break
        if all(is_coprime(a, x) for x in greedy_set):
            greedy_set.append(a)
    
    best_set = list(greedy_set)
    best_score = sum(weights_float[a] for a in best_set)
    
    # Step 2: Local search refinement
    all_candidates = sorted(range(1, N), key=lambda x: -weights_float[x])
    
    def can_add_fast(candidate, current_set):
        for x in current_set:
            if candidate in conflict[x]:
                return False
        return True
    
    improved = True
    while improved and time.time() < deadline - 60:
        improved = False
        
        # Single additions
        for a in all_candidates:
            if time.time() > deadline - 60:
                break
            if a not in best_set and can_add_fast(a, best_set):
                best_set.append(a)
                best_score += weights_float[a]
                improved = True
        
        # 1-for-1 swaps
        for i, remove_a in enumerate(list(best_set)):
            if time.time() > deadline - 60:
                break
            for add_b in all_candidates:
                if time.time() > deadline - 60:
                    break
                if add_b == remove_a or add_b in best_set:
                    continue
                valid = all(add_b not in conflict[x] for x in best_set if x != remove_a)
                if valid and weights_float[add_b] > weights_float[remove_a] + 1e-12:
                    best_set[i] = add_b
                    best_score += weights_float[add_b] - weights_float[remove_a]
                    improved = True
                    break
        
        # 1-for-2 swaps
        if time.time() < deadline - 65:
            for i, remove_a in enumerate(list(best_set)):
                if time.time() > deadline - 65:
                    break
                valid_cands = [b for b in all_candidates 
                              if b not in best_set and b != remove_a
                              and all(b not in conflict[x] for x in best_set if x != remove_a)]
                for idx1 in range(len(valid_cands)):
                    if time.time() > deadline - 65:
                        break
                    for idx2 in range(idx1 + 1, len(valid_cands)):
                        if time.time() > deadline - 65:
                            break
                        b1, b2 = valid_cands[idx1], valid_cands[idx2]
                        if b1 in conflict[b2]:
                            continue
                        if weights_float[b1] + weights_float[b2] > weights_float[remove_a] + 1e-12:
                            best_set[i] = b1
                            best_set.append(b2)
                            best_score += weights_float[b1] + weights_float[b2] - weights_float[remove_a]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # Step 3: Try to add more elements by removing blockers
    if time.time() < deadline - 10:
        sorted_by_weight = sorted(range(1, N), key=lambda x: -weights_float[x])
        best_set_set = set(best_set)
        
        for target in sorted_by_weight:
            if time.time() > deadline - 10:
                break
            if target in best_set_set:
                continue
            
            blockers = [x for x in best_set if target in conflict[x]]
            
            if not blockers:
                best_set.append(target)
                best_set_set.add(target)
                best_score += weights_float[target]
                continue
            
            blocker_score = sum(weights_float[b] for b in blockers)
            if weights_float[target] > blocker_score + 1e-12:
                for b in blockers:
                    best_set.remove(b)
                    best_set_set.remove(b)
                    best_score -= weights_float[b]
                best_set.append(target)
                best_set_set.add(target)
                best_score += weights_float[target]
            
            for b in blockers:
                if weights_float[target] > weights_float[b] + 1e-12:
                    other_blockers = [x for x in best_set if x != b and target in conflict[x]]
                    if not other_blockers:
                        best_set.remove(b)
                        best_set_set.remove(b)
                        best_score -= weights_float[b]
                        best_set.append(target)
                        best_set_set.add(target)
                        best_score += weights_float[target]
                        break
    
    # Final validation
    final_set = list(best_set)
    for i in range(len(final_set)):
        for j in range(i + 1, len(final_set)):
            if not is_coprime(final_set[i], final_set[j]):
                return greedy_set
    
    return final_set
