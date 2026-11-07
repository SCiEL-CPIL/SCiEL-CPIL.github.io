"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

def construct():
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill
    
    N = 256
    candidates = list(range(2, N + 1))
    n = len(candidates)  # 255 elements
    
    # Prime factorization for each candidate
    def factorize(x):
        factors = {}
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors[d] = factors.get(d, 0) + 1
                x //= d
            d += 1
        if x > 1:
            factors[x] = factors.get(x, 0) + 1
        return factors
    
    factorizations = [factorize(x) for x in candidates]
    
    # Precompute conflict pairs using bitmask approach
    # For each a, conflict_mask[a] is a list of (b, c) pairs where a | candidates[b]*candidates[c]
    # and b != a, c != a, b != c
    
    conflict_pairs = [[] for _ in range(n)]
    
    for a in range(n):
        fa = factorizations[a]
        for b in range(n):
            if b == a:
                continue
            fb = factorizations[b]
            for c in range(b + 1, n):
                if c == a:
                    continue
                fc = factorizations[c]
                # Check if a | candidates[b] * candidates[c]
                ok = True
                for p, ea in fa.items():
                    eb = fb.get(p, 0)
                    ec = fc.get(p, 0)
                    if ea > eb + ec:
                        ok = False
                        break
                if ok:
                    conflict_pairs[a].append((b, c))
    
    def is_valid_set(S_set):
        """Check if S_set is a valid independent set"""
        S_list = list(S_set)
        for a in S_list:
            for (b, c) in conflict_pairs[a]:
                if b in S_set and c in S_set:
                    return False
        return True
    
    def greedy_remove(S_set, seed=None):
        """Greedy removal: iteratively remove the most conflicted element"""
        if seed is not None:
            random.seed(seed)
        S = set(S_set)
        changed = True
        while changed:
            changed = False
            worst = -1
            worst_count = 0
            for a in S:
                count = 0
                for (b, c) in conflict_pairs[a]:
                    if b in S and c in S:
                        count += 1
                        if count >= worst_count:
                            break
                if count > worst_count:
                    worst_count = count
                    worst = a
            if worst_count > 0:
                S.discard(worst)
                changed = True
        return S
    
    def try_add_back(S_set, removed_list):
        """Try to add back removed elements"""
        S = set(S_set)
        removed = list(removed_list)
        improved = True
        while improved:
            improved = False
            for i, r in enumerate(removed):
                new_S = S | {r}
                valid = True
                for a in new_S:
                    if a == r:
                        continue
                    for (b, c) in conflict_pairs[a]:
                        if b in new_S and c in new_S:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    S = new_S
                    removed.pop(i)
                    improved = True
                    break
        return S, removed
    
    def one_swap(S_set, removed_list):
        """Try 1-swap: remove one from S, add one from removed"""
        S = set(S_set)
        removed = list(removed_list)
        best_swap = None
        best_size = len(S)
        
        for r in removed:
            for s in S:
                new_S = (S - {s}) | {r}
                if len(new_S) <= best_size:
                    continue
                valid = True
                for a in new_S:
                    for (b, c) in conflict_pairs[a]:
                        if b in new_S and c in new_S:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_swap = (s, r)
                    best_size = len(new_S)
                    break
            if best_swap:
                break
        
        if best_swap:
            s, r = best_swap
            S = (S - {s}) | {r}
            removed.remove(r)
            removed.append(s)
        
        return S, removed
    
    # Run multiple randomized restarts
    best_S = set()
    best_size = 0
    
    # First, try deterministic greedy
    S_det = greedy_remove(set(range(n)))
    removed_det = list(set(range(n)) - S_det)
    S_det, removed_det = try_add_back(S_det, removed_det)
    S_det, removed_det = one_swap(S_det, removed_det)
    
    if len(S_det) > best_size:
        best_S = S_det
        best_size = len(S_det)
    
    # Randomized restarts
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Randomize the order of removal by shuffling
        S_rand = set(range(n))
        indices = list(range(n))
        random.shuffle(indices)
        
        # Build a weighted removal: remove elements in random order but prioritize high conflict
        changed = True
        while changed:
            changed = False
            # Find all elements with conflicts
            candidates_to_remove = []
            for a in S_rand:
                count = 0
                for (b, c) in conflict_pairs[a]:
                    if b in S_rand and c in S_rand:
                        count += 1
                if count > 0:
                    candidates_to_remove.append((a, count))
            
            if not candidates_to_remove:
                break
            
            # Weighted random selection: higher conflict = higher weight
            total_weight = sum(c for _, c in candidates_to_remove)
            r = random.random() * total_weight
            cumulative = 0
            chosen = candidates_to_remove[0][0]
            for a, count in candidates_to_remove:
                cumulative += count
                if r <= cumulative:
                    chosen = a
                    break
            
            S_rand.discard(chosen)
            changed = True
        
        removed_rand = list(set(range(n)) - S_rand)
        S_rand, removed_rand = try_add_back(S_rand, removed_rand)
        S_rand, removed_rand = one_swap(S_rand, removed_rand)
        
        if len(S_rand) > best_size:
            best_S = S_rand
            best_size = len(S_rand)
    
    # Additional polish: more swap iterations
    if time.time() - start_time < DEADLINE - 10:
        removed_final = list(set(range(n)) - best_S)
        improved = True
        max_iterations = 100
        for _ in range(max_iterations):
            if time.time() - start_time > DEADLINE - 5:
                break
            if not improved:
                break
            improved = False
            # Try multiple swaps
            for _ in range(50):
                if time.time() - start_time > DEADLINE - 5:
                    break
                S_new, rem_new = one_swap(best_S, removed_final)
                if len(S_new) > len(best_S):
                    best_S = S_new
                    removed_final = rem_new
                    improved = True
                    break
    
    # Convert back to actual values
    result = [candidates[i] for i in sorted(best_S)]
    
    # Verify
    def verify(result_list):
        for i in range(len(result_list)):
            for j in range(len(result_list)):
                if i == j:
                    continue
                for k in range(len(result_list)):
                    if i == k or j == k:
                        continue
                    a, b, c = result_list[i], result_list[j], result_list[k]
                    if (b * c) % a == 0:
                        return False
        return True
    
    if not verify(result):
        # Fallback to primes
        result = [a for a in range(2, N+1) if all(a % d for d in range(2, isqrt(a)+1))]
    
    return result
# EVOLVE-BLOCK-END
