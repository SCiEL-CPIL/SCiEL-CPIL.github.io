"""Combined program: conflict-aware greedy + improved local search with merge moves + B&B."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    N = 512
    
    # Sieve for primes up to 511
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return is_prime
    
    is_prime = sieve(511)
    
    # For each number 1..511, compute its set of prime factors
    prime_factors = [set() for _ in range(512)]
    for p in range(2, 512):
        if is_prime[p]:
            for multiple in range(p, 512, p):
                prime_factors[multiple].add(p)
    
    # Score for each a: 1/(512-a)
    scores = [0.0] * 512
    for a in range(1, 512):
        scores[a] = 1.0 / (512 - a)
    
    # Precompute sorted candidates by score descending
    candidates_by_score = sorted(range(511, 0, -1), key=lambda x: scores[x], reverse=True)
    
    # Precompute conflict sets
    conflict = [set() for _ in range(512)]
    for a in range(1, 512):
        for b in range(a+1, 512):
            if prime_factors[a] & prime_factors[b]:
                conflict[a].add(b)
                conflict[b].add(a)
    
    def get_score(s):
        return sum(scores[a] for a in s)
    
    def find_best_add(used_primes, current_set, exclude=None):
        """Find the best number to add that is coprime to used_primes and not in current_set."""
        if exclude is None:
            exclude = set()
        for a in candidates_by_score:
            if a in current_set or a in exclude:
                continue
            if not (prime_factors[a] & used_primes):
                return a
        return None
    
    def hill_climb(initial_set, max_iters=8000):
        current = list(initial_set)
        current_score = get_score(current)
        current_set = set(current)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # 1. Try adding any a not in current that is coprime
            used_primes = set()
            for a in current:
                used_primes |= prime_factors[a]
            
            best_add = find_best_add(used_primes, current_set)
            if best_add is not None:
                current.append(best_add)
                current_set.add(best_add)
                current_score += scores[best_add]
                improved = True
                continue

            # 2. Try 1-for-1 swap: remove one, add one
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                remaining = current[:i] + current[i+1:]
                rem_set = set(remaining)
                rem_score = current_score - scores[removed]
                
                rem_used = set()
                for a in remaining:
                    rem_used |= prime_factors[a]
                
                best_add = find_best_add(rem_used, rem_set)
                if best_add is not None:
                    new_score = rem_score + scores[best_add]
                    if new_score > current_score + 1e-15:
                        current = remaining + [best_add]
                        current_set = rem_set | {best_add}
                        current_score = new_score
                        improved = True
                        break
            
            if improved:
                continue

            # 3. Try 1-for-2 swap: remove one, add two
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                remaining = current[:i] + current[i+1:]
                rem_set = set(remaining)
                rem_score = current_score - scores[removed]
                
                rem_used = set()
                for a in remaining:
                    rem_used |= prime_factors[a]
                
                # Find best pair to add
                best_pair = None
                best_pair_score = 0
                
                valid_cands = []
                for a in candidates_by_score:
                    if a in rem_set:
                        continue
                    if prime_factors[a] & rem_used:
                        continue
                    valid_cands.append(a)
                    if len(valid_cands) > 100:
                        break
                
                for idx_a in range(len(valid_cands)):
                    a = valid_cands[idx_a]
                    for idx_b in range(idx_a + 1, len(valid_cands)):
                        b = valid_cands[idx_b]
                        if prime_factors[a] & prime_factors[b]:
                            continue
                        s = rem_score + scores[a] + scores[b]
                        if s > best_pair_score:
                            best_pair_score = s
                            best_pair = (a, b)
                
                if best_pair is not None and best_pair_score > current_score + 1e-15:
                    current = remaining + list(best_pair)
                    current_set = rem_set | set(best_pair)
                    current_score = best_pair_score
                    improved = True
                    break
            
            if improved:
                continue

            # 4. Try 2-for-1 swap (MERGE)
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(current)):
                    if time.time() > deadline:
                        break
                    removed1 = current[i]
                    removed2 = current[j]
                    remaining = [x for k, x in enumerate(current) if k != i and k != j]
                    rem_set = set(remaining)
                    rem_score = current_score - scores[removed1] - scores[removed2]
                    
                    rem_used = set()
                    for a in remaining:
                        rem_used |= prime_factors[a]
                    
                    best_new_score = 0
                    best_add_candidate = None
                    for a in candidates_by_score:
                        if a in rem_set:
                            continue
                        if prime_factors[a] & rem_used:
                            continue
                        new_score = rem_score + scores[a]
                        if new_score > best_new_score:
                            best_new_score = new_score
                            best_add_candidate = a
                        if best_new_score > current_score + 1e-15:
                            break
                    
                    if best_add_candidate is not None and best_new_score > current_score + 1e-15:
                        current = remaining + [best_add_candidate]
                        current_set = rem_set | {best_add_candidate}
                        current_score = best_new_score
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue

            # 5. Try 2-for-2 swap
            if len(current) >= 2:
                for i in range(len(current)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(current)):
                        if time.time() > deadline:
                            break
                        removed1 = current[i]
                        removed2 = current[j]
                        remaining = [x for k, x in enumerate(current) if k != i and k != j]
                        rem_set = set(remaining)
                        rem_score = current_score - scores[removed1] - scores[removed2]
                        
                        rem_used = set()
                        for a in remaining:
                            rem_used |= prime_factors[a]
                        
                        best_pair = None
                        best_pair_score = 0
                        
                        valid_cands = []
                        for a in candidates_by_score:
                            if a in rem_set:
                                continue
                            if prime_factors[a] & rem_used:
                                continue
                            valid_cands.append(a)
                            if len(valid_cands) > 80:
                                break
                        
                        for idx_a in range(len(valid_cands)):
                            a = valid_cands[idx_a]
                            for idx_b in range(idx_a + 1, len(valid_cands)):
                                b = valid_cands[idx_b]
                                if prime_factors[a] & prime_factors[b]:
                                    continue
                                s = rem_score + scores[a] + scores[b]
                                if s > best_pair_score:
                                    best_pair_score = s
                                    best_pair = (a, b)
                        
                        if best_pair is not None and best_pair_score > current_score + 1e-15:
                            current = remaining + list(best_pair)
                            current_set = rem_set | set(best_pair)
                            current_score = best_pair_score
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return current, current_score
    
    def branch_and_bound(initial_set, time_budget=40.0):
        """
        B&B with tighter upper bound using greedy independent set.
        """
        bb_start = time.time()
        best_score_bb = get_score(initial_set)
        best_set_bb = list(initial_set)
        
        current = list(initial_set)
        current_score = best_score_bb
        current_set = set(current)
        
        def greedy_ub(used_primes, excluded_set):
            ub = 0.0
            taken = set()
            taken_primes = set(used_primes)
            for a in candidates_by_score:
                if a in excluded_set or a in taken:
                    continue
                if prime_factors[a] & taken_primes:
                    continue
                ub += scores[a]
                taken.add(a)
                taken_primes |= prime_factors[a]
            return ub
        
        def dfs(used_primes, current_score, current_set, candidate_list, depth=0):
            nonlocal best_score_bb, best_set_bb
            
            if time.time() - bb_start > time_budget:
                return
            
            if depth > 60:
                return
            
            ub = current_score + greedy_ub(used_primes, current_set)
            
            if ub <= best_score_bb + 1e-15:
                return
            
            valid_candidates = [a for a in candidate_list if a not in current_set and not (prime_factors[a] & used_primes)]
            
            if not valid_candidates:
                return
            
            for idx, a in enumerate(valid_candidates):
                if time.time() - bb_start > time_budget:
                    return
                
                new_used = used_primes | prime_factors[a]
                new_score = current_score + scores[a]
                new_set = current_set | {a}
                
                if new_score > best_score_bb + 1e-15:
                    best_score_bb = new_score
                    best_set_bb = list(new_set)
                
                remaining_after = valid_candidates[idx+1:]
                ub_after = new_score + greedy_ub(new_used, new_set)
                if ub_after <= best_score_bb + 1e-15:
                    continue
                
                dfs(new_used, new_score, new_set, remaining_after, depth + 1)
        
        initial_used = set()
        for a in initial_set:
            initial_used |= prime_factors[a]
        
        candidate_list = [a for a in candidates_by_score if a not in current_set]
        
        dfs(initial_used, current_score, current_set, candidate_list)
        
        return best_set_bb, best_score_bb
    
    # Initial strategy: Start with all primes
    initial_set = [p for p in range(2, 512) if is_prime[p]]
    
    # Hill climb from all primes
    best_set, best_score = hill_climb(initial_set, max_iters=5000)
    
    # Try B&B from the current best
    if time.time() < deadline - 55:
        bb_set, bb_score = branch_and_bound(best_set, time_budget=45.0)
        if bb_score > best_score + 1e-15:
            best_set = bb_set
            best_score = bb_score
    
    # Polish after B&B
    if time.time() < deadline - 25:
        result, score = hill_climb(best_set, max_iters=8000)
        if score > best_score + 1e-15:
            best_set = result
            best_score = score
    
    # Random restarts with more aggressive perturbation
    rng = random.Random(42)
    
    for restart in range(40):
        if time.time() > deadline - 20:
            break
        
        perturbed = best_set[:]
        k = rng.randint(1, min(6, len(perturbed)))
        for _ in range(k):
            if len(perturbed) > 1:
                idx = rng.randrange(len(perturbed))
                perturbed.pop(idx)
        
        if len(perturbed) < len(best_set) - 1:
            for a in candidates_by_score[:15]:
                if a not in perturbed:
                    used = set()
                    for x in perturbed:
                        used |= prime_factors[x]
                    if not (prime_factors[a] & used):
                        perturbed.append(a)
                        break
        
        result, score = hill_climb(perturbed, max_iters=2000)
        
        if score > best_score + 1e-15:
            best_set = result
            best_score = score
    
    # Final polish with more iterations
    if time.time() < deadline - 5:
        result, score = hill_climb(best_set, max_iters=5000)
        if score > best_score + 1e-15:
            best_set = result
            best_score = score
            
    return best_set
# EVOLVE-BLOCK-END
