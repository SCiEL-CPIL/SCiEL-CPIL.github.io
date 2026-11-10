import time
import numpy as np
from math import gcd, isqrt
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 15s margin for safety

    MAX_N = 256

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(MAX_N)
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    # Pre-compute divisibility: for each n, which primes divide it
    divisors = [[] for _ in range(MAX_N + 1)]
    for n in range(1, MAX_N + 1):
        for p in primes:
            if n % p == 0:
                divisors[n].append(p)

    # Branch and Bound: exact solver
    # Key insight: each prime can be "used" by at most one number in the set.
    # For each prime p, the maximum contribution is the largest power of p <= 256
    # that doesn't share other primes with already-selected numbers.
    
    # For upper bound: for each prime p not yet "claimed", the best we can add
    # is the largest multiple of p (by p-power) that is coprime to all selected.
    # Simpler bound: sum over unclaimed primes of max prime power <= 256
    
    # Precompute: for each prime, the largest power p^k <= 256
    max_prime_power = {}
    for p in primes:
        power = p
        while power * p <= MAX_N:
            power *= p
        max_prime_power[p] = power

    # For B&B: state = set of selected numbers, set of used primes
    # Upper bound = current_sum + sum of max_prime_power[p] for p not in used_primes
    # But this is loose. Better: for each unused prime p, find the largest n<=256
    # divisible by p, coprime to all selected, and not using other unused primes
    # that could conflict... this gets complex.
    
    # Let's use a tighter bound: for each unused prime p, the candidate is the
    # largest number <= 256 that is a power of p (p, p^2, ...) since mixing primes
    # reduces the total. Actually no - we can mix, e.g., 3*5=15 uses both 3 and 5.
    
    # Better upper bound approach:
    # For each unused prime p, compute the max value we can assign to "slots" 
    # associated with p. The simplest valid upper bound: sum of max_prime_power[p]
    # for unused p. This is valid because any number using prime p contributes at most
    # its p-part to p's "budget", and other primes' contributions are accounted for separately.
    
    # Actually, a number like 30 = 2*3*5 uses all three primes. Its value 30 > 2+3+5=10.
    # So sum of prime powers is NOT a valid upper bound!
    
    # Correct upper bound: For each unused prime p, the maximum contribution from numbers
    # that include p is bounded by the maximum n <= 256 that is coprime to selected and 
    # divisible by p. But multiple primes can share a number...
    
    # Let me use a different approach: the LP relaxation of the set packing problem.
    # Or simply: upper bound = current_sum + sum of all remaining numbers that are coprime to selected
    # (this is very loose but valid)
    
    # Actually, let me think differently. The key structure:
    # - Each prime p can appear in at most one selected number
    # - A number n = prod(p_i^e_i) "uses" all primes p_i
    # - We want to maximize sum of selected numbers
    
    # Upper bound idea: For each unused prime p, let f(p) = max{n <= 256 : p|n, gcd(n,S)=1, 
    # and for all q|n, q is unused}. Then UB = sum(S) + sum(f(p) for p unused) is NOT valid
    # because one number can use multiple primes.
    
    # Valid UB: sum(S) + max over all subsets T of unused primes of sum of max n using exactly primes in T
    # This is exponential.
    
    # Practical UB: sum(S) + sum of largest available numbers that are coprime to S, 
    # but each prime can only be counted once. Use greedy: sort candidates by value desc,
    # add if coprime to S and to previously added candidates.
    
    def compute_upper_bound(selected_set, used_primes_set):
        """Compute an upper bound on the total sum."""
        current_sum = sum(selected_set)
        # Find all candidates coprime to selected
        candidates = []
        for n in range(MAX_N, 0, -1):
            if n in selected_set:
                continue
            if gcd(n, 1) == 1 and all(gcd(n, x) == 1 for x in selected_set):
                # Check if n uses only unused primes
                n_primes = set(divisors[n])
                if n_primes.issubset(set(primes) - used_primes_set):
                    candidates.append(n)
        
        # Greedy pack: pick largest coprime numbers
        bound_add = 0
        temp_used = set()
        for n in candidates:
            if all(p not in temp_used for p in divisors[n]):
                bound_add += n
                temp_used.update(divisors[n])
        
        return current_sum + bound_add

    # Branch and Bound
    best_sum = 0
    best_selection = []
    
    # Initial solution: greedy
    greedy_set = set()
    greedy_sum = 0
    for n in range(MAX_N, 0, -1):
        if all(gcd(n, x) == 1 for x in greedy_set):
            greedy_set.add(n)
            greedy_sum += n
    best_sum = greedy_sum
    best_selection = list(greedy_set)
    
    # B&B search
    # State: (selected_list, used_primes_set, current_sum)
    # Branch on: include or exclude next candidate
    
    # Candidates sorted by value descending
    all_candidates = list(range(MAX_N, 0, -1))
    
    nodes_explored = 0
    
    def bb_search(selected_list, used_primes_set, current_sum, start_idx):
        nonlocal best_sum, best_selection, nodes_explored
        
        if time.time() > deadline - 5:
            return
        
        nodes_explored += 1
        
        # Upper bound check
        if start_idx >= len(all_candidates):
            if current_sum > best_sum:
                best_sum = current_sum
                best_selection = list(selected_list)
            return
        
        # Compute upper bound
        ub = compute_upper_bound(set(selected_list), used_primes_set)
        if ub <= best_sum:
            return
        
        # Try to include candidates from start_idx
        for idx in range(start_idx, len(all_candidates)):
            n = all_candidates[idx]
            n_primes = set(divisors[n])
            
            # Check if n can be added
            if n_primes.issubset(set(primes) - used_primes_set):
                # Check coprime with selected
                can_add = True
                for x in selected_list:
                    if gcd(n, x) != 1:
                        can_add = False
                        break
                
                if can_add:
                    # Include n
                    new_selected = selected_list + [n]
                    new_used = used_primes_set | n_primes
                    new_sum = current_sum + n
                    
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_selection = list(new_selected)
                    
                    bb_search(new_selected, new_used, new_sum, idx + 1)
        
        # Exclude: try next candidate without including current
        # This is handled by the loop structure (we try all candidates from start_idx)
        # But we also need to explore the "skip" branch
        # Actually the loop already handles trying each candidate. We need to also
        # explore subsets that skip some candidates.
        # Let me restructure: at each level, decide include/exclude for all_candidates[start_idx]
    
    # The above B&B is not quite right. Let me use a cleaner formulation.
    
    def bb_search2(selected_list, used_primes_set, current_sum, idx):
        nonlocal best_sum, best_selection, nodes_explored
        
        if time.time() > deadline - 5:
            return
        
        nodes_explored += 1
        
        if idx >= len(all_candidates):
            if current_sum > best_sum:
                best_sum = current_sum
                best_selection = list(selected_list)
            return
        
        n = all_candidates[idx]
        n_primes = set(divisors[n])
        
        # Option 1: Exclude n
        bb_search2(selected_list, used_primes_set, current_sum, idx + 1)
        
        # Option 2: Include n (if valid)
        if n_primes.issubset(set(primes) - used_primes_set):
            can_add = True
            for x in selected_list:
                if gcd(n, x) != 1:
                    can_add = False
                    break
            
            if can_add:
                new_sum = current_sum + n
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_selection = list(selected_list + [n])
                
                bb_search2(selected_list + [n], used_primes_set | n_primes, new_sum, idx + 1)
    
    # This is O(2^256) which is way too much. We need much tighter bounds.
    # Let me use the greedy upper bound to prune.
    
    def bb_search3(selected_list, used_primes_set, current_sum, idx):
        nonlocal best_sum, best_selection, nodes_explored
        
        if time.time() > deadline - 5:
            return
        
        nodes_explored += 1
        
        # Upper bound: current_sum + greedy pack of remaining candidates
        remaining = [all_candidates[i] for i in range(idx, len(all_candidates))]
        bound_add = 0
        temp_used = set(used_primes_set)
        for n in remaining:  # already sorted desc
            n_primes = set(divisors[n])
            if n_primes.issubset(set(primes) - temp_used):
                can_add = True
                for x in selected_list:
                    if gcd(n, x) != 1:
                        can_add = False
                        break
                if can_add:
                    bound_add += n
                    temp_used |= n_primes
        
        if current_sum + bound_add <= best_sum:
            return
        
        if idx >= len(all_candidates):
            if current_sum > best_sum:
                best_sum = current_sum
                best_selection = list(selected_list)
            return
        
        n = all_candidates[idx]
        n_primes = set(divisors[n])
        
        # Option 1: Exclude n
        bb_search3(selected_list, used_primes_set, current_sum, idx + 1)
        
        # Option 2: Include n
        if n_primes.issubset(set(primes) - used_primes_set):
            can_add = True
            for x in selected_list:
                if gcd(n, x) != 1:
                    can_add = False
                    break
            
            if can_add:
                new_sum = current_sum + n
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_selection = list(selected_list + [n])
                
                bb_search3(selected_list + [n], used_primes_set | n_primes, new_sum, idx + 1)
    
    # Even this might be too slow for 256 candidates. The bound computation is O(n^2) per node.
    # Let me limit the search depth or use a better approach.
    
    # Alternative: Use the structure more. Since numbers > 128 can only be used if they're
    # prime powers or products of large primes, the effective search space is smaller.
    
    # Let me try a hybrid: B&B on top numbers, then local search to polish.
    
    # Filter candidates: only consider numbers that could potentially be in optimal solution
    # A number n is "dominated" if there's a larger number m with same prime set and m coprime to all
    # Actually, let's just run B&B with time limit and see how far we get.
    
    # Optimize: precompute prime sets
    n_prime_sets = [set(divisors[n]) for n in range(MAX_N + 1)]
    prime_set_list = [frozenset(s) for s in n_prime_sets]
    
    # For faster coprime check, precompute gcd matrix
    gcd_matrix = [[0]*(MAX_N+1) for _ in range(MAX_N+1)]
    for i in range(1, MAX_N+1):
        for j in range(i, MAX_N+1):
            gcd_matrix[i][j] = gcd(i, j)
            gcd_matrix[j][i] = gcd_matrix[i][j]
    
    # B&B with tighter bound and early termination
    def bb_search4(selected_list, used_primes_frozen, current_sum, idx):
        nonlocal best_sum, best_selection, nodes_explored
        
        if time.time() > deadline - 5:
            return
        
        nodes_explored += 1
        
        if idx >= len(all_candidates):
            if current_sum > best_sum:
                best_sum = current_sum
                best_selection = list(selected_list)
            return
        
        # Quick upper bound: sum of all remaining that don't share primes with used
        bound_add = 0
        for i in range(idx, len(all_candidates)):
            n = all_candidates[i]
            if not (prime_set_list[n] & used_primes_frozen):
                # Check coprime with selected
                ok = True
                for x in selected_list:
                    if gcd_matrix[n][x] != 1:
                        ok = False
                        break
                if ok:
                    bound_add += n
        
        if current_sum + bound_add <= best_sum:
            return
        
        n = all_candidates[idx]
        n_primes_frozen = prime_set_list[n]
        
        # Option 1: Exclude
        bb_search4(selected_list, used_primes_frozen, current_sum, idx + 1)
        
        # Option 2: Include
        if not (n_primes_frozen & used_primes_frozen):
            ok = True
            for x in selected_list:
                if gcd_matrix[n][x] != 1:
                    ok = False
                    break
            if ok:
                new_sum = current_sum + n
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_selection = list(selected_list + [n])
                bb_search4(selected_list + [n], used_primes_frozen | n_primes_frozen, new_sum, idx + 1)
    
    # The bound is still O(n * |selected|) per node. With 256 candidates this could be slow.
    # Let me limit to top K candidates for B&B, then polish with local search.
    
    # Actually, let me just run it and see. The pruning should be effective.
    # But 2^256 is too much. Let me only branch on candidates > some threshold.
    
    # Better approach: only branch on "important" candidates. Numbers <= 2 can always be added
    # (1 is always addable, 2 is addable if no even number selected).
    
    # Let me use a different strategy: branch on primes, not numbers.
    # For each prime, decide which number (power or product) uses it.
    
    # This is getting complex. Let me just run the B&B with a time limit and 
    # combine with local search.
    
    # Run B&B on a reduced set: top 50 candidates by value
    reduced_candidates = list(range(MAX_N, MAX_N - 50, -1))  # 256 down to 207
    # Plus some mid-range numbers that are prime powers
    for n in range(100, 207):
        if len(divisors[n]) == 1:  # prime power
            reduced_candidates.append(n)
    reduced_candidates = sorted(set(reduced_candidates), reverse=True)
    
    # Map to indices for B&B
    bb_candidates = reduced_candidates
    
    def bb_search5(selected_list, used_primes_frozen, current_sum, idx):
        nonlocal best_sum, best_selection, nodes_explored
        
        if time.time() > deadline - 10:
            return
        
        nodes_explored += 1
        
        if idx >= len(bb_candidates):
            if current_sum > best_sum:
                best_sum = current_sum
                best_selection = list(selected_list)
            return
        
        # Upper bound
        bound_add = 0
        for i in range(idx, len(bb_candidates)):
            n = bb_candidates[i]
            if not (prime_set_list[n] & used_primes_frozen):
                ok = True
                for x in selected_list:
                    if gcd_matrix[n][x] != 1:
                        ok = False
                        break
                if ok:
                    bound_add += n
        
        if current_sum + bound_add <= best_sum:
            return
        
        n = bb_candidates[idx]
        n_primes_frozen = prime_set_list[n]
        
        # Exclude
        bb_search5(selected_list, used_primes_frozen, current_sum, idx + 1)
        
        # Include
        if not (n_primes_frozen & used_primes_frozen):
            ok = True
            for x in selected_list:
                if gcd_matrix[n][x] != 1:
                    ok = False
                    break
            if ok:
                new_sum = current_sum + n
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_selection = list(selected_list + [n])
                bb_search5(selected_list + [n], used_primes_frozen | n_primes_frozen, new_sum, idx + 1)
    
    # Initialize with greedy
    bb_search5([], frozenset(), 0, 0)
    
    # Now polish with local search
    best_set = set(best_selection)
    best_sum = sum(best_set)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        S_list = sorted(list(best_set))
        
        # Try to add any missing number
        for b in range(MAX_N, 1, -1):
            if b in best_set:
                continue
            if all(gcd(b, x) == 1 for x in best_set):
                best_set.add(b)
                best_sum += b
                improved = True
        
        # 1-for-1 swaps
        for a in S_list:
            if a == 1:
                continue
            for b in range(MAX_N, a, -1):
                if b in best_set:
                    continue
                if all(gcd(b, x) == 1 for x in best_set if x != a):
                    new_sum = best_sum - a + b
                    if new_sum > best_sum:
                        best_set.remove(a)
                        best_set.add(b)
                        best_sum = new_sum
                        improved = True
        
        # 2-for-1 swaps
        if time.time() < deadline - 5:
            S_list = sorted(list(best_set))
            for i in range(len(S_list)):
                if time.time() >= deadline - 5:
                    break
                a1 = S_list[i]
                if a1 == 1:
                    continue
                for j in range(i + 1, len(S_list)):
                    a2 = S_list[j]
                    if a2 == 1:
                        continue
                    for b in range(MAX_N, a1 + a2, -1):
                        if b in best_set or b == a1 or b == a2:
                            continue
                        if all(gcd(b, x) == 1 for x in best_set if x not in (a1, a2)):
                            best_set.remove(a1)
                            best_set.remove(a2)
                            best_set.add(b)
                            best_sum = best_sum - a1 - a2 + b
                            improved = True
                            break
    
    # Final polish: add 1
    if 1 not in best_set:
        best_set.add(1)
        best_sum += 1

    # Verify
    final_list = sorted(best_set)
    valid = True
    for i in range(len(final_list)):
        for j in range(i + 1, len(final_list)):
            if gcd(final_list[i], final_list[j]) != 1:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to greedy
        S = []
        for n in range(MAX_N, 0, -1):
            if all(gcd(n, x) == 1 for x in S):
                S.append(n)
        return [int(x) for x in S]
    
    return [int(x) for x in final_list]
