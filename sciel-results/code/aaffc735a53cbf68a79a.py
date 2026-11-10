"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from fractions import Fraction

def _sieve_primes(limit):
    """Return list of primes <= limit."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def _factorize(n, primes):
    """Return list of prime factors of n (with multiplicity not needed, distinct primes)."""
    factors = []
    temp = n
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            factors.append(p)
            while temp % p == 0:
                temp //= p
    if temp > 1:
        factors.append(temp)
    return factors

def _score_float(chosen):
    """Compute sum 1/(512 - a) for chosen integers as float."""
    return sum(1.0 / (512 - a) for a in chosen)

def _score_frac(chosen):
    """Compute sum 1/(512 - a) for chosen integers as exact Fraction."""
    total = Fraction(0)
    for a in chosen:
        total += Fraction(1, 512 - a)
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin within 150s limit
    
    N = 512
    primes = _sieve_primes(N - 1)
    
    # Precompute prime factors for each number 1..511
    factor_cache = {}
    for a in range(1, N):
        if a == 1:
            factor_cache[a] = []
        else:
            factor_cache[a] = _factorize(a, primes)
    
    # Precompute weight as Fraction
    weight_frac = {}
    weight_float = {}
    for a in range(1, N):
        weight_frac[a] = Fraction(1, N - a)
        weight_float[a] = 1.0 / (N - a)
    
    # Greedy solution (high to low by weight)
    def greedy_solution():
        chosen = []
        chosen_factors = set()
        for a in range(N - 1, 0, -1):
            fa = factor_cache[a]
            conflict = any(p in chosen_factors for p in fa)
            if not conflict:
                chosen.append(a)
                chosen_factors.update(fa)
        return chosen
    
    best_chosen = greedy_solution()
    best_score_float = _score_float(best_chosen)
    best_score_frac = _score_frac(best_chosen)
    
    # Local search with prime-factor conflict detection
    best_set = set(best_chosen)
    best_factors = set()
    for a in best_set:
        best_factors.update(factor_cache[a])
    
    def local_search_improve(best_set, best_factors, best_score_float, best_score_frac, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding a new element
            for a in range(N - 1, 0, -1):
                if a in best_set:
                    continue
                fa = factor_cache[a]
                if any(p in best_factors for p in fa):
                    continue
                new_score = best_score_float + weight_float[a]
                if new_score > best_score_float + 1e-12:
                    best_set.add(a)
                    best_factors.update(fa)
                    best_score_float = new_score
                    best_score_frac = best_score_frac + weight_frac[a]
                    improved = True
                    break
            
            if time.time() >= deadline:
                break
            
            # Try 1-for-1 swaps
            for a_remove in sorted(best_set, key=lambda x: weight_float[x]):
                if a_remove == 1:
                    continue
                score_without = best_score_float - weight_float[a_remove]
                factors_without = set(best_factors)
                for p in factor_cache[a_remove]:
                    used_by_other = False
                    for b in best_set:
                        if b != a_remove and p in factor_cache[b]:
                            used_by_other = True
                            break
                    if not used_by_other:
                        factors_without.discard(p)
                
                for a_add in range(N - 1, 0, -1):
                    if a_add in best_set:
                        continue
                    fa = factor_cache[a_add]
                    if any(p in factors_without for p in fa):
                        continue
                    new_score = score_without + weight_float[a_add]
                    if new_score > best_score_float + 1e-12:
                        best_set.remove(a_remove)
                        best_set.add(a_add)
                        best_factors = factors_without
                        best_factors.update(fa)
                        best_score_float = new_score
                        best_score_frac = best_score_frac - weight_frac[a_remove] + weight_frac[a_add]
                        improved = True
                        break
                if improved:
                    break
        
        return best_set, best_factors, best_score_float, best_score_frac
    
    best_set, best_factors, best_score_float, best_score_frac = local_search_improve(
        best_set, best_factors, best_score_float, best_score_frac, deadline
    )
    
    # Branch and bound with clique-based upper bounds
    # Items sorted by weight descending
    items = list(range(N - 1, 0, -1))
    
    # Precompute conflict mask: for each item, which other items conflict
    # Use bitsets for efficiency
    # Map each prime factor to a set of items that have it
    prime_to_items = {}
    for a in range(1, N):
        for p in factor_cache[a]:
            if p not in prime_to_items:
                prime_to_items[p] = set()
            prime_to_items[p].add(a)
    
    # For each item, compute the set of conflicting items (as a set of item indices)
    # We'll use a dictionary: item -> set of conflicting items
    conflict_map = {}
    for a in range(1, N):
        conflicts = set()
        for p in factor_cache[a]:
            for b in prime_to_items[p]:
                if b != a:
                    conflicts.add(b)
        conflict_map[a] = conflicts
    
    # Branch and bound
    # State: current set of chosen items, current score, index of next item to consider
    # Upper bound: current_score + sum of weights of remaining non-conflicting items
    
    # For tighter bounds, use a greedy coloring / clique upper bound
    # The conflict graph is a union of cliques (one per prime factor)
    # Upper bound: for each prime factor, at most one item can be chosen from items having that factor
    
    # Simpler upper bound: sum of weights of all remaining items that don't conflict with chosen
    # This is an overestimate but fast to compute
    
    best_bb_score = best_score_float
    best_bb_set = set(best_set)
    
    # Use a priority queue or recursive DFS
    # Given time constraints, do a limited B&B
    
    def bb_upper_bound(chosen_factors, start_idx, remaining_weights):
        """
        Upper bound: sum of weights of items from start_idx onward that don't conflict with chosen_factors.
        This is optimistic since we ignore conflicts among remaining items.
        """
        ub = 0.0
        for i in range(start_idx, len(items)):
            a = items[i]
            if any(p in chosen_factors for p in factor_cache[a]):
                continue
            ub += weight_float[a]
        return ub
    
    # Recursive B&B with time check
    # To avoid deep recursion, limit depth and use iterative approach
    
    # Actually, with 511 items, full B&B is infeasible.
    # Instead, do a focused B&B on a subset of high-weight items.
    
    # Focus on top K items by weight where K is manageable
    K = 60  # Consider top 60 items for B&B
    top_items = items[:K]
    
    # For B&B, we need to consider which of these top items to include
    # The rest (lower weight) can be greedily added if they don't conflict
    
    def greedy_fill(chosen_factors, chosen_set):
        """Greedily add remaining items that don't conflict."""
        added = []
        for a in items:
            if a in chosen_set:
                continue
            if any(p in chosen_factors for p in factor_cache[a]):
                continue
            chosen_set.add(a)
            chosen_factors.update(factor_cache[a])
            added.append(a)
        return added
    
    # B&B over top K items
    # State: index into top_items, chosen set, chosen factors, current score
    # We use DFS with pruning
    
    bb_deadline = deadline - 5.0  # Leave 5s for final local search
    
    best_bb_result = (set(best_set), best_score_float)
    
    # Precompute for each top item, the set of other top items it conflicts with
    top_conflict = {}
    for a in top_items:
        top_conflict[a] = set(conflict_map[a]) & set(top_items)
    
    def bb_dfs(idx, chosen_set, chosen_factors, current_score):
        nonlocal best_bb_result, best_bb_score
        
        if time.time() > bb_deadline:
            return
        
        # Upper bound: current score + sum of weights of remaining top items not conflicting
        ub = current_score
        for j in range(idx, len(top_items)):
            a = top_items[j]
            if any(p in chosen_factors for p in factor_cache[a]):
                continue
            ub += weight_float[a]
        
        # Also add optimistic bound from non-top items
        # (skip for speed; the top items dominate)
        
        if ub <= best_bb_score + 1e-12:
            return
        
        if idx == len(top_items):
            # Greedily fill with non-top items
            temp_set = set(chosen_set)
            temp_factors = set(chosen_factors)
            fill_score = current_score
            for a in items:
                if a in temp_set:
                    continue
                if any(p in temp_factors for p in factor_cache[a]):
                    continue
                temp_set.add(a)
                temp_factors.update(factor_cache[a])
                fill_score += weight_float[a]
            
            if fill_score > best_bb_score + 1e-12:
                best_bb_score = fill_score
                best_bb_result = (temp_set, fill_score)
            return
        
        a = top_items[idx]
        
        # Option 1: include a (if no conflict)
        if not any(p in chosen_factors for p in factor_cache[a]):
            new_set = chosen_set | {a}
            new_factors = chosen_factors | set(factor_cache[a])
            bb_dfs(idx + 1, new_set, new_factors, current_score + weight_float[a])
        
        # Option 2: exclude a
        bb_dfs(idx + 1, chosen_set, chosen_factors, current_score)
    
    # Start B&B from empty set over top items
    bb_dfs(0, set(), set(), 0.0)
    
    if best_bb_score > best_score_float + 1e-12:
        best_set = best_bb_result[0]
        best_score_float = best_bb_result[1]
        best_factors = set()
        for a in best_set:
            best_factors.update(factor_cache[a])
        best_score_frac = _score_frac(best_set)
    
    # Final local search polish
    if time.time() < deadline - 2.0:
        best_set, best_factors, best_score_float, best_score_frac = local_search_improve(
            best_set, best_factors, best_score_float, best_score_frac, deadline
        )
    
    return list(best_set)
# EVOLVE-BLOCK-END
