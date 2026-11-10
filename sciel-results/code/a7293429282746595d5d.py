import time
from math import gcd

def construct():
    start = time.time()
    deadline = start + 142.0  # safety margin under 150s kill
    N = 512

    # Precompute prime factors for each number 1..511
    prime_factors = [None] * N
    for a in range(1, N):
        n = a
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        prime_factors[a] = factors

    # Precompute conflict bitmask: conflict[a] has bit b set if gcd(a,b) != 1
    conflict = [0] * N
    for a in range(1, N):
        m = 0
        for b in range(1, N):
            if a != b and not prime_factors[a].isdisjoint(prime_factors[b]):
                m |= (1 << b)
        conflict[a] = m

    def weight(a):
        return 1.0 / (N - a)

    def score(lst):
        return sum(weight(a) for a in lst)

    def greedy(order, exclude=None):
        S = []
        chosen_mask = 0
        for a in order:
            if exclude is not None and a in exclude:
                continue
            if (chosen_mask & conflict[a]) == 0:
                S.append(a)
                chosen_mask |= (1 << a)
        return S

    # Multiple initial solutions
    best = None
    best_score = 0.0

    # Descending greedy
    S1 = greedy(range(511, 0, -1))
    s1 = score(S1)
    if s1 > best_score:
        best = S1
        best_score = s1

    # Greedy sorted by weight descending (same as descending here since weight decreases with a)
    # Try a few different orderings
    for order_fn in [
        lambda: range(511, 0, -1),
        lambda: sorted(range(1, N), key=lambda a: -weight(a)),
    ]:
        S = greedy(list(order_fn()))
        s = score(S)
        if s > best_score:
            best = S
            best_score = s

    # Local search: drop one element, re-greedy
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in list(best):
            if time.time() >= deadline:
                break
            S2 = greedy(range(511, 0, -1), exclude={x})
            s2 = score(S2)
            if s2 > best_score + 1e-12:
                best = S2
                best_score = s2
                improved = True
                break

    # Branch and bound: maximum weight independent set
    # Use the current best as initial lower bound
    # Candidates are all numbers 1..511
    # We'll use a recursive branch-and-bound with fractional knapsack upper bound

    def branch_and_bound(candidates, chosen_mask, current_score, best_score_ref):
        """
        candidates: list of candidate vertices (sorted by weight descending)
        chosen_mask: bitmask of already chosen vertices
        current_score: score of chosen set so far
        best_score_ref: list containing [best_score] for mutation
        Returns: (best_solution, best_score)
        """
        # Try to add candidates in order
        # Upper bound: current_score + sum of weights of candidates that are independent
        # Fractional knapsack: sort candidates by weight, take as many as fit
        
        if not candidates:
            return current_score

        # Quick upper bound: current_score + sum of all candidate weights
        # (loose but fast)
        total_candidate_weight = sum(weight(a) for a in candidates)
        if current_score + total_candidate_weight <= best_score_ref[0] + 1e-12:
            return current_score

        # Fractional knapsack upper bound
        # Sort candidates by weight descending (they should already be)
        # But we need to check which are actually independent
        # For a tighter bound, consider only candidates that don't conflict with chosen
        feasible = [a for a in candidates if (chosen_mask & conflict[a]) == 0]
        
        if not feasible:
            return current_score
        
        # Fractional knapsack: take weights in descending order
        # Since all "items" have value=weight and "weight"=1 (each vertex counts as 1),
        # the fractional knapsack bound is just the sum of the top k weights where k = len(feasible)
        # But we need to account for conflicts among feasible candidates too.
        # A simpler bound: sum of all feasible weights
        fb = current_score + sum(weight(a) for a in feasible)
        if fb <= best_score_ref[0] + 1e-12:
            return current_score

        # Branch on the first candidate
        # Try including it
        a = feasible[0]
        # New mask: add a
        new_mask = chosen_mask | (1 << a)
        # New candidates: remove a and all that conflict with a
        new_candidates = [x for x in feasible[1:] if (new_mask & conflict[x]) == 0]
        
        # Recurse with a included
        score_with = branch_and_bound(new_candidates, new_mask, current_score + weight(a), best_score_ref)
        
        # Try excluding it
        score_without = branch_and_bound(feasible[1:], chosen_mask, current_score, best_score_ref)
        
        return max(score_with, score_without)

    # The full B&B on 511 vertices is too large. Instead, use it on a reduced problem.
    # Strategy: identify a "core" of high-weight candidates and use B&B on those,
    # then fill in the rest greedily.

    # Alternative: use B&B with aggressive pruning on the full set but with a time limit
    # Let's try a different approach: iterative deepening B&B

    # Actually, let's use a smarter approach:
    # 1. Get a good initial solution via local search (done above)
    # 2. Use B&B to prove optimality or improve, with a time budget

    # For the B&B to be effective, we need to order vertices well.
    # Let's try B&B on the top K candidates by weight, where K is chosen to fit in time.

    # First, let's see how many elements we have
    n_best = len(best)
    
    # Use B&B to try to improve: for each element in best, try replacing it
    # with a better combination from the remaining candidates
    
    # Actually, let's try a direct B&B approach with the following strategy:
    # - Order vertices by weight descending
    # - At each node, branch on include/exclude
    # - Use fractional knapsack bound
    # - Time limit: use remaining time
    
    remaining_time = deadline - time.time()
    if remaining_time < 10:
        # Not enough time for B&B, do final polish
        pass
    else:
        # B&B approach
        # Order all candidates by weight descending
        all_candidates = list(range(511, 0, -1))  # already in weight-descending order
        
        best_score_ref = [best_score]
        best_solution_ref = [best[:]]
        
        # We need to track the actual solution, not just the score
        # Let's modify B&B to track solutions
        
        def bnb_solve(candidates, chosen_mask, chosen_list, current_score):
            """Branch and bound with solution tracking."""
            nonlocal best_score_ref, best_solution_ref
            
            if time.time() >= deadline:
                return
            
            if not candidates:
                if current_score > best_score_ref[0] + 1e-12:
                    best_score_ref[0] = current_score
                    best_solution_ref[0] = chosen_list[:]
                return
            
            # Upper bound
            feasible = [a for a in candidates if (chosen_mask & conflict[a]) == 0]
            if not feasible:
                if current_score > best_score_ref[0] + 1e-12:
                    best_score_ref[0] = current_score
                    best_solution_ref[0] = chosen_list[:]
                return
            
            # Fractional knapsack bound (sum of feasible weights)
            ub = current_score + sum(weight(a) for a in feasible)
            if ub <= best_score_ref[0] + 1e-12:
                return
            
            # Branch on first feasible candidate
            a = feasible[0]
            
            # Include a
            new_mask = chosen_mask | (1 << a)
            new_candidates = [x for x in feasible[1:] if (new_mask & conflict[x]) == 0]
            chosen_list.append(a)
            bnb_solve(new_candidates, new_mask, chosen_list, current_score + weight(a))
            chosen_list.pop()
            
            # Exclude a
            bnb_solve(feasible[1:], chosen_mask, chosen_list, current_score)
        
        # Start B&B from scratch (or from a partial solution)
        # Starting from scratch on 511 vertices will be slow, but let's try with the bound
        bnb_solve(all_candidates, 0, [], 0.0)
        
        if best_score_ref[0] > best_score + 1e-12:
            best = best_solution_ref[0]
            best_score = best_score_ref[0]

    # Final polish: local search on the B&B result
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in list(best):
            if time.time() >= deadline:
                break
            S2 = greedy(range(511, 0, -1), exclude={x})
            s2 = score(S2)
            if s2 > best_score + 1e-12:
                best = S2
                best_score = s2
                improved = True
                break

    # Try adding any remaining coprime elements
    best_set = set(best)
    chosen_mask = 0
    for a in best:
        chosen_mask |= (1 << a)
    for a in range(511, 0, -1):
        if a not in best_set and (chosen_mask & conflict[a]) == 0:
            best.append(a)
            best_set.add(a)
            chosen_mask |= (1 << a)
            best_score += weight(a)

    return best
