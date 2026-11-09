"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110

    N = 512

    def get_primes(n):
        sieve = [True] * n
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    primes = get_primes(N)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    def get_factor_mask(n):
        mask = 0
        tmp = n
        for i, p in enumerate(primes):
            if p * p > tmp:
                break
            if tmp % p == 0:
                mask |= (1 << i)
                while tmp % p == 0:
                    tmp //= p
        if tmp > 1:
            mask |= (1 << prime_to_idx[tmp])
        return mask

    factor_masks = [0] * N
    for a in range(1, N):
        factor_masks[a] = get_factor_mask(a)

    scores = [0.0] * N
    for a in range(1, N):
        scores[a] = 1.0 / (N - a)

    def npf(a):
        return bin(factor_masks[a]).count('1')

    def compute_score(chosen):
        return sum(scores[a] for a in chosen)

    def greedy_construct(ordering):
        chosen = []
        chosen_set = set()
        chosen_mask = 0
        for a in ordering:
            if factor_masks[a] & chosen_mask == 0:
                chosen.append(a)
                chosen_set.add(a)
                chosen_mask |= factor_masks[a]
        return chosen, chosen_set, chosen_mask

    best_chosen = None
    best_score = -1.0
    best_chosen_set = None
    best_chosen_mask = 0

    def update_best(chosen, chosen_set, chosen_mask, score):
        nonlocal best_chosen, best_score, best_chosen_set, best_chosen_mask
        if score > best_score:
            best_score = score
            best_chosen = chosen
            best_chosen_set = chosen_set
            best_chosen_mask = chosen_mask

    # Strategy 1: Pure score descending
    ordering1 = list(range(511, 0, -1))
    c1, s1, m1 = greedy_construct(ordering1)
    sc1 = compute_score(c1)
    update_best(c1, s1, m1, sc1)

    # Strategy 2: Score / num_prime_factors
    ordering2 = sorted(range(1, 512), key=lambda x: scores[x] / max(1, npf(x)), reverse=True)
    c2, s2, m2 = greedy_construct(ordering2)
    sc2 = compute_score(c2)
    update_best(c2, s2, m2, sc2)

    # Strategy 3: More randomized greedy seeds
    for trial in range(50):
        if time.time() > deadline:
            break
        ordering = list(range(1, 512))
        random.shuffle(ordering)
        weighted = [(random.random() * 0.3 + scores[x] / max(1, npf(x)), x) for x in ordering]
        weighted.sort(reverse=True)
        ordering_w = [x for _, x in weighted]
        cw, sw, mw = greedy_construct(ordering_w)
        scw = compute_score(cw)
        update_best(cw, sw, mw, scw)

    # Best-improvement local search with multiple move types
    def best_improvement_ls(chosen, chosen_set, chosen_mask, current_score, max_rounds=30):
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            best_gain = 0.0
            best_move = None  # (move_type, args)
            
            # Try 1-add: add one element if possible
            for b in range(511, 0, -1):
                if b in chosen_set:
                    continue
                if factor_masks[b] & chosen_mask:
                    continue
                gain = scores[b]
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_move = ('add', (b,))
            
            # Try 2-add: add two elements if possible
            if len(chosen) < 511:
                for b1 in range(511, 0, -1):
                    if b1 in chosen_set:
                        continue
                    if factor_masks[b1] & chosen_mask:
                        continue
                    temp_mask_1 = chosen_mask | factor_masks[b1]
                    for b2 in range(b1 - 1, 0, -1):
                        if b2 in chosen_set:
                            continue
                        if factor_masks[b2] & temp_mask_1:
                            continue
                        gain = scores[b1] + scores[b2]
                        if gain > best_gain + 1e-12:
                            best_gain = gain
                            best_move = ('2add', (b1, b2))
            
            # Try 1-swap: remove one, add one
            for a in chosen:
                temp_mask = chosen_mask & ~factor_masks[a]
                for b in range(511, 0, -1):
                    if b in chosen_set:
                        continue
                    if factor_masks[b] & temp_mask:
                        continue
                    gain = scores[b] - scores[a]
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_move = ('swap', (a, b))
            
            # Try 1-remove: remove one element (only if it frees up space for better additions)
            for a in chosen:
                temp_mask = chosen_mask & ~factor_masks[a]
                # Check if removing a allows adding at least one better element
                for b in range(511, 0, -1):
                    if b in chosen_set:
                        continue
                    if factor_masks[b] & temp_mask:
                        continue
                    # If we can add b after removing a, the net gain is scores[b] - scores[a]
                    # This is already covered by swap, but let's also consider just removing
                    # if it enables a 2-add that wasn't possible before
                    break
            
            # Try 2-swap: remove two, add two
            n = len(chosen)
            if n >= 2:
                for i in range(n):
                    if time.time() > deadline:
                        break
                    a1 = chosen[i]
                    for j in range(i + 1, n):
                        a2 = chosen[j]
                        temp_mask = chosen_mask & ~factor_masks[a1] & ~factor_masks[a2]
                        valid = []
                        for b in range(511, 0, -1):
                            if b in chosen_set:
                                continue
                            if factor_masks[b] & temp_mask:
                                continue
                            valid.append(b)
                        for idx1 in range(len(valid)):
                            b1 = valid[idx1]
                            for idx2 in range(idx1 + 1, len(valid)):
                                b2 = valid[idx2]
                                if factor_masks[b1] & factor_masks[b2]:
                                    continue
                                gain = scores[b1] + scores[b2] - scores[a1] - scores[a2]
                                if gain > best_gain + 1e-12:
                                    best_gain = gain
                                    best_move = ('2swap', (a1, a2, b1, b2))
            
            if best_move is None:
                break
            
            mt = best_move[0]
            args = best_move[1]
            
            if mt == 'add':
                b = args[0]
                chosen.append(b)
                chosen_set.add(b)
                chosen_mask |= factor_masks[b]
                current_score += best_gain
            elif mt == '2add':
                b1, b2 = args
                chosen.append(b1)
                chosen.append(b2)
                chosen_set.add(b1)
                chosen_set.add(b2)
                chosen_mask |= factor_masks[b1] | factor_masks[b2]
                current_score += best_gain
            elif mt == 'swap':
                a, b = args
                chosen.remove(a)
                chosen.append(b)
                chosen_set.remove(a)
                chosen_set.add(b)
                chosen_mask = (chosen_mask & ~factor_masks[a]) | factor_masks[b]
                current_score += best_gain
            elif mt == '2swap':
                a1, a2, b1, b2 = args
                chosen.remove(a1)
                chosen.remove(a2)
                chosen.append(b1)
                chosen.append(b2)
                chosen_set.remove(a1)
                chosen_set.remove(a2)
                chosen_set.add(b1)
                chosen_set.add(b2)
                chosen_mask = (chosen_mask & ~factor_masks[a1] & ~factor_masks[a2]) | factor_masks[b1] | factor_masks[b2]
                current_score += best_gain
        
        return chosen, chosen_set, chosen_mask, current_score

    # Run best-improvement local search from the best greedy solution
    best_chosen, best_chosen_set, best_chosen_mask, best_score = best_improvement_ls(
        best_chosen, best_chosen_set, best_chosen_mask, best_score, max_rounds=50
    )

    # Simulated annealing for additional improvement
    def simulated_annealing(chosen, chosen_set, chosen_mask, current_score):
        temperature = 0.008
        cooling_rate = 0.999
        min_temperature = 0.000005

        while temperature > min_temperature and time.time() < deadline:
            move_type = random.random()
            if move_type < 0.5 and len(chosen) > 0:
                a = random.choice(chosen)
                temp_mask = chosen_mask & ~factor_masks[a]
                candidates = [b for b in range(1, 512) if b not in chosen_set and (factor_masks[b] & temp_mask) == 0]
                if not candidates:
                    temperature *= cooling_rate
                    continue
                b = random.choice(candidates)
                delta = scores[b] - scores[a]
                if delta > 0 or random.random() < math.exp(delta / temperature):
                    chosen.remove(a)
                    chosen.append(b)
                    chosen_set.remove(a)
                    chosen_set.add(b)
                    chosen_mask = (chosen_mask & ~factor_masks[a]) | factor_masks[b]
                    current_score += delta
            elif move_type < 0.8 and len(chosen) >= 2:
                a1, a2 = random.sample(chosen, 2)
                temp_mask = chosen_mask & ~factor_masks[a1] & ~factor_masks[a2]
                candidates = [b for b in range(1, 512) if b not in chosen_set and (factor_masks[b] & temp_mask) == 0]
                if len(candidates) < 2:
                    temperature *= cooling_rate
                    continue
                b1, b2 = random.sample(candidates, 2)
                if (factor_masks[b1] & factor_masks[b2]) != 0:
                    temperature *= cooling_rate
                    continue
                delta = scores[b1] + scores[b2] - scores[a1] - scores[a2]
                if delta > 0 or random.random() < math.exp(delta / temperature):
                    chosen.remove(a1)
                    chosen.remove(a2)
                    chosen.append(b1)
                    chosen.append(b2)
                    chosen_set.remove(a1)
                    chosen_set.remove(a2)
                    chosen_set.add(b1)
                    chosen_set.add(b2)
                    chosen_mask = (chosen_mask & ~factor_masks[a1] & ~factor_masks[a2]) | factor_masks[b1] | factor_masks[b2]
                    current_score += delta
            elif move_type < 0.9:
                # Try adding an element
                candidates = [b for b in range(1, 512) if b not in chosen_set and (factor_masks[b] & chosen_mask) == 0]
                if not candidates:
                    temperature *= cooling_rate
                    continue
                b = random.choice(candidates)
                delta = scores[b]
                if delta > 0 or random.random() < math.exp(delta / temperature):
                    chosen.append(b)
                    chosen_set.add(b)
                    chosen_mask |= factor_masks[b]
                    current_score += delta

            temperature *= cooling_rate

        return chosen, chosen_set, chosen_mask, current_score

    best_chosen, best_chosen_set, best_chosen_mask, best_score = simulated_annealing(
        best_chosen, best_chosen_set, best_chosen_mask, best_score
    )

    # Final best-improvement local search pass with tighter convergence
    best_chosen, best_chosen_set, best_chosen_mask, best_score = best_improvement_ls(
        best_chosen, best_chosen_set, best_chosen_mask, best_score, max_rounds=80
    )

    # Exact polish: try all possible single additions and swaps one more time
    # with a tighter tolerance to ensure we're at a local optimum
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all single adds
        for b in range(511, 0, -1):
            if b in best_chosen_set:
                continue
            if factor_masks[b] & best_chosen_mask:
                continue
            best_chosen.append(b)
            best_chosen_set.add(b)
            best_chosen_mask |= factor_masks[b]
            best_score += scores[b]
            improved = True
        # Try all single swaps
        for a in list(best_chosen):
            temp_mask = best_chosen_mask & ~factor_masks[a]
            for b in range(511, 0, -1):
                if b in best_chosen_set:
                    continue
                if factor_masks[b] & temp_mask:
                    continue
                gain = scores[b] - scores[a]
                if gain > 1e-15:
                    best_chosen.remove(a)
                    best_chosen.append(b)
                    best_chosen_set.remove(a)
                    best_chosen_set.add(b)
                    best_chosen_mask = (best_chosen_mask & ~factor_masks[a]) | factor_masks[b]
                    best_score += gain
                    improved = True
                    break

    # Branch-and-bound exact solver using bitmask conflict structure
    # This models the problem as Maximum Weight Independent Set on a conflict graph
    # where nodes are candidates 1..511 and edges connect non-coprime pairs.
    # We use pivot-based branch-and-bound with a fractional knapsack upper bound.
    
    if time.time() < deadline - 5:  # Leave 5s safety margin
        # Precompute: for each candidate, the set of candidates it conflicts with
        # Represented as a bitmask over the candidate indices
        # We'll work with candidates sorted by score descending for tighter bounds
        
        # Build list of (score, candidate, mask) sorted by score descending
        candidates = [(scores[a], a, factor_masks[a]) for a in range(1, 512)]
        candidates.sort(key=lambda x: -x[0])
        
        # Re-index: position in sorted list -> candidate
        # For B&B, we need to work with the sorted order
        n_cand = len(candidates)
        cand_scores = [c[0] for c in candidates]
        cand_masks = [c[2] for c in candidates]
        
        # Precompute suffix sums of scores for upper bound
        suffix_sum = [0.0] * (n_cand + 1)
        for i in range(n_cand - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + cand_scores[i]
        
        # Precompute conflict masks in the new indexing
        # conflict_mask[i] = bitmask of all j > i that conflict with candidate i
        # Actually, we need: for a given set of chosen (as bitmask over positions),
        # which candidates are still available?
        
        # For the B&B, we maintain:
        # - pos: current position in the sorted candidate list
        # - chosen_mask: bitmask of chosen positions (in the sorted order)
        # - conflict_mask: bitmask of positions that conflict with at least one chosen
        
        # Upper bound: sum of scores of all unchosen, non-conflicting candidates from pos onward
        # Plus the current score
        
        # Pivot selection: choose a pivot among available candidates to branch on
        # Use a simple heuristic: the candidate with highest score among available
        
        best_bb_score = best_score
        best_bb_solution = list(best_chosen)
        
        # Time check helper
        def time_ok():
            return time.time() < deadline - 3
        
        # Recursive B&B with iterative deepening on branching
        # We use a stack-based approach to avoid recursion depth issues
        
        # For each starting position, we try to build a solution
        # Branching: at each step, pick an available candidate and either include or exclude it
        
        # To make this tractable, we use a greedy lower bound and a simple upper bound
        # and only branch when the upper bound exceeds the current best
        
        # Let's implement a simpler but effective B&B:
        # Start from the current best solution, then try to improve via B&B
        # on the remaining candidates
        
        # Actually, let's do a full B&B from scratch but with aggressive pruning
        # using the current best as initial lower bound
        
        # State: (pos, chosen_mask, conflict_mask, current_score)
        # We process candidates in order, at each step deciding include/exclude
        
        # But with 511 candidates, this is too large for full enumeration.
        # Instead, we use a pivot-based approach:
        
        # At each node, select a pivot candidate p from the available set.
        # Branch 1: include p, then recurse on candidates that don't conflict with p
        # Branch 2: exclude p, then recurse on remaining candidates
        
        # Upper bound at a node: current_score + sum of scores of all available candidates
        # (this is a loose bound, but with good ordering it can be tight)
        
        # Let's use a tighter bound: fractional knapsack style
        # Sort available candidates by score, take them in order until we hit a conflict
        
        def bb_search(pos, chosen_mask, conflict_mask, current_score, depth=0):
            nonlocal best_bb_score, best_bb_solution
            if not time_ok():
                return
            if pos >= n_cand:
                return
            
            # Find the next available candidate (not chosen, not conflicting)
            # and compute upper bound
            # Available candidates are those with index >= pos, not in chosen_mask, not in conflict_mask
            
            # Quick upper bound: current_score + suffix_sum[pos]
            # But this includes conflicting candidates, so it's an overestimate
            # For a tighter bound, we need to account for conflicts
            
            # Simple approach: greedily select from available candidates to get a lower bound
            # and use that to prune
            
            # Find first available candidate
            first_avail = -1
            for i in range(pos, n_cand):
                if not (chosen_mask >> i) & 1 and not (conflict_mask >> i) & 1:
                    first_avail = i
                    break
            
            if first_avail == -1:
                if current_score > best_bb_score:
                    best_bb_score = current_score
                    # Reconstruct solution
                    sol = []
                    for i in range(n_cand):
                        if (chosen_mask >> i) & 1:
                            sol.append(candidates[i][1])
                    best_bb_solution = sol
                return
            
            # Upper bound: current_score + sum of scores of all candidates from first_avail onward
            # that are not in conflict_mask
            ub = current_score
            for i in range(first_avail, n_cand):
                if not (conflict_mask >> i) & 1:
                    ub += cand_scores[i]
            
            if ub <= best_bb_score + 1e-12:
                return
            
            # Pivot: select the candidate with highest score among available
            # This is first_avail since we're in score-descending order
            p = first_avail
            
            # Branch 1: include p
            new_chosen = chosen_mask | (1 << p)
            new_conflict = conflict_mask | cand_masks[p]
            bb_search(p + 1, new_chosen, new_conflict, current_score + cand_scores[p], depth + 1)
            
            # Branch 2: exclude p
            bb_search(p + 1, chosen_mask, conflict_mask, current_score, depth + 1)
        
        # Start B&B from scratch
        bb_search(0, 0, 0, 0.0)
        
        if best_bb_score > best_score + 1e-12:
            best_score = best_bb_score
            best_chosen = best_bb_solution
            best_chosen_set = set(best_chosen)
            best_chosen_mask = 0
            for a in best_chosen:
                best_chosen_mask |= factor_masks[a]

    return best_chosen
# EVOLVE-BLOCK-END
