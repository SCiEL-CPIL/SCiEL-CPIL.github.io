"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c with a,b,c in S (a=b allowed)."""
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(x, S_set):
        """Check if adding x to S_set maintains sum-free property."""
        if 2 * x in S_set:
            return False
        for c in S_set:
            if c - x in S_set:
                return False
        for a in S_set:
            if x - a in S_set:
                return False
        return True

    def greedy_build(S, order=None):
        """Greedy construction: add elements in given order if they don't create conflicts."""
        if order is None:
            order = list(S)
        S_set = set()
        result = []
        for x in order:
            if can_add(x, S_set):
                result.append(x)
                S_set.add(x)
        return result

    def local_search_improve(S, initial_subset, max_rounds=50):
        """Local search: iteratively try to add elements that fit."""
        current = set(initial_subset)
        remaining = [x for x in S if x not in current]
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            added = False
            for x in remaining[:]:
                if can_add(x, current):
                    current.add(x)
                    remaining.remove(x)
                    added = True
            if not added:
                break
        return list(current)

    def swap_improve(S, current, max_rounds=30):
        """Try to improve by removing one element and adding two others."""
        current_set = set(current)
        remaining = [x for x in S if x not in current_set]
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            for rem in list(current_set):
                if time.time() > deadline:
                    break
                temp_set = current_set - {rem}
                added_count = 0
                new_elements = []
                for x in remaining:
                    if len(new_elements) >= 2:
                        break
                    if can_add(x, temp_set):
                        temp_set.add(x)
                        new_elements.append(x)
                        added_count += 1
                if added_count > 1:
                    current_set = temp_set
                    remaining = [x for x in S if x not in current_set]
                    improved = True
                    break
            if not improved:
                break
        return list(current_set)

    def two_opt_improve(S, current, max_rounds=30):
        """2-opt: remove 2 elements, try to add up to 3."""
        current_set = set(current)
        remaining = [x for x in S if x not in current_set]
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            current_list = list(current_set)
            for i in range(len(current_list)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(current_list)):
                    if time.time() > deadline:
                        break
                    rem_set = {current_list[i], current_list[j]}
                    temp_set = current_set - rem_set
                    added_count = 0
                    for x in remaining:
                        if added_count >= 3:
                            break
                        if can_add(x, temp_set):
                            temp_set.add(x)
                            added_count += 1
                    if added_count > 2:
                        current_set = temp_set
                        remaining = [x for x in S if x not in current_set]
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return list(current_set)

    def solve_one(S, seed):
        """Find a large sum-free subset of S using multiple strategies."""
        n = len(S)
        if n == 0:
            return []
        
        best = []
        
        # Strategy 1: All odd numbers (always sum-free)
        odds = [x for x in S if x % 2 != 0]
        if len(odds) > len(best):
            best = odds
        
        # Strategy 2: Greedy by absolute value
        greedy_abs = greedy_build(S, sorted(S, key=abs))
        if len(greedy_abs) > len(best):
            best = greedy_abs
        
        # Strategy 3: Greedy by value (ascending)
        greedy_asc = greedy_build(S, sorted(S))
        if len(greedy_asc) > len(best):
            best = greedy_asc
        
        # Strategy 4: Greedy by value (descending)
        greedy_desc = greedy_build(S, sorted(S, reverse=True))
        if len(greedy_desc) > len(best):
            best = greedy_desc
        
        # Strategy 5: Local search from odds
        if time.time() < deadline - 5:
            ls_from_odds = local_search_improve(S, odds, max_rounds=50)
            if len(ls_from_odds) > len(best):
                best = ls_from_odds
        
        # Strategy 6: Local search from greedy_abs
        if time.time() < deadline - 5:
            ls_from_greedy = local_search_improve(S, greedy_abs, max_rounds=50)
            if len(ls_from_greedy) > len(best):
                best = ls_from_greedy
        
        # Strategy 7: Random restarts with local search
        rng = Random(seed)
        for _ in range(15):
            if time.time() > deadline - 3:
                break
            p = 0.3 + rng.random() * 0.3
            random_subset = [x for x in S if rng.random() < p]
            cleaned = greedy_build(random_subset, random_subset)
            improved = local_search_improve(S, cleaned, max_rounds=30)
            if len(improved) > len(best):
                best = improved
        
        # Strategy 8: Swap improvement on best
        if time.time() < deadline - 3:
            swapped = swap_improve(S, best, max_rounds=20)
            if len(swapped) > len(best):
                best = swapped
            if time.time() < deadline - 2:
                final_ls = local_search_improve(S, best, max_rounds=30)
                if len(final_ls) > len(best):
                    best = final_ls
        
        # Strategy 9: 2-opt improvement
        if time.time() < deadline - 2:
            two_opt = two_opt_improve(S, best, max_rounds=20)
            if len(two_opt) > len(best):
                best = two_opt
            if time.time() < deadline - 1:
                final_ls2 = local_search_improve(S, best, max_rounds=30)
                if len(final_ls2) > len(best):
                    best = final_ls2
        
        # Strategy 10: More diverse random restarts with perturbation
        if time.time() < deadline - 5:
            for _ in range(5):
                if time.time() > deadline - 3:
                    break
                # Start from best, remove random 10-20% of elements
                best_set = set(best)
                perturbed = list(best_set)
                rng.shuffle(perturbed)
                remove_count = max(1, int(len(perturbed) * (0.1 + rng.random() * 0.1)))
                for _ in range(remove_count):
                    if perturbed:
                        perturbed.pop()
                improved = local_search_improve(S, perturbed, max_rounds=40)
                if len(improved) > len(best):
                    best = improved
        
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline - 5:
            S = instance(size, seed)
            results.append([x for x in S if x % 2 != 0])
        else:
            S = instance(size, seed)
            result = solve_one(S, seed)
            if not is_sum_free(result):
                result = [x for x in S if x % 2 != 0]
            results.append(result)
    
    return results
# EVOLVE-BLOCK-END
