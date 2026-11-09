import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    n = 192
    max_size = 48
    
    def is_valid(S):
        """
        Check if S is valid.
        Optimized: For each a, check if any multiple of a in the sum range
        can be formed by 3 distinct other elements.
        """
        S_list = list(S)
        k = len(S_list)
        if k < 4:
            return True
        
        # Precompute min and max sum of 3 distinct elements in S
        S_sorted = sorted(S_list)
        min_sum = S_sorted[0] + S_sorted[1] + S_sorted[2]
        max_sum = S_sorted[-1] + S_sorted[-2] + S_sorted[-3]
        
        # For each a in S
        for a in S_list:
            # Find multiples of a in [min_sum, max_sum]
            start_m = (min_sum + a - 1) // a
            end_m = max_sum // a
            
            for m in range(start_m, end_m + 1):
                target = m * a
                # Check if target can be formed by 3 distinct elements from S \ {a}
                others = [x for x in S_list if x != a]
                others_set = set(others)
                
                found = False
                for i in range(len(others)):
                    b = others[i]
                    for j in range(i + 1, len(others)):
                        c = others[j]
                        d = target - b - c
                        if d <= c:
                            continue
                        if d in others_set:
                            found = True
                            break
                    if found:
                        break
                
                if found:
                    return False
        
        return True

    def count_violations_per_element(S):
        """
        For each element, count how many violations it's involved in.
        Returns dict: element -> count
        """
        S_list = list(S)
        counts = {x: 0 for x in S_list}
        if len(S_list) < 4:
            return counts
        
        S_sorted = sorted(S_list)
        min_sum = S_sorted[0] + S_sorted[1] + S_sorted[2]
        max_sum = S_sorted[-1] + S_sorted[-2] + S_sorted[-3]
        
        for a in S_list:
            start_m = (min_sum + a - 1) // a
            end_m = max_sum // a
            
            for m in range(start_m, end_m + 1):
                target = m * a
                others = [x for x in S_list if x != a]
                others_set = set(others)
                
                # Find all triplets (b,c,d) from others that sum to target
                triplets_found = []
                for i in range(len(others)):
                    b = others[i]
                    for j in range(i + 1, len(others)):
                        c = others[j]
                        d = target - b - c
                        if d <= c:
                            continue
                        if d in others_set:
                            triplets_found.append((b, c, d))
                
                # Each triplet contributes to violations for a and for each element in triplet
                for b, c, d in triplets_found:
                    counts[a] += 1
                    counts[b] += 1
                    counts[c] += 1
                    counts[d] += 1
        
        return counts

    def get_violation_info(S):
        """
        Returns (a, b, c, d, m) for the first violation found, or None.
        """
        S_list = list(S)
        k = len(S_list)
        if k < 4:
            return None
        
        S_sorted = sorted(S_list)
        min_sum = S_sorted[0] + S_sorted[1] + S_sorted[2]
        max_sum = S_sorted[-1] + S_sorted[-2] + S_sorted[-3]
        
        for a in S_list:
            start_m = (min_sum + a - 1) // a
            end_m = max_sum // a
            
            for m in range(start_m, end_m + 1):
                target = m * a
                others = [x for x in S_list if x != a]
                others_set = set(others)
                
                for i in range(len(others)):
                    b = others[i]
                    for j in range(i + 1, len(others)):
                        c = others[j]
                        d = target - b - c
                        if d <= c:
                            continue
                        if d in others_set:
                            return (a, b, c, d, m)
        return None

    def min_conflicts_search(initial_S, max_iters=5000):
        """
        Randomized min-conflicts local search.
        Starts with initial_S and iteratively swaps elements to reduce violations.
        Returns the best valid set found, or None if no valid set found.
        """
        S = set(initial_S)
        best_S = set()
        best_violations = float('inf')
        
        # Compute initial violation count
        if is_valid(S):
            return S
        
        counts = count_violations_per_element(S)
        current_violations = sum(counts.values())
        
        if current_violations == 0:
            return S
        
        best_violations = current_violations
        best_S = set(S)
        
        all_numbers = set(range(1, n + 1))
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            # Find element with most violations
            if not counts:
                break
                
            worst_element = max(counts.keys(), key=lambda x: counts[x])
            
            # Randomly choose a replacement from numbers not in S
            candidates = [x for x in all_numbers if x not in S]
            if not candidates:
                break
            
            # Weighted random selection: prefer larger numbers (they're "safer")
            # and numbers that are far from the worst_element
            weights = []
            for x in candidates:
                # Larger numbers are safer (fewer multiples in range)
                weight = x / 100.0
                # Prefer numbers that are different from worst_element
                if x != worst_element:
                    weight *= 1.5
                weights.append(weight)
            
            total_weight = sum(weights)
            r = random.random() * total_weight
            cumulative = 0
            chosen = candidates[0]
            for x, w in zip(candidates, weights):
                cumulative += w
                if r <= cumulative:
                    chosen = x
                    break
            
            # Perform the swap
            S.remove(worst_element)
            S.add(chosen)
            
            # Check if valid
            if is_valid(S):
                return S
            
            # Update violation counts
            new_counts = count_violations_per_element(S)
            new_violations = sum(new_counts.values())
            
            # Accept if better or with some probability (simulated annealing style)
            if new_violations < best_violations:
                best_violations = new_violations
                best_S = set(S)
            
            # Always accept if it's an improvement, or with small probability
            if new_violations <= current_violations:
                current_violations = new_violations
                counts = new_counts
            else:
                # Revert if worse, but with small probability accept to escape local min
                if random.random() < 0.1:
                    current_violations = new_violations
                    counts = new_counts
                else:
                    S.remove(chosen)
                    S.add(worst_element)
                    # Keep old counts
                    pass
        
        # Return best found
        if is_valid(best_S):
            return best_S
        return None

    def greedy_large_first():
        """
        Greedy construction: add numbers from largest to smallest if valid.
        """
        S = set()
        for x in range(n, 0, -1):
            if len(S) >= max_size:
                break
            if time.time() > deadline - 3:
                break
            S_trial = S | {x}
            if is_valid(S_trial):
                S = S_trial
        return S

    def greedy_with_divisibility_awareness():
        """
        Greedy that prioritizes numbers with fewer divisors in range.
        """
        S = set()
        # Sort by number of divisors in [1,192] ascending (fewer divisors = safer)
        def num_divisors_in_range(x):
            count = 0
            for d in range(1, n + 1):
                if x % d == 0:
                    count += 1
            return count
        
        candidates = list(range(1, n + 1))
        candidates.sort(key=lambda x: (num_divisors_in_range(x), -x))
        
        for x in candidates:
            if len(S) >= max_size:
                break
            if time.time() > deadline - 3:
                break
            S_trial = S | {x}
            if is_valid(S_trial):
                S = S_trial
        return S

    # Strategy:
    # 1. Try greedy constructions
    # 2. Try min-conflicts search from various starting points
    # 3. Keep the best valid set found
    
    best_S = set()
    
    # Try greedy large-first
    S_greedy = greedy_large_first()
    if is_valid(S_greedy) and len(S_greedy) > len(best_S):
        best_S = S_greedy
    
    # Try greedy with divisibility awareness
    S_div = greedy_with_divisibility_awareness()
    if is_valid(S_div) and len(S_div) > len(best_S):
        best_S = S_div
    
    # Try min-conflicts search from various starting points
    starting_points = [
        list(range(145, 193)),  # 48 elements
        list(range(140, 188)),
        list(range(135, 183)),
        list(range(130, 178)),
        list(range(125, 173)),
        list(range(120, 168)),
    ]
    
    # Also try random starting points
    random.seed(42)
    for _ in range(5):
        random_start = random.sample(range(1, n + 1), max_size)
        starting_points.append(random_start)
    
    for start_point in starting_points:
        if time.time() > deadline - 5:
            break
        
        # Try min-conflicts search
        S_result = min_conflicts_search(set(start_point), max_iters=3000)
        
        if S_result is not None:
            if is_valid(S_result) and len(S_result) > len(best_S):
                best_S = S_result
        
        # Also try repair from this starting point
        S_repair = set(start_point)
        while time.time() < deadline - 5:
            if is_valid(S_repair):
                if len(S_repair) > len(best_S):
                    best_S = S_repair
                break
            
            viol = get_violation_info(S_repair)
            if viol:
                a, b, c, d, m = viol
                # Remove the element with most violations
                counts = count_violations_per_element(S_repair)
                worst = max([a, b, c, d], key=lambda x: counts.get(x, 0))
                S_repair.remove(worst)
            else:
                break
    
    # Final check and return
    if len(best_S) == 0:
        # Fallback: return a small valid set
        S = set(range(183, 193))
        best_S = S

    return list(best_S)
