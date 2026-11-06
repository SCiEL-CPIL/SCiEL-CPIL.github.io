"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from fractions import Fraction
    from itertools import combinations
    
    deadline = time.time() + 85.0  # 5s safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute Fraction values
    vals = [Fraction(1, d) for d in denominators]
    one = Fraction(1, 1)
    
    # Greedy seed: start with the seed program's greedy solution
    # Greedy: add denominators in order if sum stays < 1
    greedy_idx = []
    greedy_sum = Fraction(0)
    for i in range(n):
        if greedy_sum + vals[i] < one:
            greedy_idx.append(i)
            greedy_sum += vals[i]
    
    # Also try greedy by descending denominator (smallest reciprocals first)
    greedy2_idx = []
    greedy2_sum = Fraction(0)
    for i in range(n - 1, -1, -1):
        if greedy2_sum + vals[i] < one:
            greedy2_idx.append(i)
            greedy2_sum += vals[i]
    
    # Also try greedy by ascending denominator
    greedy3_idx = []
    greedy3_sum = Fraction(0)
    for i in range(n):
        if greedy3_sum + vals[i] < one:
            greedy3_idx.append(i)
            greedy3_sum += vals[i]
    
    # Pick the best greedy
    candidates = [greedy_idx, greedy2_idx, greedy3_idx]
    best_deficit = None
    best_set = None
    for c in candidates:
        if not c:
            continue
        s = sum(vals[i] for i in c)
        if s < one:
            d = one - s
            if best_deficit is None or d < best_deficit:
                best_deficit = d
                best_set = c
    
    if best_set is None:
        # Fallback: just pick denominator 2
        return [0]
    
    best_deficit = one - sum(vals[i] for i in best_set)
    
    # Local search: try swapping elements to reduce deficit
    # Strategy: try to add an element not in the set, remove one or more from the set
    # to keep sum < 1, and minimize deficit
    
    current_set = set(best_set)
    current_sum = sum(vals[i] for i in best_set)
    current_deficit = one - current_sum
    
    # Try adding each element not in set
    for i in range(n):
        if i in current_set:
            continue
        new_sum = current_sum + vals[i]
        if new_sum < one:
            new_deficit = one - new_sum
            if new_deficit < current_deficit:
                current_set.add(i)
                current_sum = new_sum
                current_deficit = new_deficit
    
    # Now try local moves: remove one element, add another
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in list(current_set):
            for j in range(n):
                if j in current_set:
                    continue
                # Try swapping i for j
                new_sum = current_sum - vals[i] + vals[j]
                if new_sum < one:
                    new_deficit = one - new_sum
                    if new_deficit < current_deficit:
                        current_set.remove(i)
                        current_set.add(j)
                        current_sum = new_sum
                        current_deficit = new_deficit
                        improved = True
                        break
            if improved:
                continue
            if time.time() > deadline:
                break
        # Also try adding without removing
        for j in range(n):
            if j not in current_set:
                new_sum = current_sum + vals[j]
                if new_sum < one:
                    new_deficit = one - new_sum
                    if new_deficit < current_deficit:
                        current_set.add(j)
                        current_sum = new_sum
                        current_deficit = new_deficit
                        improved = True
                        break
            if time.time() > deadline:
                break
    
    # Try removing elements and adding multiple others
    if time.time() < deadline:
        # For each pair of elements to remove, try adding up to 2 others
        for i in list(current_set):
            for j in list(current_set):
                if i >= j:
                    continue
                base_sum = current_sum - vals[i] - vals[j]
                # Try adding one
                for k in range(n):
                    if k == i or k == j or k in current_set:
                        continue
                    new_sum = base_sum + vals[k]
                    if new_sum < one:
                        new_deficit = one - new_sum
                        if new_deficit < current_deficit:
                            current_set.discard(i)
                            current_set.discard(j)
                            current_set.add(k)
                            current_sum = new_sum
                            current_deficit = new_deficit
                            improved = True
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
    
    # Try adding pairs
    if time.time() < deadline:
        for i in range(n):
            if i in current_set:
                continue
            for j in range(n):
                if j in current_set or j <= i:
                    continue
                new_sum = current_sum + vals[i] + vals[j]
                if new_sum < one:
                    new_deficit = one - new_sum
                    if new_deficit < current_deficit:
                        current_set.add(i)
                        current_set.add(j)
                        current_sum = new_sum
                        current_deficit = new_deficit
                        improved = True
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
    
    # Final check: try removing single elements to see if we can add more
    if time.time() < deadline:
        for i in list(current_set):
            new_sum = current_sum - vals[i]
            if new_sum < one:
                # Try adding back i and also others
                pass
    
    # Convert to sorted list of indices
    result = sorted(current_set)
    
    # Verify
    final_sum = sum(vals[i] for i in result)
    if final_sum >= one:
        # Fallback to greedy
        return greedy_idx if greedy_idx else [0]
    
    return result
# EVOLVE-BLOCK-END
