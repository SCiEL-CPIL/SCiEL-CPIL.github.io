def construct():
    from fractions import Fraction
    
    denominators = list(range(2, 49))
    n = len(denominators)
    values = [Fraction(1, d) for d in denominators]
    
    # Start with known good solution: indices for 1/2, 1/3, 1/7, 1/43
    # 1/2 is index 0, 1/3 is index 1, 1/7 is index 5, 1/43 is index 41
    best_indices = {0, 1, 5, 41}
    best_sum = Fraction(1,2) + Fraction(1,3) + Fraction(1,7) + Fraction(1,43)
    best_deficit = Fraction(1) - best_sum
    
    # Local search: try adding, removing, or swapping one element
    improved = True
    while improved:
        improved = False
        current_sum = best_sum
        
        # Try swaps
        for i in list(best_indices):
            for j in range(n):
                if j in best_indices:
                    continue
                new_sum = current_sum - values[i] + values[j]
                if 0 < new_sum < 1:
                    new_deficit = Fraction(1) - new_sum
                    if new_deficit < best_deficit:
                        best_indices = (best_indices - {i}) | {j}
                        best_sum = new_sum
                        best_deficit = new_deficit
                        improved = True
                        # Restart loop to update current_sum and indices
                        # Breaking out to restart the while loop is safe
                        break
            if improved:
                break
        
        if improved:
            continue
            
        # Try adds
        for j in range(n):
            if j not in best_indices:
                new_sum = best_sum + values[j]
                if 0 < new_sum < 1:
                    new_deficit = Fraction(1) - new_sum
                    if new_deficit < best_deficit:
                        best_indices = best_indices | {j}
                        best_sum = new_sum
                        best_deficit = new_deficit
                        improved = True
                        break
        
        if improved:
            continue

        # Try removes
        for i in list(best_indices):
            new_sum = best_sum - values[i]
            if 0 < new_sum < 1:
                new_deficit = Fraction(1) - new_sum
                if new_deficit < best_deficit:
                    best_indices = best_indices - {i}
                    best_sum = new_sum
                    best_deficit = new_deficit
                    improved = True
                    break
    
    return sorted(best_indices)
