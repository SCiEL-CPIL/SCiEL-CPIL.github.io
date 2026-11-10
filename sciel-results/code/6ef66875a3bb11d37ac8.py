def construct():
    from fractions import Fraction
    import time
    
    deadline = time.time() + 80
    
    denominators = list(range(2, 49))
    n = len(denominators)
    fractions = [Fraction(1, d) for d in denominators]
    
    best_sum = Fraction(0)
    best_indices = []
    
    # Split: first 20 are "large", rest are "small"
    split = 20
    large_fracs = fractions[:split]
    small_fracs = fractions[split:]
    
    def greedy_fill(current_sum, small_start_idx):
        """Greedily add small fractions to maximize sum while keeping < 1."""
        total = current_sum
        selected = []
        for i, f in enumerate(small_fracs):
            if total + f < 1:
                total += f
                selected.append(small_start_idx + i)
        return total, selected
    
    # Enumerate subsets of large fractions
    def enumerate_large(idx, current_sum, selected_large):
        nonlocal best_sum, best_indices
        
        if time.time() > deadline:
            return
        
        if idx == split:
            # Greedily fill in small fractions
            total, selected_small = greedy_fill(current_sum, split)
            if total < 1 and total > best_sum:
                best_sum = total
                best_indices = selected_large + selected_small
            return
        
        # Try including
        new_sum = current_sum + large_fracs[idx]
        if new_sum < 1:
            selected_large.append(idx)
            enumerate_large(idx + 1, new_sum, selected_large)
            selected_large.pop()
        
        # Try excluding
        enumerate_large(idx + 1, current_sum, selected_large)
    
    enumerate_large(0, Fraction(0), [])
    
    # Local search to refine the solution
    if best_indices and time.time() < deadline - 5:
        curr_set = set(best_indices)
        curr_sum = best_sum
        curr_indices = list(best_indices)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Try adding elements
            for i in range(n):
                if i not in curr_set:
                    if curr_sum + fractions[i] < 1:
                        new_sum = curr_sum + fractions[i]
                        if new_sum > best_sum:
                            curr_indices = curr_indices + [i]
                            curr_set.add(i)
                            curr_sum = new_sum
                            best_sum = new_sum
                            best_indices = curr_indices
                            improved = True
                            break
            
            if improved:
                continue
            
            # Try swaps: remove i, add j where j > i in value
            for i in curr_indices[:]:
                for j in range(n):
                    if j in curr_set:
                        continue
                    if fractions[j] > fractions[i]:
                        new_sum = curr_sum - fractions[i] + fractions[j]
                        if new_sum < 1 and new_sum > best_sum:
                            new_indices = [x for x in curr_indices if x != i] + [j]
                            curr_indices = new_indices
                            curr_set = set(curr_indices)
                            curr_sum = new_sum
                            best_sum = new_sum
                            best_indices = curr_indices
                            improved = True
                            break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try removing one element if it allows a better configuration
            # This is less likely to help but can escape local optima
            for i in curr_indices[:]:
                if len(curr_indices) > 1:
                    new_indices = [x for x in curr_indices if x != i]
                    new_sum = curr_sum - fractions[i]
                    # After removing, try to greedily add back
                    temp_sum = new_sum
                    temp_set = set(new_indices)
                    temp_indices = list(new_indices)
                    for j in range(n):
                        if j not in temp_set:
                            if temp_sum + fractions[j] < 1:
                                temp_sum += fractions[j]
                                temp_set.add(j)
                                temp_indices.append(j)
                    if temp_sum > best_sum:
                        curr_indices = temp_indices
                        curr_set = temp_set
                        curr_sum = temp_sum
                        best_sum = temp_sum
                        best_indices = curr_indices
                        improved = True
                        break
    
    return best_indices
