import random
from itertools import combinations

def construct():
    n = 192
    max_elements = 48
    
    def is_valid_addition(selected, candidate):
        """Check if adding candidate to selected maintains validity."""
        sel_set = set(selected)
        # For the new candidate: check if it divides sum of any 3 others in selected
        if len(selected) >= 3:
            for a, b, c in combinations(selected, 3):
                if (a + b + c) % candidate == 0:
                    return False
        # For each existing element a: check triples that include candidate
        # i.e., a | (b + c + candidate) for b,c in selected, b!=a, c!=a, b!=c
        for a in selected:
            others = [x for x in selected if x != a]
            if len(others) >= 2:
                for b, c in combinations(others, 2):
                    if (b + c + candidate) % a == 0:
                        return False
        return True

    def is_valid(selected):
        """Check if the entire set is valid."""
        for a in selected:
            others = [x for x in selected if x != a]
            for b, c, d in combinations(others, 3):
                if (b + c + d) % a == 0:
                    return False
        return True

    def greedy_large_first():
        selected = []
        for x in range(n, 0, -1):
            if is_valid_addition(selected, x):
                selected.append(x)
                if len(selected) >= max_elements:
                    break
        return selected

    def greedy_random():
        selected = []
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        for x in candidates:
            if is_valid_addition(selected, x):
                selected.append(x)
                if len(selected) >= max_elements:
                    break
        return selected

    def local_search(selected):
        selected = list(selected)
        # Use a set for faster membership check
        selected_set = set(selected)
        
        for iteration in range(100):  # Increased iterations
            improved = False
            
            # 1. Try adding
            for x in range(1, n + 1):
                if x not in selected_set and len(selected) < max_elements:
                    if is_valid_addition(selected, x):
                        selected.append(x)
                        selected_set.add(x)
                        improved = True
                        break
            
            if improved:
                continue
            
            # 2. Try single swap
            swapped = False
            for i in range(len(selected)):
                removed = selected[i]
                temp_set = selected_set - {removed}
                temp_list = [x for x in selected if x != removed]
                for x in range(1, n + 1):
                    if x not in temp_set:
                        if is_valid_addition(temp_list, x):
                            # Accept swap
                            selected = temp_list + [x]
                            selected_set = temp_set | {x}
                            swapped = True
                            improved = True
                            break
                if swapped:
                    break
            
            if improved:
                continue
            
            # 3. Try 2-swap (remove 2, add 2)
            # Only try if stuck, and limit the search space
            found_2swap = False
            # Randomize the order of removals to explore different areas
            indices = list(range(len(selected)))
            random.shuffle(indices)
            
            for i_idx in range(len(indices)):
                i = indices[i_idx]
                for j_idx in range(i_idx+1, len(indices)):
                    j = indices[j_idx]
                    r1, r2 = selected[i], selected[j]
                    
                    # Create temp list excluding r1, r2
                    temp_list = [x for k, x in enumerate(selected) if k != i and k != j]
                    temp_set = selected_set - {r1, r2}
                    
                    # Try to find 2 elements to add
                    # We can restrict the candidates to those not in temp_set
                    candidates_to_add = [x for x in range(1, n+1) if x not in temp_set]
                    random.shuffle(candidates_to_add)
                    
                    for x in candidates_to_add:
                        if not is_valid_addition(temp_list, x):
                            continue
                        
                        temp_list_1 = temp_list + [x]
                        temp_set_1 = temp_set | {x}
                        
                        for y in candidates_to_add:
                            if y == x:
                                continue
                            if y in temp_set_1:
                                continue
                            
                            if is_valid_addition(temp_list_1, y):
                                # Accept 2-swap
                                selected = temp_list_1 + [y]
                                selected_set = temp_set_1 | {y}
                                found_2swap = True
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
                
        return selected

    best = greedy_large_first()
    
    for trial in range(50):
        random.seed(1000 + trial)
        s = greedy_random()
        s = local_search(s)
        if len(s) > len(best):
            best = s
            if len(best) >= max_elements:
                break
        
        # Also try large first with some randomization in local search
        random.seed(2000 + trial)
        s2 = greedy_large_first()
        s2 = local_search(s2)
        if len(s2) > len(best):
            best = s2
            if len(best) >= max_elements:
                break

    if not is_valid(best):
        best = greedy_large_first()
    
    return best[:max_elements]
