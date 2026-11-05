import time
import random
from itertools import combinations

def _can_add(x, selected):
    """Check if x can be added to selected without violating the constraint."""
    n = len(selected)
    if n >= 3:
        # x must not divide sum of any 3 distinct elements in selected
        for i in range(n):
            si = selected[i]
            for j in range(i + 1, n):
                sij = si + selected[j]
                for k in range(j + 1, n):
                    if (sij + selected[k]) % x == 0:
                        return False

    # For each a in selected, a must not divide x + u + v for any u, v in selected \ {a}
    for a_idx, a in enumerate(selected):
        for i in range(n):
            if i == a_idx:
                continue
            si = selected[i]
            for j in range(i + 1, n):
                if j == a_idx:
                    continue
                sj = selected[j]
                if (x + si + sj) % a == 0:
                    return False
    return True

def _greedy(order, deadline):
    """Run greedy in the given order."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, deadline):
    """Improve the set by adding elements or performing swaps that enable growth."""
    selected = list(selected)
    in_set = set(selected)
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False

        # 1. Try to add any element
        for x in range(192, 0, -1):
            if x not in in_set and _can_add(x, selected):
                selected.append(x)
                in_set.add(x)
                improved = True
                break

        # 2. If no addition, try swaps
        if not improved:
            candidates_out = list(selected)
            random.shuffle(candidates_out)

            for x_out in candidates_out:
                if time.time() > deadline - 2.0:
                    return selected

                selected_without = [v for v in selected if v != x_out]

                for x_in in range(192, 0, -1):
                    if x_in in in_set:
                        continue

                    if _can_add(x_in, selected_without):
                        new_set = selected_without + [x_in]
                        new_in_set = set(new_set)

                        can_grow = False
                        for x_add in range(192, 0, -1):
                            if x_add not in new_in_set:
                                if _can_add(x_add, new_set):
                                    can_grow = True
                                    break

                        if can_grow:
                            selected = new_set
                            in_set = new_in_set
                            improved = True
                            break

                if improved:
                    break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    best = []

    # Strategy 1: Greedy descending
    result = _greedy(range(192, 0, -1), deadline)
    if len(result) > len(best):
        best = result

    # Strategy 2: Greedy ascending
    if time.time() < deadline - 5:
        result = _greedy(range(1, 193), deadline)
        if len(result) > len(best):
            best = result

    # Strategy 3: Multiple random shuffles
    random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 20:
            break
        order = list(range(1, 193))
        random.shuffle(order)
        result = _greedy(order, deadline)
        if len(result) > len(best):
            best = result

    # Strategy 4: Local search on the best greedy result
    if time.time() < deadline - 10:
        best = _local_search(best, deadline)

    # Strategy 5: Simulated Annealing-inspired global search
    # This is the key improvement from the inspirations: use stochastic global
    # optimization over the full state space to escape local optima.
    
    if time.time() < deadline - 15:
        # Start from the current best
        current_set = set(best)
        current_list = list(best)
        
        # Precompute all numbers
        all_numbers = list(range(1, 193))
        
        # Simulated annealing parameters
        T = 1.0  # Temperature
        cooling_rate = 0.999
        min_T = 0.01
        
        # Number of iterations based on time budget
        sa_deadline = deadline - 2.0
        iterations = 0
        
        while time.time() < sa_deadline and T > min_T:
            iterations += 1
            
            # Randomly choose a move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3:
                # Try to add a random element
                candidates = [x for x in all_numbers if x not in current_set]
                if not candidates:
                    continue
                x_add = random.choice(candidates)
                new_set = current_set | {x_add}
                new_list = current_list + [x_add]
                
                # Check validity
                valid = True
                for a in current_list:
                    if a == x_add:
                        continue
                    # Check if a divides x_add + u + v for any u, v in current_list \ {a}
                    others = [v for v in current_list if v != a]
                    if len(others) >= 2:
                        for i in range(len(others)):
                            for j in range(i + 1, len(others)):
                                if (x_add + others[i] + others[j]) % a == 0:
                                    valid = False
                                    break
                            if not valid:
                                break
                    if not valid:
                        break
                
                if valid:
                    # Also check if x_add divides sum of any 3 in current_list
                    if len(current_list) >= 3:
                        for combo in combinations(current_list, 3):
                            if sum(combo) % x_add == 0:
                                valid = False
                                break
                
                if valid:
                    current_set = new_set
                    current_list = new_list
                    if len(current_set) > len(best):
                        best = list(current_set)
                else:
                    # Accept with probability based on temperature
                    if random.random() < T:
                        current_set = new_set
                        current_list = new_list
            
            elif move_type < 0.6:
                # Try to remove a random element
                if not current_list:
                    continue
                x_remove = random.choice(current_list)
                new_set = current_set - {x_remove}
                new_list = [v for v in current_list if v != x_remove]
                
                # Always accept removals (they don't create violations)
                current_set = new_set
                current_list = new_list
            
            else:
                # Try to swap: remove one, add one
                if not current_list:
                    continue
                x_remove = random.choice(current_list)
                candidates = [x for x in all_numbers if x not in current_set and x != x_remove]
                if not candidates:
                    continue
                x_add = random.choice(candidates)
                
                # Check if the swap is valid
                temp_set = current_set - {x_remove}
                temp_list = [v for v in current_list if v != x_remove]
                
                if _can_add(x_add, temp_list):
                    new_set = temp_set | {x_add}
                    new_list = temp_list + [x_add]
                    current_set = new_set
                    current_list = new_list
                    if len(current_set) > len(best):
                        best = list(current_set)
            
            # Cool down
            T *= cooling_rate
    
    return best
