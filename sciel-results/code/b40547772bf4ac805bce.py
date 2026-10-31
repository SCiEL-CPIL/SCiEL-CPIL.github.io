import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 40s safety margin for 150s limit

    def compute_score(U):
        if not U or len(U) <= 1:
            return 1.0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        u_arr = np.array(U, dtype=np.int64)
        n = len(u_arr)
        
        # Vectorized computation of sumset and diffset
        sums = np.add.outer(u_arr, u_arr)
        sumset = set(np.unique(sums).tolist())
        
        diffs = np.subtract.outer(u_arr, u_arr)
        diffset = set(np.unique(diffs).tolist())
        
        size_sum = len(sumset)
        size_diff = len(diffset)
        
        if size_sum == 0:
            return 1.0
            
        ratio = size_diff / size_sum
        if ratio <= 1.0:
            return 1.0
        
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
            
        return 1.0 + math.log(ratio) / denom

    def is_golomb(U):
        """Check if U is a Golomb ruler (all positive pairwise differences are unique)."""
        diffs = set()
        n = len(U)
        for i in range(n):
            for j in range(i + 1, n):
                d = U[j] - U[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    # Known optimal Golomb rulers
    golomb_rulers = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 18, 23, 25],
        8: [0, 1, 4, 10, 18, 23, 29, 34],
        9: [0, 1, 4, 10, 18, 23, 29, 37, 44],
        10: [0, 1, 4, 10, 18, 23, 29, 37, 44, 55],
        11: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 67],
        12: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 80],
    }

    # Initial candidates: known Golomb rulers and some variations
    initial_sets = []
    for n in range(1, 13):
        if n in golomb_rulers:
            initial_sets.append(golomb_rulers[n])
    
    # Add some hand-crafted sets that might have good properties
    extra_sets = [
        [0, 1, 3, 7],
        [0, 1, 4, 6, 10],
        [0, 2, 5, 11],
        [0, 1, 5, 11],
        [0, 3, 7, 12],
        [0, 1, 4, 9, 11, 18],
        [0, 2, 7, 12, 18],
        [0, 1, 5, 10, 16],
        [0, 1, 2, 5, 10], # Arithmetic progression like
        [0, 1, 3, 6, 10], # Triangular numbers
    ]
    initial_sets.extend(extra_sets)
    
    # Deduplicate
    seen = set()
    unique_initial = []
    for s in initial_sets:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique_initial.append(list(s))
            
    best_score = 0.0
    best_set = [0]
    
    for s in unique_initial:
        score = compute_score(s)
        if score > best_score:
            best_score = score
            best_set = list(s)
    
    # Local search with multiple restarts
    max_restarts = 10
    best_overall_score = best_score
    best_overall_set = best_set
    rng = random.Random(42)
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        # Start from best set or a random Golomb ruler
        if restart == 0:
            current_set = list(best_overall_set)
        else:
            # Pick a random Golomb ruler as starting point
            n = rng.randint(3, 12)
            if n in golomb_rulers:
                current_set = list(golomb_rulers[n])
            else:
                current_set = list(best_overall_set)
        
        current_score = compute_score(current_set)
        
        # Local search phase
        no_improve_count = 0
        max_no_improve = 1000
        start_search_time = time.time()
        
        while time.time() < deadline - 5 and no_improve_count < max_no_improve:
            # Generate a neighbor
            neighbor = list(current_set)
            op = rng.randint(0, 6)
            
            if op == 0 and len(neighbor) > 1:
                # Remove a random element (not 0)
                idx = rng.randint(1, len(neighbor) - 1)
                neighbor.pop(idx)
                if len(neighbor) == 0:
                    continue
                if 0 not in neighbor:
                    neighbor.append(0)
                    neighbor.sort()
            elif op == 1:
                # Add a random element
                max_val = max(neighbor) if neighbor else 0
                # Bias towards adding elements that keep the set relatively compact
                # but also allow for expansion
                upper_bound = max(2 * max_val + 20, 30)
                new_val = rng.randint(1, upper_bound)
                if new_val not in neighbor:
                    neighbor.append(new_val)
                    neighbor.sort()
            elif op == 2:
                # Modify a random element (not 0) to a nearby value
                if len(neighbor) > 1:
                    idx = rng.randint(1, len(neighbor) - 1)
                    old_val = neighbor[idx]
                    # Try small perturbations
                    deltas = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
                    rng.shuffle(deltas)
                    for delta in deltas:
                        new_val = old_val + delta
                        if new_val >= 0 and new_val not in neighbor:
                            neighbor[idx] = new_val
                            neighbor.sort()
                            break
                    else:
                        continue
            elif op == 3:
                # Swap: remove one element and add another
                if len(neighbor) > 1:
                    idx = rng.randint(1, len(neighbor) - 1)
                    removed_val = neighbor.pop(idx)
                    max_val = max(neighbor) if neighbor else 0
                    upper_bound = max(2 * max_val + 20, 30)
                    new_val = rng.randint(1, upper_bound)
                    if new_val not in neighbor:
                        neighbor.append(new_val)
                        neighbor.sort()
                    else:
                        # Revert
                        neighbor.append(removed_val)
                        neighbor.sort()
                        continue
            elif op == 4:
                # Add two elements
                max_val = max(neighbor) if neighbor else 0
                upper_bound = max(2 * max_val + 20, 30)
                new_val1 = rng.randint(1, upper_bound)
                new_val2 = rng.randint(1, upper_bound)
                if new_val1 not in neighbor and new_val2 not in neighbor and new_val1 != new_val2:
                    neighbor.append(new_val1)
                    neighbor.append(new_val2)
                    neighbor.sort()
            elif op == 5:
                # Remove two elements
                if len(neighbor) > 3:
                    idx1 = rng.randint(1, len(neighbor) - 1)
                    idx2 = rng.randint(1, len(neighbor) - 1)
                    if idx1 == idx2:
                        continue
                    if idx1 > idx2:
                        idx1, idx2 = idx2, idx1
                    neighbor.pop(idx1)
                    neighbor.pop(idx2)
                    if len(neighbor) == 0:
                        continue
                    if 0 not in neighbor:
                        neighbor.append(0)
                        neighbor.sort()
            else:
                # "Dual-view" inspired move: try to fix Golomb property
                # If not Golomb, try to resolve a collision by moving an element
                if not is_golomb(neighbor):
                    # Find a collision
                    diffs = {}
                    n = len(neighbor)
                    collision = None
                    for i in range(n):
                        for j in range(i + 1, n):
                            d = neighbor[j] - neighbor[i]
                            if d in diffs:
                                # Collision found between (diffs[d]) and (i,j)
                                # Try to move one of the elements involved
                                # Let's try moving neighbor[j]
                                old_val = neighbor[j]
                                # Try small shifts
                                for delta in [-2, -1, 1, 2]:
                                    new_val = old_val + delta
                                    if new_val >= 0 and new_val not in neighbor:
                                        candidate = list(neighbor)
                                        candidate[j] = new_val
                                        candidate.sort()
                                        if is_golomb(candidate):
                                            # Check if score improves
                                            s = compute_score(candidate)
                                            if s > current_score:
                                                neighbor = candidate
                                                break
                                else:
                                    continue
                    else:
                        # No specific collision fix applied, just a random small move
                        if len(neighbor) > 1:
                            idx = rng.randint(1, len(neighbor) - 1)
                            old_val = neighbor[idx]
                            new_val = old_val + rng.randint(-3, 3)
                            if new_val >= 0 and new_val not in neighbor:
                                neighbor[idx] = new_val
                                neighbor.sort()
                            else:
                                continue
                else:
                    # Already Golomb, do a standard random move
                    if len(neighbor) > 1:
                        idx = rng.randint(1, len(neighbor) - 1)
                        old_val = neighbor[idx]
                        new_val = old_val + rng.randint(-5, 5)
                        if new_val >= 0 and new_val not in neighbor:
                            neighbor[idx] = new_val
                            neighbor.sort()
                        else:
                            continue

            if not neighbor or 0 not in neighbor:
                continue
                
            new_score = compute_score(neighbor)
            
            if new_score > current_score:
                current_set = neighbor
                current_score = new_score
                no_improve_count = 0
                if current_score > best_overall_score:
                    best_overall_score = current_score
                    best_overall_set = list(current_set)
            else:
                no_improve_count += 1
                # Small probability of accepting worse move to escape local minima
                if rng.random() < 0.005:
                    current_set = neighbor
                    current_score = new_score
    
    # Final polish: systematically try small perturbations on the best set
    if time.time() < deadline - 2:
        polish_set = list(best_overall_set)
        polish_score = compute_score(polish_set)
        
        # Try removing each element one at a time
        for i in range(1, len(polish_set)):
            if time.time() > deadline - 1:
                break
            candidate = list(polish_set)
            candidate.pop(i)
            if len(candidate) > 0 and 0 in candidate:
                candidate.sort()
                s = compute_score(candidate)
                if s > polish_score:
                    polish_set = candidate
                    polish_score = s
        
        # Try adding elements near existing ones
        if time.time() < deadline - 1:
            max_val = max(polish_set)
            for base in list(polish_set):
                for delta in range(-3, 4):
                    if time.time() > deadline - 1:
                        break
                    new_val = base + delta
                    if new_val >= 0 and new_val not in polish_set:
                        candidate = list(polish_set)
                        candidate.append(new_val)
                        candidate.sort()
                        s = compute_score(candidate)
                        if s > polish_score:
                            polish_set = candidate
                            polish_score = s
        
        if polish_score > best_overall_score:
            best_overall_score = polish_score
            best_overall_set = polish_set
    
    return best_overall_set
