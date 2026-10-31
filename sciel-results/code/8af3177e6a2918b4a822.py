import math
import random
import time
import itertools

def score_set(U):
    """
    Calculates the score: 1 + log(|U-U|/|U+U|)/log(2 max U + 1)
    """
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    sums = set()
    diffs = set()
    # Optimization: use nested loops
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
            
    size_sum = len(sums)
    size_diff = len(diffs)
    
    if size_sum == 0:
        return 0
        
    ratio = size_diff / size_sum
    # If ratio <= 1, log is <= 0, score <= 1. 
    # The problem states score is the lower bound. 
    # Usually |U-U| >= |U+U| for sets containing 0? 
    # |U-U| contains 0. |U+U| contains 0.
    # For {0, 1}: sums {0,1,2}, diffs {-1,0,1}. ratio 1.
    # For {0, 1, 3}: sums {0,1,2,3,4,6}, diffs {-3,-2,-1,0,1,2,3}. ratio 7/6 > 1.
    
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0
        
    return 1 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Initial candidates: Small brute-force bests and Golomb rulers
    # Golomb rulers have distinct differences, which maximizes |U-U| for a given n
    # but might not minimize |U+U| optimally. Hill climbing will adjust.
    
    initial_candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 4, 10, 12, 17, 25, 34],
        # Some AP-like or mixed sets
        [0, 1, 2, 4, 8],
        [0, 1, 3, 7, 15],
    ]
    
    best_set = initial_candidates[0]
    best_score = score_set(best_set)
    
    # Quick brute force for small sizes to ensure we start well
    # We already did this in the seed, but let's verify top few
    # Size 4, max 10
    for comb in itertools.combinations(range(1, 11), 3):
        U = [0] + list(comb)
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U
            
    # Size 5, max 15
    for comb in itertools.combinations(range(1, 16), 4):
        U = [0] + list(comb)
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U

    current_set = list(best_set)
    current_score = best_score
    
    # Hill Climbing
    # We will alternate between random moves and local polish
    no_improvement_count = 0
    
    while time.time() < deadline:
        # 1. Try a random move
        neighbor = list(current_set)
        move_type = random.choice(['add', 'remove', 'shift', 'swap', 'polish'])
        
        if move_type == 'add':
            max_val = max(neighbor)
            # Try adding a number close to existing ones or slightly beyond
            # Limit max_val growth to keep log denom small
            max_allowed = max_val * 2 + 10
            candidates = [i for i in range(1, max_allowed + 1) if i not in neighbor]
            if candidates:
                # Bias towards smaller numbers to keep max_u low? 
                # Or bias towards gaps?
                # Let's pick randomly but weighted towards lower values
                # Actually, uniform in range is fine, but restrict range
                # If max is large, adding a small number increases |U-U| without increasing max much
                # Let's try adding a small number
                small_candidates = [i for i in candidates if i < max_val // 2]
                if small_candidates and random.random() < 0.5:
                    neighbor.append(random.choice(small_candidates))
                else:
                    neighbor.append(random.choice(candidates))
                    
        elif move_type == 'remove':
            if len(neighbor) > 2: # Keep at least {0, x, y}
                # Don't remove 0
                rem_candidates = [x for x in neighbor if x != 0]
                rem = random.choice(rem_candidates)
                neighbor.remove(rem)
                
        elif move_type == 'shift':
            if len(neighbor) > 1:
                idx = random.choice([i for i, x in enumerate(neighbor) if x != 0])
                old_val = neighbor[idx]
                # Shift by small amount
                delta = random.choice([-3, -2, -1, 1, 2, 3, -5, 5, -10, 10])
                new_val = old_val + delta
                if new_val > 0 and new_val not in neighbor:
                    neighbor[idx] = new_val
                else:
                    # If invalid, try a random new value
                    new_val = random.randint(1, max(neighbor) + 5)
                    if new_val not in neighbor:
                        neighbor[idx] = new_val
                    else:
                        continue # Skip this iteration if move is invalid
                        
        elif move_type == 'swap':
            if len(neighbor) > 2:
                # Pick two non-zero elements to swap
                non_zero_indices = [i for i, x in enumerate(neighbor) if x != 0]
                if len(non_zero_indices) >= 2:
                    i, j = random.sample(non_zero_indices, 2)
                    # Swap values? No, that does nothing to the set.
                    # "Swap" in set context usually means replace one with the other's value?
                    # Better: Replace one element with the value of another, and the other with a new value?
                    # Or just replace one element with a random value.
                    # Let's do: Replace element i with a random value, and element j with a random value.
                    val_i = neighbor[i]
                    val_j = neighbor[j]
                    new_val_i = random.randint(1, max(neighbor) + 5)
                    new_val_j = random.randint(1, max(neighbor) + 5)
                    
                    # Ensure no duplicates
                    if new_val_i != val_j and new_val_j != val_i and new_val_i != new_val_j:
                        # Check against rest of set
                        rest = [x for k, x in enumerate(neighbor) if k != i and k != j]
                        if new_val_i not in rest and new_val_j not in rest:
                            neighbor[i] = new_val_i
                            neighbor[j] = new_val_j
                        else:
                            continue
                    else:
                        continue
                        
        else: # polish
            # Try all small shifts for each element
            improved = False
            for i in range(len(neighbor)):
                if neighbor[i] == 0:
                    continue
                original_val = neighbor[i]
                for delta in range(-2, 3):
                    if delta == 0:
                        continue
                    new_val = original_val + delta
                    if new_val <= 0:
                        continue
                    if new_val in neighbor:
                        continue
                    neighbor[i] = new_val
                    s = score_set(neighbor)
                    if s > current_score:
                        current_score = s
                        current_set = list(neighbor)
                        best_score = max(best_score, current_score)
                        best_set = list(neighbor)
                        improved = True
                    neighbor[i] = original_val
            
            if improved:
                no_improvement_count = 0
                continue
            else:
                # If no improvement in polish, try a more drastic move
                # Reset to best_set to avoid drifting into bad local minima
                if random.random() < 0.1:
                    current_set = list(best_set)
                    current_score = best_score

        neighbor.sort()
        new_score = score_set(neighbor)
        
        if new_score > current_score:
            current_set = neighbor
            current_score = new_score
            no_improvement_count = 0
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
        else:
            no_improvement_count += 1
            # Accept worse moves with low probability to escape local optima
            # Simulated Annealing style
            if random.random() < 0.05:
                current_set = neighbor
                current_score = new_score
        
        # If we are stuck for a while, restart from best_set
        if no_improvement_count > 50:
            current_set = list(best_set)
            current_score = best_score
            no_improvement_count = 0

    return best_set

# Helper to ensure we don't run too long in case of bugs
if __name__ == "__main__":
    # Quick test
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time taken: {end - start:.2f}s")
    print(f"Best set: {res}")
    print(f"Score: {score_set(res)}")
