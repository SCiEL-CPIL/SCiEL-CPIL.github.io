import time
import random
import math
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin for 150s limit
    
    def compute_score(U):
        if not U or len(U) <= 1:
            return 1.0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        u_arr = np.array(U, dtype=np.int64)
        n = len(u_arr)
        
        # Use broadcasting for efficient computation
        # Sumset: all pairwise sums
        sumset = set()
        diffset = set()
        
        for i in range(n):
            for j in range(i, n):
                s = u_arr[i] + u_arr[j]
                sumset.add(s)
                if i != j:
                    d = u_arr[i] - u_arr[j]
                    diffset.add(d)
                    diffset.add(-d)
        
        diffset.add(0)
        
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

    # Known optimal Golomb rulers
    golomb_rulers = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 25],
        8: [0, 1, 4, 10, 12, 17, 25, 29],
        9: [0, 1, 4, 10, 12, 17, 25, 29, 34],
        10: [0, 1, 4, 10, 12, 17, 25, 29, 34, 41],
        11: [0, 1, 4, 10, 12, 17, 25, 29, 34, 41, 48],
        12: [0, 1, 4, 10, 12, 17, 25, 29, 34, 41, 48, 55],
    }
    
    # Generate more diverse initial candidates
    initial_sets = []
    
    # Golomb rulers
    for n in range(1, 13):
        if n in golomb_rulers:
            initial_sets.append(list(golomb_rulers[n]))
    
    # Subsets of Golomb rulers (remove some elements)
    for base_n in [5, 6, 7, 8, 9, 10]:
        if base_n in golomb_rulers:
            base = golomb_rulers[base_n]
            for remove_count in range(1, 3):
                for idxs in combinations(range(1, len(base)), remove_count):
                    subset = [x for i, x in enumerate(base) if i not in idxs]
                    if 0 in subset and len(subset) >= 2:
                        initial_sets.append(sorted(subset))
    
    # Hand-crafted sets with good properties
    extra_sets = [
        [0, 1, 3, 7],
        [0, 1, 4, 6, 10],
        [0, 2, 5, 11],
        [0, 1, 5, 11],
        [0, 3, 7, 12],
        [0, 1, 4, 9, 11, 18],
        [0, 2, 7, 12, 18],
        [0, 1, 5, 10, 16],
        [0, 2, 5, 9, 14, 20],
        [0, 1, 4, 8, 13, 19],
        [0, 3, 8, 14, 20],
        [0, 1, 6, 10, 15],
        [0, 2, 7, 11, 16],
        [0, 4, 9, 14, 20],
        [0, 1, 3, 8, 14, 21],
        [0, 2, 6, 11, 17],
        [0, 1, 5, 12, 18],
        [0, 3, 7, 13, 19],
        [0, 1, 4, 10, 16, 23],
        [0, 2, 5, 12, 18, 25],
        [0, 1, 6, 11, 17, 24],
        [0, 3, 8, 14, 20, 27],
    ]
    initial_sets.extend(extra_sets)
    
    # Deduplicate
    unique_sets = []
    seen = set()
    for s in initial_sets:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique_sets.append(s)
    initial_sets = unique_sets
    
    best_score = 0.0
    best_set = [0]
    
    for s in initial_sets:
        score = compute_score(s)
        if score > best_score:
            best_score = score
            best_set = list(s)
    
    # Enhanced local search with multiple restarts and better neighborhood
    best_overall_score = best_score
    best_overall_set = best_set
    
    def generate_neighbor(current_set, move_type):
        """Generate a neighbor of current_set using the specified move type."""
        neighbor = list(current_set)
        n = len(neighbor)
        
        if move_type == 0 and n > 1:
            # Remove a random element (not 0)
            idx = random.randint(1, n - 1)
            neighbor.pop(idx)
            if not neighbor or 0 not in neighbor:
                return None
            neighbor.sort()
            
        elif move_type == 1:
            # Add a random element
            max_val = max(neighbor) if neighbor else 0
            # Try different ranges for new values
            if random.random() < 0.3:
                # Small addition
                upper_bound = max_val + random.randint(1, 5)
            elif random.random() < 0.6:
                # Medium addition
                upper_bound = max_val + random.randint(5, 15)
            else:
                # Larger addition
                upper_bound = max(2 * max_val + 10, 30)
            
            new_val = random.randint(1, upper_bound)
            if new_val not in neighbor:
                neighbor.append(new_val)
                neighbor.sort()
            else:
                return None
                
        elif move_type == 2:
            # Modify a random element to a nearby value
            if n <= 1:
                return None
            idx = random.randint(1, n - 1)
            old_val = neighbor[idx]
            # Try various perturbation sizes
            max_delta = random.choice([3, 5, 8, 12])
            delta = random.randint(-max_delta, max_delta)
            if delta == 0:
                return None
            new_val = old_val + delta
            if new_val < 0 or new_val in neighbor:
                return None
            neighbor[idx] = new_val
            neighbor.sort()
            
        elif move_type == 3:
            # Swap: remove one element and add another
            if n <= 1:
                return None
            idx = random.randint(1, n - 1)
            removed_val = neighbor.pop(idx)
            max_val = max(neighbor) if neighbor else 0
            upper_bound = max(2 * max_val + 15, 30)
            new_val = random.randint(1, upper_bound)
            if new_val not in neighbor:
                neighbor.append(new_val)
                neighbor.sort()
            else:
                neighbor.append(removed_val)
                neighbor.sort()
                return None
                
        elif move_type == 4:
            # Double swap: remove two elements, add two new ones
            if n <= 2:
                return None
            idx1, idx2 = random.sample(range(1, n), 2)
            removed1 = neighbor.pop(idx1)
            removed2 = neighbor.pop(idx2)
            max_val = max(neighbor) if neighbor else 0
            upper_bound = max(2 * max_val + 15, 30)
            new1 = random.randint(1, upper_bound)
            new2 = random.randint(1, upper_bound)
            if new1 not in neighbor and new2 not in neighbor and new1 != new2:
                neighbor.append(new1)
                neighbor.append(new2)
                neighbor.sort()
            else:
                neighbor.append(removed1)
                neighbor.append(removed2)
                neighbor.sort()
                return None
                
        elif move_type == 5:
            # Scale: multiply all non-zero elements by a small factor
            if n <= 1:
                return None
            factor = random.choice([2, 3])
            new_neighbor = [0] + [x * factor for x in neighbor[1:]]
            # Remove duplicates
            new_neighbor = sorted(set(new_neighbor))
            if len(new_neighbor) != n:
                return None
            neighbor = new_neighbor
            neighbor.sort()
            
        else:
            return None
        
        if not neighbor or 0 not in neighbor:
            return None
            
        return neighbor
    
    # Phase 1: Hill climbing from best initial set
    current_set = list(best_overall_set)
    current_score = best_overall_score
    no_improve_count = 0
    max_no_improve = 300
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        move_type = random.randint(0, 5)
        neighbor = generate_neighbor(current_set, move_type)
        
        if neighbor is None:
            no_improve_count += 1
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
            # Accept worse moves with small probability
            if random.random() < 0.01:
                current_set = neighbor
                current_score = new_score
    
    # Phase 2: Multiple restarts from diverse starting points
    if time.time() < deadline:
        for restart in range(3):
            if time.time() > deadline - 20:
                break
            
            # Choose a random starting point
            if random.random() < 0.5:
                start_n = random.randint(4, 10)
                if start_n in golomb_rulers:
                    current_set = list(golomb_rulers[start_n])
                else:
                    current_set = list(best_overall_set)
            else:
                current_set = list(best_overall_set)
            
            current_score = compute_score(current_set)
            no_improve_count = 0
            max_no_improve = 200
            
            while time.time() < deadline - 10 and no_improve_count < max_no_improve:
                move_type = random.randint(0, 5)
                neighbor = generate_neighbor(current_set, move_type)
                
                if neighbor is None:
                    no_improve_count += 1
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
                    if random.random() < 0.008:
                        current_set = neighbor
                        current_score = new_score
    
    # Phase 3: Final polish - try all single-element modifications
    if time.time() < deadline - 5:
        current_set = list(best_overall_set)
        current_score = best_overall_score
        improved = True
        
        while improved and time.time() < deadline - 2:
            improved = False
            n = len(current_set)
            
            # Try removing each element
            for i in range(1, n):
                test_set = current_set[:i] + current_set[i+1:]
                if test_set and 0 in test_set:
                    test_score = compute_score(test_set)
                    if test_score > current_score + 1e-12:
                        current_set = test_set
                        current_score = test_score
                        improved = True
                        break
            
            if improved:
                continue
                
            # Try adding new elements
            max_val = max(current_set)
            for new_val in range(1, max_val + 10):
                if new_val not in current_set:
                    test_set = sorted(current_set + [new_val])
                    test_score = compute_score(test_set)
                    if test_score > current_score + 1e-12:
                        current_set = test_set
                        current_score = test_score
                        improved = True
                        break
            
            if improved:
                continue
                
            # Try modifying each element
            for i in range(1, n):
                old_val = current_set[i]
                for new_val in range(max(1, old_val - 5), old_val + 6):
                    if new_val != old_val and new_val not in current_set:
                        test_set = current_set[:i] + [new_val] + current_set[i+1:]
                        test_set = sorted(test_set)
                        if len(test_set) == n:
                            test_score = compute_score(test_set)
                            if test_score > current_score + 1e-12:
                                current_set = test_set
                                current_score = test_score
                                improved = True
                                break
                if improved:
                    break
        
        if current_score > best_overall_score:
            best_overall_score = current_score
            best_overall_set = current_set
    
    return best_overall_set
# EVOLVE-BLOCK-END
