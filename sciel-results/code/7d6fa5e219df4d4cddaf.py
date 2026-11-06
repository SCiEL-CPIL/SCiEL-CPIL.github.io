import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    base = [1, 2, 5, 11]
    base_set = set(base)
    max_val = 284
    max_size = 20
    
    def calculate_cost(marks):
        """
        Returns the number of duplicate sums.
        Also checks if all marks are in [1, 284] and distinct.
        """
        if len(marks) != len(set(marks)):
            return float('inf')
        if min(marks) < 1 or max(marks) > max_val:
            return float('inf')
        
        sum_counts = {}
        n = len(marks)
        for i in range(n):
            a = marks[i]
            for j in range(i, n):
                b = marks[j]
                s = a + b
                if s in sum_counts:
                    sum_counts[s] += 1
                else:
                    sum_counts[s] = 1
        
        collisions = 0
        for count in sum_counts.values():
            if count > 1:
                collisions += count - 1
        return collisions

    def random_marks(size, base):
        """
        Generate a random set of 'size' distinct integers in [1, 284]
        that includes the base elements.
        """
        marks = list(base)
        remaining_size = size - len(base)
        available = [i for i in range(1, max_val + 1) if i not in base_set]
        random.shuffle(available)
        marks.extend(available[:remaining_size])
        random.shuffle(marks)
        return marks

    best_solution = None
    best_size_found = len(base)
    
    # Try to find a solution of size 20, then 19, etc.
    for target_size in range(max_size, len(base), -1):
        if time.time() > deadline - 5:
            break
            
        # Try multiple restarts for this target size
        found = False
        restarts = 0
        max_restarts = 50 # Limit restarts to avoid wasting time on one size
        
        while not found and restarts < max_restarts and time.time() < deadline - 5:
            restarts += 1
            
            # Initialize random marks
            marks = random_marks(target_size, base)
            current_cost = calculate_cost(marks)
            
            # Simulated Annealing
            # If cost is 0, we are done
            if current_cost == 0:
                best_solution = sorted(marks)
                best_size_found = target_size
                found = True
                break
                
            temperature = 10.0
            cooling_rate = 0.99
            min_temperature = 0.1
            max_steps = 5000
            
            for step in range(max_steps):
                if time.time() > deadline - 2:
                    break
                    
                # Pick a random non-base mark to move
                non_base_indices = [i for i, m in enumerate(marks) if m not in base_set]
                if not non_base_indices:
                    break
                idx = random.choice(non_base_indices)
                old_val = marks[idx]
                
                # New random value
                new_val = random.randint(1, max_val)
                
                # Ensure new_val is not equal to any other mark
                if new_val in set(marks) - {old_val}:
                    continue
                    
                # Calculate new cost
                # To speed up, we can update the sum counts incrementally, 
                # but for N=20, full recalculation is O(N^2) = 400 ops, which is fast.
                marks[idx] = new_val
                new_cost = calculate_cost(marks)
                
                # Acceptance criterion
                if new_cost < current_cost:
                    current_cost = new_cost
                else:
                    # Metropolis criterion
                    delta = new_cost - current_cost
                    if delta < 0:
                        pass
                    else:
                        accept_prob = 1.0 / (1.0 + delta / temperature)
                        if random.random() > accept_prob:
                            # Reject move
                            marks[idx] = old_val
                            continue
                        else:
                            current_cost = new_cost
                
                if current_cost == 0:
                    best_solution = sorted(marks)
                    best_size_found = target_size
                    found = True
                    break
                    
                temperature *= cooling_rate
                if temperature < min_temperature:
                    break
            
        if found:
            break

    if best_solution is None:
        # Fallback: return the base set if nothing better found
        return sorted(base)
        
    return best_solution
