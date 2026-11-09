import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit (5s buffer)
    
    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    N = 284
    
    def is_valid_sidon(marks):
        """Check if the set is a valid Sidon set (distinct pairwise sums)."""
        if len(marks) != len(set(marks)):
            return False
        if any(m < 1 or m > N for m in marks):
            return False
        if not prescribed_set.issubset(set(marks)):
            return False
        
        sums = set()
        n = len(marks)
        for i in range(n):
            for j in range(i, n):
                s = marks[i] + marks[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def solve_for_size(k, time_budget):
        """
        Try to find a valid Sidon set of size k using simulated annealing.
        """
        if k < len(prescribed):
            return None
            
        # Greedy initialization: start with prescribed, add random elements
        marks = list(prescribed)
        pool = [x for x in range(1, N + 1) if x not in prescribed_set]
        random.shuffle(pool)
        marks.extend(pool[:k - len(prescribed)])
        
        # Calculate initial conflicts and sum counts
        sum_count = {}
        n = len(marks)
        for i in range(n):
            for j in range(i, n):
                s = marks[i] + marks[j]
                sum_count[s] = sum_count.get(s, 0) + 1
                
        def get_conflicts_from_counts(counts):
            c = 0
            for val in counts.values():
                if val > 1:
                    c += val * (val - 1) // 2
            return c
            
        current_conflicts = get_conflicts_from_counts(sum_count)
        
        if current_conflicts == 0:
            return marks
            
        best_conflicts = current_conflicts
        best_marks = list(marks)
        
        # Simulated Annealing
        T = 10.0
        cooling = 0.9999
        min_T = 0.01
        
        # We maintain sum_count to update conflicts in O(k)
        # Note: sum_count is already initialized above
        
        steps = 0
        max_steps = 500000
        
        # Pre-compute a set of current marks for O(1) membership check
        marks_set = set(marks)
        
        while T > min_T and steps < max_steps:
            if time.time() > start_time + time_budget:
                break
                
            # Pick a random index to change
            idx = random.randint(0, k - 1)
            if marks[idx] in prescribed_set:
                T *= cooling
                continue
                
            old_val = marks[idx]
            
            # Pick a new value
            new_val = random.randint(1, N)
            if new_val == old_val:
                T *= cooling
                continue
                
            # Check if new_val is already in marks (excluding idx)
            if new_val in marks_set:
                T *= cooling
                continue
                
            # Calculate delta conflicts
            # Remove old sums
            old_sums = []
            for j in range(k):
                if j == idx:
                    continue
                s = old_val + marks[j]
                old_sums.append(s)
            old_sums.append(old_val + old_val)  # i == j case
            
            new_sums = []
            for j in range(k):
                if j == idx:
                    continue
                s = new_val + marks[j]
                new_sums.append(s)
            new_sums.append(new_val + new_val)  # i == j case
            
            # Update sum_count
            for s in old_sums:
                sum_count[s] -= 1
                if sum_count[s] == 0:
                    del sum_count[s]
                    
            for s in new_sums:
                sum_count[s] = sum_count.get(s, 0) + 1
                
            new_conflicts = get_conflicts_from_counts(sum_count)
            
            delta = new_conflicts - current_conflicts
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                # Accept move
                marks_set.remove(old_val)
                marks[idx] = new_val
                marks_set.add(new_val)
                current_conflicts = new_conflicts
                
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    best_marks = list(marks)
                    
                if current_conflicts == 0:
                    return marks
            else:
                # Revert
                for s in old_sums:
                    sum_count[s] = sum_count.get(s, 0) + 1
                for s in new_sums:
                    sum_count[s] -= 1
                    if sum_count[s] == 0:
                        del sum_count[s]
                        
            T *= cooling
            steps += 1
            
        if best_conflicts == 0:
            return best_marks
        return None

    # Try sizes from 20 down to 4
    # We have ~115s.
    # Allocate time dynamically.
    
    for k in range(20, len(prescribed) - 1, -1):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 1:
            break
        
        # Allocate a portion of remaining time for this size
        # We want to try larger sizes more aggressively if we have time
        # But we also want to guarantee we can try smaller sizes if larger ones fail
        # Let's give a fixed chunk for the first few, then decay
        if k == 20:
            budget = min(60, remaining * 0.6)
        elif k == 19:
            budget = min(30, remaining * 0.4)
        else:
            budget = min(10, remaining * 0.2)
            
        # Multiple restarts for better chance
        # If we have lots of time, do more restarts
        # Each restart takes some time, but we pass the remaining budget to solve_for_size
        # So solve_for_size will stop when it hits the deadline.
        
        # Actually, the current design passes `time_budget` to solve_for_size,
        # but solve_for_size checks `time.time() > start_time + time_budget`.
        # This `time_budget` is relative to `start_time` of construct().
        # So we should pass the absolute deadline or a relative time from now?
        # The current code does: `result = solve_for_size(k, 110 - (time.time() - start_time))`
        # This effectively sets the deadline for this specific call to be `start_time + (110 - elapsed)`
        # which is `start_time + 110 - elapsed`.
        # Since `deadline = start_time + 115`, this is 5s before the hard deadline.
        
        # Let's just use the remaining time minus a small buffer for safety.
        effective_budget = remaining - 0.5
        
        result = solve_for_size(k, effective_budget)
        if result is not None:
            if is_valid_sidon(result):
                return result
        
    # Fallback: return the prescribed set if we run out of time or fail
    return prescribed

if __name__ == "__main__":
    res = construct()
    print(len(res))
