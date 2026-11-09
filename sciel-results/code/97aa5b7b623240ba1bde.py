import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
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

    def count_conflicts_sums(marks):
        """Count number of duplicate sums (conflicts)."""
        sums = {}
        n = len(marks)
        for i in range(n):
            for j in range(i, n):
                s = marks[i] + marks[j]
                sums[s] = sums.get(s, 0) + 1
        conflicts = 0
        for c in sums.values():
            if c > 1:
                conflicts += c * (c - 1) // 2
        return conflicts

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
        
        # Calculate initial conflicts
        current_conflicts = count_conflicts_sums(marks)
        
        if current_conflicts == 0:
            return marks
            
        best_conflicts = current_conflicts
        best_marks = list(marks)
        
        # Simulated Annealing
        T = 10.0
        cooling = 0.9999
        min_T = 0.01
        
        # We maintain sum_count to update conflicts in O(k)
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
        
        steps = 0
        max_steps = 500000
        
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
            if new_val in marks:
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
                marks[idx] = new_val
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
    for k in range(20, len(prescribed) - 1, -1):
        if time.time() > deadline:
            break
        
        time_budget = 110 - (time.time() - start_time)
        if time_budget < 5:
            break
        
        # Multiple restarts for better chance
        max_restarts = min(5, max(2, int(time_budget / 15)))
        
        result = None
        for r in range(max_restarts):
            if time.time() > start_time + 110:
                break
            result = solve_for_size(k, 110 - (time.time() - start_time))
            if result is not None:
                if is_valid_sidon(result):
                    return result
        
    # Fallback: return the prescribed set if we run out of time or fail
    return prescribed

if __name__ == "__main__":
    res = construct()
    print(len(res))
