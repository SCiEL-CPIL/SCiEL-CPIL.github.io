import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
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
        # But we want to try to reach size k.
        marks = list(prescribed)
        pool = [x for x in range(1, N + 1) if x not in prescribed_set]
        random.shuffle(pool)
        
        # If k is larger than prescribed, add randoms
        if k > len(prescribed):
            marks.extend(pool[:k - len(prescribed)])
            
        # If k is smaller (not expected in this loop, but for safety)
        # The loop goes 20 down to 4.
        
        # Calculate initial conflicts
        sum_count = {}
        n = len(marks)
        for i in range(n):
            for j in range(i, n):
                s = marks[i] + marks[j]
                sum_count[s] = sum_count.get(s, 0) + 1
                
        def get_conflicts(counts):
            c = 0
            for val in counts.values():
                if val > 1:
                    c += val * (c - 1) // 2 # Wait, this is wrong. 
                    # Conflicts are pairs of pairs.
                    # If a sum appears 'val' times, it contributes val*(val-1)/2 to the conflict count?
                    # No, the objective is usually just the number of sums that are not unique?
                    # Or the number of "bad" pairs?
                    # Let's stick to: sum of (count - 1) for all counts > 1? 
                    # Or just count of sums where count > 1?
                    # Minimizing the number of duplicate sums is a good proxy.
                    # Let's use: sum(1 for val in counts.values() if val > 1)
                    pass
            return 0 # Placeholder

        # Let's define a better conflict metric:
        # Number of pairs of indices (i,j) and (k,l) such that i<=j, k<=l, (i,j)!=(k,l) and marks[i]+marks[j] == marks[k]+marks[l].
        # This is equivalent to sum over all sums s of C(count[s], 2).
        
        def calc_conflicts(counts):
            c = 0
            for val in counts.values():
                if val > 1:
                    c += val * (val - 1) // 2
            return c

        current_conflicts = calc_conflicts(sum_count)
        if current_conflicts == 0:
            return marks

        best_conflicts = current_conflicts
        best_marks = list(marks)
        
        T = 10.0
        cooling = 0.999
        min_T = 0.001
        
        steps = 0
        max_steps = 1000000 # High limit, controlled by time
        
        # Use a set for fast membership check of current marks
        marks_set = set(marks)
        
        while T > min_T:
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
            # Remove old sums involving old_val
            old_sums = []
            for j in range(k):
                if j == idx:
                    continue
                s = old_val + marks[j]
                old_sums.append(s)
            old_sums.append(old_val + old_val)
            
            new_sums = []
            for j in range(k):
                if j == idx:
                    continue
                s = new_val + marks[j]
                new_sums.append(s)
            new_sums.append(new_val + new_val)
            
            # Update sum_count
            for s in old_sums:
                sum_count[s] -= 1
                if sum_count[s] == 0:
                    del sum_count[s]
                    
            for s in new_sums:
                sum_count[s] = sum_count.get(s, 0) + 1
                
            new_conflicts = calc_conflicts(sum_count)
            
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
    # Try 20 with 60s.
    # Try 19 with 30s.
    # Try 18 with 10s.
    # Etc.
    
    time_alloc = {
        20: 60,
        19: 30,
        18: 15,
        17: 5,
        16: 2,
        15: 1,
        14: 0.5,
        13: 0.5,
        12: 0.5,
        11: 0.5,
        10: 0.5,
        9: 0.5,
        8: 0.5,
        7: 0.5,
        6: 0.5,
        5: 0.5,
        4: 0.5
    }
    
    for k in range(20, 3, -1):
        if time.time() > deadline:
            break
        
        budget = time_alloc.get(k, 0.5)
        # Adjust budget if we are running low on total time
        remaining = deadline - time.time()
        if remaining < budget:
            budget = max(0.1, remaining * 0.5) # Use half remaining time
            
        result = solve_for_size(k, budget)
        if result is not None:
            if is_valid_sidon(result):
                return result
        
    # Fallback
    return prescribed

if __name__ == "__main__":
    res = construct()
    print(len(res))
