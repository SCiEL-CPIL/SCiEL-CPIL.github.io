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
        current_conflicts = 0
        
        for i in range(n):
            for j in range(i, n):
                s = marks[i] + marks[j]
                # Add sum s
                c = sum_count.get(s, 0)
                current_conflicts += c
                sum_count[s] = c + 1

        if current_conflicts == 0:
            return marks
            
        best_conflicts = current_conflicts
        best_marks = list(marks)
        
        # Simulated Annealing
        T = 10.0
        cooling = 0.99995  # Slower cooling to explore more
        min_T = 0.001
        
        steps = 0
        max_steps = 2000000  # Higher limit, but time will break it
        
        # Pre-compute valid indices (non-prescribed)
        # Actually, we can just check if marks[idx] in prescribed_set
        
        while T > min_T and steps < max_steps:
            if time.time() > start_time + time_budget:
                break
                
            # Pick a random index to change
            idx = random.randint(0, k - 1)
            if marks[idx] in prescribed_set:
                # Try a few times to find a non-prescribed index, or just skip
                # Skipping wastes a step. Better to pick again.
                # But to avoid infinite loop if k is small and all prescribed, 
                # we just continue.
                T *= cooling
                steps += 1
                continue
                
            old_val = marks[idx]
            
            # Pick a new value
            new_val = random.randint(1, N)
            if new_val == old_val:
                T *= cooling
                steps += 1
                continue
                
            # Check if new_val is already in marks (excluding idx)
            # This is O(k). We can use a set for faster lookup?
            # marks is a list. Converting to set every time is O(k).
            # Since k is small (~20), linear scan is fine, but let's keep a set.
            # We'll maintain a set of current values.
            
            # For now, linear scan in the list (excluding idx)
            # Optimization: maintain a set `marks_set`
            # Let's add that to the state
            
            # Update state:
            # 1. Remove old_val from marks_set
            # 2. Check if new_val in marks_set
            # 3. If yes, reject
            # 4. If no, add new_val to marks_set, update sums, update conflicts
            
            # We need to maintain marks_set.
            
            # Let's refactor to maintain marks_set.
            
            # --- Refactored Loop Body ---
            
            # We need marks_set defined outside.
            # Let's restart the function structure to include marks_set.
            
            pass # Placeholder for logic rewrite

    # Rewrite solve_for_size with marks_set
    
    def solve_for_size_optimized(k, time_budget):
        if k < len(prescribed):
            return None
            
        marks = list(prescribed)
        pool = [x for x in range(1, N + 1) if x not in prescribed_set]
        random.shuffle(pool)
        marks.extend(pool[:k - len(prescribed)])
        
        marks_set = set(marks)
        
        sum_count = {}
        n = len(marks)
        current_conflicts = 0
        
        for i in range(n):
            for j in range(i, n):
                s = marks[i] + marks[j]
                c = sum_count.get(s, 0)
                current_conflicts += c
                sum_count[s] = c + 1

        if current_conflicts == 0:
            return marks
            
        best_conflicts = current_conflicts
        best_marks = list(marks)
        
        T = 10.0
        cooling = 0.99995
        min_T = 0.001
        
        steps = 0
        max_steps = 5000000
        
        while T > min_T and steps < max_steps:
            if time.time() > start_time + time_budget:
                break
                
            idx = random.randint(0, k - 1)
            if marks[idx] in prescribed_set:
                T *= cooling
                steps += 1
                continue
                
            old_val = marks[idx]
            new_val = random.randint(1, N)
            
            if new_val == old_val:
                T *= cooling
                steps += 1
                continue
                
            if new_val in marks_set:
                T *= cooling
                steps += 1
                continue
            
            # Calculate delta conflicts
            # Remove old sums
            # Sums involving old_val: (old_val, old_val) and (old_val, marks[j]) for j != idx
            # Note: In the sum_count dict, we store sums for i<=j.
            # When we change marks[idx] to new_val:
            # 1. Remove sums where one element is old_val.
            #    - (old_val, old_val)
            #    - (old_val, marks[j]) for all j != idx.
            # 2. Add sums where one element is new_val.
            #    - (new_val, new_val)
            #    - (new_val, marks[j]) for all j != idx.
            
            # Update sum_count and current_conflicts
            
            # Helper to remove a sum
            def remove_sum(s):
                nonlocal current_conflicts
                c = sum_count.get(s, 0)
                if c == 0:
                    return
                # Removing one instance from count c -> c-1
                # Conflicts change from c(c-1)/2 to (c-1)(c-2)/2
                # Delta = (c-1)(c-2)/2 - c(c-1)/2 = (c-1)(c-2-c)/2 = -(c-1)
                current_conflicts -= (c - 1)
                sum_count[s] = c - 1
                if sum_count[s] == 0:
                    del sum_count[s]
            
            def add_sum(s):
                nonlocal current_conflicts
                c = sum_count.get(s, 0)
                # Adding one instance from count c -> c+1
                # Conflicts change from c(c-1)/2 to (c+1)c/2
                # Delta = c
                current_conflicts += c
                sum_count[s] = c + 1

            # Remove old sums
            # (old_val, old_val)
            remove_sum(old_val + old_val)
            for j in range(k):
                if j == idx:
                    continue
                remove_sum(old_val + marks[j])
            
            # Add new sums
            add_sum(new_val + new_val)
            for j in range(k):
                if j == idx:
                    continue
                add_sum(new_val + marks[j])
            
            # Accept or reject
            if current_conflicts < best_conflicts:
                best_conflicts = current_conflicts
                best_marks = list(marks)
                if best_conflicts == 0:
                    return best_marks
            
            # Metropolis criterion
            # We need the delta. 
            # delta = new_conflicts - old_conflicts
            # We didn't store old_conflicts before removal.
            # Let's store it.
            
            # Refined Loop Body with Delta Tracking
            
            pass # Need to rewrite loop body properly

    # Let's do a clean rewrite of solve_for_size in the final code.

    def solve_for_size_final(k, time_budget):
        if k < len(prescribed):
            return None
            
        marks = list(prescribed)
        pool = [x for x in range(1, N + 1) if x not in prescribed_set]
        random.shuffle(pool)
        marks.extend(pool[:k - len(prescribed)])
        
        marks_set = set(marks)
        
        sum_count = {}
        n = len(marks)
        current_conflicts = 0
        
        for i in range(n):
            for j in range(i, n):
                s = marks[i] + marks[j]
                c = sum_count.get(s, 0)
                current_conflicts += c
                sum_count[s] = c + 1

        if current_conflicts == 0:
            return marks
            
        best_conflicts = current_conflicts
        best_marks = list(marks)
        
        T = 10.0
        cooling = 0.99995
        min_T = 0.001
        
        steps = 0
        max_steps = 10000000
        
        while T > min_T and steps < max_steps:
            if time.time() > start_time + time_budget:
                break
                
            idx = random.randint(0, k - 1)
            if marks[idx] in prescribed_set:
                T *= cooling
                steps += 1
                continue
                
            old_val = marks[idx]
            new_val = random.randint(1, N)
            
            if new_val == old_val or new_val in marks_set:
                T *= cooling
                steps += 1
                continue
            
            # Calculate delta
            # Remove old sums
            old_conflicts = current_conflicts
            
            # (old_val, old_val)
            s = old_val + old_val
            c = sum_count.get(s, 0)
            current_conflicts -= (c - 1)
            sum_count[s] = c - 1
            if sum_count[s] == 0:
                del sum_count[s]
                
            for j in range(k):
                if j == idx:
                    continue
                s = old_val + marks[j]
                c = sum_count.get(s, 0)
                current_conflicts -= (c - 1)
                sum_count[s] = c - 1
                if sum_count[s] == 0:
                    del sum_count[s]
            
            # Add new sums
            s = new_val + new_val
            c = sum_count.get(s, 0)
            current_conflicts += c
            sum_count[s] = c + 1
            
            for j in range(k):
                if j == idx:
                    continue
                s = new_val + marks[j]
                c = sum_count.get(s, 0)
                current_conflicts += c
                sum_count[s] = c + 1
            
            delta = current_conflicts - old_conflicts
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                # Accept
                marks[idx] = new_val
                marks_set.remove(old_val)
                marks_set.add(new_val)
                
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    best_marks = list(marks)
                    if best_conflicts == 0:
                        return best_marks
            else:
                # Revert
                # Remove new sums
                s = new_val + new_val
                c = sum_count.get(s, 0)
                current_conflicts -= (c - 1)
                sum_count[s] = c - 1
                if sum_count[s] == 0:
                    del sum_count[s]
                
                for j in range(k):
                    if j == idx:
                        continue
                    s = new_val + marks[j]
                    c = sum_count.get(s, 0)
                    current_conflicts -= (c - 1)
                    sum_count[s] = c - 1
                    if sum_count[s] == 0:
                        del sum_count[s]
                    
                # Add old sums
                s = old_val + old_val
                c = sum_count.get(s, 0)
                current_conflicts += c
                sum_count[s] = c + 1
                
                for j in range(k):
                    if j == idx:
                        continue
                    s = old_val + marks[j]
                    c = sum_count.get(s, 0)
                    current_conflicts += c
                    sum_count[s] = c + 1
            
            T *= cooling
            steps += 1
            
        if best_conflicts == 0:
            return best_marks
        return None

    # Try sizes from 20 down to 4
    for k in range(20, len(prescribed) - 1, -1):
        if time.time() > deadline:
            break
        
        time_budget = 115 - (time.time() - start_time)
        if time_budget < 5:
            break
        
        # Multiple restarts
        # Since SA is fast now, we can do more restarts or longer runs
        # Let's try 3 restarts of ~30s each? Or just one long run?
        # One long run is usually better for SA.
        
        result = solve_for_size_final(k, time_budget)
        
        if result is not None:
            if is_valid_sidon(result):
                return result
        
    return prescribed

if __name__ == "__main__":
    res = construct()
    print(len(res))
