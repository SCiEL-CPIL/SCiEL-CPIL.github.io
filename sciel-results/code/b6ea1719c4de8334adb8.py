import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    N = 284
    
    def is_valid_sidon(marks):
        if len(marks) != len(set(marks)):
            return False
        if any(m < 1 or m > N for m in marks):
            return False
        if not prescribed_set.issubset(set(marks)):
            return False
        sums = {}
        n = len(marks)
        for i in range(n):
            for j in range(i, n):
                s = marks[i] + marks[j]
                if s in sums:
                    return False
                sums[s] = 1
        return True

    def solve_for_size(k, time_budget, rng):
        if k < len(prescribed):
            return None
            
        # Initialization: Prescribed + Greedy
        marks = list(prescribed)
        marks_set = set(marks)
        available = [x for x in range(1, N + 1) if x not in marks_set]
        
        # Greedy init to get a good starting point
        sum_count = {}
        for i in range(len(marks)):
            for j in range(i, len(marks)):
                s = marks[i] + marks[j]
                sum_count[s] = sum_count.get(s, 0) + 1
                
        # Sort available by some heuristic? Maybe just random order but check conflicts.
        # To speed up, we can just iterate through 1..N and add if valid.
        # This ensures we find a maximal set quickly.
        for x in range(1, N + 1):
            if len(marks) >= k:
                break
            if x in marks_set:
                continue
            ok = True
            if sum_count.get(x + x, 0) > 0:
                ok = False
            if ok:
                for m in marks:
                    if sum_count.get(x + m, 0) > 0:
                        ok = False
                        break
            if ok:
                marks.append(x)
                marks_set.add(x)
                sum_count[x + x] = sum_count.get(x + x, 0) + 1
                for m in marks[:-1]:
                    s = x + m
                    sum_count[s] = sum_count.get(s, 0) + 1
        
        # If we didn't reach k, fill with random (should be rare for k<=20)
        if len(marks) < k:
            remaining = [x for x in range(1, N + 1) if x not in marks_set]
            rng.shuffle(remaining)
            for x in remaining:
                if len(marks) >= k:
                    break
                marks.append(x)
                marks_set.add(x)
                
        # Rebuild sum_count for the full set
        sum_count = {}
        n = len(marks)
        for i in range(n):
            for j in range(i, n):
                s = marks[i] + marks[j]
                sum_count[s] = sum_count.get(s, 0) + 1
                
        current_conflicts = sum(c * (c - 1) // 2 for c in sum_count.values())
        
        if current_conflicts == 0:
            return marks

        best_conflicts = current_conflicts
        best_marks = list(marks)
        
        # Simulated Annealing
        T = 10.0
        cooling = 0.9995
        min_T = 0.01
        
        marks_set = set(marks)
        prescribed_indices = [i for i, m in enumerate(marks) if m in prescribed_set]
        
        steps = 0
        max_steps = 500000 # Increased steps
        no_improve = 0
        max_no_improve = 50000 # Increased patience
        
        while T > min_T and steps < max_steps and no_improve < max_no_improve:
            if time.time() > start_time + time_budget:
                break
                
            # Pick random non-prescribed index
            idx = rng.randint(0, k - 1)
            if idx in prescribed_indices:
                T *= cooling
                continue
                
            old_val = marks[idx]
            
            # Pick new value
            new_val = rng.randint(1, N)
            if new_val == old_val or new_val in marks_set:
                T *= cooling
                continue
                
            # Calculate Delta
            delta = 0
            
            # Remove old
            for j in range(k):
                if j == idx:
                    s = old_val + old_val
                else:
                    s = old_val + marks[j]
                c = sum_count.get(s, 0)
                if c > 1:
                    delta -= (c - 1)
                sum_count[s] = c - 1
                if sum_count[s] == 0:
                    del sum_count[s]
                    
            # Add new
            for j in range(k):
                if j == idx:
                    s = new_val + new_val
                else:
                    s = new_val + marks[j]
                c = sum_count.get(s, 0)
                if c >= 1:
                    delta += c
                sum_count[s] = c + 1
                
            new_conflicts = current_conflicts + delta
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                # Accept
                marks[idx] = new_val
                marks_set.discard(old_val)
                marks_set.add(new_val)
                current_conflicts = new_conflicts
                
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    best_marks = list(marks)
                    no_improve = 0
                    
                    if current_conflicts == 0:
                        return marks
                else:
                    no_improve += 1
            else:
                # Revert
                # Re-add old
                for j in range(k):
                    if j == idx:
                        s = old_val + old_val
                    else:
                        s = old_val + marks[j]
                    c = sum_count.get(s, 0)
                    sum_count[s] = c + 1
                # Remove new
                for j in range(k):
                    if j == idx:
                        s = new_val + new_val
                    else:
                        s = new_val + marks[j]
                    c = sum_count.get(s, 0)
                    sum_count[s] = c - 1
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
            
        time_elapsed = time.time() - start_time
        time_remaining = 110 - time_elapsed
        if time_remaining < 10:
            break
            
        # Allocate time based on k. Higher k gets more time.
        # We want to spend most time on k=20.
        # If k=20, spend 60s. If k=19, spend 30s. etc.
        if k == 20:
            budget_for_k = min(60, time_remaining)
        elif k == 19:
            budget_for_k = min(30, time_remaining)
        else:
            budget_for_k = min(10, time_remaining)
            
        # Multiple restarts
        max_restarts = 10
        for r in range(max_restarts):
            if time.time() > start_time + 110:
                break
            rng = random.Random(hash((k, r)) & 0xFFFFFFFF)
            result = solve_for_size(k, 110 - (time.time() - start_time), rng)
            if result is not None:
                if is_valid_sidon(result):
                    return result
        # If we failed for this k, try next smaller k

    # Fallback
    return prescribed

if __name__ == "__main__":
    res = construct()
    print(len(res))
