import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    N = 284
    
    def get_conflicts(marks):
        """Returns the number of duplicate sums."""
        sums = {}
        n = len(marks)
        for i in range(n):
            for j in range(i, n):
                s = marks[i] + marks[j]
                sums[s] = sums.get(s, 0) + 1
        conflicts = 0
        for c in sums.values():
            if c > 1:
                conflicts += c - 1 # Or c*(c-1)//2? 
                # "number of duplicate sums" usually means count of pairs that are duplicates.
                # If sum s appears 3 times, there are 3 pairs (0,1), (0,2), (1,2) that are "conflicting" in a pairwise sense?
                # Actually, for the objective function, we just want to minimize the number of "bad" sums.
                # Let's use sum(max(0, c-1)) as a proxy for "excess" sums.
                # Or better, sum(c*(c-1)//2) which is the number of colliding pairs.
                conflicts += c * (c - 1) // 2
        return conflicts

    def is_valid_sidon(marks):
        if len(marks) != len(set(marks)):
            return False
        if any(m < 1 or m > N for m in marks):
            return False
        if not prescribed_set.issubset(set(marks)):
            return False
        return get_conflicts(marks) == 0

    def solve_for_size(k, time_budget, rng):
        if k < len(prescribed):
            return None
            
        # Initialization: Prescribed + Random
        marks = list(prescribed)
        marks_set = set(marks)
        available = [x for x in range(1, N + 1) if x not in marks_set]
        rng.shuffle(available)
        
        # Try to add random elements that don't immediately cause conflicts? 
        # Or just random? Random is safer for SA to explore.
        # Let's do a mix: try to add 'conflict-free' elements first to get a good start.
        # But if we can't find enough, fill with random.
        
        # Greedy-ish init:
        sum_count = {}
        for i in range(len(marks)):
            for j in range(i, len(marks)):
                s = marks[i] + marks[j]
                sum_count[s] = sum_count.get(s, 0) + 1
                
        for x in available:
            if len(marks) >= k:
                break
            # Check if x can be added without creating a conflict
            # x+x and x+marks[j] must not be in sum_count
            ok = True
            if sum_count.get(x + x, 0) > 0:
                ok = False
            if ok:
                for j in range(len(marks)):
                    if sum_count.get(x + marks[j], 0) > 0:
                        ok = False
                        break
            if ok:
                marks.append(x)
                marks_set.add(x)
                # Update sum_count
                sum_count[x + x] = sum_count.get(x + x, 0) + 1
                for j in range(len(marks) - 1):
                    s = x + marks[j]
                    sum_count[s] = sum_count.get(s, 0) + 1
        
        # If still not enough, fill with random
        if len(marks) < k:
            remaining = [x for x in range(1, N + 1) if x not in marks_set]
            rng.shuffle(remaining)
            for x in remaining:
                if len(marks) >= k:
                    break
                marks.append(x)
                marks_set.add(x)
                
        # Rebuild sum_count for the full set (in case greedy didn't finish or we want to be sure)
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
        
        T = 10.0
        cooling = 0.9995
        min_T = 0.01
        
        marks_set = set(marks)
        prescribed_indices = [i for i, m in enumerate(marks) if m in prescribed_set]
        
        steps = 0
        max_steps = 200000
        no_improve = 0
        max_no_improve = 20000
        
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
            # To speed up, we can pick from a random subset or just 1..N
            new_val = rng.randint(1, N)
            if new_val == old_val:
                T *= cooling
                continue
                
            if new_val in marks_set:
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
                # If c > 1, removing one reduces conflicts by (c-1)
                # Specifically, if c was 3, it contributed 3 pairs. Now c becomes 2, contributes 1 pair. Delta = -2.
                # Formula: new_pairs - old_pairs = (c-1)(c-2)/2 - c(c-1)/2 = (c-1)(c-2-c)/2 = (c-1)(-2)/2 = -(c-1)
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
                # Adding one increases c by 1. New pairs - Old pairs = c(c-1)/2 - (c-1)(c-2)/2 = (c-1)(c - (c-2))/2 = (c-1)(2)/2 = c-1?
                # Wait. If c was 0, becomes 1. Pairs 0 -> 0. Delta 0.
                # If c was 1, becomes 2. Pairs 0 -> 1. Delta 1.
                # If c was 2, becomes 3. Pairs 1 -> 3. Delta 2.
                # So delta += c
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
            
        # Allocate time based on k. Higher k gets more time? Or equal?
        # Let's give 10-20 seconds per k, or split remaining time.
        # If we are trying 20, we want to be sure.
        budget_for_k = min(30, time_remaining / (20 - k + 1)) # Rough split
        
        # Multiple restarts
        max_restarts = 5
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
