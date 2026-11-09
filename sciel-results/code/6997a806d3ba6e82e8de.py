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

    def greedy_init(k, prescribed, prescribed_set, N, rng):
        """Greedy initialization: start with prescribed, add elements that cause fewest new conflicts."""
        marks = list(prescribed)
        marks_set = set(marks)
        sum_count = {}
        
        # Build initial sum counts
        n = len(marks)
        for i in range(n):
            for j in range(i, n):
                s = marks[i] + marks[j]
                sum_count[s] = sum_count.get(s, 0) + 1
        
        pool = [x for x in range(1, N + 1) if x not in marks_set]
        rng.shuffle(pool)
        
        for x in pool:
            if len(marks) >= k:
                break
            # Check if adding x causes any new conflict
            conflict = False
            for j in range(len(marks)):
                s = x + marks[j]
                if sum_count.get(s, 0) >= 1:
                    conflict = True
                    break
            if not conflict:
                # Also check x+x
                if sum_count.get(x + x, 0) >= 1:
                    conflict = True
            if not conflict:
                marks.append(x)
                marks_set.add(x)
                for j in range(len(marks) - 1):
                    s = x + marks[j]
                    sum_count[s] = sum_count.get(s, 0) + 1
                sum_count[x + x] = sum_count.get(x + x, 0) + 1
        
        # If we still need more, fill randomly
        if len(marks) < k:
            remaining = [x for x in range(1, N + 1) if x not in marks_set]
            rng.shuffle(remaining)
            for x in remaining:
                if len(marks) >= k:
                    break
                marks.append(x)
                marks_set.add(x)
        
        return marks[:k]

    def solve_for_size(k, time_budget, rng):
        """
        Try to find a valid Sidon set of size k using simulated annealing with O(1) conflict updates.
        """
        if k < len(prescribed):
            return None
            
        # Greedy initialization
        marks = greedy_init(k, prescribed, prescribed_set, N, rng)
        
        # Build sum_count and compute initial conflicts
        sum_count = {}
        n = len(marks)
        for i in range(n):
            for j in range(i, n):
                s = marks[i] + marks[j]
                sum_count[s] = sum_count.get(s, 0) + 1
        
        # Compute total conflicts
        total_conflicts = 0
        for c in sum_count.values():
            if c > 1:
                total_conflicts += c * (c - 1) // 2
        
        if total_conflicts == 0:
            return marks
            
        best_conflicts = total_conflicts
        best_marks = list(marks)
        
        # Simulated Annealing with O(1) delta computation
        T = 5.0
        cooling = 0.99995
        min_T = 0.005
        
        marks_set = set(marks)
        prescribed_in_marks = [m in prescribed_set for m in marks]
        
        steps = 0
        max_steps = 2000000
        no_improve_steps = 0
        max_no_improve = 50000
        
        while T > min_T and steps < max_steps and no_improve_steps < max_no_improve:
            if time.time() > start_time + time_budget:
                break
                
            # Pick a random index to change (prefer non-prescribed)
            idx = rng.randint(0, k - 1)
            if prescribed_in_marks[idx]:
                T *= cooling
                continue
                
            old_val = marks[idx]
            
            # Pick a new value
            new_val = rng.randint(1, N)
            if new_val == old_val:
                T *= cooling
                continue
                
            # Check if new_val is already in marks (excluding idx)
            if new_val in marks_set:
                T *= cooling
                continue
                
            # Calculate delta conflicts in O(k) time
            delta = 0
            
            # Remove old_val's contributions
            for j in range(k):
                if j == idx:
                    s = old_val + old_val
                else:
                    s = old_val + marks[j]
                c = sum_count.get(s, 0)
                if c <= 1:
                    pass
                else:
                    delta -= (c - 1)
                if s in sum_count:
                    sum_count[s] = c - 1
                    if sum_count[s] == 0:
                        del sum_count[s]
            
            # Add new_val's contributions
            for j in range(k):
                if j == idx:
                    s = new_val + new_val
                else:
                    s = new_val + marks[j]
                c = sum_count.get(s, 0)
                if c >= 1:
                    delta += c
                sum_count[s] = c + 1
            
            new_conflicts = total_conflicts + delta
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                # Accept move
                marks[idx] = new_val
                marks_set.discard(old_val)
                marks_set.add(new_val)
                total_conflicts = new_conflicts
                
                if total_conflicts < best_conflicts:
                    best_conflicts = total_conflicts
                    best_marks = list(marks)
                    no_improve_steps = 0
                    
                    if total_conflicts == 0:
                        return marks
                else:
                    no_improve_steps += 1
            else:
                # Revert
                for j in range(k):
                    if j == idx:
                        s = old_val + old_val
                    else:
                        s = old_val + marks[j]
                    c = sum_count.get(s, 0)
                    sum_count[s] = c + 1
                
                for j in range(k):
                    if j == idx:
                        s = new_val + new_val
                    else:
                        s = new_val + marks[j]
                    c = sum_count.get(s, 0)
                    if c == 1:
                        del sum_count[s]
                    else:
                        sum_count[s] = c - 1
                        
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
        max_restarts = min(10, max(2, int(time_budget / 10)))
        
        result = None
        for r in range(max_restarts):
            if time.time() > start_time + 110:
                break
            rng = random.Random(hash((k, r)) & 0xFFFFFFFF)
            result = solve_for_size(k, 110 - (time.time() - start_time), rng)
            if result is not None:
                if is_valid_sidon(result):
                    return result
        
    # Fallback: return the prescribed set if we run out of time or fail
    return prescribed

if __name__ == "__main__":
    res = construct()
    print(len(res))
