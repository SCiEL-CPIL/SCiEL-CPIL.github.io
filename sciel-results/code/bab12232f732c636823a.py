import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    N = 284
    
    # We want to maximize the size. The theoretical max is 20.
    # We will try sizes from 20 down to 4.
    
    def count_conflicts(marks):
        """
        Calculate the number of duplicate differences.
        Returns the number of pairs of pairs that have the same difference.
        If diff_count[d] is c, the number of conflicts contributed by d is c*(c-1)/2.
        """
        diff_count = {}
        n = len(marks)
        for i in range(n):
            for j in range(i + 1, n):
                d = abs(marks[i] - marks[j])
                diff_count[d] = diff_count.get(d, 0) + 1
        
        conflicts = 0
        for d, c in diff_count.items():
            if c > 1:
                conflicts += c * (c - 1) // 2
        return conflicts

    def is_valid_sidon(marks):
        """Check if the set is a valid Sidon set."""
        if len(marks) != len(set(marks)):
            return False
        if any(m < 1 or m > N for m in marks):
            return False
        if not prescribed_set.issubset(set(marks)):
            return False
        
        diff_count = {}
        n = len(marks)
        for i in range(n):
            for j in range(i + 1, n):
                d = abs(marks[i] - marks[j])
                if d in diff_count:
                    return False
                diff_count[d] = 1
        return True

    def solve_for_size(k):
        """
        Try to find a valid Sidon set of size k.
        """
        if k < len(prescribed):
            return None
            
        # Initialize
        marks = list(prescribed)
        pool = [x for x in range(1, N + 1) if x not in prescribed_set]
        random.shuffle(pool)
        marks.extend(pool[:k - len(prescribed)])
        
        # Calculate initial conflicts
        current_conflicts = count_conflicts(marks)
        
        if current_conflicts == 0:
            return marks
            
        best_conflicts = current_conflicts
        best_marks = list(marks)
        
        # Simulated Annealing
        T = 10.0
        cooling = 0.9999
        min_T = 0.01
        
        # We maintain diff_count to update conflicts in O(k)
        diff_count = {}
        n = len(marks)
        for i in range(n):
            for j in range(i + 1, n):
                d = abs(marks[i] - marks[j])
                diff_count[d] = diff_count.get(d, 0) + 1
        
        # Helper to get conflicts from diff_count
        def get_conflicts_from_counts(counts):
            c = 0
            for val in counts.values():
                if val > 1:
                    c += val * (val - 1) // 2
            return c
            
        current_conflicts = get_conflicts_from_counts(diff_count)
        
        steps = 0
        max_steps = 500000
        
        while T > min_T and steps < max_steps:
            if time.time() > deadline:
                break
                
            # Pick a random index to change
            idx = random.randint(0, k - 1)
            if marks[idx] in prescribed_set:
                T *= cooling
                continue
                
            old_val = marks[idx]
            
            # Pick a new value
            # To ensure distinctness, we pick a random value and check if it's in the set
            # This is inefficient if k is large, but k is small (<=20)
            new_val = random.randint(1, N)
            if new_val == old_val:
                T *= cooling
                continue
                
            # Check if new_val is already in marks (excluding idx)
            # Since marks is a list, we can just check if new_val is in marks
            # But we need to be careful: if new_val is in marks, it's a duplicate element, which is invalid.
            # The problem requires distinct integers.
            if new_val in marks:
                T *= cooling
                continue
                
            # Calculate delta conflicts
            # Remove old differences
            old_diffs = []
            for j in range(k):
                if j == idx:
                    continue
                d = abs(old_val - marks[j])
                old_diffs.append(d)
                
            new_diffs = []
            for j in range(k):
                if j == idx:
                    continue
                d = abs(new_val - marks[j])
                new_diffs.append(d)
                
            delta_conflicts = 0
            # We need to simulate the changes in diff_count
            # For each old diff, decrement count
            # For each new diff, increment count
            # This is O(k).
            
            # To avoid modifying diff_count if we reject the move, we can calculate delta directly
            # But it's easier to just update and revert if needed.
            # Given the small size, updating a dict is fast.
            
            # Let's just calculate the new conflict count directly for simplicity and safety, 
            # as k is small. O(k^2) is 400 ops. 500k steps * 400 = 200M ops. Might be too slow in Python.
            # Let's stick to the O(k) update.
            
            # We will update diff_count, calculate new_conflicts, and if rejected, revert.
            
            for d in old_diffs:
                diff_count[d] -= 1
                if diff_count[d] == 0:
                    del diff_count[d]
                    
            for d in new_diffs:
                diff_count[d] = diff_count.get(d, 0) + 1
                
            new_conflicts = get_conflicts_from_counts(diff_count)
            
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
                for d in old_diffs:
                    diff_count[d] = diff_count.get(d, 0) + 1
                for d in new_diffs:
                    diff_count[d] -= 1
                    if diff_count[d] == 0:
                        del diff_count[d]
                        
            T *= cooling
            steps += 1
            
        # If we didn't find a perfect solution, return the best one if it's valid? 
        # No, the task requires a valid Sidon set. 
        # If best_conflicts > 0, we cannot return best_marks.
        # We should return None to try a smaller size.
        if best_conflicts == 0:
            return best_marks
        return None

    # Try sizes from 20 down to 4
    for k in range(20, len(prescribed) - 1, -1):
        if time.time() > deadline:
            break
        result = solve_for_size(k)
        if result is not None:
            # Verify one last time
            if is_valid_sidon(result):
                return result
            else:
                # Should not happen
                pass

    # Fallback: return the prescribed set if we run out of time or fail
    return prescribed

if __name__ == "__main__":
    res = construct()
    print(len(res))
