import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    n = 128
    target_lo = 128
    target_hi = 256
    target_range = list(range(target_lo, target_hi + 1))
    
    best_A = None
    best_peak = float('inf')
    best_uncovered = float('inf')
    
    # Helper to calculate score
    def get_score(A_set, counts):
        uncovered = 0
        peak = 0
        for s in target_range:
            c = counts[s]
            if c == 0:
                uncovered += 1
            elif c > peak:
                peak = c
        # We prioritize coverage. Then minimize peak.
        # Score for comparison: (uncovered, peak)
        return uncovered, peak

    # Helper to update counts for adding/removing x
    # A is a set. counts is a list/array.
    # This function returns the list of changed sums and their delta?
    # It's faster to just recompute the affected parts or keep a global counts array and update it.
    
    def apply_move(A_list, A_set, counts, x, y):
        """
        Swap x (in A) with y (not in A).
        Or if y is None, remove x.
        Or if x is None, add y.
        Returns new A_list, A_set, counts, and whether move was applied (for rollback)
        """
        # We will modify counts in place and return the changes to allow rollback
        # Changes: list of (sum, delta)
        changes = []
        
        if x is not None:
            # Remove x
            A_set.remove(x)
            A_list.remove(x)
            for z in A_list:
                s = x + z
                if target_lo <= s <= target_hi:
                    counts[s] -= 1
                    changes.append((s, 1)) # delta was -1, so to rollback add 1
            s = 2 * x
            if target_lo <= s <= target_hi:
                counts[s] -= 1
                changes.append((s, 1))
        
        if y is not None:
            # Add y
            A_set.add(y)
            A_list.append(y)
            for z in A_list:
                if z == y:
                    continue
                s = y + z
                if target_lo <= s <= target_hi:
                    counts[s] += 1
                    changes.append((s, -1)) # delta was +1, so to rollback subtract 1
            s = 2 * y
            if target_lo <= s <= target_hi:
                counts[s] += 1
                changes.append((s, -1))
                
        return changes

    def rollback(A_list, A_set, counts, changes):
        for s, delta in changes:
            counts[s] += delta
        # Note: A_list and A_set modifications must be reversed manually by caller
        # This is getting messy. Let's just copy the state for simplicity if k is small?
        # k ~ 40. Copying set and list is fast. Copying counts (size 257) is fast.
        
    # Simpler approach:
    # State: A (list), counts (list)
    # Move: Generate new A, new counts. Compare scores.
    # Cost of generating new counts from scratch: O(k^2). k=40 -> 1600 ops.
    # Cost of local update: O(k). k=40 -> 40 ops.
    # Local update is better.
    
    def calculate_initial_counts(A):
        counts = [0] * 257
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if target_lo <= s <= target_hi:
                    counts[s] += 1
        return counts

    def local_search(initial_A, num_steps):
        A = list(initial_A)
        A_set = set(A)
        counts = calculate_initial_counts(A)
        curr_unc, curr_peak = get_score(A_set, counts)
        curr_score = (curr_unc, curr_peak)
        
        best_A_local = A[:]
        best_score_local = curr_score
        
        for step in range(num_steps):
            if time.time() > deadline:
                break
                
            # Generate random move
            move_type = random.random()
            
            if move_type < 0.7:
                # Swap
                if not A: continue
                x = random.choice(A)
                # Pick y not in A
                y = random.randint(0, n)
                attempts = 0
                while y in A_set and attempts < 10:
                    y = random.randint(0, n)
                    attempts += 1
                if y in A_set:
                    continue
                
                # Apply swap
                old_A = A[:]
                old_counts = counts[:]
                
                # Remove x
                A.remove(x)
                A_set.remove(x)
                for z in A:
                    s = x + z
                    if target_lo <= s <= target_hi:
                        counts[s] -= 1
                s = 2 * x
                if target_lo <= s <= target_hi:
                    counts[s] -= 1
                
                # Add y
                A.append(y)
                A_set.add(y)
                for z in A:
                    if z == y:
                        continue
                    s = y + z
                    if target_lo <= s <= target_hi:
                        counts[s] += 1
                s = 2 * y
                if target_lo <= s <= target_hi:
                    counts[s] += 1
                
                new_unc, new_peak = get_score(A_set, counts)
                new_score = (new_unc, new_peak)
                
                if new_score < curr_score:
                    curr_score = new_score
                    if curr_score < best_score_local:
                        best_score_local = curr_score
                        best_A_local = A[:]
                else:
                    # Rollback
                    A = old_A
                    A_set = set(A)
                    counts = old_counts
                    
            elif move_type < 0.85:
                # Remove
                if not A: continue
                x = random.choice(A)
                old_A = A[:]
                old_counts = counts[:]
                
                A.remove(x)
                A_set.remove(x)
                for z in A:
                    s = x + z
                    if target_lo <= s <= target_hi:
                        counts[s] -= 1
                s = 2 * x
                if target_lo <= s <= target_hi:
                    counts[s] -= 1
                
                new_unc, new_peak = get_score(A_set, counts)
                new_score = (new_unc, new_peak)
                
                if new_score < curr_score:
                    curr_score = new_score
                    if curr_score < best_score_local:
                        best_score_local = curr_score
                        best_A_local = A[:]
                else:
                    A = old_A
                    A_set = set(A)
                    counts = old_counts
                    
            else:
                # Add
                y = random.randint(0, n)
                attempts = 0
                while y in A_set and attempts < 10:
                    y = random.randint(0, n)
                    attempts += 1
                if y in A_set:
                    continue
                    
                old_A = A[:]
                old_counts = counts[:]
                
                A.append(y)
                A_set.add(y)
                for z in A:
                    if z == y:
                        continue
                    s = y + z
                    if target_lo <= s <= target_hi:
                        counts[s] += 1
                s = 2 * y
                if target_lo <= s <= target_hi:
                    counts[s] += 1
                
                new_unc, new_peak = get_score(A_set, counts)
                new_score = (new_unc, new_peak)
                
                if new_score < curr_score:
                    curr_score = new_score
                    if curr_score < best_score_local:
                        best_score_local = curr_score
                        best_A_local = A[:]
                else:
                    A = old_A
                    A_set = set(A)
                    counts = old_counts
                    
        return best_A_local, best_score_local

    # Try different initial sizes
    sizes = [25, 30, 35, 40, 45, 50]
    
    for k in sizes:
        if time.time() > deadline:
            break
            
        # Multiple restarts for each size
        num_restarts = 20
        steps_per_restart = 2000
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Random initial set of size k
            init_A = random.sample(range(n + 1), k)
            
            A_res, score_res = local_search(init_A, steps_per_restart)
            
            if score_res < (best_uncovered, best_peak):
                best_uncovered, best_peak = score_res
                best_A = A_res
                
            # If we found a perfect coverage with very low peak, we can stop early?
            # If best_peak <= 3 and best_uncovered == 0, it's likely near optimal.
            if best_uncovered == 0 and best_peak <= 4:
                break
                
        if best_uncovered == 0 and best_peak <= 4:
            break

    if best_A is None:
        # Fallback to seed
        return list(range(64, 129))
        
    return best_A
