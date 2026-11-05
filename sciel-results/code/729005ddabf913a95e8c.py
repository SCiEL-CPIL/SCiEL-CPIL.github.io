import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = list(range(min_sum, max_sum + 1))
    num_targets = len(target_sums)  # 129

    def get_counts(A):
        """Calculate counts for all sums up to max_sum"""
        A_list = list(A)
        k = len(A_list)
        counts = np.zeros(max_sum + 1, dtype=int)
        for i in range(k):
            ai = A_list[i]
            # Self pair
            s = 2 * ai
            if s <= max_sum:
                counts[s] += 1
            # Pairs with others
            for j in range(i + 1, k):
                aj = A_list[j]
                s = ai + aj
                if s <= max_sum:
                    counts[s] += 1
        return counts

    def calculate_cost(A, counts):
        """
        Returns a cost tuple: (uncovered, max_mult, sum_sq)
        We want to minimize this lexicographically.
        """
        target_counts = counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            # High penalty for uncovered
            max_mult = int(np.max(target_counts))
            # Sum of squares to measure spread
            sum_sq = np.sum(target_counts ** 2)
            return (1000 + uncovered, max_mult, sum_sq)
        
        max_mult = int(np.max(target_counts))
        sum_sq = np.sum(target_counts ** 2)
        return (0, max_mult, sum_sq)

    def incremental_swap(A, x_in, x_out, counts):
        """
        Simulate swapping x_in (in A) with x_out (not in A).
        Returns new_counts.
        """
        new_counts = counts.copy()
        
        # Remove x_in contributions
        s = 2 * x_in
        if s <= max_sum:
            new_counts[s] -= 1
        for y in A:
            if y == x_in:
                continue
            s = x_in + y
            if s <= max_sum:
                new_counts[s] -= 1

        # Add x_out contributions
        s = 2 * x_out
        if s <= max_sum:
            new_counts[s] += 1
        for y in A:
            if y == x_in:
                continue
            s = x_out + y
            if s <= max_sum:
                new_counts[s] += 1

        return new_counts

    def incremental_add(A, x_add, counts):
        """
        Simulate adding x_add (not in A) to A.
        Returns new_counts.
        """
        new_counts = counts.copy()
        s = 2 * x_add
        if s <= max_sum:
            new_counts[s] += 1
        for y in A:
            s = x_add + y
            if s <= max_sum:
                new_counts[s] += 1
        return new_counts

    def incremental_remove(A, x_rem, counts):
        """
        Simulate removing x_rem (in A) from A.
        Returns new_counts.
        """
        new_counts = counts.copy()
        s = 2 * x_rem
        if s <= max_sum:
            new_counts[s] -= 1
        for y in A:
            if y == x_rem:
                continue
            s = x_rem + y
            if s <= max_sum:
                new_counts[s] -= 1
        return new_counts

    best_A = None
    best_cost = (10000, 1000, 1000000) # (uncovered, max_mult, sum_sq)

    # Try multiple random restarts
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Choose a size for A
        # k=22 gives avg mult ~2. k=20 gives avg ~1.5.
        # Let's try k in [18, 28]
        k = random.randint(18, 28)
        
        # Generate random set
        # Bias towards covering the range?
        # Just random subset of [0, 128]
        A = set(random.sample(range(129), k))
        
        # Ensure we have some structure?
        # Maybe start with evenly spaced and perturb?
        if r % 2 == 0:
            # Evenly spaced
            base = np.linspace(0, 128, k, dtype=int)
            A = set(int(x) for x in base)
            # Add some noise
            if len(A) < k:
                # handle duplicates
                A = set()
                used = set()
                for x in base:
                    xx = int(x)
                    while xx in used:
                        xx += 1
                        if xx > 128: xx = 0
                    A.add(xx)
                    used.add(xx)
            # Perturb 20% of elements
            A_list = list(A)
            for _ in range(k // 5):
                idx = random.randint(0, k-1)
                old_val = A_list[idx]
                new_val = random.randint(max(0, old_val-10), min(128, old_val+10))
                if new_val not in A:
                    A.remove(old_val)
                    A.add(new_val)
                    A_list[idx] = new_val
        else:
            # Pure random
            A = set(random.sample(range(129), k))

        current_counts = get_counts(A)
        current_cost = calculate_cost(A, current_counts)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_A = set(A)

        # Local Search
        A = set(A)
        current_counts = get_counts(A)
        current_cost = calculate_cost(A, current_counts)
        
        outside = list(set(range(129)) - A)
        A_list = list(A)
        
        stagnation_counter = 0
        max_stagnation = 500
        
        while time.time() < deadline - 5.0:
            if stagnation_counter > max_stagnation:
                break
                
            move_type = random.random()
            
            new_cost = None
            new_counts = None
            move_applied = False
            
            if move_type < 0.7 and len(outside) > 0:
                # Swap
                x_in = random.choice(A_list)
                x_out = random.choice(outside)
                
                new_counts = incremental_swap(A, x_in, x_out, current_counts)
                new_cost = calculate_cost(A, new_counts)
                
                if new_cost < current_cost:
                    # Apply
                    A.remove(x_in)
                    A.add(x_out)
                    outside.remove(x_out)
                    outside.append(x_in)
                    A_list.remove(x_in)
                    A_list.append(x_out)
                    
                    current_counts = new_counts
                    current_cost = new_cost
                    move_applied = True
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                        
            elif move_type < 0.85 and len(A) > 15:
                # Remove
                x_rem = random.choice(A_list)
                
                new_counts = incremental_remove(A, x_rem, current_counts)
                new_cost = calculate_cost(A, new_counts)
                
                if new_cost < current_cost:
                    A.remove(x_rem)
                    outside.append(x_rem)
                    A_list.remove(x_rem)
                    
                    current_counts = new_counts
                    current_cost = new_cost
                    move_applied = True
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)

            elif move_type < 1.0 and len(A) < 35:
                # Add
                x_add = random.choice(outside)
                
                new_counts = incremental_add(A, x_add, current_counts)
                new_cost = calculate_cost(A, new_counts)
                
                if new_cost < current_cost:
                    A.add(x_add)
                    outside.remove(x_add)
                    A_list.append(x_add)
                    
                    current_counts = new_counts
                    current_cost = new_cost
                    move_applied = True
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)

            if not move_applied:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

    # Final verification
    if best_A is None:
        best_A = set(range(0, 129, 8))
        
    return list(best_A)
