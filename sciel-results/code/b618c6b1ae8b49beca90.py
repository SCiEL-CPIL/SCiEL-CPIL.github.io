import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80  # 80s out of 90s, safety margin
    
    def greedy_add(order):
        """Greedy addition in given order. Returns (set, reachable_sums)."""
        A = set()
        reachable_sums = {0}
        
        for x in order:
            if time.time() > deadline:
                break
            target_rem = 200 - x
            if target_rem in reachable_sums:
                continue
            else:
                A.add(x)
                new_sums = set()
                for s in reachable_sums:
                    new_s = s + x
                    if new_s <= 200:
                        new_sums.add(new_s)
                reachable_sums.update(new_sums)
        
        return A, reachable_sums
    
    def local_improve(A, reachable_sums, excluded):
        """Try to add excluded elements. Returns (A, reachable_sums)."""
        if not excluded:
            return A, reachable_sums
        
        improved = False
        for e in list(excluded):
            if time.time() > deadline:
                break
            target_rem = 200 - e
            if target_rem not in reachable_sums:
                A.add(e)
                excluded.discard(e)
                new_sums = set()
                for s in reachable_sums:
                    new_s = s + e
                    if new_s <= 200:
                        new_sums.add(new_s)
                reachable_sums.update(new_sums)
                improved = True
                break
        
        if improved:
            return local_improve(A, reachable_sums, excluded)
        
        return A, reachable_sums
    
    all_numbers = list(range(1, 101))
    
    # Try multiple greedy orders
    orders = [
        list(range(100, 0, -1)),  # Descending
        list(range(1, 101)),      # Ascending
    ]
    
    # Add random orders with fixed seed
    rng = random.Random(42)
    for _ in range(5):
        order = all_numbers[:]
        rng.shuffle(order)
        orders.append(order)
    
    best_A = set()
    best_size = 0
    
    for order in orders:
        if time.time() > deadline:
            break
        
        A, reachable_sums = greedy_add(order)
        excluded = set(all_numbers) - A
        
        # Local improvement: try to add excluded elements
        A, reachable_sums = local_improve(A, reachable_sums, excluded)
        
        if len(A) > best_size:
            best_size = len(A)
            best_A = A.copy()
    
    # Ensure non-empty
    if not best_A:
        return [100]
        
    return list(best_A)
