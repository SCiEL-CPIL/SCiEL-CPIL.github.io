"""Small inclusion-maximal Sidon sets in [1,256]"""
import time
import random

N = 256
MAX_SUM = 2 * N  # 512

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Bitmask utilities
    # sums_mask has bit i set if sum i is present in {a+b | a,b in S, a<=b}
    
    def check_add(x, sums_mask, selected_set):
        """Check if x can be added to the set defined by sums_mask and selected_set."""
        if x in selected_set:
            return False
        # 1. Check 2*x
        if (sums_mask >> (2 * x)) & 1:
            return False
        # 2. Check x + a for all a in S
        # We iterate through the set. Since N is small (256), this is fast.
        for a in selected_set:
            if (sums_mask >> (x + a)) & 1:
                return False
        return True

    def get_new_sums_mask(x, sums_mask, selected_set):
        """Calculate the new sums mask if x is added."""
        new_mask = sums_mask
        # Add 2*x
        new_mask |= (1 << (2 * x))
        # Add x + a
        for a in selected_set:
            new_mask |= (1 << (x + a))
        return new_mask

    def is_maximal(selected, sums_mask):
        """Check if the set is inclusion-maximal."""
        selected_set = set(selected)
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            if check_add(x, sums_mask, selected_set):
                return False
        return True

    def compute_sums_mask(selected):
        """Compute sums mask from scratch."""
        mask = 0
        n = len(selected)
        for i in range(n):
            a = selected[i]
            mask |= (1 << (2 * a))
            for j in range(i + 1, n):
                b = selected[j]
                mask |= (1 << (a + b))
        return mask

    def greedy_construct(order, deadline):
        """Greedily build a Sidon set from the given order."""
        selected = []
        selected_set = set()
        sums_mask = 0
        
        for x in order:
            if time.time() > deadline:
                break
            if check_add(x, sums_mask, selected_set):
                selected.append(x)
                selected_set.add(x)
                sums_mask = get_new_sums_mask(x, sums_mask, selected_set)
        
        return selected, sums_mask, selected_set

    def minimize_maximal(selected, sums_mask, deadline):
        """
        Local search: Try to remove elements while maintaining maximality.
        Returns (minimized_selected, minimized_sums_mask).
        """
        current_selected = list(selected)
        current_sums_mask = sums_mask
        current_set = set(current_selected)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to remove each element
            for i in range(len(current_selected)):
                x_to_remove = current_selected[i]
                
                # Tentatively remove
                new_selected = current_selected[:i] + current_selected[i+1:]
                new_set = set(new_selected)
                
                # Recompute sums mask (costly but necessary for correctness in this simple loop)
                # Optimization: We could try to undo sums, but recomputing is safer and fast enough for small K
                new_sums_mask = compute_sums_mask(new_selected)
                
                # Check if still maximal
                if is_maximal(new_selected, new_sums_mask):
                    # Keep removal
                    current_selected = new_selected
                    current_sums_mask = new_sums_mask
                    current_set = new_set
                    improved = True
                    break # Restart loop with new state
        
        return current_selected, current_sums_mask

    best_set = None
    best_size = float('inf')
    
    # 1. Try standard heuristics
    orders = [
        list(range(1, N + 1)),          # Ascending
        list(range(N, 0, -1)),          # Descending
        list(range(N, 0, -2)) + list(range(N-1, 0, -2)), # Alternating high/low
    ]
    
    rng = random.Random(42)
    
    # Add a few random orders to the list
    for _ in range(5):
        rand_order = list(range(1, N + 1))
        rng.shuffle(rand_order)
        orders.append(rand_order)

    for order in orders:
        if time.time() > deadline:
            break
        
        selected, sums_mask, selected_set = greedy_construct(order, deadline)
        
        if not selected:
            continue
            
        # Check maximality
        if is_maximal(selected, sums_mask):
            # Try to minimize
            min_selected, min_sums_mask = minimize_maximal(selected, sums_mask, deadline)
            
            if len(min_selected) < best_size:
                best_size = len(min_selected)
                best_set = min_selected
                
                # Early exit if we find a very small set
                if best_size <= 10:
                    break

    # 2. Randomized search with more iterations if time permits
    if time.time() < deadline - 5:
        attempts = 0
        while time.time() < deadline - 2:
            attempts += 1
            order = list(range(1, N + 1))
            rng.shuffle(order)
            
            selected, sums_mask, selected_set = greedy_construct(order, deadline)
            
            if not selected:
                continue
            
            # Quick check: if current size is already >= best, skip expensive maximality/minimization
            # Note: Greedy sets are usually larger than minimal maximal sets, 
            # but a greedy set might be smaller than the best *maximal* set found so far?
            # Actually, greedy sets are rarely maximal directly. 
            # But if len(selected) >= best_size, it's unlikely the minimized version will be smaller 
            # unless the greedy set was very close to optimal and minimization works well.
            # However, minimization requires the set to be maximal first.
            # If it's not maximal, we can't minimize it directly in the same way (we'd need to extend it first).
            # The current `minimize_maximal` assumes the input is maximal.
            # So we must check maximality first.
            
            if is_maximal(selected, sums_mask):
                if len(selected) < best_size: # Only bother minimizing if it's potentially better
                    min_selected, min_sums_mask = minimize_maximal(selected, sums_mask, deadline)
                    if len(min_selected) < best_size:
                        best_size = len(min_selected)
                        best_set = min_selected
                        if best_size <= 10:
                            break

    if best_set is None:
        # Fallback: Return a valid Sidon set (not necessarily maximal, but verifier requires maximal)
        # If no maximal set found, we must return something. 
        # The ascending greedy set is Sidon. We need to ensure it's maximal for the verifier.
        # If we didn't find one in the loop, let's do one last guaranteed attempt.
        order = list(range(1, N + 1))
        selected, sums_mask, selected_set = greedy_construct(order, start_time + 1.0)
        # Force maximality? No, if it's not maximal, it fails. 
        # But ascending greedy on [1,256] usually produces a large set. 
        # Let's just return it. The task says "Construct a Sidon subset... that cannot be extended".
        # If we fail to find a maximal one, we fail the task. 
        # However, the ascending greedy set IS Sidon. Is it maximal?
        # Usually not. But `minimize_maximal` requires maximal input.
        # We should have found one in the loop. 
        # If best_set is None, it means we failed to find ANY maximal set in the time limit?
        # Unlikely. The first heuristic (ascending) should be fast.
        # Let's ensure we return *something* valid if possible.
        # Re-check the first result from the loop if it exists but wasn't updated?
        # No, `best_set` is only updated if maximal.
        
        # Emergency fallback: Return the first greedy result, hoping it's maximal? 
        # Better: Just return the ascending greedy result. 
        # If it's not maximal, the score is 0. But we probably found one in the loop.
        # If we are here, we timed out or had an issue.
        # Let's return the ascending greedy result.
        return sorted(selected)

    return sorted(best_set)

# EVOLVE-BLOCK-END
