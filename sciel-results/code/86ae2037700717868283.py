import time
from random import Random
import sys
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_instance(vals):
        n = len(vals)
        if n == 0:
            return []
        
        # Precompute value to index map for quick lookup
        val_to_idx = {v: i for i, v in enumerate(vals)}
        
        def can_add(val, current_set_vals):
            """Check if val can be added to current_set_vals (a set of values)."""
            for x in current_set_vals:
                if val + x in current_set_vals:
                    return False
                if val - x in current_set_vals:
                    return False
            return True

        def greedy_fill(order, start_set=None, start_vals=None):
            """Greedy fill given an order of indices."""
            if start_set is None:
                current_set = set()
                current_vals = set()
            else:
                current_set = set(start_set)
                current_vals = set(start_vals)
            
            for idx in order:
                if idx in current_set:
                    continue
                val = vals[idx]
                if can_add(val, current_vals):
                    current_set.add(idx)
                    current_vals.add(val)
            return current_set, current_vals

        # 1. Multiple Greedy Initializations
        best_set = set()
        best_vals = set()
        best_size = 0
        
        rng_init = Random(12345)
        
        # Strategy 1: Descending absolute value
        order1 = sorted(range(n), key=lambda i: abs(vals[i]), reverse=True)
        s, v = greedy_fill(order1)
        if len(s) > best_size:
            best_size = len(s)
            best_set, best_vals = s, v

        # Strategy 2: Positive values first (descending), then negative (ascending)
        pos_idx = [i for i in range(n) if vals[i] > 0]
        neg_idx = [i for i in range(n) if vals[i] < 0]
        order2 = sorted(pos_idx, key=lambda i: vals[i], reverse=True) + \
                 sorted(neg_idx, key=lambda i: vals[i])
        s, v = greedy_fill(order2)
        if len(s) > best_size:
            best_size = len(s)
            best_set, best_vals = s, v

        # Strategy 3: Random shuffles
        for _ in range(5):
            order3 = list(range(n))
            rng_init.shuffle(order3)
            s, v = greedy_fill(order3)
            if len(s) > best_size:
                best_size = len(s)
                best_set, best_vals = s, v

        # 2. Simulated Annealing / Local Search
        # We will use a "remove and re-greedy" move.
        # State is (best_set, best_vals).
        
        current_set = set(best_set)
        current_vals = set(best_vals)
        current_size = best_size
        
        # Temperature schedule
        # Start high, cool down
        T = 1.0
        alpha = 0.995  # Cooling rate
        min_T = 1e-6
        
        rng_sa = Random(54321)
        
        # We perform a fixed number of iterations or until time runs out
        # Each iteration: pick a random element to remove, then greedily fill.
        
        while time.time() < deadline - 5.0:
            if T < min_T:
                break
            
            # Pick a random element in current_set to remove
            if not current_set:
                break
                
            remove_idx = rng_sa.choice(list(current_set))
            remove_val = vals[remove_idx]
            
            # Create a temporary set without this element
            temp_set = current_set - {remove_idx}
            temp_vals = current_vals - {remove_val}
            
            # Greedily fill the temp_set with remaining elements
            # To make it faster, we can just iterate through all indices not in temp_set
            # and try to add them.
            # We use a random order for the fill to allow different outcomes?
            # Or a fixed order for consistency?
            # Let's use a fixed order based on index for speed and determinism in the inner loop,
            # but the random removal provides the diversity.
            
            # Actually, to escape local minima better, we can try to add elements 
            # that were previously conflicting with the removed element.
            # But simple greedy fill is robust.
            
            new_set, new_vals = greedy_fill(range(n), start_set=temp_set, start_vals=temp_vals)
            
            new_size = len(new_set)
            delta = new_size - current_size
            
            if delta > 0:
                current_set = new_set
                current_vals = new_vals
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
                    best_vals = set(current_vals)
            else:
                # Accept worse move with probability exp(delta / T)
                # delta is <= 0
                prob = math.exp(delta / T)
                if rng_sa.random() < prob:
                    current_set = new_set
                    current_vals = new_vals
                    current_size = new_size
            
            T *= alpha
        
        # Final polish: Try simple 1-swap hill climbing on the best solution
        # Remove one, add one if it fits and improves (which it won't if we just removed one)
        # Actually, the SA already does remove-and-fill which is stronger.
        # Let's do a final pass of "try to add any missing element" just in case.
        # This is redundant if greedy_fill was exhaustive, but safe.
        
        final_set, final_vals = greedy_fill(range(n), start_set=best_set, start_vals=best_vals)
        
        return [vals[i] for i in sorted(final_set)]

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            # Fallback to odd numbers if time runs out
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
            
        vals = instance(size, seed)
        results.append(solve_instance(vals))

    return results
