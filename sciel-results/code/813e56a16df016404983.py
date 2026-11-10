import time
import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
        start_time = time.time()
        deadline = 120.0 # Total budget, but we don't know when we start.
        # Actually, the evaluator kills the *program* after 120s.
        # We don't have access to the global start time easily unless we store it.
        # But we can just be efficient.
        
        # If no items, return -1
        if len(remaining_weights) == 0:
            return -1

        # Filter feasible items
        feasible_indices = [i for i, w in enumerate(remaining_weights) if w <= remaining_capacity]
        if not feasible_indices:
            return -1

        # Sort feasible items by density
        # density = value / weight
        # We want to iterate through all feasible items as candidates
        # For each candidate, we simulate the rest
        
        # Pre-calculate densities for feasible items
        # To avoid division by zero (though weights should be > 0)
        
        best_candidate = -1
        best_score = -1
        best_value = -1
        best_weight = float('inf')

        # Sort feasible items by density descending to use as the "greedy order" for simulation
        # This order is fixed for the simulation of *any* candidate, we just skip the candidate itself.
        feasible_sorted = sorted(feasible_indices, key=lambda i: remaining_values[i] / remaining_weights[i], reverse=True)

        for i in feasible_indices:
            # Check time?
            # if time.time() > deadline: break # Not applicable per-call easily without global state
            
            w_i = remaining_weights[i]
            v_i = remaining_values[i]
            c_rem = remaining_capacity - w_i
            v_total = v_i
            
            # Simulate greedy fill with remaining items (excluding i)
            for j in feasible_sorted:
                if j == i:
                    continue
                if remaining_weights[j] <= c_rem:
                    v_total += remaining_values[j]
                    c_rem -= remaining_weights[j]
                    if c_rem <= 0:
                        break
            
            # Compare scores
            # Primary: v_total
            # Secondary: v_i (higher is better)
            # Tertiary: w_i (lower is better)
            if v_total > best_score or (v_total == best_score and v_i > best_value) or (v_total == best_score and v_i == best_value and w_i < best_weight):
                best_score = v_total
                best_value = v_i
                best_weight = w_i
                best_candidate = i

        return best_candidate
