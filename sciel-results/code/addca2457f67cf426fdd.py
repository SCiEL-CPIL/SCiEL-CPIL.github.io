"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import math
    from fractions import Fraction

    start_time = time.time()
    deadline = start_time + 55  # 55 second deadline for safety

    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    # Using floats for speed in search
    unit_vals = [1.0 / d for d in denominators]
    
    # Precompute total sum
    total_sum = sum(unit_vals)
    
    best_indices = []
    best_float_sum = 0.0
    best_fraction_deficit = Fraction(1)
    best_fraction_sum = Fraction(0)

    def evaluate_fraction(indices):
        """Calculate exact fraction sum and deficit."""
        if not indices:
            return Fraction(0), Fraction(1)
        s = sum(Fraction(1, denominators[i]) for i in indices)
        deficit = 1 - s
        return s, deficit

    def greedy_order(order):
        """Greedy selection based on given order of indices."""
        selected = set()
        total = 0.0
        for idx in order:
            if total + unit_vals[idx] < 1.0:
                selected.add(idx)
                total += unit_vals[idx]
        return selected, total

    def simulated_annealing(start_set, max_time):
        """
        Runs SA on the selected set.
        Returns the best set found.
        """
        cur = set(start_set)
        cur_sum = sum(unit_vals[i] for i in cur)
        if cur_sum >= 1.0:
            # Should not happen if start is valid, but just in case
            # Remove largest until valid
            while cur_sum >= 1.0 and cur:
                # Remove largest to drop sum fastest? Or random?
                # Let's remove the one that contributes most to being over?
                # Actually, if it's over, we just need to get it under.
                # Remove the largest value.
                max_idx = max(cur, key=lambda i: unit_vals[i])
                cur.remove(max_idx)
                cur_sum -= unit_vals[max_idx]
        
        best_local = set(cur)
        best_local_sum = cur_sum
        
        temperature = 0.1
        cooling = 0.999
        min_temperature = 1e-6
        
        while time.time() < max_time and temperature > min_temperature:
            # Choose a move
            if not cur:
                # Add a random item
                if n > 0:
                    add = random.randrange(n)
                    if add not in cur:
                        new_sum = cur_sum + unit_vals[add]
                        if new_sum < 1.0:
                            cur.add(add)
                            cur_sum = new_sum
                            if cur_sum > best_local_sum:
                                best_local_sum = cur_sum
                                best_local = set(cur)
                temperature *= cooling
                continue

            move_type = random.random()
            new_set = None
            new_sum = None
            
            if move_type < 0.6:
                # Swap: remove i, add j
                if len(cur) > 0:
                    i = random.choice(tuple(cur))
                    # Find candidates for j not in cur
                    # To be fast, just pick random j and check if in cur
                    # Or maintain a list of not_in_set? 
                    # For n=47, checking 'in cur' (set) is fast.
                    j = random.randrange(n)
                    if j not in cur:
                        new_sum = cur_sum - unit_vals[i] + unit_vals[j]
                        if new_sum < 1.0:
                            new_set = cur - {i} | {j}
            elif move_type < 0.85:
                # Add: add j
                j = random.randrange(n)
                if j not in cur:
                    new_sum = cur_sum + unit_vals[j]
                    if new_sum < 1.0:
                        new_set = cur | {j}
            else:
                # Remove: remove i
                if len(cur) > 0:
                    i = random.choice(tuple(cur))
                    new_sum = cur_sum - unit_vals[i]
                    # new_sum is always < 1.0 if cur_sum < 1.0
                    new_set = cur - {i}
            
            if new_set is not None and new_sum is not None:
                delta = new_sum - cur_sum
                # Accept if better or with probability exp(delta/T)
                if delta > 0:
                    accept = True
                else:
                    if temperature > 0:
                        accept = random.random() < math.exp(delta / temperature)
                    else:
                        accept = False
                
                if accept:
                    cur = new_set
                    cur_sum = new_sum
                    if cur_sum > best_local_sum:
                        best_local_sum = cur_sum
                        best_local = set(cur)
            
            temperature *= cooling

        return best_local, best_local_sum

    # Initial Greedy Start
    random.seed(42)
    
    # Try a few different greedy orders
    orders = [
        list(range(n)),
        list(reversed(range(n))),
        sorted(range(n), key=lambda i: -unit_vals[i]), # Largest first
        sorted(range(n), key=lambda i: unit_vals[i])   # Smallest first
    ]
    
    for order in orders:
        start_set, start_sum = greedy_order(order)
        if start_set:
            # Run SA with a short time limit for each start
            # Total time budget 55s. Let's do 4 starts * 5s + 1 long run?
            # Or just run SA in a loop until deadline.
            pass

    # Main Search Loop
    # We will alternate between starting from greedy and random restarts
    # Or just run SA from the best found so far, and occasionally restart.
    
    current_best_set, current_best_sum = greedy_order(sorted(range(n), key=lambda i: -unit_vals[i]))
    
    if current_best_set:
        # Update global best
        best_indices = sorted(current_best_set)
        best_float_sum = current_best_sum
        s_frac, d_frac = evaluate_fraction(best_indices)
        best_fraction_sum = s_frac
        best_fraction_deficit = d_frac

    # Run SA continuously, restarting if stuck or periodically
    # To ensure exploration, we will do random restarts every few seconds or if no improvement for a while.
    
    last_improvement_time = time.time()
    no_improvement_limit = 2.0 # seconds
    
    while time.time() < deadline:
        # Run SA for a short burst
        burst_end = min(time.time() + 2.0, deadline)
        improved_set, improved_sum = simulated_annealing(current_best_set, burst_end)
        
        if improved_sum > best_float_sum:
            best_float_sum = improved_sum
            best_indices = sorted(improved_set)
            current_best_set = improved_set
            last_improvement_time = time.time()
            
            # Update exact fraction
            s_frac, d_frac = evaluate_fraction(best_indices)
            best_fraction_sum = s_frac
            best_fraction_deficit = d_frac
        else:
            # No improvement in this burst
            # Check if we should restart
            if time.time() - last_improvement_time > no_improvement_limit:
                # Random restart
                # Generate a random valid set
                # Try greedy with random order
                order = list(range(n))
                random.shuffle(order)
                restart_set, restart_sum = greedy_order(order)
                if restart_set and restart_sum > best_float_sum * 0.9: # Only restart if decent
                    current_best_set = restart_set
                else:
                    # Try a completely random small set?
                    # Or just keep the best and let SA explore from there?
                    # SA from best is good. Let's just continue SA from best.
                    current_best_set = set(best_indices)
                last_improvement_time = time.time()
        
        # If time is running out, do a final check
        if time.time() > deadline - 5.0:
            break

    # Final verification and return
    if not best_indices:
        # Fallback: just 1/2
        best_indices = [0]
    else:
        # Ensure the returned indices are valid and sum < 1
        # Re-evaluate to be absolutely sure
        s_frac, d_frac = evaluate_fraction(best_indices)
        if d_frac <= 0:
            # This shouldn't happen if our float check was robust, 
            # but if it does, we need to fix it.
            # Remove the smallest fraction from the set?
            # Or just return a known valid set like [0] (1/2)
            # Let's try to repair by removing the smallest element
            # This is a rare edge case.
            # For safety, return [0]
            best_indices = [0]

    return best_indices
# EVOLVE-BLOCK-END
