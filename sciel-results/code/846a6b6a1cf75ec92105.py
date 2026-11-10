"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Return a nonempty subset of [1,100] with no subset summing to 200.
    Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    target = 200
    max_val = 100
    universe = list(range(1, max_val + 1))

    def greedy_select(order):
        """
        Greedily select elements in the given order.
        Returns the selected list.
        """
        selected = []
        reachable = {0}
        
        for x in order:
            # Check if adding x creates a sum of 200
            # i.e., is (target - x) in reachable?
            if (target - x) in reachable:
                continue
            
            # If safe, add x
            selected.append(x)
            # Update reachable sums
            # Only sums <= target matter
            new_sums = set()
            for s in reachable:
                ns = s + x
                if ns <= target:
                    new_sums.add(ns)
            reachable |= new_sums
            
        return selected

    best_solution = []
    best_size = 0

    def update_best(sol):
        nonlocal best_solution, best_size
        if sol and len(sol) > best_size:
            best_solution = sol
            best_size = len(sol)

    # 1. Deterministic Greedy Strategies
    # Descending (Seed approach)
    update_best(greedy_select(range(max_val, 0, -1)))
    
    # Ascending
    update_best(greedy_select(range(1, max_val + 1)))
    
    # Odd then Even
    odds = list(range(1, max_val + 1, 2))
    evens = list(range(2, max_val + 1, 2))
    update_best(greedy_select(odds + evens))
    update_best(greedy_select(evens + odds))
    
    # Structured Block {66..100} (Known valid, size 35)
    # Greedy descending usually finds something similar or better.
    # Let's explicitly check a few other structural ideas if time permits, 
    # but greedy is the main engine.

    # 2. Randomized Greedy Search
    # Use fixed seed for reproducibility of the search process
    random.seed(42)
    
    # Pre-shuffled lists for speed? No, just shuffle in loop.
    # We can run many iterations.
    
    # Optimization: 
    # If best_size is already very high (e.g., > 40), random search might be 
    # less likely to beat it unless it's close to optimal.
    # However, greedy descending usually yields around 40-50 elements? 
    # Let's check typical sizes.
    # For target 200, max element 100.
    # If we take 100, 99, 98... 
    # 100+99 = 199. Next 98 -> 199+98 > 200. 100+98=198. 99+98=197.
    # This is effectively a knapsack-like problem but with subset sum constraint.
    
    # Let's run random shuffles until deadline.
    # To speed up, we can skip if the potential max size of a random order 
    # is unlikely to beat current best? Hard to know. 
    # Just run them.
    
    # We can also try "smart" random orders biased towards larger numbers,
    # as larger numbers "fill" the sum budget faster, potentially allowing
    # more smaller numbers to be added? Or does it block too many sums?
    # Actually, adding large numbers first blocks fewer small combinations?
    # No, adding a large number x blocks sum T-x. 
    # If x is large, T-x is small. Blocking a small sum prevents many combinations of small numbers.
    # If x is small, T-x is large. Blocking a large sum prevents fewer combinations?
    # Intuitively, descending greedy (large first) is often good for maximizing count 
    # because it leaves the "small sum" space open? 
    # Wait, if I pick 100, I can't pick any subset summing to 100.
    # If I pick 1, I can't pick any subset summing to 199.
    # Picking small numbers first blocks huge sums, which are hard to form anyway with remaining small numbers?
    # Actually, picking small numbers first allows many large numbers to be added later?
    # Let's rely on the random search to find the best heuristic.

    while time.time() < deadline:
        # Generate a random order
        shuffled = universe[:]
        random.shuffle(shuffled)
        
        sol = greedy_select(shuffled)
        update_best(sol)
        
        # Occasionally try a biased order
        if random.random() < 0.1:
            # Bias towards larger numbers
            # Sort by -value + noise
            biased_order = sorted(universe, key=lambda x: -x + random.gauss(0, 10))
            sol_w = greedy_select(biased_order)
            update_best(sol_w)
            
        # Occasionally try a "reverse biased" order
        if random.random() < 0.1:
            biased_order = sorted(universe, key=lambda x: x + random.gauss(0, 10))
            sol_w = greedy_select(biased_order)
            update_best(sol_w)

    # Final verification
    if best_solution:
        reachable = {0}
        valid = True
        for x in best_solution:
            if (target - x) in reachable:
                valid = False
                break
            new_sums = {s + x for s in reachable if s + x <= target}
            reachable |= new_sums
        
        if not valid:
            # Fallback to deterministic descending
            return greedy_select(range(max_val, 0, -1))
            
    if not best_solution:
        return [1]
        
    return best_solution
# EVOLVE-BLOCK-END
