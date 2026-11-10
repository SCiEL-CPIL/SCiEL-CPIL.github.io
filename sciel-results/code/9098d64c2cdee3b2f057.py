"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import math
    import random
    from fractions import Fraction

    deadline = time.time() + 85.0
    random.seed(42)

    N = 256
    universe = list(range(2, N + 1))

    # Precompute conflict matrix for O(1) conflict checks
    # conflict_mask[i] is a bitmask (or set) of indices in universe that conflict with universe[i]
    # Since universe size is 255, we can use a set of indices or a list of booleans.
    # Let's use a list of sets for clarity and speed.
    # Actually, for local search, checking compatibility with all selected elements is O(|S|).
    # Precomputing a 255x255 boolean matrix is 65k entries, very fast.
    
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(universe)}
    n_univ = len(universe)
    
    # conflict_matrix[i][j] is True if universe[i] and universe[j] conflict
    conflict_matrix = [[False] * n_univ for _ in range(n_univ)]
    for i in range(n_univ):
        for j in range(i + 1, n_univ):
            if math.lcm(universe[i], universe[j]) <= N:
                conflict_matrix[i][j] = True
                conflict_matrix[j][i] = True

    def conflicts_val(a, b):
        return conflict_matrix[val_to_idx[a]][val_to_idx[b]]

    def is_valid(subset):
        s = list(subset)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if conflicts_val(s[i], s[j]):
                    return False
        return True

    def reciprocal_sum(subset):
        return sum(1.0 / x for x in subset)

    def greedy_by_weight(order):
        """Greedy: process in given order, add if no conflict."""
        selected = []
        selected_set = set()
        for x in order:
            if all(not conflicts_val(x, y) for y in selected):
                selected.append(x)
                selected_set.add(x)
        return selected

    # --- Phase 1: Initial Greedy Solutions ---
    best = []
    best_score = 0.0

    # Pass 1: ascending (original seed logic)
    sol = greedy_by_weight(universe)
    if reciprocal_sum(sol) > best_score:
        best_score = reciprocal_sum(sol)
        best = sol[:]

    # Pass 2: descending
    sol = greedy_by_weight(sorted(universe, reverse=True))
    if reciprocal_sum(sol) > best_score:
        best_score = reciprocal_sum(sol)
        best = sol[:]

    # Pass 3: random orderings
    for _ in range(50):
        if time.time() > deadline:
            break
        order = universe[:]
        random.shuffle(order)
        sol = greedy_by_weight(order)
        s = reciprocal_sum(sol)
        if s > best_score:
            best_score = s
            best = sol[:]

    # --- Phase 2: Local Search with Add/Remove/Swap ---
    # We work with sets of values for easy lookup, but maintain a list for iteration.
    
    def try_improve(initial_subset, initial_score):
        """Local search with add/remove/swap moves."""
        cur_set = set(initial_subset)
        cur_score = initial_score
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try adding each unselected element
            for x in universe:
                if time.time() > deadline:
                    return list(cur_set), cur_score
                if x in cur_set:
                    continue
                # Check compatibility
                compatible = True
                for y in cur_set:
                    if conflicts_val(x, y):
                        compatible = False
                        break
                if compatible:
                    new_score = cur_score + 1.0 / x
                    if new_score > cur_score:
                        cur_set.add(x)
                        cur_score = new_score
                        improved = True
                        break  # restart loop after improvement
            
            if not improved and time.time() < deadline:
                # 2. Try swapping: remove one, add another
                for x in list(cur_set):
                    if time.time() > deadline:
                        return list(cur_set), cur_score
                    
                    # Temporarily remove x
                    cur_set.remove(x)
                    base_score = cur_score - 1.0 / x
                    
                    best_swap_y = None
                    best_swap_score = base_score
                    
                    for y in universe:
                        if y in cur_set:
                            continue
                        # Check compatibility of y with remaining cur_set
                        compatible = True
                        for z in cur_set:
                            if conflicts_val(y, z):
                                compatible = False
                                break
                        if compatible:
                            new_score = base_score + 1.0 / y
                            if new_score > best_swap_score:
                                best_swap_score = new_score
                                best_swap_y = y
                    
                    if best_swap_y is not None and best_swap_score > cur_score:
                        cur_set.add(best_swap_y)
                        cur_score = best_swap_score
                        improved = True
                    else:
                        # Put x back
                        cur_set.add(x)
                    
                    if improved:
                        break  # restart loop after improvement
            
            if not improved and time.time() < deadline:
                # 3. Try removing one element to allow adding two better ones?
                # This is a 2-exchange move. It's expensive. 
                # For now, let's stick to 1-exchange (add/remove/swap) which is standard.
                # If stuck, we might need 2-exchange, but let's see if 1-exchange is enough.
                # Actually, the previous code had a "remove one" step that didn't explicitly 
                # try to add two. It just tried to add one. 
                # Let's add a simple "remove if it doesn't hurt" check? No, we want to maximize sum.
                # Removing an element always decreases the score unless it allows adding something much larger reciprocal.
                # The swap step above covers "remove x, add y". 
                # What about "remove x, add y, add z"? That's a 2-add 1-remove move.
                # Let's implement a simple version: for each x in set, remove it, then greedily add best compatible elements.
                
                for x in list(cur_set):
                    if time.time() > deadline:
                        return list(cur_set), cur_score
                    
                    cur_set.remove(x)
                    base_score = cur_score - 1.0 / x
                    
                    # Try to add as many compatible elements as possible from the remaining universe
                    # We want to maximize the sum of reciprocals of added elements.
                    # Greedy: sort remaining compatible elements by 1/y descending?
                    # Or just try all single additions first (covered by swap).
                    # Let's try a 2-add move: find y, z such that y, z are compatible with each other and with cur_set\{x}.
                    
                    # This is O(N^2) per removal, which is 255^2 * 255 ~ 16M ops. Might be too slow.
                    # Let's stick to 1-exchange for now, but maybe do a few random restarts of the local search.
                    
                    # Restore x
                    cur_set.add(x)
                    
                    # If we don't improve, we break out of the while loop.
                    # The while loop condition `improved` is False, so it will exit.
                    break

        return list(cur_set), cur_score

    best, current_score = try_improve(best, best_score)

    # --- Phase 3: More Random Restarts with Local Search ---
    for _ in range(50):
        if time.time() > deadline:
            break
        order = universe[:]
        random.shuffle(order)
        sol = greedy_by_weight(order)
        s = reciprocal_sum(sol)
        if s > current_score:
            best, current_score = try_improve(sol, s)
        else:
            # Even if greedy score is lower, local search might improve it?
            # Unlikely to beat the current best if it starts lower, but possible.
            # Let's only run local search if it's close or if we have time.
            # To be safe, let's just run it if it's within some margin or if we have lots of time.
            if time.time() < deadline - 10.0:
                best2, score2 = try_improve(sol, s)
                if score2 > current_score:
                    best, current_score = best2, score2

    # --- Final Polish: Exact Fractional Check? ---
    # The score is rational. 1.0/x is float. 
    # Using Fraction might be overkill and slow.
    # The float precision is likely sufficient for comparison.
    
    return best
# EVOLVE-BLOCK-END
