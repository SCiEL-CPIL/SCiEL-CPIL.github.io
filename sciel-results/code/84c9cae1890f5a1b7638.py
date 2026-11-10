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
    universe_set = set(universe)
    
    # Precompute conflict information
    # conflict_mask[x] is a set of y in universe that conflict with x
    # Actually, let's use a boolean matrix or just check on the fly with caching
    # Since N is small (255 elements), we can precompute a compatibility matrix
    # compat[i][j] = True if lcm(universe[i], universe[j]) > N
    
    n_univ = len(universe)
    # Use a list of sets for conflicts
    conflict_sets = [set() for _ in range(n_univ)]
    for i in range(n_univ):
        for j in range(i + 1, n_univ):
            a = universe[i]
            b = universe[j]
            if math.lcm(a, b) <= N:
                conflict_sets[i].add(j)
                conflict_sets[j].add(i)
    
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(universe)}
    
    def reciprocal_sum(subset):
        return sum(1.0 / x for x in subset)
    
    def greedy_by_weight(order):
        """Greedy: process in given order, add if no conflict."""
        selected = []
        selected_indices = set()
        for x in order:
            idx = val_to_idx[x]
            if all(j not in selected_indices for j in conflict_sets[idx]):
                selected.append(x)
                selected_indices.add(idx)
        return selected
    
    # Multiple greedy passes with different orderings
    best = []
    best_score = 0.0
    
    # Pass 1: ascending
    sol = greedy_by_weight(universe)
    s = reciprocal_sum(sol)
    if s > best_score:
        best_score = s
        best = sol[:]
    
    # Pass 2: descending
    sol = greedy_by_weight(sorted(universe, reverse=True))
    s = reciprocal_sum(sol)
    if s > best_score:
        best_score = s
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
    
    # Local search with exact rational arithmetic for polish
    def try_improve(subset_list, score):
        """Local search with add/remove/swap moves."""
        cur_set = set(subset_list)
        cur_score = score
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding each unselected element
            for x in universe:
                if x in cur_set:
                    continue
                idx_x = val_to_idx[x]
                if all(val_to_idx[y] not in conflict_sets[idx_x] for y in cur_set):
                    new_score = cur_score + 1.0 / x
                    if new_score > cur_score:
                        cur_set.add(x)
                        cur_score = new_score
                        improved = True
                        break  # restart after improvement
            
            if not improved:
                # Try swapping: remove one, add another
                for x in list(cur_set):
                    if time.time() > deadline:
                        break
                    idx_x = val_to_idx[x]
                    cur_set.remove(x)
                    base_score = cur_score - 1.0 / x
                    
                    # Try adding any compatible element
                    best_add = None
                    best_add_score = -1.0
                    for y in universe:
                        if y in cur_set:
                            continue
                        idx_y = val_to_idx[y]
                        if all(val_to_idx[z] not in conflict_sets[idx_y] for z in cur_set):
                            new_score = base_score + 1.0 / y
                            if new_score > cur_score:
                                if new_score > best_add_score:
                                    best_add_score = new_score
                                    best_add = y
                        if time.time() > deadline:
                            break
                    
                    if best_add is not None:
                        cur_set.add(best_add)
                        cur_score = best_add_score
                        improved = True
                    else:
                        cur_set.add(x)  # put x back
                    
                    if improved:
                        break
            
            if not improved:
                # Try 2-swap: remove two, add two
                # This is more expensive, so limit it
                if time.time() < deadline - 10:
                    found_2swap = False
                    list_cur = list(cur_set)
                    for i in range(len(list_cur)):
                        if found_2swap or time.time() > deadline:
                            break
                        for j in range(i + 1, len(list_cur)):
                            if found_2swap or time.time() > deadline:
                                break
                            x1 = list_cur[i]
                            x2 = list_cur[j]
                            cur_set.remove(x1)
                            cur_set.remove(x2)
                            base_score = cur_score - 1.0 / x1 - 1.0 / x2
                            
                            best_pair = None
                            best_pair_score = -1.0
                            
                            # Find best pair to add
                            # This is O(n^2) which is 255^2 ~ 65k, acceptable if rare
                            for y1 in universe:
                                if y1 in cur_set:
                                    continue
                                idx_y1 = val_to_idx[y1]
                                if all(val_to_idx[z] not in conflict_sets[idx_y1] for z in cur_set):
                                    for y2 in universe:
                                        if y2 in cur_set or y2 == y1:
                                            continue
                                        idx_y2 = val_to_idx[y2]
                                        if y2 in conflict_sets[idx_y1]:
                                            continue
                                        if all(val_to_idx[z] not in conflict_sets[idx_y2] for z in cur_set):
                                            new_score = base_score + 1.0 / y1 + 1.0 / y2
                                            if new_score > cur_score:
                                                if new_score > best_pair_score:
                                                    best_pair_score = new_score
                                                    best_pair = (y1, y2)
                            if best_pair is not None:
                                cur_set.add(best_pair[0])
                                cur_set.add(best_pair[1])
                                cur_score = best_pair_score
                                improved = True
                                found_2swap = True
                            else:
                                cur_set.add(x1)
                                cur_set.add(x2)
        
        return list(cur_set), cur_score
    
    best, current_score = try_improve(best, current_score)
    
    # More random restarts with local search
    for _ in range(20):
        if time.time() > deadline:
            break
        order = universe[:]
        random.shuffle(order)
        sol = greedy_by_weight(order)
        s = reciprocal_sum(sol)
        if s > current_score:
            best, current_score = try_improve(sol, s)
    
    # Final polish: exact rational check and local refinement
    # Use Fraction to ensure no floating point errors in the final decision
    # However, the problem asks for maximizing the sum, and the evaluator uses raw value.
    # The current best is a list of integers.
    
    # One more pass of local search with a different random seed or strategy
    # Let's try a "shaking" heuristic: remove a random subset of elements and re-greedy
    if time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline:
                break
            # Randomly remove 20% of elements
            current_set = set(best)
            to_remove = random.sample(list(current_set), max(1, len(current_set) // 5))
            for x in to_remove:
                current_set.remove(x)
            
            # Re-greedy the rest
            remaining = list(current_set)
            random.shuffle(remaining)
            new_sol = []
            new_set = set()
            for x in remaining:
                idx = val_to_idx[x]
                if all(val_to_idx[y] not in conflict_sets[idx] for y in new_set):
                    new_sol.append(x)
                    new_set.add(x)
            
            # Try to add more
            for x in universe:
                if x in new_set:
                    continue
                idx = val_to_idx[x]
                if all(val_to_idx[y] not in conflict_sets[idx] for y in new_set):
                    new_sol.append(x)
                    new_set.add(x)
            
            s = reciprocal_sum(new_sol)
            if s > current_score:
                best, current_score = try_improve(new_sol, s)
    
    return best
# EVOLVE-BLOCK-END
