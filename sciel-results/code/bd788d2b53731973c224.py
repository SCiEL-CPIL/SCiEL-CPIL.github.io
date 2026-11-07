<<<<<<< SEARCH
    best_res = None
    best_score = 0
    
    # Try multiple seeds and orders
    seed_counter = 0
    for attempt in range(200):
        if time.time() > deadline:
            break
        seed_counter += 1
        
        # Alternate between different orderings
        if attempt % 5 == 0:
            order = tree_order  # descending
        elif attempt % 5 == 1:
            order = tree_order[::-1]  # ascending
        else:
            rng_tmp = Random(attempt)
            order = list(range(n))
            rng_tmp.shuffle(order)
        
        res, score = solve_with_backtracking(seed_counter, order)
        
        if score > best_score:
            best_score = score
            best_res = res
        
        if best_score == 190:
            break

    if best_res is None:
        fallback = [None] * n
        fallback[0] = [0]
        fallback[1] = [0, 1]
        return fallback

    final_result = []
    for i in range(n):
        if best_res[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_res[i]])
    
    return final_result
=======
    best_res = None
    best_score = 0
    
    # Try multiple seeds and orders
    seed_counter = 0
    for attempt in range(300):
        if time.time() > deadline:
            break
        seed_counter += 1
        
        # Alternate between different orderings
        # Large-first is generally best for edge-disjoint packing
        if attempt % 7 == 0:
            order = tree_order  # descending
        elif attempt % 7 == 1:
            order = tree_order[::-1]  # ascending
        elif attempt % 7 == 2:
            # Large first, then medium, then small (grouped)
            order = sorted(range(n), key=lambda i: (-len(trees[i]), i))
        elif attempt % 7 == 3:
            # Reverse of large-first
            order = sorted(range(n), key=lambda i: (len(trees[i]), i))
        else:
            rng_tmp = Random(attempt * 1000 + 42)
            order = list(range(n))
            rng_tmp.shuffle(order)
        
        res, score = solve_with_backtracking(seed_counter, order)
        
        if score > best_score:
            best_score = score
            best_res = res
        
        if best_score == 190:
            break

    if best_res is None:
        fallback = [None] * n
        fallback[0] = [0]
        fallback[1] = [0, 1]
        return fallback

    final_result = []
    for i in range(n):
        if best_res[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_res[i]])
    
    return final_result
>>>>>>> REPLACE
