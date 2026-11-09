<<<<<<< SEARCH
    # 4. Try ILP if time permits
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 60:
        # ILP might take a while; give it up to 80s
        ilp_time_budget = min(80, DEADLINE - elapsed - 10)
        if ilp_time_budget > 15:
            ilp_classes, ilp_score = solve_ilp(ilp_time_budget)
            if ilp_classes is not None and ilp_score < best_score:
                best_score = ilp_score
                best_classes = ilp_classes
    
    # 5. Local search from best
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        remaining = DEADLINE - elapsed - 2
        if remaining > 5:
            classes, score = local_search(best_classes, remaining)
            if score < best_score:
                best_score = score
                best_classes = classes
    
    # 6. More restarts if time remains
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        for trial in range(20):
            if time.time() - start_time >= DEADLINE - 5:
                break
            order = MODULI[:]
            rng.shuffle(order)
            classes = greedy_init(order)
            score = count_uncovered(classes)
            if score < best_score:
                best_score = score
                best_classes = list(classes)
            # Quick local search
            remaining = DEADLINE - (time.time() - start_time)
            if remaining > 15 and score <= best_score + 50:
                classes, score = local_search(classes, min(remaining * 0.5, 10))
                if score < best_score:
                    best_score = score
                    best_classes = classes
=======
    # 4. Iterated Local Search from best
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        remaining = DEADLINE - elapsed - 2
        if remaining > 10:
            classes, score = iterated_local_search(best_classes, remaining, rng)
            if score < best_score:
                best_score = score
                best_classes = classes
    
    # 5. More restarts with ILS if time remains
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        for trial in range(20):
            if time.time() - start_time >= DEADLINE - 5:
                break
            order = MODULI[:]
            rng.shuffle(order)
            classes = greedy_init(order)
            score = count_uncovered(classes)
            # Quick local search
            remaining = DEADLINE - (time.time() - start_time)
            if remaining > 10:
                classes, score = local_search(classes, min(remaining * 0.3, 5))
            if score < best_score:
                best_score = score
                best_classes = list(classes)
                # If we found a new best, do a short ILS on it
                remaining = DEADLINE - (time.time() - start_time)
                if remaining > 5:
                    classes, score = iterated_local_search(best_classes, min(remaining * 0.5, 10), rng)
                    if score < best_score:
                        best_score = score
                        best_classes = classes
>>>>>>> REPLACE
