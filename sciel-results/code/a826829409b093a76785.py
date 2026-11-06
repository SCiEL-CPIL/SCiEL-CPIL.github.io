<<<<<<< SEARCH
    # Branch and bound
    if time.time() < deadline - 10:
        bb_result = branch_and_bound(best_solution, deadline - 5)
        bb_score = calc_score(bb_result)
        if bb_score > best_score:
            best_score = bb_score
            best_solution = bb_result
=======
    # Branch and bound
    if time.time() < deadline - 10:
        bb_result = branch_and_bound(best_solution, deadline - 5)
        bb_score = calc_score(bb_result)
        if bb_score > best_score:
            best_score = bb_score
            best_solution = bb_result

    # Second B&B pass with different ordering if time permits
    if time.time() < deadline - 10:
        # Try B&B on a slightly different starting point
        alt_sol = [list(x) for x in best_solution]
        # Remove the lowest-weight element and try to re-optimize
        if len(alt_sol) > 5:
            # Find the element with smallest 1/m (largest m)
            min_idx = max(range(len(alt_sol)), key=lambda i: alt_sol[i][0])
            removed = alt_sol.pop(min_idx)
            bb_result2 = branch_and_bound(alt_sol, deadline - 5)
            bb_score2 = calc_score(bb_result2)
            if bb_score2 > best_score:
                best_score = bb_score2
                best_solution = bb_result2
>>>>>>> REPLACE

