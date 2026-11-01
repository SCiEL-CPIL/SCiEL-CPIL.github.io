<<<<<<< SEARCH
    if len(best_solution) < len(solution):
        solution = best_solution
    
    return solution
=======
    if len(best_solution) < len(solution):
        solution = best_solution
    
    # 9. Final polish: more aggressive redundancy removal with multiple passes
    for _ in range(30):
        if time.time() > deadline - 1.0:
            break
        solution, changed = remove_redundant(solution)
        if not changed:
            break
    
    # Final pairwise merge
    for _ in range(50):
        if time.time() > deadline - 0.5:
            break
        solution, changed = pairwise_merge(solution)
        if not changed:
            break
    
    for _ in range(20):
        if time.time() > deadline - 0.2:
            break
        solution, changed = remove_redundant(solution)
        if not changed:
            break
    
    return solution
>>>>>>> REPLACE
