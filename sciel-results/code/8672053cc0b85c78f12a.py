<<<<<<< SEARCH
    # Run local search for a few iterations
    for _ in range(50):
        if time.time() > deadline - 2.0:
            break
        best_solution, improved = local_search_swap(best_solution)
        if improved:
            best_solution = optimize_redundancy(best_solution)
            if len(best_solution) < best_len:
                best_len = len(best_solution)
        else:
            # Try 2-exchange
            best_solution, improved2 = local_search_2exchange(best_solution)
            if improved2:
                best_solution = optimize_redundancy(best_solution)
                if len(best_solution) < best_len:
                    best_len = len(best_solution)
            else:
                # Try a few more random restarts of local search
                for _ in range(5):
                    if time.time() > deadline - 2.0:
                        break
                    best_solution, improved3 = local_search_swap(best_solution)
                    if improved3:
                        best_solution = optimize_redundancy(best_solution)
                        if len(best_solution) < best_len:
                            best_len = len(best_solution)
                break

    return best_solution
=======
    def run_local_search_pipeline(blocks):
        """Run swap + redundancy + 2-exchange pipeline until no improvement."""
        for _ in range(30):
            if time.time() > deadline - 2.0:
                break
            blocks, improved = local_search_swap(blocks)
            if improved:
                blocks = optimize_redundancy(blocks)
            else:
                blocks, improved2 = local_search_2exchange(blocks)
                if improved2:
                    blocks = optimize_redundancy(blocks)
                else:
                    break
        return blocks

    # Run local search for the current best
    best_solution = run_local_search_pipeline(best_solution)
    if len(best_solution) < best_len:
        best_len = len(best_solution)

    # Random restarts: use remaining budget to try more diverse starting points
    restart_seed = 200
    while time.time() < deadline - 8.0:
        sol = randomized_greedy(restart_seed)
        restart_seed += 1
        if sol is None:
            break
        if len(sol) < best_len:
            sol = run_local_search_pipeline(sol)
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol

    return best_solution
>>>>>>> REPLACE

Now I'll replace the slow `set.intersection_update` in `local_search_2exchange` with a faster list-based intersection:

