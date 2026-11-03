        valid = (uncovered == 0)
        if valid and len(solution) < best_len:
            best_len = len(solution)
            best_solution = solution
