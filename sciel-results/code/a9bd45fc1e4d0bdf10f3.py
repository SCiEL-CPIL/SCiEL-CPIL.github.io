<<<<<<< SEARCH
    sa_end = start_time + 92.0
    while time.time() < sa_end:
        if random.random() < 0.3:
            init_rows = [r[:] for r in best_rows]
        else:
            bound = best_max + 5
            init_rows = []
            for _ in range(n_rows):
                pts = random.sample(range(1, bound + 1), k)
                pts.sort()
                init_rows.append([0] + pts)

        sa_time = min(6.0, sa_end - time.time())
        if sa_time < 1.0:
            break

        local_rows, local_max, local_collisions = simulate_annealing(init_rows, sa_time)

        if local_collisions < compute_cost(best_rows)[0] or \
           (local_collisions == 0 and local_max < best_max):
            best_rows = local_rows
            best_max = local_max

        # If we have a valid solution, try to polish it (interleaved).
        if compute_cost(best_rows)[0] == 0 and time.time() < sa_end - 3.0:
            polish_time = min(2.0, sa_end - time.time())
            polished_rows, polished_max, _ = local_polish(best_rows, polish_time)
            if polished_max < best_max:
                best_rows = polished_rows
                best_max = polished_max
=======
    sa_end = start_time + 85.0
    while time.time() < sa_end:
        # Adaptive initialization: if we have a good solution, perturb it;
        # otherwise use random or greedy
        if random.random() < 0.4 and best_collisions == 0:
            # Perturb the best solution slightly
            init_rows = [r[:] for r in best_rows]
            # Randomly swap or shift a few elements
            for _ in range(random.randint(1, 3)):
                r_idx = random.randint(0, n_rows - 1)
                i_idx = random.randint(1, row_len - 1)
                old_val = init_rows[r_idx][i_idx]
                lower = init_rows[r_idx][i_idx - 1] + 1
                if i_idx < row_len - 1:
                    upper = init_rows[r_idx][i_idx + 1] - 1
                else:
                    upper = old_val + 5
                if lower <= upper:
                    init_rows[r_idx][i_idx] = random.randint(lower, upper)
        elif random.random() < 0.3:
            init_rows = [r[:] for r in best_rows]
        else:
            bound = best_max + 3
            init_rows = []
            for _ in range(n_rows):
                pts = random.sample(range(1, bound + 1), k)
                pts.sort()
                init_rows.append([0] + pts)

        sa_time = min(5.0, sa_end - time.time())
        if sa_time < 1.0:
            break

        local_rows, local_max, local_collisions = simulate_annealing(init_rows, sa_time)

        if local_collisions < best_collisions or \
           (local_collisions == 0 and local_max < best_max):
            best_rows = local_rows
            best_max = local_max
            best_collisions = local_collisions

        # If we have a valid solution, try to polish it (interleaved).
        if best_collisions == 0 and time.time() < sa_end - 5.0:
            polish_time = min(3.0, sa_end - time.time())
            polished_rows, polished_max, _ = local_polish(best_rows, polish_time)
            if polished_max < best_max:
                best_rows = polished_rows
                best_max = polished_max
>>>>>>> REPLACE

