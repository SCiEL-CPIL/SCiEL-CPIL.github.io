<<<<<<< SEARCH
def _simulated_annealing(sequence, chain, rng, t_start=1.0, t_end=0.01, max_iters=3000):
    """
    Simulated annealing with pivot, reversal, kink, and reinsert moves.
    Returns (best_chain, best_energy).
    """
    best_chain = list(chain)
    best_energy = _energy_fast(sequence, best_chain)
    cur_chain = list(chain)
    cur_energy = best_energy
    n = len(cur_chain)

    # Adaptive move weights
    move_weights = [1.0, 1.0, 1.0, 0.5]  # pivot, reversal, kink, reinsert
    move_success = [0, 0, 0, 0]
    move_attempts = [0, 0, 0, 0]

    total_steps = max(100, max_iters)
    for it in range(total_steps):
        if _remaining() < 5.0:
            break

        # Geometric cooling
        if total_steps > 1:
            frac = it / total_steps
            T = t_start * (t_end / t_start) ** frac
        else:
            T = t_start

        # Choose move type
        r = rng.random()
        cumulative = [0]
        for w in move_weights:
            cumulative.append(cumulative[-1] + w)
        total_w = cumulative[-1]
        r *= total_w
        if r < cumulative[1]:
            move_type = 0  # pivot
        elif r < cumulative[2]:
            move_type = 1  # reversal
        elif r < cumulative[3]:
            move_type = 2  # kink
        else:
            move_type = 3  # reinsert

        move_attempts[move_type] += 1

        if move_type == 0:
            new_chain = _pivot_move(cur_chain, rng)
        elif move_type == 1:
            new_chain = _reversal_move(cur_chain, rng)
        elif move_type == 2:
            new_chain = _kink_move(cur_chain, rng)
        else:
            new_chain = _reinsert_fragment(cur_chain, sequence, rng)

        if new_chain is None:
            continue

        new_energy = _energy_fast(sequence, new_chain)
        delta = new_energy - cur_energy

        if delta <= 0:
            cur_chain = new_chain
            cur_energy = new_energy
            move_success[move_type] += 1
            if cur_energy < best_energy:
                best_chain = list(cur_chain)
                best_energy = cur_energy
        else:
            # Accept with probability exp(-delta/T)
            if T > 0 and rng.random() < math.exp(-delta / T):
                cur_chain = new_chain
                cur_energy = new_energy
                move_success[move_type] += 1

        # Adapt move weights every 200 iterations
        if it % 200 == 199:
            for k in range(4):
                if move_attempts[k] > 0:
                    rate = move_success[k] / move_attempts[k]
                    move_weights[k] = max(0.1, rate * 2.0)
                move_attempts[k] = 0
                move_success[k] = 0

    return best_chain, best_energy
=======
def _simulated_annealing(sequence, chain, rng, t_start=1.0, t_end=0.01, max_iters=3000):
    """
    Simulated annealing with pivot, reversal, kink, reinsert, and fragment relocation moves.
    Returns (best_chain, best_energy).
    """
    best_chain = list(chain)
    best_energy = _energy_fast(sequence, best_chain)
    cur_chain = list(chain)
    cur_energy = best_energy
    n = len(cur_chain)

    # Adaptive move weights: pivot, reversal, kink, reinsert, fragment_relocation
    move_weights = [1.0, 1.0, 0.8, 0.5, 0.7]
    move_success = [0, 0, 0, 0, 0]
    move_attempts = [0, 0, 0, 0, 0]

    total_steps = max(100, max_iters)
    for it in range(total_steps):
        if _remaining() < 5.0:
            break

        if total_steps > 1:
            frac = it / total_steps
            T = t_start * (t_end / t_start) ** frac
        else:
            T = t_start

        # Choose move type
        r = rng.random()
        cumulative = [0]
        for w in move_weights:
            cumulative.append(cumulative[-1] + w)
        total_w = cumulative[-1]
        r *= total_w
        if r < cumulative[1]:
            move_type = 0  # pivot
        elif r < cumulative[2]:
            move_type = 1  # reversal
        elif r < cumulative[3]:
            move_type = 2  # kink
        elif r < cumulative[4]:
            move_type = 3  # reinsert
        else:
            move_type = 4  # fragment relocation

        move_attempts[move_type] += 1

        if move_type == 0:
            new_chain = _pivot_move(cur_chain, rng)
        elif move_type == 1:
            new_chain = _reversal_move(cur_chain, rng)
        elif move_type == 2:
            new_chain = _kink_move(cur_chain, rng)
        elif move_type == 3:
            new_chain = _reinsert_fragment(cur_chain, sequence, rng)
        else:
            new_chain = _fragment_relocation(cur_chain, sequence, rng)

        if new_chain is None:
            continue

        new_energy = _energy_fast(sequence, new_chain)
        delta = new_energy - cur_energy

        if delta <= 0:
            cur_chain = new_chain
            cur_energy = new_energy
            move_success[move_type] += 1
            if cur_energy < best_energy:
                best_chain = list(cur_chain)
                best_energy = cur_energy
        else:
            if T > 0 and rng.random() < math.exp(-delta / T):
                cur_chain = new_chain
                cur_energy = new_energy
                move_success[move_type] += 1

        # Adapt move weights every 150 iterations
        if it % 150 == 149:
            for k in range(5):
                if move_attempts[k] > 0:
                    rate = move_success[k] / move_attempts[k]
                    move_weights[k] = max(0.1, rate * 2.5)
                move_attempts[k] = 0
                move_success[k] = 0

    return best_chain, best_energy
>>>>>>> REPLACE

