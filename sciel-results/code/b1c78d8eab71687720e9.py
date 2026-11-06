"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import math
    from fractions import Fraction

    deadline = time.time() + 82  # 82 second deadline for safety (90s limit)

    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    unit_vals = [Fraction(1, d) for d in denominators]
    
    # Precompute float values for fast comparison
    unit_floats = [float(1.0 / d) for d in denominators]

    def greedy(order):
        selected = set()
        total = Fraction(0)
        total_f = 0.0
        for idx in order:
            v = unit_vals[idx]
            vf = unit_floats[idx]
            if total_f + vf < 0.9999999999:  # use float for speed, verify with Fraction
                selected.add(idx)
                total += v
                total_f += vf
        # Verify strictly below 1 with exact arithmetic
        if total >= 1:
            # Remove elements until strictly below 1 (shouldn't happen often with float guard)
            while total >= 1 and selected:
                # Remove the largest element to drop below 1 fastest
                max_idx = max(selected, key=lambda i: unit_vals[i])
                selected.remove(max_idx)
                total -= unit_vals[max_idx]
                total_f -= unit_floats[max_idx]
        return selected, total

    best_indices = []
    best_total = Fraction(0)
    best_deficit = Fraction(1)

    # Try multiple orderings for greedy to get good starting points
    random.seed(42)
    orderings = [
        list(range(n)),  # ascending: 2,3,...,48
        list(reversed(range(n))),  # descending: 48,...,3,2
        sorted(range(n), key=lambda i: -denominators[i]),  # same as reversed
        sorted(range(n), key=lambda i: denominators[i]),  # same as ascending
        # Alternating
        list(range(0, n, 2)) + list(range(1, n, 2)),
        list(range(1, n, 2)) + list(range(0, n, 2)),
    ]
    for _ in range(100):
        o = list(range(n))
        random.shuffle(o)
        orderings.append(o)

    for order in orderings:
        idxs, total = greedy(order)
        if idxs:
            deficit = Fraction(1) - total
            if deficit > 0 and deficit < best_deficit:
                best_deficit = deficit
                best_total = total
                best_indices = sorted(idxs)

    def local_search(start_set, max_iterations=200000, temperature_init=0.01):
        nonlocal best_indices, best_total, best_deficit

        if not start_set:
            return

        cur = set(start_set)
        cur_total = sum(unit_vals[i] for i in cur)
        cur_deficit = Fraction(1) - cur_total
        if cur_deficit <= 0:
            return

        not_in_set = set(range(n)) - cur
        temperature = Fraction(temperature_init)
        cooling = 0.9995
        min_temperature = Fraction(1, 1000000)

        iters = 0
        while iters < max_iterations and time.time() < deadline:
            iters += 1
            if iters % 500 == 0 and temperature < min_temperature:
                temperature = min_temperature

            # Choose a random move
            move_type = random.random()
            best_new_def = cur_deficit
            best_new_set = None
            best_new_total = None

            if move_type < 0.25 and not_in_set:
                # Add a random element
                add = random.choice(tuple(not_in_set))
                nt = cur_total + unit_vals[add]
                if nt < 1:
                    nd = Fraction(1) - nt
                    if nd < best_new_def:
                        best_new_def = nd
                        best_new_set = cur | {add}
                        best_new_total = nt
            elif move_type < 0.45 and len(cur) > 1:
                # Remove a random element
                rm = random.choice(tuple(cur))
                nt = cur_total - unit_vals[rm]
                nd = Fraction(1) - nt
                if nd < best_new_def:
                    best_new_def = nd
                    best_new_set = cur - {rm}
                    best_new_total = nt
            elif move_type < 0.75 and len(cur) > 0 and not_in_set:
                # 1-swap
                rm = random.choice(tuple(cur))
                add = random.choice(tuple(not_in_set))
                nt = cur_total - unit_vals[rm] + unit_vals[add]
                if nt < 1:
                    nd = Fraction(1) - nt
                    if nd < best_new_def:
                        best_new_def = nd
                        best_new_set = (cur - {rm}) | {add}
                        best_new_total = nt
            elif move_type < 0.9 and len(cur) >= 2 and len(not_in_set) >= 2:
                # 2-swap
                rm1, rm2 = random.sample(tuple(cur), 2)
                add1, add2 = random.sample(tuple(not_in_set), 2)
                nt = cur_total - unit_vals[rm1] - unit_vals[rm2] + unit_vals[add1] + unit_vals[add2]
                if nt < 1:
                    nd = Fraction(1) - nt
                    if nd < best_new_def:
                        best_new_def = nd
                        best_new_set = (cur - {rm1, rm2}) | {add1, add2}
                        best_new_total = nt
            else:
                # 3-swap (more aggressive)
                if len(cur) >= 3 and len(not_in_set) >= 3:
                    rm1, rm2, rm3 = random.sample(tuple(cur), 3)
                    add1, add2, add3 = random.sample(tuple(not_in_set), 3)
                    nt = cur_total - unit_vals[rm1] - unit_vals[rm2] - unit_vals[rm3] + unit_vals[add1] + unit_vals[add2] + unit_vals[add3]
                    if nt < 1:
                        nd = Fraction(1) - nt
                        if nd < best_new_def:
                            best_new_def = nd
                            best_new_set = (cur - {rm1, rm2, rm3}) | {add1, add2, add3}
                            best_new_total = nt

            if best_new_set is not None:
                # Accept if better or with probability exp(-delta/T)
                delta = best_new_def - cur_deficit
                if delta < 0:
                    # Better, always accept
                    cur = best_new_set
                    cur_total = best_new_total
                    cur_deficit = best_new_def
                    not_in_set = set(range(n)) - cur
                    if cur_deficit < best_deficit:
                        best_deficit = cur_deficit
                        best_total = cur_total
                        best_indices = sorted(cur)
                else:
                    # Worse, accept with probability
                    # Convert to float for exp computation
                    delta_f = float(delta)
                    temp_f = float(temperature)
                    if temp_f > 0 and delta_f < 100 * temp_f:
                        if random.random() < math.exp(-delta_f / temp_f):
                            cur = best_new_set
                            cur_total = best_new_total
                            cur_deficit = best_new_def
                            not_in_set = set(range(n)) - cur
                    temperature *= cooling

        return cur_deficit

    # Run local search from best greedy
    local_search(set(best_indices), max_iterations=300000, temperature_init=0.02)

    # Try random restarts with local search
    restart_count = 0
    while time.time() < deadline - 2:
        o = list(range(n))
        random.shuffle(o)
        idxs, total = greedy(o)
        if idxs:
            deficit = Fraction(1) - total
            if deficit > 0 and deficit < best_deficit:
                best_deficit = deficit
                best_total = total
                best_indices = sorted(idxs)
            local_search(idxs, max_iterations=50000, temperature_init=0.01)
        restart_count += 1
        if restart_count % 10 == 0:
            # Also try a fresh random set
            k = random.randint(5, 15)
            random_set = set(random.sample(range(n), k))
            rt = sum(unit_vals[i] for i in random_set)
            if rt < 1:
                rd = Fraction(1) - rt
                if rd < best_deficit:
                    best_deficit = rd
                    best_total = rt
                    best_indices = sorted(random_set)
                local_search(random_set, max_iterations=20000, temperature_init=0.01)

    # Final: ensure we return a valid solution
    if not best_indices:
        best_indices = [0]
        best_total = Fraction(1, 2)

    return best_indices
# EVOLVE-BLOCK-END
