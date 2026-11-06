"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import math
    from fractions import Fraction

    start_time = time.time()
    deadline = start_time + 55  # 55s deadline for safety (60s limit)

    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    unit_floats = [1.0 / d for d in denominators]
    
    # Precompute Fraction values only when needed for final verification
    unit_fracs = [Fraction(1, d) for d in denominators]

    def greedy(order):
        selected = set()
        total_f = 0.0
        for idx in order:
            vf = unit_floats[idx]
            if total_f + vf < 0.9999999999:
                selected.add(idx)
                total_f += vf
        # Verify strictly below 1 with exact arithmetic
        total = sum(unit_fracs[i] for i in selected)
        if total >= 1:
            while total >= 1 and selected:
                max_idx = max(selected, key=lambda i: unit_floats[i])
                selected.remove(max_idx)
                total -= unit_fracs[max_idx]
        return selected, total

    best_indices = []
    best_total_f = 0.0
    best_deficit_f = 1.0

    random.seed(42)
    
    # Try multiple orderings for greedy to get good starting points
    orderings = [
        list(range(n)),  # ascending
        list(reversed(range(n))),  # descending
        list(range(0, n, 2)) + list(range(1, n, 2)),
        list(range(1, n, 2)) + list(range(0, n, 2)),
    ]
    for _ in range(50):
        o = list(range(n))
        random.shuffle(o)
        orderings.append(o)

    for order in orderings:
        idxs, total = greedy(order)
        if idxs:
            deficit = Fraction(1) - total
            if deficit > 0:
                deficit_f = float(deficit)
                if deficit_f < best_deficit_f:
                    best_deficit_f = deficit_f
                    best_total_f = float(total)
                    best_indices = sorted(idxs)

    def local_search(start_set, max_iterations=50000, temperature_init=0.01):
        nonlocal best_indices, best_total_f, best_deficit_f

        if not start_set:
            return

        cur = set(start_set)
        cur_total_f = sum(unit_floats[i] for i in cur)
        cur_deficit_f = 1.0 - cur_total_f
        if cur_deficit_f <= 0:
            return

        not_in_set = list(set(range(n)) - cur)
        cur_list = list(cur)
        temperature = temperature_init
        cooling = 0.999
        min_temperature = 1e-8

        iters = 0
        while iters < max_iterations and time.time() < deadline:
            iters += 1
            if iters % 1000 == 0 and temperature < min_temperature:
                temperature = min_temperature

            move_type = random.random()
            best_new_def_f = cur_deficit_f
            best_new_set = None
            best_new_total_f = None

            if move_type < 0.25 and not_in_set:
                add = random.choice(not_in_set)
                nt = cur_total_f + unit_floats[add]
                if nt < 1.0:
                    nd = 1.0 - nt
                    if nd < best_new_def_f:
                        best_new_def_f = nd
                        best_new_set = cur | {add}
                        best_new_total_f = nt
            elif move_type < 0.45 and len(cur) > 1:
                rm = random.choice(cur_list)
                nt = cur_total_f - unit_floats[rm]
                nd = 1.0 - nt
                if nd < best_new_def_f:
                    best_new_def_f = nd
                    best_new_set = cur - {rm}
                    best_new_total_f = nt
            elif move_type < 0.75 and len(cur) > 0 and not_in_set:
                rm = random.choice(cur_list)
                add = random.choice(not_in_set)
                nt = cur_total_f - unit_floats[rm] + unit_floats[add]
                if nt < 1.0:
                    nd = 1.0 - nt
                    if nd < best_new_def_f:
                        best_new_def_f = nd
                        best_new_set = (cur - {rm}) | {add}
                        best_new_total_f = nt
            elif move_type < 0.9 and len(cur) >= 2 and len(not_in_set) >= 2:
                rm1, rm2 = random.sample(cur_list, 2)
                add1, add2 = random.sample(not_in_set, 2)
                nt = cur_total_f - unit_floats[rm1] - unit_floats[rm2] + unit_floats[add1] + unit_floats[add2]
                if nt < 1.0:
                    nd = 1.0 - nt
                    if nd < best_new_def_f:
                        best_new_def_f = nd
                        best_new_set = (cur - {rm1, rm2}) | {add1, add2}
                        best_new_total_f = nt
            else:
                if len(cur) >= 3 and len(not_in_set) >= 3:
                    rm1, rm2, rm3 = random.sample(cur_list, 3)
                    add1, add2, add3 = random.sample(not_in_set, 3)
                    nt = cur_total_f - unit_floats[rm1] - unit_floats[rm2] - unit_floats[rm3] + unit_floats[add1] + unit_floats[add2] + unit_floats[add3]
                    if nt < 1.0:
                        nd = 1.0 - nt
                        if nd < best_new_def_f:
                            best_new_def_f = nd
                            best_new_set = (cur - {rm1, rm2, rm3}) | {add1, add2, add3}
                            best_new_total_f = nt

            if best_new_set is not None:
                delta = best_new_def_f - cur_deficit_f
                if delta < 0:
                    cur = best_new_set
                    cur_total_f = best_new_total_f
                    cur_deficit_f = best_new_def_f
                    not_in_set = list(set(range(n)) - cur)
                    cur_list = list(cur)
                    if cur_deficit_f < best_deficit_f:
                        best_deficit_f = cur_deficit_f
                        best_total_f = cur_total_f
                        best_indices = sorted(cur)
                else:
                    if temperature > 0 and delta < 100 * temperature:
                        if random.random() < math.exp(-delta / temperature):
                            cur = best_new_set
                            cur_total_f = best_new_total_f
                            cur_deficit_f = best_new_def_f
                            not_in_set = list(set(range(n)) - cur)
                            cur_list = list(cur)
                temperature *= cooling

        return cur_deficit_f

    # Run local search from best greedy
    if best_indices:
        local_search(set(best_indices), max_iterations=100000, temperature_init=0.02)

    # Try random restarts with local search
    restart_count = 0
    while time.time() < deadline - 2:
        o = list(range(n))
        random.shuffle(o)
        idxs, total = greedy(o)
        if idxs:
            deficit = Fraction(1) - total
            if deficit > 0:
                deficit_f = float(deficit)
                if deficit_f < best_deficit_f:
                    best_deficit_f = deficit_f
                    best_total_f = float(total)
                    best_indices = sorted(idxs)
            local_search(idxs, max_iterations=20000, temperature_init=0.01)
        restart_count += 1
        if restart_count % 5 == 0:
            k = random.randint(5, 15)
            random_set = set(random.sample(range(n), k))
            rt = sum(unit_floats[i] for i in random_set)
            if rt < 1.0:
                rd = 1.0 - rt
                if rd < best_deficit_f:
                    best_deficit_f = rd
                    best_total_f = rt
                    best_indices = sorted(random_set)
                local_search(random_set, max_iterations=10000, temperature_init=0.01)

    # Final: ensure we return a valid solution
    if not best_indices:
        best_indices = [0]

    # Verify final solution with exact arithmetic
    final_total = sum(unit_fracs[i] for i in best_indices)
    if final_total >= 1:
        # Fallback to a simple valid solution
        best_indices = [0]

    return best_indices
# EVOLVE-BLOCK-END
