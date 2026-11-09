def construct():
    import time
    import random
    import math
    from fractions import Fraction

    deadline = time.time() + 85

    denominators = list(range(2, 49))
    n = len(denominators)
    vals = [1.0 / d for d in denominators]

    def greedy_init():
        sel = set()
        tot = 0.0
        for i in range(n):
            if tot + vals[i] < 1.0:
                sel.add(i)
                tot += vals[i]
        return sel, tot

    best_selected, best_total = greedy_init()
    best_deficit = 1.0 - best_total

    for restart in range(20):
        if time.time() > deadline:
            break

        if restart == 0:
            current, current_total = set(best_selected), best_total
        else:
            current = set()
            current_total = 0.0
            for i in range(n):
                if random.random() < 0.5:
                    current.add(i)
                    current_total += vals[i]
            while current_total >= 1.0 and current:
                i = random.choice(list(current))
                current.discard(i)
                current_total -= vals[i]

        current_deficit = 1.0 - current_total
        in_set = [i in current for i in range(n)]

        temperature = 0.5
        cooling = 0.99999
        no_improve = 0
        restart_limit = 50000

        for step in range(restart_limit):
            if time.time() > deadline:
                break

            move_type = random.random()

            if move_type < 0.33:
                cands = [i for i in range(n) if not in_set[i]]
                if not cands:
                    continue
                i = random.choice(cands)
                new_total = current_total + vals[i]
                if new_total >= 1.0:
                    continue
                new_deficit = 1.0 - new_total
                if new_deficit < current_deficit:
                    current.add(i)
                    in_set[i] = True
                    current_total = new_total
                    current_deficit = new_deficit
                    no_improve = 0
                    if current_deficit < best_deficit:
                        best_deficit = current_deficit
                        best_total = current_total
                        best_selected = set(current)
                elif temperature > 1e-15:
                    delta = new_deficit - current_deficit
                    if delta > 0:
                        prob = math.exp(-delta / max(temperature, 1e-15))
                        if random.random() < prob:
                            current.add(i)
                            in_set[i] = True
                            current_total = new_total
                            current_deficit = new_deficit
                            no_improve = 0
            elif move_type < 0.66:
                cands = [i for i in range(n) if in_set[i]]
                if len(cands) <= 1:
                    continue
                i = random.choice(cands)
                new_total = current_total - vals[i]
                new_deficit = 1.0 - new_total
                if new_deficit < current_deficit:
                    current.discard(i)
                    in_set[i] = False
                    current_total = new_total
                    current_deficit = new_deficit
                    no_improve = 0
                    if current_deficit < best_deficit:
                        best_deficit = current_deficit
                        best_total = current_total
                        best_selected = set(current)
                elif temperature > 1e-15:
                    delta = new_deficit - current_deficit
                    if delta > 0:
                        prob = math.exp(-delta / max(temperature, 1e-15))
                        if random.random() < prob:
                            current.discard(i)
                            in_set[i] = False
                            current_total = new_total
                            current_deficit = new_deficit
                            no_improve = 0
            else:
                in_c = [i for i in range(n) if in_set[i]]
                out_c = [i for i in range(n) if not in_set[i]]
                if not in_c or not out_c:
                    continue
                i_out = random.choice(in_c)
                i_in = random.choice(out_c)
                new_total = current_total - vals[i_out] + vals[i_in]
                if new_total >= 1.0:
                    continue
                new_deficit = 1.0 - new_total
                if new_deficit < current_deficit:
                    current.discard(i_out)
                    current.add(i_in)
                    in_set[i_out] = False
                    in_set[i_in] = True
                    current_total = new_total
                    current_deficit = new_deficit
                    no_improve = 0
                    if current_deficit < best_deficit:
                        best_deficit = current_deficit
                        best_total = current_total
                        best_selected = set(current)
                elif temperature > 1e-15:
                    delta = new_deficit - current_deficit
                    if delta > 0:
                        prob = math.exp(-delta / max(temperature, 1e-15))
                        if random.random() < prob:
                            current.discard(i_out)
                            current.add(i_in)
                            in_set[i_out] = False
                            in_set[i_in] = True
                            current_total = new_total
                            current_deficit = new_deficit
                            no_improve = 0

            temperature *= cooling
            no_improve += 1
            if no_improve > 5000:
                break

        if current_deficit < best_deficit:
            best_deficit = current_deficit
            best_total = current_total
            best_selected = set(current)

    # Verify with exact arithmetic
    exact_total = Fraction(0)
    for i in best_selected:
        exact_total += Fraction(1, denominators[i])

    if exact_total < 1:
        return list(best_selected)

    # Fallback greedy
    selected2 = []
    total2 = Fraction(0)
    for i, d in enumerate(denominators):
        if total2 + Fraction(1, d) < 1:
            selected2.append(i)
            total2 += Fraction(1, d)
    return selected2
