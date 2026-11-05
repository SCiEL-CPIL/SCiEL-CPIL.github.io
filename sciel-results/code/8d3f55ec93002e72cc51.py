def construct():
    import time
    start = time.time()
    deadline = start + 80

    def get_subset_sums(elements):
        sums = [0.0]
        for a in elements:
            inv = 1.0 / a
            sums = sums + [s + inv for s in sums]
        return sums

    def can_add(current_sums, new_inv):
        sset = set(current_sums)
        for s in current_sums:
            if (new_inv + s) in sset:
                return False
        return True

    def greedy_build(order):
        current = []
        current_sums = [0.0]
        for x in order:
            if time.time() > deadline - 5:
                break
            inv = 1.0 / x
            if can_add(current_sums, inv):
                current.append(x)
                current_sums = current_sums + [s + inv for s in current_sums]
        return current

    orders = [
        list(range(22, 0, -1)),
        list(range(1, 23)),
        [2, 3, 5, 7, 11, 13, 17, 19, 1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22],
    ]

    best = []
    for order in orders:
        if time.time() > deadline - 10:
            break
        result = greedy_build(order)
        if len(result) > len(best):
            best = result

    if time.time() < deadline - 15 and len(best) > 0:
        best = list(best)
        in_best = set(best)
        not_in_best = [x for x in range(1, 23) if x not in in_best]
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            for new_x in not_in_best[:]:
                if time.time() > deadline - 15:
                    break
                trial_sums = get_subset_sums(best)
                inv = 1.0 / new_x
                if can_add(trial_sums, inv):
                    best.append(new_x)
                    not_in_best.remove(new_x)
                    improved = True
                    break
            if not improved:
                for i, old_x in enumerate(best):
                    if time.time() > deadline - 15:
                        break
                    for new_x in not_in_best[:]:
                        if time.time() > deadline - 15:
                            break
                        trial = best[:i] + best[i + 1:]
                        trial_sums = get_subset_sums(trial)
                        inv = 1.0 / new_x
                        if can_add(trial_sums, inv):
                            best = trial + [new_x]
                            in_best.discard(old_x)
                            in_best.add(new_x)
                            not_in_best.remove(new_x)
                            not_in_best.append(old_x)
                            improved = True
                            break
                    if improved:
                        break

    return best if best else [1]
