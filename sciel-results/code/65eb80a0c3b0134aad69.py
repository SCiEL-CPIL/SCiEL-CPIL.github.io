def construct():
    from fractions import Fraction
    import time

    deadline = time.time() + 50  # 50 second deadline for safety

    # Denominators are 2 to 48, indices 0 to 46
    denoms = list(range(2, 49))
    n = len(denoms)
    fracs = [Fraction(1, d) for d in denoms]

    # DP: track all achievable sums strictly below 1, with one witness each
    # Use a dict: sum -> list of indices
    achievable = {Fraction(0): []}

    for i, frac in enumerate(fracs):
        new_achievable = {}
        # Carry over existing sums (without adding current item)
        for s, idxs in achievable.items():
            if s < 1:
                if s not in new_achievable:
                    new_achievable[s] = idxs[:]
        # Add current item to existing sums
        for s, idxs in achievable.items():
            if s < 1:
                new_s = s + frac
                if new_s < 1:
                    if new_s not in new_achievable:
                        new_achievable[new_s] = idxs + [i]
        achievable = new_achievable

        if time.time() > deadline:
            break

    # Find the sum closest to 1 from below
    best_sum = Fraction(0)
    best_idxs = []
    for s, idxs in achievable.items():
        if s < 1 and s > best_sum:
            best_sum = s
            best_idxs = idxs

    if not best_idxs:
        return [0]  # 1/2 fallback

    # Local search refinement: try 1-swap to see if we can get closer to 1
    # (in case the DP witness isn't optimal for local improvements)
    chosen = list(best_idxs)
    total = best_sum
    deficit = Fraction(1, 1) - total
    best_def = deficit
    chosen_set = set(chosen)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for pos in range(len(chosen)):
            cur_i = chosen[pos]
            cur_val = Fraction(1, denoms[cur_i])
            for j in range(n):
                if j in chosen_set:
                    continue
                delta = Fraction(1, denoms[j]) - cur_val
                new_def = deficit - delta
                if new_def <= 0:
                    continue
                if new_def < best_def:
                    new_chosen = chosen[:pos] + [j] + chosen[pos + 1:]
                    new_total = total - cur_val + Fraction(1, denoms[j])
                    chosen = new_chosen
                    total = new_total
                    deficit = new_def
                    best_def = new_def
                    chosen_set = set(chosen)
                    improved = True
                    break
            if improved:
                break

    if not chosen:
        return [0]

    # Final validity check
    s = Fraction(0)
    for i in chosen:
        s += Fraction(1, denoms[i])
    if s >= 1:
        return best_idxs  # fallback to DP result

    return chosen
