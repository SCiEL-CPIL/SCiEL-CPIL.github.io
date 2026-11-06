"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_smooth(n):
    y = n
    for p in (2, 3, 5):
        while y % p == 0:
            y //= p
    return y == 1

def _divides(a, b):
    return b % a == 0

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    # Generate all 2,3,5-smooth numbers up to 128
    smooth = [x for x in range(1, 129) if _is_smooth(x)]
    smooth_set = set(smooth)
    n_smooth = len(smooth)

    # Map value -> index in smooth list
    val_to_idx = {v: i for i, v in enumerate(smooth)}

    # Pre-compute divisibility: divides[i] = set of indices j where smooth[i] divides smooth[j]
    divides = [set() for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if smooth[j] % smooth[i] == 0:
                divides[i].add(j)
                # Also record reverse for quick check
                divides[j].add(i)  # mark that i divides j (for antichain check we need: neither divides other)

    # Actually let's be precise: divides[i] contains j if smooth[i] | smooth[j]
    # For antichain check: for pair (i,j), check j not in divides[i] and i not in divides[j]

    # Rebuild cleanly
    divides = [set() for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if smooth[j] % smooth[i] == 0:
                divides[i].add(j)

    def is_antichain_pair(i, j):
        """Check if smooth[i] and smooth[j] form an antichain (neither divides the other)."""
        if i == j:
            return False
        if j in divides[i]:
            return False
        if i in divides[j]:
            return False
        return True

    def is_antichain_set(indices):
        """Check if a set of indices forms a divisibility antichain."""
        n = len(indices)
        for a in range(n):
            for b in range(a + 1, n):
                ia, ib = indices[a], indices[b]
                if ib in divides[ia] or ia in divides[ib]:
                    return False
        return True

    # For each target, find minimal antichain representation
    answers = [None] * 129

    # Pre-compute antichain pairs by sum
    pairs_by_sum = {}
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if not is_antichain_pair(i, j):
                continue
            s = smooth[i] + smooth[j]
            if s <= 128:
                if s not in pairs_by_sum:
                    pairs_by_sum[s] = []
                pairs_by_sum[s].append((i, j))

    # Pre-compute antichain triples by sum
    triples_by_sum = {}
    for i in range(n_smooth):
        si = smooth[i]
        for j in range(i + 1, n_smooth):
            if not is_antichain_pair(i, j):
                continue
            sij = si + smooth[j]
            if sij > 128:
                continue
            for k in range(j + 1, n_smooth):
                if not is_antichain_pair(i, k):
                    continue
                if not is_antichain_pair(j, k):
                    continue
                s = sij + smooth[k]
                if s <= 128:
                    if s not in triples_by_sum:
                        triples_by_sum[s] = []
                    triples_by_sum[s].append((i, j, k))

    # Pre-compute antichain quadruples by sum
    quads_by_sum = {}
    for i in range(n_smooth):
        si = smooth[i]
        for j in range(i + 1, n_smooth):
            if not is_antichain_pair(i, j):
                continue
            sij = si + smooth[j]
            if sij > 128:
                continue
            for k in range(j + 1, n_smooth):
                if not is_antichain_pair(i, k):
                    continue
                if not is_antichain_pair(j, k):
                    continue
                sijk = sij + smooth[k]
                if sijk > 128:
                    continue
                for l in range(k + 1, n_smooth):
                    if not is_antichain_pair(i, l):
                        continue
                    if not is_antichain_pair(j, l):
                        continue
                    if not is_antichain_pair(k, l):
                        continue
                    s = sijk + smooth[l]
                    if s <= 128:
                        if s not in quads_by_sum:
                            quads_by_sum[s] = []
                        quads_by_sum[s].append((i, j, k, l))

    # Solve each target
    for n in range(1, 129):
        if time.time() > deadline:
            break

        answer = None

        # Try 1 term
        if n in smooth_set:
            answer = [n]
        else:
            # Try 2 terms
            if n in pairs_by_sum and pairs_by_sum[n]:
                idxs = pairs_by_sum[n][0]
                answer = [smooth[idxs[0]], smooth[idxs[1]]]
            elif n in triples_by_sum and triples_by_sum[n]:
                idxs = triples_by_sum[n][0]
                answer = [smooth[idxs[0]], smooth[idxs[1]], smooth[idxs[2]]]
            elif n in quads_by_sum and quads_by_sum[n]:
                idxs = quads_by_sum[n][0]
                answer = [smooth[idxs[0]], smooth[idxs[1]], smooth[idxs[2]], smooth[idxs[3]]]
            else:
                # DFS for 5+ terms
                found = False
                for count in range(5, 11):
                    if time.time() > deadline:
                        break
                    result = None
                    # Iterative DFS with stack
                    stack = [(0, [], 0)]
                    while stack:
                        if time.time() > deadline:
                            break
                        idx_start, chosen, cur_sum = stack.pop()
                        if len(chosen) == count:
                            if cur_sum == n:
                                # Verify antichain
                                ok = True
                                for a in range(len(chosen)):
                                    for b in range(a + 1, len(chosen)):
                                        ia, ib = chosen[a], chosen[b]
                                        if ib in divides[ia] or ia in divides[ib]:
                                            ok = False
                                            break
                                    if not ok:
                                        break
                                if ok:
                                    result = [smooth[i] for i in chosen]
                            continue
                        remaining = count - len(chosen)
                        for t in range(idx_start, n_smooth):
                            new_sum = cur_sum + smooth[t]
                            if new_sum > n:
                                break
                            # Check antichain with all chosen
                            valid = True
                            for c in chosen:
                                if t in divides[c] or c in divides[t]:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            # Pruning: can we still reach n?
                            # Minimum possible sum for remaining-1 more terms (smallest available after t)
                            if remaining - 1 > 0:
                                min_add = 0
                                for m in range(remaining - 1):
                                    if t + 1 + m < n_smooth:
                                        min_add += smooth[t + 1 + m]
                                    else:
                                        min_add = n + 1
                                        break
                                if new_sum + min_add > n:
                                    continue
                            stack.append((t + 1, chosen + [t], new_sum))
                    if result is not None:
                        answer = result
                        found = True
                        break

        if answer is not None:
            answers[n] = answer

    # Fill any remaining
    for n in range(1, 129):
        if answers[n] is None:
            # Emergency: try harder
            for count in range(1, 15):
                if time.time() > deadline:
                    break
                stack = [(0, [], 0)]
                while stack:
                    if time.time() > deadline:
                        break
                    idx_start, chosen, cur_sum = stack.pop()
                    if len(chosen) == count:
                        if cur_sum == n:
                            ok = True
                            for a in range(len(chosen)):
                                for b in range(a + 1, len(chosen)):
                                    ia, ib = chosen[a], chosen[b]
                                    if ib in divides[ia] or ia in divides[ib]:
                                        ok = False
                                        break
                                if not ok:
                                    break
                            if ok:
                                answers[n] = [smooth[i] for i in chosen]
                                break
                        continue
                    remaining = count - len(chosen)
                    for t in range(idx_start, n_smooth):
                        new_sum = cur_sum + smooth[t]
                        if new_sum > n:
                            break
                        valid = True
                        for c in chosen:
                            if t in divides[c] or c in divides[t]:
                                valid = False
                                break
                        if not valid:
                            continue
                        stack.append((t + 1, chosen + [t], new_sum))
                if answers[n] is not None:
                    break
            if answers[n] is None:
                # Last resort - should not happen
                answers[n] = [n] if n in smooth_set else [1]

    return answers[1:129]
# EVOLVE-BLOCK-END
