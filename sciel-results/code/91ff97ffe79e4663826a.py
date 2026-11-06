"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _is_smooth(n):
    y = n
    for p in (2, 3, 5):
        while y % p == 0:
            y //= p
    return y == 1

def _divides(a, b):
    """Return True if a divides b."""
    return b % a == 0

def _is_antichain(terms, divides, val_to_idx):
    """Check if terms form a divisibility antichain."""
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            a, b = terms[i], terms[j]
            if a > b:
                a, b = b, a
            if _divides(a, b):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit

    # Generate all 2,3,5-smooth numbers up to 128
    smooth = [x for x in range(1, 129) if _is_smooth(x)]
    smooth_set = set(smooth)
    val_to_idx = {v: i for i, v in enumerate(smooth)}

    # Pre-compute divisibility relationships
    # div[i][j] = True if smooth[i] divides smooth[j]
    n_smooth = len(smooth)
    divides = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth[j] % smooth[i] == 0:
                divides[i][j] = True

    # For each target, find minimal antichain representation
    answers = [None] * 129  # 1-indexed, index 0 unused

    # Pre-compute antichains by sum for sizes 2, 3, 4, 5, 6
    # Only store the FIRST (smallest) antichain per sum to save memory
    antichains_by_sum = {2: {}, 3: {}, 4: {}, 5: {}, 6: {}}

    # Size 2
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            s = smooth[i] + smooth[j]
            if s > 128:
                break
            if divides[i][j] or divides[j][i]:
                continue
            if s not in antichains_by_sum[2]:
                antichains_by_sum[2][s] = (smooth[i], smooth[j])

    # Size 3
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if divides[i][j] or divides[j][i]:
                continue
            for k in range(j + 1, n_smooth):
                s = smooth[i] + smooth[j] + smooth[k]
                if s > 128:
                    break
                if divides[i][k] or divides[k][i]:
                    continue
                if divides[j][k] or divides[k][j]:
                    continue
                if s not in antichains_by_sum[3]:
                    antichains_by_sum[3][s] = (smooth[i], smooth[j], smooth[k])

    # Size 4
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if divides[i][j] or divides[j][i]:
                continue
            for k in range(j + 1, n_smooth):
                if divides[i][k] or divides[k][i]:
                    continue
                if divides[j][k] or divides[k][j]:
                    continue
                for l in range(k + 1, n_smooth):
                    s = smooth[i] + smooth[j] + smooth[k] + smooth[l]
                    if s > 128:
                        break
                    if divides[i][l] or divides[l][i]:
                        continue
                    if divides[j][l] or divides[l][j]:
                        continue
                    if divides[k][l] or divides[l][k]:
                        continue
                    if s not in antichains_by_sum[4]:
                        antichains_by_sum[4][s] = (smooth[i], smooth[j], smooth[k], smooth[l])

    # Size 5
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if divides[i][j] or divides[j][i]:
                continue
            for k in range(j + 1, n_smooth):
                if divides[i][k] or divides[k][i]:
                    continue
                if divides[j][k] or divides[k][j]:
                    continue
                for l in range(k + 1, n_smooth):
                    if divides[i][l] or divides[l][i]:
                        continue
                    if divides[j][l] or divides[l][j]:
                        continue
                    if divides[k][l] or divides[l][k]:
                        continue
                    for m in range(l + 1, n_smooth):
                        s = smooth[i] + smooth[j] + smooth[k] + smooth[l] + smooth[m]
                        if s > 128:
                            break
                        if divides[i][m] or divides[m][i]:
                            continue
                        if divides[j][m] or divides[m][j]:
                            continue
                        if divides[k][m] or divides[m][k]:
                            continue
                        if divides[l][m] or divides[m][l]:
                            continue
                        if s not in antichains_by_sum[5]:
                            antichains_by_sum[5][s] = (smooth[i], smooth[j], smooth[k], smooth[l], smooth[m])

    # Size 6
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if divides[i][j] or divides[j][i]:
                continue
            for k in range(j + 1, n_smooth):
                if divides[i][k] or divides[k][i]:
                    continue
                if divides[j][k] or divides[k][j]:
                    continue
                for l in range(k + 1, n_smooth):
                    if divides[i][l] or divides[l][i]:
                        continue
                    if divides[j][l] or divides[l][j]:
                        continue
                    if divides[k][l] or divides[l][k]:
                        continue
                    for m in range(l + 1, n_smooth):
                        if divides[i][m] or divides[m][i]:
                            continue
                        if divides[j][m] or divides[m][j]:
                            continue
                        if divides[k][m] or divides[m][k]:
                            continue
                        if divides[l][m] or divides[m][l]:
                            continue
                        for n2 in range(m + 1, n_smooth):
                            s = smooth[i] + smooth[j] + smooth[k] + smooth[l] + smooth[m] + smooth[n2]
                            if s > 128:
                                break
                            if divides[i][n2] or divides[n2][i]:
                                continue
                            if divides[j][n2] or divides[n2][j]:
                                continue
                            if divides[k][n2] or divides[n2][k]:
                                continue
                            if divides[l][n2] or divides[n2][l]:
                                continue
                            if divides[m][n2] or divides[n2][m]:
                                continue
                            if s not in antichains_by_sum[6]:
                                antichains_by_sum[6][s] = (smooth[i], smooth[j], smooth[k], smooth[l], smooth[m], smooth[n2])

    # Now solve each target
    total_terms = 0
    for n in range(1, 129):
        if time.time() > deadline:
            break

        answer = None

        # Try 1 term
        if n in smooth_set:
            answer = [n]
        else:
            # Try 2-6 terms from precomputed tables
            for count in range(2, 7):
                if n in antichains_by_sum[count]:
                    answer = list(antichains_by_sum[count][n])
                    break

        # Fallback for targets not covered by precomputed tables
        if answer is None:
            found = False
            for count in range(7, 11):
                if time.time() > deadline:
                    break
                # Recursive search with index-based antichain checking
                def search(idx, chosen_idx, current_sum, remaining_count):
                    if time.time() > deadline:
                        return None
                    if remaining_count == 0:
                        if current_sum == n:
                            return [smooth[i] for i in chosen_idx]
                        return None
                    if current_sum > n:
                        return None
                    for t in range(idx, n_smooth):
                        if current_sum + smooth[t] > n:
                            break
                        # Check antichain property with all chosen
                        valid = True
                        for ci in chosen_idx:
                            if divides[ci][t] or divides[t][ci]:
                                valid = False
                                break
                        if not valid:
                            continue
                        chosen_idx.append(t)
                        result = search(t + 1, chosen_idx, current_sum + smooth[t], remaining_count - 1)
                        if result is not None:
                            return result
                        chosen_idx.pop()
                    return None

                result = search(0, [], 0, count)
                if result is not None:
                    answer = result
                    found = True
                    break

            if not found and answer is None:
                # Last resort: try larger counts
                for count in range(11, 16):
                    if time.time() > deadline:
                        break
                    def search2(idx, chosen_idx, current_sum, remaining_count):
                        if time.time() > deadline:
                            return None
                        if remaining_count == 0:
                            if current_sum == n:
                                return [smooth[i] for i in chosen_idx]
                            return None
                        if current_sum > n:
                            return None
                        for t in range(idx, n_smooth):
                            if current_sum + smooth[t] > n:
                                break
                            valid = True
                            for ci in chosen_idx:
                                if divides[ci][t] or divides[t][ci]:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            chosen_idx.append(t)
                            result = search2(t + 1, chosen_idx, current_sum + smooth[t], remaining_count - 1)
                            if result is not None:
                                return result
                            chosen_idx.pop()
                        return None

                    result = search2(0, [], 0, count)
                    if result is not None:
                        answer = result
                        break

        if answer is None:
            # Should not happen, but provide a fallback
            if n in smooth_set:
                answer = [n]
            else:
                answer = [n] if n in smooth_set else [1, n-1] if n-1 in smooth_set and n-1 != 1 else [n]
        else:
            total_terms += len(answer)

        answers[n] = answer

    # Fill any remaining unfilled targets
    for n in range(1, 129):
        if answers[n] is None:
            for count in range(1, 16):
                if time.time() > deadline:
                    break
                def search_final(idx, chosen_idx, current_sum, remaining_count):
                    if time.time() > deadline:
                        return None
                    if remaining_count == 0:
                        if current_sum == n:
                            return [smooth[i] for i in chosen_idx]
                        return None
                    if current_sum > n:
                        return None
                    for t in range(idx, n_smooth):
                        if current_sum + smooth[t] > n:
                            break
                        valid = True
                        for ci in chosen_idx:
                            if divides[ci][t] or divides[t][ci]:
                                valid = False
                                break
                        if not valid:
                            continue
                        chosen_idx.append(t)
                        result = search_final(t + 1, chosen_idx, current_sum + smooth[t], remaining_count - 1)
                        if result is not None:
                            return result
                        chosen_idx.pop()
                    return None

                result = search_final(0, [], 0, count)
                if result is not None:
                    answers[n] = result
                    break
            else:
                answers[n] = [n] if n in smooth_set else [1, 2, 3]

    return answers[1:129]
# EVOLVE-BLOCK-END
