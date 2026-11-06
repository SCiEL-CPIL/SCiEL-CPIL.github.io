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

def _is_antichain(terms):
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

    # Pre-compute divisibility relationships
    # div[i][j] = True if smooth[i] divides smooth[j]
    n_smooth = len(smooth)
    divides = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth[j] % smooth[i] == 0:
                divides[i][j] = True

    # For each target, find minimal antichain representation
    # We'll search by increasing number of terms (1, 2, 3, ...)
    answers = [None] * 129  # 1-indexed, index 0 unused

    # Pre-compute all valid antichains up to size 3 (likely sufficient for minimal)
    # For larger targets, we may need more terms, but let's start with small counts

    # Strategy: For each target n, try counts 1, 2, 3, ... until found
    # Use bitmasks or direct combination enumeration

    # To speed up, pre-compute sums of all antichains of size 1, 2, 3
    # Size 1: just the smooth numbers themselves
    # Size 2: pairs (a,b) where a+b <= 128, a!=b, neither divides the other
    # Size 3: triples that form antichains and sum <= 128

    # Let's build lookup: sum -> list of antichains (by term count)
    # But we want minimal count per target

    # Efficient approach: For each target, search combinations in order of count
    # Cache results to avoid recomputation

    # Pre-compute all antichains of size 2
    antichains_by_sum_2 = {}
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a, b = smooth[i], smooth[j]
            s = a + b
            if s > 128:
                continue
            # Check antichain: neither divides the other
            if divides[i][j] or divides[j][i]:
                continue
            if s not in antichains_by_sum_2:
                antichains_by_sum_2[s] = []
            antichains_by_sum_2[s].append((a, b))

    # Pre-compute all antichains of size 3
    antichains_by_sum_3 = {}
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a, b = smooth[i], smooth[j]
            # Quick check: a and b don't divide each other
            if divides[i][j] or divides[j][i]:
                continue
            for k in range(j + 1, n_smooth):
                c = smooth[k]
                s = a + b + c
                if s > 128:
                    continue
                # Check c doesn't divide or get divided by a or b
                if divides[i][k] or divides[k][i]:
                    continue
                if divides[j][k] or divides[k][j]:
                    continue
                if s not in antichains_by_sum_3:
                    antichains_by_sum_3[s] = []
                antichains_by_sum_3[s].append((a, b, c))

    # Pre-compute all antichains of size 4 (may be needed for some targets)
    antichains_by_sum_4 = {}
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a, b = smooth[i], smooth[j]
            if divides[i][j] or divides[j][i]:
                continue
            for k in range(j + 1, n_smooth):
                c = smooth[k]
                if divides[i][k] or divides[k][i]:
                    continue
                if divides[j][k] or divides[k][j]:
                    continue
                for l in range(k + 1, n_smooth):
                    d = smooth[l]
                    s = a + b + c + d
                    if s > 128:
                        continue
                    # Check d against a, b, c
                    if divides[i][l] or divides[l][i]:
                        continue
                    if divides[j][l] or divides[l][j]:
                        continue
                    if divides[k][l] or divides[l][k]:
                        continue
                    if s not in antichains_by_sum_4:
                        antichains_by_sum_4[s] = []
                    antichains_by_sum_4[s].append((a, b, c, d))

    # Now solve each target
    total_terms = 0
    for n in range(1, 129):
        if time.time() > deadline:
            # Return what we have; fill remaining with fallback
            break

        answer = None

        # Try 1 term
        if n in smooth_set:
            answer = [n]
        else:
            # Try 2 terms
            if n in antichains_by_sum_2:
                answer = list(antichains_by_sum_2[n][0])
            else:
                # Try 3 terms
                if n in antichains_by_sum_3:
                    answer = list(antichains_by_sum_3[n][0])
                else:
                    # Try 4 terms
                    if n in antichains_by_sum_4:
                        answer = list(antichains_by_sum_4[n][0])
                    else:
                        # Fallback: brute force search for 5+ terms
                        # This should be rare
                        found = False
                        for count in range(5, 10):
                            if time.time() > deadline:
                                break
                            # Generate combinations of smooth numbers summing to n
                            # Use recursive search with pruning
                            def search(idx, remaining, chosen, current_sum):
                                if time.time() > deadline:
                                    return None
                                if len(chosen) == count:
                                    if current_sum == n and _is_antichain(chosen):
                                        return list(chosen)
                                    return None
                                if current_sum > n:
                                    return None
                                # Pruning: even if we add the smallest possible (count - len(chosen)) times, can we reach n?
                                min_possible = current_sum
                                for t in range(idx, n_smooth):
                                    if min_possible + smooth[t] > n:
                                        break
                                    if len(chosen) > 0:
                                        # Check antichain property with last chosen
                                        last = chosen[-1]
                                        if divides[smooth.index(last)][t] or divides[t][smooth.index(last)]:
                                            continue
                                    chosen.append(smooth[t])
                                    result = search(t + 1, remaining - smooth[t], chosen, current_sum + smooth[t])
                                    if result is not None:
                                        return result
                                    chosen.pop()
                                return None

                            result = search(0, n, [], 0)
                            if result is not None:
                                answer = result
                                found = True
                                break

                        if not found and answer is None:
                            # Last resort: use a valid but possibly non-minimal representation
                            # Try larger counts
                            for count in range(10, 15):
                                if time.time() > deadline:
                                    break
                                def search2(idx, chosen, current_sum):
                                    if time.time() > deadline:
                                        return None
                                    if len(chosen) == count:
                                        if current_sum == n and _is_antichain(chosen):
                                            return list(chosen)
                                        return None
                                    if current_sum > n:
                                        return None
                                    for t in range(idx, n_smooth):
                                        if current_sum + smooth[t] > n:
                                            break
                                        if len(chosen) > 0:
                                            last = chosen[-1]
                                            if divides[smooth.index(last)][t] or divides[t][smooth.index(last)]:
                                                continue
                                        chosen.append(smooth[t])
                                        result = search2(t + 1, chosen, current_sum + smooth[t])
                                        if result is not None:
                                            return result
                                        chosen.pop()
                                    return None

                                result = search2(0, [], 0)
                                if result is not None:
                                    answer = result
                                    break

        if answer is None:
            # Should not happen, but provide a fallback
            # Use n itself if smooth, else find any valid representation
            if n in smooth_set:
                answer = [n]
            else:
                # This is a problem; try a simple construction
                # For now, return empty which will be invalid, but shouldn't happen
                answer = [n] if n in smooth_set else [1, n-1] if n-1 in smooth_set and n-1 != 1 else [n]
        else:
            total_terms += len(answer)

        answers[n] = answer

    # Fill any remaining unfilled targets with a valid (possibly non-optimal) representation
    for n in range(1, 129):
        if answers[n] is None:
            # Try to find any valid antichain
            for count in range(1, 15):
                if time.time() > deadline:
                    break
                def search_final(idx, chosen, current_sum):
                    if time.time() > deadline:
                        return None
                    if len(chosen) == count:
                        if current_sum == n and _is_antichain(chosen):
                            return list(chosen)
                        return None
                    if current_sum > n:
                        return None
                    for t in range(idx, n_smooth):
                        if current_sum + smooth[t] > n:
                            break
                        if len(chosen) > 0:
                            last = chosen[-1]
                            if smooth.index(last) < n_smooth and t < n_smooth:
                                if divides[smooth.index(last)][t] or divides[t][smooth.index(last)]:
                                    continue
                        chosen.append(smooth[t])
                        result = search_final(t + 1, chosen, current_sum + smooth[t])
                        if result is not None:
                            return result
                        chosen.pop()
                    return None

                result = search_final(0, [], 0)
                if result is not None:
                    answers[n] = result
                    break
            else:
                # Last resort
                answers[n] = [n] if n in smooth_set else [1, 2, 3]

    return answers[1:129]
# EVOLVE-BLOCK-END
