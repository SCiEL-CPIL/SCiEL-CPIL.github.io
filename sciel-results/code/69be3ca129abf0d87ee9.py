# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin below 90s limit

    # Generate all 5-smooth numbers up to 128
    smooth_numbers = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth_numbers.append(x)

    n_smooth = len(smooth_numbers)

    # Precompute divisibility relation: divs[i][j] = True if smooth_numbers[i] divides smooth_numbers[j]
    divs = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if smooth_numbers[j] % smooth_numbers[i] == 0:
                divs[i][j] = True

    # Precompute sum-to-index lookup for quick k=1 and k=2 checks
    val_to_idx = {}
    for i, v in enumerate(smooth_numbers):
        val_to_idx[v] = i

    answers = []
    total_terms = 0

    for n in range(1, 129):
        if time.time() - start_time > timeout:
            # Return best effort
            break

        # Candidates are smooth numbers <= n
        cand_indices = [i for i in range(n_smooth) if smooth_numbers[i] <= n]
        nc = len(cand_indices)

        found = None

        # k=1
        if n in val_to_idx:
            found = [n]
        else:
            # k=2
            found = None
            for i in range(nc):
                ai = smooth_numbers[cand_indices[i]]
                rem = n - ai
                if rem <= ai:
                    break
                if rem in val_to_idx:
                    j = val_to_idx[rem]
                    # Check antichain: neither divides the other
                    if not divs[cand_indices[i]][j] and not divs[j][cand_indices[i]]:
                        found = [ai, rem]
                        break

        if found is None:
            # k=3
            found = None
            for i in range(nc):
                ai = smooth_numbers[cand_indices[i]]
                if ai * 3 >= n:
                    break
                for j in range(i + 1, nc):
                    aj = smooth_numbers[cand_indices[j]]
                    if ai + aj * 2 >= n:
                        break
                    # Check i,j antichain
                    if divs[cand_indices[i]][cand_indices[j]] or divs[cand_indices[j]][cand_indices[i]]:
                        continue
                    rem = n - ai - aj
                    if rem <= aj:
                        break
                    if rem in val_to_idx:
                        k_idx = val_to_idx[rem]
                        # Check k with i and j
                        if (not divs[cand_indices[i]][k_idx] and not divs[k_idx][cand_indices[i]] and
                            not divs[cand_indices[j]][k_idx] and not divs[k_idx][cand_indices[j]]):
                            found = [ai, aj, rem]
                            break
                if found:
                    break

        if found is None:
            # k=4
            found = None
            for i in range(nc):
                ai = smooth_numbers[cand_indices[i]]
                if ai * 4 >= n:
                    break
                for j in range(i + 1, nc):
                    aj = smooth_numbers[cand_indices[j]]
                    if ai + aj * 3 >= n:
                        break
                    if divs[cand_indices[i]][cand_indices[j]] or divs[cand_indices[j]][cand_indices[i]]:
                        continue
                    for k_idx in range(j + 1, nc):
                        ak = smooth_numbers[cand_indices[k_idx]]
                        if ai + aj + ak * 2 >= n:
                            break
                        if (divs[cand_indices[i]][cand_indices[k_idx]] or divs[cand_indices[k_idx]][cand_indices[i]] or
                            divs[cand_indices[j]][cand_indices[k_idx]] or divs[cand_indices[k_idx]][cand_indices[j]]):
                            continue
                        rem = n - ai - aj - ak
                        if rem <= ak:
                            break
                        if rem in val_to_idx:
                            l_idx = val_to_idx[rem]
                            if (not divs[cand_indices[i]][l_idx] and not divs[l_idx][cand_indices[i]] and
                                not divs[cand_indices[j]][l_idx] and not divs[l_idx][cand_indices[j]] and
                                not divs[cand_indices[k_idx]][l_idx] and not divs[l_idx][cand_indices[k_idx]]):
                                found = [ai, aj, ak, rem]
                                break
                    if found:
                        break
                if found:
                    break

        if found is None:
            # k=5
            found = None
            for i in range(nc):
                ai = smooth_numbers[cand_indices[i]]
                if ai * 5 >= n:
                    break
                for j in range(i + 1, nc):
                    aj = smooth_numbers[cand_indices[j]]
                    if ai + aj * 4 >= n:
                        break
                    if divs[cand_indices[i]][cand_indices[j]] or divs[cand_indices[j]][cand_indices[i]]:
                        continue
                    for k_idx in range(j + 1, nc):
                        ak = smooth_numbers[cand_indices[k_idx]]
                        if ai + aj + ak * 3 >= n:
                            break
                        if (divs[cand_indices[i]][cand_indices[k_idx]] or divs[cand_indices[k_idx]][cand_indices[i]] or
                            divs[cand_indices[j]][cand_indices[k_idx]] or divs[cand_indices[k_idx]][cand_indices[j]]):
                            continue
                        for l_idx in range(k_idx + 1, nc):
                            al = smooth_numbers[cand_indices[l_idx]]
                            if ai + aj + ak + al * 2 >= n:
                                break
                            if (divs[cand_indices[i]][cand_indices[l_idx]] or divs[cand_indices[l_idx]][cand_indices[i]] or
                                divs[cand_indices[j]][cand_indices[l_idx]] or divs[cand_indices[l_idx]][cand_indices[j]] or
                                divs[cand_indices[k_idx]][cand_indices[l_idx]] or divs[cand_indices[l_idx]][cand_indices[k_idx]]):
                                continue
                            rem = n - ai - aj - ak - al
                            if rem <= al:
                                break
                            if rem in val_to_idx:
                                m_idx = val_to_idx[rem]
                                if (not divs[cand_indices[i]][m_idx] and not divs[m_idx][cand_indices[i]] and
                                    not divs[cand_indices[j]][m_idx] and not divs[m_idx][cand_indices[j]] and
                                    not divs[cand_indices[k_idx]][m_idx] and not divs[m_idx][cand_indices[k_idx]] and
                                    not divs[cand_indices[l_idx]][m_idx] and not divs[m_idx][cand_indices[l_idx]]):
                                    found = [ai, aj, ak, al, rem]
                                    break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break

        if found is None:
            # k=6 (should be rare, but handle it)
            found = None
            for i in range(nc):
                ai = smooth_numbers[cand_indices[i]]
                if ai * 6 >= n:
                    break
                for j in range(i + 1, nc):
                    aj = smooth_numbers[cand_indices[j]]
                    if ai + aj * 5 >= n:
                        break
                    if divs[cand_indices[i]][cand_indices[j]] or divs[cand_indices[j]][cand_indices[i]]:
                        continue
                    for k_idx in range(j + 1, nc):
                        ak = smooth_numbers[cand_indices[k_idx]]
                        if ai + aj + ak * 4 >= n:
                            break
                        if (divs[cand_indices[i]][cand_indices[k_idx]] or divs[cand_indices[k_idx]][cand_indices[i]] or
                            divs[cand_indices[j]][cand_indices[k_idx]] or divs[cand_indices[k_idx]][cand_indices[j]]):
                            continue
                        for l_idx in range(k_idx + 1, nc):
                            al = smooth_numbers[cand_indices[l_idx]]
                            if ai + aj + ak + al * 3 >= n:
                                break
                            if (divs[cand_indices[i]][cand_indices[l_idx]] or divs[cand_indices[l_idx]][cand_indices[i]] or
                                divs[cand_indices[j]][cand_indices[l_idx]] or divs[cand_indices[l_idx]][cand_indices[j]] or
                                divs[cand_indices[k_idx]][cand_indices[l_idx]] or divs[cand_indices[l_idx]][cand_indices[k_idx]]):
                                continue
                            rem = n - ai - aj - ak - al
                            if rem <= al:
                                break
                            if rem in val_to_idx:
                                m_idx = val_to_idx[rem]
                                if (not divs[cand_indices[i]][m_idx] and not divs[m_idx][cand_indices[i]] and
                                    not divs[cand_indices[j]][m_idx] and not divs[m_idx][cand_indices[j]] and
                                    not divs[cand_indices[k_idx]][m_idx] and not divs[m_idx][cand_indices[k_idx]] and
                                    not divs[cand_indices[l_idx]][m_idx] and not divs[m_idx][cand_indices[l_idx]]):
                                    found = [ai, aj, ak, al, rem]
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break

        if found is None:
            # Fallback: use a single term if n is smooth, else raise error
            if n in smooth_numbers:
                found = [n]
            else:
                # This should not happen based on problem statement
                # Try a quick brute force for higher k
                from itertools import combinations
                for k in range(7, nc + 1):
                    if time.time() - start_time > timeout:
                        break
                    for terms in combinations(smooth_numbers[:nc], k):
                        if sum(terms) != n:
                            continue
                        is_antichain = True
                        for p in range(len(terms)):
                            for q in range(p + 1, len(terms)):
                                a, b = terms[p], terms[q]
                                if b % a == 0 or a % b == 0:
                                    is_antichain = False
                                    break
                            if not is_antichain:
                                break
                        if is_antichain:
                            found = list(terms)
                            break
                    if found:
                        break

        if found is None:
            # Absolute fallback to prevent crash
            found = [1, n - 1] if n > 1 else [1]

        answers.append(found)
        total_terms += len(found)

    return answers
# EVOLVE-BLOCK-END
