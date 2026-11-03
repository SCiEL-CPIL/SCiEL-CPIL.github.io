import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115
    rng = random.Random(42)

    total_space = q ** n
    q_powers = [1] * n
    for i in range(1, n):
        q_powers[i] = q_powers[i - 1] * q

    # Precompute neighbor masks as Python ints (bitsets)
    neighbor_masks = [0] * total_space
    for i in range(total_space):
        digits = [0] * n
        temp = i
        for p in range(n):
            digits[p] = temp % q
            temp //= q
        mask = 1 << i
        if radius >= 1:
            for pos in range(n):
                orig = digits[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    delta = (val - orig) * q_powers[pos]
                    new_idx = i + delta
                    if 0 <= new_idx < total_space:
                        mask |= (1 << new_idx)
        neighbor_masks[i] = mask

    full_mask = (1 << total_space) - 1

    # --- Fast greedy to get a feasible cover ---
    def greedy_cover():
        uncovered = full_mask
        code = []
        while uncovered:
            best_w = -1
            best_count = 0
            for w in range(total_space):
                cnt = (uncovered & neighbor_masks[w]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_w = w
            if best_w < 0:
                break
            code.append(best_w)
            uncovered &= ~neighbor_masks[best_w]
        # Remove redundant
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            cov = [0] * total_space
            for idx in code:
                m = neighbor_masks[idx]
                while m:
                    lb = m & (-m)
                    pos = lb.bit_length() - 1
                    cov[pos] += 1
                    m ^= lb
            new_code = []
            for idx in code:
                m = neighbor_masks[idx]
                redundant = True
                while m:
                    lb = m & (-m)
                    pos = lb.bit_length() - 1
                    if cov[pos] <= 1:
                        redundant = False
                        break
                    m ^= lb
                if redundant:
                    m = neighbor_masks[idx]
                    while m:
                        lb = m & (-m)
                        pos = lb.bit_length() - 1
                        cov[pos] -= 1
                        m ^= lb
                else:
                    new_code.append(idx)
            if len(new_code) < len(code):
                code = new_code
                changed = True
        return code

    # --- Simulated annealing local search ---
    def anneal(code, time_limit):
        code = list(code)
        code_set = set(code)
        # Build coverage counts
        cov = [0] * total_space
        for idx in code:
            m = neighbor_masks[idx]
            while m:
                lb = m & (-m)
                pos = lb.bit_length() - 1
                cov[pos] += 1
                m ^= lb

        # Check coverage
        covered = 0
        for idx in code:
            covered |= neighbor_masks[idx]
        if covered != full_mask:
            return code  # infeasible, skip

        T = 50.0
        T_min = 0.01
        cooling = 0.995
        current_len = len(code)
        best_code = list(code)
        best_len = current_len

        while time.time() < time_limit and T > T_min:
            # Pick a random codeword to try removing
            i = rng.randrange(len(code))
            idx = code[i]
            # Check if removable
            m = neighbor_masks[idx]
            removable = True
            while m:
                lb = m & (-m)
                pos = lb.bit_length() - 1
                if cov[pos] <= 1:
                    removable = False
                    break
                m ^= lb

            if removable:
                # Remove it
                m = neighbor_masks[idx]
                while m:
                    lb = m & (-m)
                    pos = lb.bit_length() - 1
                    cov[pos] -= 1
                    m ^= lb
                code.pop(i)
                code_set.discard(idx)
                current_len -= 1

                # Try to add a random word that covers uncovered or reduces redundancy
                # Find words that cover at least one currently-uncovered-by-this-word position
                # Actually, we need to maintain feasibility. After removal, check coverage.
                # Let's check if coverage is still full
                covered = 0
                for j in code:
                    covered |= neighbor_masks[j]
                if covered == full_mask:
                    # Good, we reduced size
                    if current_len < best_len:
                        best_len = current_len
                        best_code = list(code)
                    T *= cooling
                    continue
                else:
                    # Need to add a word back. Pick one that covers most uncovered.
                    uncovered = full_mask & ~covered
                    best_add = -1
                    best_cnt = 0
                    # Sample a few candidates for speed
                    candidates = rng.sample(range(total_space), min(200, total_space))
                    for w in candidates:
                        if w in code_set:
                            continue
                        cnt = (uncovered & neighbor_masks[w]).bit_count()
                        if cnt > best_cnt:
                            best_cnt = cnt
                            best_add = w
                    if best_add >= 0 and best_cnt > 0:
                        code.append(best_add)
                        code_set.add(best_add)
                        m = neighbor_masks[best_add]
                        while m:
                            lb = m & (-m)
                            pos = lb.bit_length() - 1
                            cov[pos] += 1
                            m ^= lb
                        current_len += 1
                    else:
                        # Revert
                        code.insert(i, idx)
                        code_set.add(idx)
                        m = neighbor_masks[idx]
                        while m:
                            lb = m & (-m)
                            pos = lb.bit_length() - 1
                            cov[pos] += 1
                            m ^= lb
                        current_len += 1
                    T *= cooling
                    continue
            else:
                # Try swap: replace idx with a random other word
                if len(code) > 1:
                    new_w = rng.randrange(total_space)
                    if new_w == idx or new_w in code_set:
                        T *= cooling
                        continue
                    # Remove idx, add new_w
                    m1 = neighbor_masks[idx]
                    while m1:
                        lb = m1 & (-m1)
                        pos = lb.bit_length() - 1
                        cov[pos] -= 1
                        m1 ^= lb
                    m2 = neighbor_masks[new_w]
                    while m2:
                        lb = m2 & (-m2)
                        pos = lb.bit_length() - 1
                        cov[pos] += 1
                        m2 ^= lb
                    code[i] = new_w
                    code_set.discard(idx)
                    code_set.add(new_w)

                    # Check coverage
                    covered = 0
                    for j in code:
                        covered |= neighbor_masks[j]
                    if covered != full_mask:
                        # Revert
                        code[i] = idx
                        code_set.discard(new_w)
                        code_set.add(idx)
                        m1 = neighbor_masks[idx]
                        while m1:
                            lb = m1 & (-m1)
                            pos = lb.bit_length() - 1
                            cov[pos] += 1
                            m1 ^= lb
                        m2 = neighbor_masks[new_w]
                        while m2:
                            lb = m2 & (-m2)
                            pos = lb.bit_length() - 1
                            cov[pos] -= 1
                            m2 ^= lb
                    T *= cooling
                else:
                    T *= cooling

        return best_code

    # --- Main: multi-restart ---
    # First, get a good initial solution via greedy
    initial = greedy_cover()
    best_code = initial
    best_len = len(initial)

    # Use remaining time for annealing restarts
    # Each restart: start from best_code with random perturbation
    while time.time() < deadline - 10:
        # Perturb: randomly remove a few and re-add greedily
        perturbed = list(best_code)
        if len(perturbed) > 5:
            k = min(3, len(perturbed) // 5)
            remove_idx = rng.sample(range(len(perturbed)), k)
            remove_set = set(remove_idx)
            for ri in sorted(remove_idx, reverse=True):
                perturbed.pop(ri)
            # Re-add greedily to restore coverage
            covered = 0
            for idx in perturbed:
                covered |= neighbor_masks[idx]
            uncovered = full_mask & ~covered
            while uncovered and time.time() < deadline - 10:
                best_w = -1
                best_cnt = 0
                for w in range(total_space):
                    cnt = (uncovered & neighbor_masks[w]).bit_count()
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_w = w
                if best_w < 0:
                    break
                perturbed.append(best_w)
                uncovered &= ~neighbor_masks[best_w]

        # Anneal from perturbed
        result = anneal(perturbed, deadline - 5)
        if len(result) < best_len:
            best_len = len(result)
            best_code = result

    # Final cleanup: remove redundant
    cov = [0] * total_space
    for idx in best_code:
        m = neighbor_masks[idx]
        while m:
            lb = m & (-m)
            pos = lb.bit_length() - 1
            cov[pos] += 1
            m ^= lb
    new_code = []
    for idx in best_code:
        m = neighbor_masks[idx]
        redundant = True
        while m:
            lb = m & (-m)
            pos = lb.bit_length() - 1
            if cov[pos] <= 1:
                redundant = False
                break
            m ^= lb
        if redundant:
            m = neighbor_masks[idx]
            while m:
                lb = m & (-m)
                pos = lb.bit_length() - 1
                cov[pos] -= 1
                m ^= lb
        else:
            new_code.append(idx)
    best_code = new_code

    # Verify coverage
    covered = 0
    for idx in best_code:
        covered |= neighbor_masks[idx]
    if covered != full_mask:
        uncovered = full_mask & ~covered
        while uncovered and time.time() < deadline - 2:
            best_w = -1
            best_cnt = 0
            for w in range(total_space):
                cnt = (uncovered & neighbor_masks[w]).bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_w = w
            if best_w < 0:
                break
            best_code.append(best_w)
            uncovered &= ~neighbor_masks[best_w]

    # Convert to result
    result = []
    for idx in best_code:
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        result.append(digits)

    return result
