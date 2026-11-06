import time

def construct():
    start_time = time.time()
    MAX_TARGET = 128

    # Generate 2-3-5 smooth numbers up to MAX_TARGET
    smooth_numbers = []
    for i in range(1, MAX_TARGET + 1):
        y = i
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth_numbers.append(i)
    smooth_numbers.sort()
    N = len(smooth_numbers)

    # Pre-compute divisibility matrix
    divs = [[False] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j:
                if smooth_numbers[j] % smooth_numbers[i] == 0:
                    divs[i][j] = True

    # compatible_mask[i] is a bitmask of indices j (j != i) that can coexist with i in an antichain
    compatible_mask = [0] * N
    for i in range(N):
        for j in range(N):
            if i != j and not divs[i][j] and not divs[j][i]:
                compatible_mask[i] |= (1 << j)

    best_rep = [None] * (MAX_TARGET + 1)
    best_k = [float('inf')] * (MAX_TARGET + 1)

    def dfs(antichain, start_idx, current_sum, k_target):
        if time.time() - start_time > 55:
            return
        if len(antichain) == k_target:
            if current_sum <= MAX_TARGET and best_k[current_sum] == float('inf'):
                best_k[current_sum] = k_target
                best_rep[current_sum] = list(antichain)
            return
        remaining = k_target - len(antichain)
        for idx in range(start_idx, N):
            # Check compatibility with all elements in antichain
            is_compat = True
            for ai in antichain:
                if not (compatible_mask[ai] >> idx) & 1:
                    is_compat = False
                    break
            if not is_compat:
                continue
            new_sum = current_sum + smooth_numbers[idx]
            if new_sum > MAX_TARGET:
                break
            if new_sum + (remaining - 1) > MAX_TARGET:
                break
            antichain.append(idx)
            dfs(antichain, idx + 1, new_sum, k_target)
            antichain.pop()

    for k in range(1, 8):
        if time.time() - start_time > 55:
            break
        if all(best_k[t] != float('inf') for t in range(1, MAX_TARGET + 1)):
            break
        dfs([], 0, 0, k)

    answers = [None] * MAX_TARGET
    for t in range(1, MAX_TARGET + 1):
        if best_rep[t] is not None:
            answers[t-1] = [smooth_numbers[i] for i in best_rep[t]]
        else:
            # Fallback: use single term if t is smooth
            if t in smooth_numbers:
                answers[t-1] = [t]
            else:
                # Should not happen; return empty as last resort
                answers[t-1] = []
    return answers
