import time
from fractions import Fraction

def construct():
    N = 256
    recips_frac = [Fraction(1, i) for i in range(1, N + 1)]
    recips_float = [1.0 / i for i in range(1, N + 1)]
    
    suffix_float = [0.0] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_float[i] = suffix_float[i + 1] + recips_float[i]

    best_set = []
    best_sum_frac = Fraction(0, 1)
    best_sum_float = 0.0
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit

    def can_add(x, current_set_set):
        for a in current_set_set:
            b = 2 * x - a
            if b in current_set_set:
                return False
            b = 2 * a - x
            if b in current_set_set:
                return False
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b in current_set_set:
                    return False
        return True

    # Greedy Initial Solution
    greedy_set = []
    greedy_set_set = set()
    for x in range(1, N + 1):
        if can_add(x, greedy_set_set):
            greedy_set.append(x)
            greedy_set_set.add(x)
    
    if greedy_set:
        best_set = greedy_set.copy()
        best_sum_float = sum(recips_float[x-1] for x in greedy_set)
        best_sum_frac = sum(recips_frac[x-1] for x in greedy_set)

    def dfs(idx, current_set_list, current_set_set, current_sum_float, current_sum_frac):
        nonlocal best_set, best_sum_frac, best_sum_float
        
        if time.time() - start_time > time_limit:
            return

        if idx == N:
            if current_sum_frac > best_sum_frac:
                best_sum_frac = current_sum_frac
                best_sum_float = current_sum_float
                best_set = current_set_list.copy()
            return

        # Prune
        if current_sum_float + suffix_float[idx] + 1e-9 <= best_sum_float:
            return

        x = idx + 1
        
        # Include x
        if can_add(x, current_set_set):
            new_set = current_set_set | {x}
            dfs(idx + 1, current_set_list + [x], new_set, current_sum_float + recips_float[idx], current_sum_frac + recips_frac[idx])
            if time.time() - start_time > time_limit:
                return

        # Exclude x
        dfs(idx + 1, current_set_list, current_set_set, current_sum_float, current_sum_frac)

    dfs(0, [], set(), 0.0, Fraction(0, 1))

    return best_set
