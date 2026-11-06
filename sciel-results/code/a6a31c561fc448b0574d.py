import time
from fractions import Fraction

def check_violation_fast(a, others):
    target = Fraction(1, a)
    m = len(others)
    if m == 0:
        return False
    total = sum(Fraction(1, x) for x in others)
    if total < target:
        return False
    if m <= 12:
        achievable = {Fraction(0)}
        for x in others:
            new_sums = {s + Fraction(1, x) for s in achievable}
            achievable |= new_sums
        return target in achievable
    else:
        others_sorted = sorted(others, key=lambda x: -Fraction(1, x))
        def dfs(idx, current_sum):
            if current_sum > target:
                return False
            if idx == m:
                return current_sum == target
            remaining_sum = sum(Fraction(1, others_sorted[j]) for j in range(idx, m))
            if current_sum + remaining_sum < target:
                return False
            if dfs(idx + 1, current_sum):
                return True
            if dfs(idx + 1, current_sum + Fraction(1, others_sorted[idx])):
                return True
            return False
        return dfs(0, Fraction(0))

def construct():
    start_time = time.time()
    deadline = 85.0
    current = set(range(1, 25))
    changed = True
    while changed and time.time() - start_time < deadline:
        changed = False
        violating = []
        for a in current:
            others = [x for x in current if x != a]
            if check_violation_fast(a, others):
                violating.append(a)
        if violating:
            elem_to_remove = min(violating)
            current.remove(elem_to_remove)
            changed = True
    for elem in sorted(range(1, 25)):
        if time.time() - start_time > deadline:
            break
        if elem not in current:
            candidate = current | {elem}
            ok = True
            for a in candidate:
                others = [x for x in candidate if x != a]
                if check_violation_fast(a, others):
                    ok = False
                    break
            if ok:
                current = candidate
    return sorted(current)
