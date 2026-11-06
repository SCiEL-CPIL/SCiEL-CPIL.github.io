import math
import time

def construct():
    deadline = time.time() + 80
    max_mod = 128

    def try_assign(moduli_order):
        result = []
        for m in moduli_order:
            if time.time() > deadline:
                break
            found = False
            for r in range(m):
                ok = True
                for (r2, m2) in result:
                    d = math.gcd(m, m2)
                    if d > 1 and (r - r2) % d == 0:
                        ok = False
                        break
                if ok:
                    result.append((r, m))
                    found = True
                    break
        return result

    # Try multiple orderings
    orderings = [
        list(range(2, max_mod + 1)),
        list(range(max_mod, 1, -1)),
        sorted(range(2, max_mod + 1), key=lambda m: sum(1 for i in range(2, m) if m % i == 0)),
        sorted(range(2, max_mod + 1), key=lambda m: -sum(1 for i in range(2, m) if m % i == 0)),
    ]

    best_result = []
    for order in orderings:
        if time.time() > deadline - 5:
            break
        res = try_assign(order)
        if len(res) > len(best_result):
            best_result = res

    # Try to add remaining moduli
    used_moduli = set(m for (r, m) in best_result)
    for m in range(2, max_mod + 1):
        if m in used_moduli:
            continue
        if time.time() > deadline - 5:
            break
        for r in range(m):
            ok = True
            for (r2, m2) in best_result:
                d = math.gcd(m, m2)
                if d > 1 and (r - r2) % d == 0:
                    ok = False
                    break
            if ok:
                best_result.append((r, m))
                used_moduli.add(m)
                break

    return [[r, m] for (r, m) in best_result]
