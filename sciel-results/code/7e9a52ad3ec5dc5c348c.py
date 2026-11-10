import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin

    selected = []  # List of (m, r)
    candidates = list(range(20, 201))
    n = len(candidates)

    for i, m in enumerate(candidates):
        if time.time() > deadline:
            break

        best_r = None
        best_cost = float('inf')

        for r in range(m):
            # Check validity against already selected
            valid = True
            for (m2, r2) in selected:
                g = gcd(m, m2)
                if r % g == r2 % g:
                    valid = False
                    break
            if not valid:
                continue

            # Compute cost: total number of residues blocked in future moduli
            cost = 0
            for j in range(i + 1, n):
                m2 = candidates[j]
                g = gcd(m, m2)
                if g > 1:
                    # Count of x in [0, m2) with x ≡ r (mod g)
                    cost += (m2 - 1 - r) // g + 1

            if cost < best_cost:
                best_cost = cost
                best_r = r

        if best_r is not None:
            selected.append((m, best_r))

    if not selected:
        return [[20, 0]]

    return [[m, r] for m, r in selected]
