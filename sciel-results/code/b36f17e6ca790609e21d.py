"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import heapq

    denominators = list(range(2, 49))
    n = len(denominators)
    vals = [Fraction(1, d) for d in denominators]

    # Greedy initial solution: include as many as possible while sum < 1
    selected = set()
    total = Fraction(0)
    for i, v in enumerate(vals):
        if total + v < 1:
            selected.add(i)
            total += v

    best_deficit = 1 - total
    best_set = set(selected)

    # Local search: try swapping one element out for one not in set
    # to minimize deficit (i.e., maximize sum while staying < 1)
    # We'll do multiple passes of local improvement

    # Precompute for efficiency
    for iteration in range(50):
        improved = False
        # Try removing one element
        for i in list(selected):
            new_total = total - vals[i]
            # Try adding one element not in selected
            for j in range(n):
                if j in selected:
                    continue
                candidate_total = new_total + vals[j]
                if candidate_total < 1:
                    new_deficit = 1 - candidate_total
                    if new_deficit < best_deficit:
                        best_deficit = new_deficit
                        best_set = selected - {i} | {j}
                        total = candidate_total
                        selected = best_set.copy()
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break

    # Also try pure additions (if any element can be added without exceeding 1)
    for j in range(n):
        if j not in selected:
            if total + vals[j] < 1:
                total += vals[j]
                selected.add(j)
                best_deficit = 1 - total
                best_set = selected.copy()

    # More thorough local search with swaps
    for iteration in range(200):
        improved = False
        for i in list(best_set):
            for j in range(n):
                if j in best_set:
                    continue
                new_total = total - vals[i] + vals[j]
                if new_total < 1:
                    new_deficit = 1 - new_total
                    if new_deficit < best_deficit:
                        best_deficit = new_deficit
                        best_set = (best_set - {i}) | {j}
                        total = new_total
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break

    return list(best_set)
# EVOLVE-BLOCK-END
