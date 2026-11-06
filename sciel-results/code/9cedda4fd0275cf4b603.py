import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # explicit guard, safety margin under 90s
    target = 200
    mask_limit = (1 << (target + 1)) - 1  # keep bits 0..200

    def greedy(order):
        selected = []
        reachable = 1  # bitset: bit s set iff sum s reachable
        for x in order:
            new_reach = (reachable << x) & mask_limit
            if (new_reach >> target) & 1:
                continue  # adding x would make target reachable
            reachable |= new_reach
            selected.append(x)
        return selected

    # Guaranteed-valid contiguous block [66,100]: 2-sums<=199, 3-sums>=201
    candidates = [list(range(66, 101))]
    # Greedy in several orderings
    candidates.append(greedy(list(range(100, 0, -1))))
    candidates.append(greedy(list(range(1, 101))))

    best = max(candidates, key=len)

    # Extra orderings within budget
    for order in (
        list(range(100, 0, -2)) + list(range(99, 0, -2)),
        list(range(1, 101, 2)) + list(range(100, 0, -2)),
    ):
        if time.time() > deadline:
            break
        cand = greedy(order)
        if len(cand) > len(best):
            best = cand

    return best
# EVOLVE-BLOCK-END
