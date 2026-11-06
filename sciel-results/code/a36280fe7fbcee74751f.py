import time
import random

def construct():
    deadline = time.time() + 80  # 80s work limit, 10s safety margin

    # Precompute all 4APs: (a, d) means a, a+d, a+2d, a+3d
    four_aps = []
    for a in range(1, 254):
        max_d = (256 - a) // 3
        for d in range(1, max_d + 1):
            four_aps.append((a, d))

    def has_4ap_violation(bits):
        """Check if bitset has any complete 4AP."""
        for a, d in four_aps:
            if ((bits >> a) & 1) and ((bits >> (a + d)) & 1) and \
               ((bits >> (a + 2 * d)) & 1) and ((bits >> (a + 3 * d)) & 1):
                return True
        return False

    def count_3aps(bits):
        """Count distinct 3-term APs (a, a+d, a+2d) with d>0."""
        count = 0
        for a in range(1, 255):
            if not (bits >> a) & 1:
                continue
            max_d = (256 - a) // 2
            for d in range(1, max_d + 1):
                if ((bits >> (a + d)) & 1) and ((bits >> (a + 2 * d)) & 1):
                    count += 1
        return count

    # Greedy initialization: add elements that don't create a 4AP
    bits = 0
    for x in range(1, 257):
        candidate = bits | (1 << x)
        if not has_4ap_violation(candidate):
            bits = candidate

    best_bits = bits
    best_score = count_3aps(bits)

    # Local search with deadline guard
    random.seed(42)
    while time.time() < deadline:
        improved = False

        # Phase 1: try adding each missing element
        for x in range(1, 257):
            if not (bits >> x) & 1:
                candidate = bits | (1 << x)
                if not has_4ap_violation(candidate):
                    new_score = count_3aps(candidate)
                    if new_score > best_score:
                        bits = candidate
                        best_bits = bits
                        best_score = new_score
                        improved = True

        # Phase 2: try 1-for-1 swaps (remove x, add y)
        for x in range(1, 257):
            if (bits >> x) & 1:
                candidate = bits & ~(1 << x)
                if not has_4ap_violation(candidate):
                    for y in range(1, 257):
                        if not (candidate >> y) & 1:
                            c2 = candidate | (1 << y)
                            if not has_4ap_violation(c2):
                                new_score = count_3aps(c2)
                                if new_score > best_score:
                                    bits = c2
                                    best_bits = bits
                                    best_score = new_score
                                    improved = True
                                    break

        # Phase 3: random perturbation to escape local optima
        if not improved:
            elems = [i for i in range(1, 257) if (bits >> i) & 1]
            if elems:
                x = random.choice(elems)
                candidate = bits & ~(1 << x)
                if not has_4ap_violation(candidate):
                    for _ in range(8):
                        y = random.randint(1, 256)
                        if not (candidate >> y) & 1:
                            c2 = candidate | (1 << y)
                            if not has_4ap_violation(c2):
                                new_score = count_3aps(c2)
                                if new_score >= best_score:
                                    bits = c2
                                    best_bits = bits
                                    best_score = new_score
                                break

    # Return sorted list of selected elements
    result = [i for i in range(1, 257) if (best_bits >> i) & 1]
    return result
