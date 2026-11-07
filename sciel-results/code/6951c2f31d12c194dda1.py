"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55

    def build_with_order(order):
        """Build a valid set by trying numbers in the given order."""
        best = []
        best_sums = set()

        for num in order:
            # Compute new subset sums
            new_sums = {num}
            for s in best_sums:
                new_sums.add(s + num)

            # Check for sum collisions
            all_sums = best_sums | new_sums
            if len(all_sums) != len(best_sums) + len(new_sums):
                continue  # duplicate sum

            # Check divisibility: new sums vs all sums
            valid = True
            for ns in new_sums:
                for s in all_sums:
                    if s == ns:
                        continue
                    if s > ns:
                        if s % ns == 0:
                            valid = False
                            break
                    else:  # s < ns
                        if ns % s == 0:
                            valid = False
                            break
                if not valid:
                    break

            if valid:
                best.append(num)
                best_sums = all_sums

        return best

    # Try multiple orderings
    best_result = []

    # Descending order
    r = build_with_order(range(256, 0, -1))
    if len(r) > len(best_result):
        best_result = r

    # Ascending
    r = build_with_order(range(1, 257))
    if len(r) > len(best_result):
        best_result = r

    # Large numbers first (129-256), then small
    r = build_with_order(list(range(256, 128, -1)) + list(range(128, 0, -1)))
    if len(r) > len(best_result):
        best_result = r

    # Random orderings
    for seed in range(200):
        if time.time() > deadline - 2:
            break
        random.seed(seed)
        nums = list(range(1, 257))
        random.shuffle(nums)
        r = build_with_order(nums)
        if len(r) > len(best_result):
            best_result = r

    # Final safety check
    if not best_result:
        return [31, 32, 34]

    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END

