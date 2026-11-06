"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85

    def subset_sums(arr):
        sums = {0}
        for x in arr:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            sums |= new_sums
        return sums

    arr = [1 << i for i in range(12)]

    for _pass in range(20):
        if time.time() > deadline - 5:
            break
        improved = False
        for idx in range(12):
            if time.time() > deadline - 5:
                break
            current = arr[idx]
            others = arr[:idx] + arr[idx+1:]
            other_sums = subset_sums(others)
            used = set(others)

            for val in range(1, current):
                if val in used:
                    continue
                shifted = {s - val for s in other_sums if s >= val}
                if not (other_sums & shifted):
                    arr[idx] = val
                    improved = True
                    break
        if not improved:
            break

    return arr
# EVOLVE-BLOCK-END
