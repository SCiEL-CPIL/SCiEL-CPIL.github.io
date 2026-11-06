# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Construct two nonempty Sidon subsets A, B of [1, 256] with at most 64 elements each.
    Pair sums inside each set (including a+a) must be unique.
    Nonzero difference sets of A and B must be disjoint.
    Maximize |A|+|B|.
    A and B MAY share elements.
    """
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s hard kill

    N = 256
    MAX_SIZE = 64

    # Bitset representation for sums (values 2..512) and diffs (values 1..255)
    # sums bit i represents sum value i (we use bits 2..512)
    # diffs bit i represents diff value i (we use bits 1..255)

    def _can_add(elements, sums_bits, diffs_own, diffs_other, x):
        """Check if x can be added. sums_bits is int bitset, diffs are int bitsets."""
        # Check 2*x sum
        if (sums_bits >> (2 * x)) & 1:
            return False
        # Check e+x sums and differences
        for e in elements:
            s = e + x
            if (sums_bits >> s) & 1:
                return False
            d = abs(e - x)
            if (diffs_own >> d) & 1 or (diffs_other >> d) & 1:
                return False
        return True

    def _add(elements, sums_bits, diffs, x):
        """Add x, return updated (elements, sums_bits, diffs)."""
        new_elements = elements + [x]
        new_sums = sums_bits | (1 << (2 * x))
        new_diffs = diffs
        for e in elements:
            new_sums |= (1 << (e + x))
            new_diffs |= (1 << abs(e - x))
        return new_elements, new_sums, new_diffs

    def _greedy_balanced(order_a, order_b):
        """Balanced greedy: alternate adding to A and B, allowing shared elements."""
        A = []
        B = []
        sums_a = 0
        sums_b = 0
        diffs_a = 0
        diffs_b = 0

        idx_a = 0
        idx_b = 0
        prefer_a = True

        while (idx_a < len(order_a) or idx_b < len(order_b)) and \
              len(A) < MAX_SIZE and len(B) < MAX_SIZE:
            if time.time() - start_time > DEADLINE:
                break

            if prefer_a:
                if idx_a < len(order_a) and len(A) < MAX_SIZE:
                    x = order_a[idx_a]
                    idx_a += 1
                    if _can_add(A, sums_a, diffs_a, diffs_b, x):
                        A, sums_a, diffs_a = _add(A, sums_a, diffs_a, x)
                    prefer_a = False
                else:
                    prefer_a = False
            else:
                if idx_b < len(order_b) and len(B) < MAX_SIZE:
                    x = order_b[idx_b]
                    idx_b += 1
                    if _can_add(B, sums_b, diffs_b, diffs_a, x):
                        B, sums_b, diffs_b = _add(B, sums_b, diffs_b, x)
                    prefer_a = True
                else:
                    prefer_a = True

        # Finish remaining
        while idx_a < len(order_a) and len(A) < MAX_SIZE:
            if time.time() - start_time > DEADLINE:
                break
            x = order_a[idx_a]
            idx_a += 1
            if _can_add(A, sums_a, diffs_a, diffs_b, x):
                A, sums_a, diffs_a = _add(A, sums_a, diffs_a, x)

        while idx_b < len(order_b) and len(B) < MAX_SIZE:
            if time.time() - start_time > DEADLINE:
                break
            x = order_b[idx_b]
            idx_b += 1
            if _can_add(B, sums_b, diffs_b, diffs_a, x):
                B, sums_b, diffs_b = _add(B, sums_b, diffs_b, x)

        return A, B, len(A) + len(B)

    def _greedy_sequential(order_a, order_b):
        """Sequential greedy: build A first, then B. Allow shared elements."""
        A = []
        B = []
        sums_a = 0
        sums_b = 0
        diffs_a = 0
        diffs_b = 0

        for x in order_a:
            if len(A) >= MAX_SIZE:
                break
            if time.time() - start_time > DEADLINE:
                break
            if _can_add(A, sums_a, diffs_a, diffs_b, x):
                A, sums_a, diffs_a = _add(A, sums_a, diffs_a, x)

        for x in order_b:
            if len(B) >= MAX_SIZE:
                break
            if time.time() - start_time > DEADLINE:
                break
            if _can_add(B, sums_b, diffs_b, diffs_a, x):
                B, sums_b, diffs_b = _add(B, sums_b, diffs_b, x)

        return A, B, len(A) + len(B)

    def _rebuild(elements):
        """Rebuild sums bitset and diffs bitset from element list."""
        sums = 0
        diffs = 0
        n = len(elements)
        for i in range(n):
            for j in range(i, n):
                sums |= (1 << (elements[i] + elements[j]))
            for j in range(i + 1, n):
                diffs |= (1 << abs(elements[i] - elements[j]))
        return sums, diffs

    def _local_search(A, B, max_rounds=200):
        """Local search: try to add elements to A or B to increase total size."""
        A = list(A)
        B = list(B)

        for rnd in range(max_rounds):
            if time.time() - start_time > DEADLINE:
                break

            improved = False
            sums_a, diffs_a = _rebuild(A)
            sums_b, diffs_b = _rebuild(B)

            # Try adding to A
            if len(A) < MAX_SIZE:
                for x in range(1, N + 1):
                    if _can_add(A, sums_a, diffs_a, diffs_b, x):
                        A, sums_a, diffs_a = _add(A, sums_a, diffs_a, x)
                        improved = True
                        break

            # Try adding to B
            if not improved and len(B) < MAX_SIZE:
                for x in range(1, N + 1):
                    if _can_add(B, sums_b, diffs_b, diffs_a, x):
                        B, sums_b, diffs_b = _add(B, sums_b, diffs_b, x)
                        improved = True
                        break

            # Try remove-and-replace: remove an element that blocks additions
            if not improved:
                # Try removing from A
                best_gain = 0
                best_idx = -1
                for i in range(len(A)):
                    A_temp = A[:i] + A[i+1:]
                    sums_at, diffs_at = _rebuild(A_temp)
                    # Count addable to A_temp
                    addable_a = 0
                    for x in range(1, N + 1):
                        if _can_add(A_temp, sums_at, diffs_at, diffs_b, x):
                            addable_a += 1
                    # Count addable to B with freed diffs
                    addable_b = 0
                    if len(B) < MAX_SIZE:
                        for x in range(1, N + 1):
                            if _can_add(B, sums_b, diffs_b, diffs_at, x):
                                addable_b += 1
                    gain = addable_a + addable_b - 1  # net gain (remove 1, add gain)
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i

                if best_idx >= 0 and best_gain > 0:
                    A = A[:best_idx] + A[best_idx+1:]
                    improved = True

            if not improved:
                # Try removing from B
                best_gain = 0
                best_idx = -1
                for i in range(len(B)):
                    B_temp = B[:i] + B[i+1:]
                    sums_bt, diffs_bt = _rebuild(B_temp)
                    addable_b = 0
                    for x in range(1, N + 1):
                        if _can_add(B_temp, sums_bt, diffs_bt, diffs_a, x):
                            addable_b += 1
                    addable_a = 0
                    if len(A) < MAX_SIZE:
                        for x in range(1, N + 1):
                            if _can_add(A, sums_a, diffs_a, diffs_bt, x):
                                addable_a += 1
                    gain = addable_a + addable_b - 1
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i

                if best_idx >= 0 and best_gain > 0:
                    B = B[:best_idx] + B[best_idx+1:]
                    improved = True

            if not improved:
                break

        return A, B, len(A) + len(B)

    # Main search
    best_A = [1]
    best_B = [2]
    best_total = 2

    # Deterministic passes
    order = list(range(1, N + 1))
    A, B, total = _greedy_sequential(order, order)
    if total > best_total:
        best_A, best_B, best_total = A, B, total

    A, B, total = _greedy_balanced(order, order)
    if total > best_total:
        best_A, best_B, best_total = A, B, total

    # Random restarts
    random.seed(42)
    for restart in range(5000):
        if time.time() - start_time > DEADLINE:
            break

        order_a = list(range(1, N + 1))
        order_b = list(range(1, N + 1))

        r = restart % 4
        if r == 0:
            random.shuffle(order_a)
            random.shuffle(order_b)
        elif r == 1:
            # Reverse
            order_a = order_a[::-1]
            order_b = order_b[::-1]
            random.shuffle(order_a)
            random.shuffle(order_b)
        elif r == 2:
            # Block shuffle
            bs = random.randint(4, 32)
            blocks_a = [order_a[i:i+bs] for i in range(0, N, bs)]
            random.shuffle(blocks_a)
            order_a = [x for b in blocks_a for x in b]
            blocks_b = [order_b[i:i+bs] for i in range(0, N, bs)]
            random.shuffle(blocks_b)
            order_b = [x for b in blocks_b for x in b]
        else:
            # Weighted random: favor certain ranges
            weights_a = [random.random() for _ in range(N)]
            order_a.sort(key=lambda x: weights_a[x-1])
            weights_b = [random.random() for _ in range(N)]
            order_b.sort(key=lambda x: weights_b[x-1])

        strategy = random.random()
        if strategy < 0.5:
            A, B, total = _greedy_balanced(order_a, order_b)
        else:
            A, B, total = _greedy_sequential(order_a, order_b)

        if total > best_total:
            # Polish with local search
            A_ls, B_ls, total_ls = _local_search(A, B)
            if total_ls > best_total:
                best_A, best_B, best_total = A_ls, B_ls, total_ls
            else:
                best_A, best_B, best_total = A, B, total

    # Final polish
    A_f, B_f, total_f = _local_search(best_A, best_B)
    if total_f > best_total:
        best_A, best_B, best_total = A_f, B_f, total_f

    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    return [best_A, best_B]
# EVOLVE-BLOCK-END
