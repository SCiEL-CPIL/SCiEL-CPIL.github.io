import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 256
    MAX_SUM = 2 * N  # 512
    MAX_DIFF = N - 1  # 255

    best_size = 0
    best_A = []
    best_B = []

    # --- Bitmask helpers ---

    def compute_sums_mask(S):
        mask = 0
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                mask |= (1 << s)
        return mask

    def compute_diffs_mask(S):
        mask = 0
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = abs(S[i] - S[j])
                mask |= (1 << d)
        return mask

    def can_add_x_to_S(x, S, sums_mask, forbidden_diff_mask):
        """Check if x can be added to S (Sidon + diff disjointness)."""
        if len(S) >= 64:
            return False
        if (sums_mask >> (2 * x)) & 1:
            return False
        for a in S:
            if (sums_mask >> (x + a)) & 1:
                return False
            d = abs(x - a)
            if d > 0 and (forbidden_diff_mask >> d) & 1:
                return False
        return True

    def add_x_to_S(x, S, sums_mask, diffs_mask):
        """Add x to S and update masks."""
        S.append(x)
        sums_mask |= (1 << (2 * x))
        for a in S[:-1]:
            sums_mask |= (1 << (x + a))
            d = abs(x - a)
            if d > 0:
                diffs_mask |= (1 << d)
        return sums_mask, diffs_mask

    def build_greedy(order, forbidden_diff_mask, max_size=64):
        """Greedy build a Sidon set from order, respecting forbidden diffs."""
        S = []
        sums_mask = 0
        diffs_mask = 0
        for x in order:
            if len(S) >= max_size:
                break
            if can_add_x_to_S(x, S, sums_mask, forbidden_diff_mask):
                sums_mask, diffs_mask = add_x_to_S(x, S, sums_mask, diffs_mask)
        return S, sums_mask, diffs_mask

    def polish(A, B, sums_A, diffs_A, sums_B, diffs_B):
        """Try to add more elements to A and B greedily."""
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break

            # Try adding to A
            for x in range(1, N + 1):
                if x in A:
                    continue
                if can_add_x_to_S(x, A, sums_A, diffs_B):
                    sums_A, diffs_A = add_x_to_S(x, A, sums_A, diffs_A)
                    improved = True
                    break

            # Try adding to B
            if not improved:
                for x in range(1, N + 1):
                    if x in B:
                        continue
                    if can_add_x_to_S(x, B, sums_B, diffs_A):
                        sums_B, diffs_B = add_x_to_S(x, B, sums_B, diffs_A)
                        improved = True
                        break

        return A, B, sums_A, diffs_A, sums_B, diffs_B

    def local_search_swap(A, B, sums_A, diffs_A, sums_B, diffs_B, max_iters=30):
        """Try moving elements between A and B."""
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break

            # Try moving x from A to B
            for x in list(A):
                if x in B:
                    continue
                if len(B) >= 64:
                    continue
                A_temp = [a for a in A if a != x]
                sA_t = compute_sums_mask(A_temp)
                dA_t = compute_diffs_mask(A_temp)
                if can_add_x_to_S(x, B, sums_B, dA_t):
                    A = A_temp
                    B = list(B) + [x]
                    sums_A = sA_t
                    diffs_A = dA_t
                    sums_B, diffs_B = add_x_to_S(x, B, sums_B, diffs_B)
                    improved = True
                    break

            if improved:
                continue

            # Try moving x from B to A
            for x in list(B):
                if x in A:
                    continue
                if len(A) >= 64:
                    continue
                B_temp = [b for b in B if b != x]
                sB_t = compute_sums_mask(B_temp)
                dB_t = compute_diffs_mask(B_temp)
                if can_add_x_to_S(x, A, sums_A, dB_t):
                    B = B_temp
                    A = list(A) + [x]
                    sums_B = sB_t
                    diffs_B = dB_t
                    sums_A, diffs_A = add_x_to_S(x, A, sums_A, diffs_A)
                    improved = True
                    break

        return A, B, sums_A, diffs_A, sums_B, diffs_B

    def local_search_replace(A, B, sums_A, diffs_A, sums_B, diffs_B):
        """Try removing one element and re-filling."""
        current_size = len(A) + len(B)
        best_local = (list(A), list(B))
        best_size_local = current_size

        rng_local = random.Random(int(time.time() * 1000) % 100000)

        # Try removing from A
        candidates_A = list(A)
        rng_local.shuffle(candidates_A)
        for x in candidates_A[:15]:
            if time.time() > deadline:
                break
            if len(A) <= 1:
                continue
            A_temp = [a for a in A if a != x]
            sA_t = compute_sums_mask(A_temp)
            dA_t = compute_diffs_mask(A_temp)
            sB_t = compute_sums_mask(B)
            dB_t = compute_diffs_mask(B)

            A_new = list(A_temp)
            B_new = list(B)
            sA_n, dA_n = sA_t, dA_t
            sB_n, dB_n = sB_t, dB_t

            for y in range(1, N + 1):
                if y in A_new or y in B_new:
                    continue
                if can_add_x_to_S(y, A_new, sA_n, dB_n):
                    sA_n, dA_n = add_x_to_S(y, A_new, sA_n, dA_n)
                elif can_add_x_to_S(y, B_new, sB_n, dA_n):
                    sB_n, dB_n = add_x_to_S(y, B_new, sB_n, dB_n)

            if len(A_new) + len(B_new) > best_size_local:
                best_size_local = len(A_new) + len(B_new)
                best_local = (A_new, B_new)

        # Try removing from B
        candidates_B = list(B)
        rng_local.shuffle(candidates_B)
        for x in candidates_B[:15]:
            if time.time() > deadline:
                break
            if len(B) <= 1:
                continue
            B_temp = [b for b in B if b != x]
            sA_t = compute_sums_mask(A)
            dA_t = compute_diffs_mask(A)
            sB_t = compute_sums_mask(B_temp)
            dB_t = compute_diffs_mask(B_temp)

            A_new = list(A)
            B_new = list(B_temp)
            sA_n, dA_n = sA_t, dA_t
            sB_n, dB_n = sB_t, dB_t

            for y in range(1, N + 1):
                if y in A_new or y in B_new:
                    continue
                if can_add_x_to_S(y, A_new, sA_n, dB_n):
                    sA_n, dA_n = add_x_to_S(y, A_new, sA_n, dA_n)
                elif can_add_x_to_S(y, B_new, sB_n, dA_n):
                    sB_n, dB_n = add_x_to_S(y, B_new, sB_n, dB_n)

            if len(A_new) + len(B_new) > best_size_local:
                best_size_local = len(A_new) + len(B_new)
                best_local = (A_new, B_new)

        if best_size_local > current_size:
            A, B = best_local
            sums_A = compute_sums_mask(A)
            diffs_A = compute_diffs_mask(A)
            sums_B = compute_sums_mask(B)
            diffs_B = compute_diffs_mask(B)

        return A, B, sums_A, diffs_A, sums_B, diffs_B

    def validate(A, B):
        """Full validation of the state."""
        if not A or not B:
            return False
        if len(A) > 64 or len(B) > 64:
            return False
        # Check Sidon for A
        sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums_A:
                    return False
                sums_A.add(s)
        # Check Sidon for B
        sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                if s in sums_B:
                    return False
                sums_B.add(s)
        # Check diff disjointness
        diffs_A = set()
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    d = A[i] - A[j]
                    if d != 0:
                        diffs_A.add(d)
        diffs_B = set()
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    d = B[i] - B[j]
                    if d != 0:
                        diffs_B.add(d)
        if diffs_A & diffs_B:
            return False
        return True

    # --- Main search loop ---

    while time.time() < deadline:
        # Generate random orderings
        order_A = list(range(1, N + 1))
        order_B = list(range(1, N + 1))
        random.shuffle(order_A)
        random.shuffle(order_B)

        # Build A first
        A, sums_A, diffs_A = build_greedy(order_A, 0, 64)
        if not A:
            continue

        # Build B avoiding diffs of A
        B, sums_B, diffs_B = build_greedy(order_B, diffs_A, 64)
        if not B:
            continue

        current_size = len(A) + len(B)

        # Polish
        A, B, sums_A, diffs_A, sums_B, diffs_B = polish(A, B, sums_A, diffs_A, sums_B, diffs_B)
        current_size = len(A) + len(B)

        # Local search: swap
        A, B, sums_A, diffs_A, sums_B, diffs_B = local_search_swap(A, B, sums_A, diffs_A, sums_B, diffs_B, max_iters=20)
        A, B, sums_A, diffs_A, sums_B, diffs_B = polish(A, B, sums_A, diffs_A, sums_B, diffs_B)

        # Local search: replace (only if time permits)
        if time.time() < deadline - 3:
            A, B, sums_A, diffs_A, sums_B, diffs_B = local_search_replace(A, B, sums_A, diffs_A, sums_B, diffs_B)
            A, B, sums_A, diffs_A, sums_B, diffs_B = polish(A, B, sums_A, diffs_A, sums_B, diffs_B)

        # Validate
        if validate(A, B):
            size = len(A) + len(B)
            if size > best_size:
                best_size = size
                best_A = list(A)
                best_B = list(B)

        # Early exit if we hit a good size
        if best_size >= 36:
            break

    # Fallback
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    return [best_A, best_B]
