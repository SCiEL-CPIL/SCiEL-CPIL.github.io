import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    n, m, p = 3, 4, 6

    def build_target():
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for c_r in range(n):
            for b in range(m):
                for c_c in range(p):
                    i = c_r * m + b
                    k = b * p + c_c
                    l = c_c * n + c_r
                    target[i, k, l] = 1.0
        return target

    TARGET = build_target()

    def verify_arrays(Ua, Va, Wa, atol=1e-9):
        R = np.einsum('ic,kc,lc->ikl', Ua, Va, Wa)
        return np.allclose(R, TARGET, atol=atol)

    def get_rank66():
        """Clean rank-66 decomposition using Strassen blocks."""
        rank = 66
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0

        def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r

            U[A0, col] = 1; V[B00, col] = 1; W[C0, col] = 1
            col += 1
            U[A1, col] = 1; V[B10, col] = 1; W[C0, col] = 1
            col += 1
            U[A0, col] = 1; V[B01, col] = 1; W[C1, col] = 1
            col += 1
            U[A1, col] = 1; V[B11, col] = 1; W[C1, col] = 1
            col += 1

        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)

            # M1 = (A00+A11)(B00+B11)
            U[A00, col] = 1; U[A11, col] = 1
            V[B00, col] = 1; V[B11, col] = 1
            W[C00, col] = 1; W[C11, col] = 1
            col += 1
            # M2 = (A10+A11)B00
            U[A10, col] = 1; U[A11, col] = 1
            V[B00, col] = 1
            W[C10, col] = 1; W[C11, col] = -1
            col += 1
            # M3 = A00(B01-B11)
            U[A00, col] = 1
            V[B01, col] = 1; V[B11, col] = -1
            W[C01, col] = 1; W[C11, col] = 1
            col += 1
            # M4 = A11(B10-B00)
            U[A11, col] = 1
            V[B10, col] = 1; V[B00, col] = -1
            W[C00, col] = 1; W[C10, col] = 1
            col += 1
            # M5 = (A00+A01)B11
            U[A00, col] = 1; U[A01, col] = 1
            V[B11, col] = 1
            W[C00, col] = -1; W[C01, col] = 1
            col += 1
            # M6 = (A10-A00)(B00+B01)
            U[A10, col] = 1; U[A00, col] = -1
            V[B00, col] = 1; V[B01, col] = 1
            W[C11, col] = 1
            col += 1
            # M7 = (A01-A11)(B10+B11)
            U[A01, col] = 1; U[A11, col] = -1
            V[B10, col] = 1; V[B11, col] = 1
            W[C00, col] = 1
            col += 1

        for j in range(3):
            bc = 2 * j
            add_1x2_2x2(0, 0, 0, bc, 0, 2*j)
            add_1x2_2x2(0, 2, 2, bc, 0, 2*j)
            add_strassen(1, 0, 0, bc, 1, 2*j)
            add_strassen(1, 2, 2, bc, 1, 2*j)

        assert col == rank, f"Expected {rank}, got {col}"
        return U, V, W

    def compute_residual(Ua, Va, Wa):
        """Compute the residual tensor R - TARGET."""
        R = np.einsum('ic,kc,lc->ikl', Ua, Va, Wa)
        return R - TARGET

    def greedy_prune_single(Ua, Va, Wa, max_iter=300):
        """Remove columns one at a time greedily, using residual for speed."""
        Ua = Ua.copy()
        Va = Va.copy()
        Wa = Wa.copy()
        current_rank = Ua.shape[1]

        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            if current_rank <= 0:
                break
            removed = False
            for col in range(current_rank - 1, -1, -1):
                if time.time() > deadline - 5.0:
                    break
                mask = [i for i in range(current_rank) if i != col]
                U_new = Ua[:, mask]
                V_new = Va[:, mask]
                W_new = Wa[:, mask]
                if verify_arrays(U_new, V_new, W_new):
                    Ua, Va, Wa = U_new, V_new, W_new
                    current_rank -= 1
                    removed = True
                    break
            if not removed:
                break
        return Ua, Va, Wa, current_rank

    def try_remove_pairs(Ua, Va, Wa, max_iter=150):
        """Try removing pairs of columns to find jointly redundant columns."""
        Ua = Ua.copy()
        Va = Va.copy()
        Wa = Wa.copy()
        current_rank = Ua.shape[1]

        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            if current_rank <= 1:
                break
            removed = False
            for c1 in range(current_rank - 1, 0, -1):
                if time.time() > deadline - 5.0:
                    break
                for c2 in range(c1 - 1, -1, -1):
                    if time.time() > deadline - 5.0:
                        break
                    mask = [i for i in range(current_rank) if i != c1 and i != c2]
                    U_new = Ua[:, mask]
                    V_new = Va[:, mask]
                    W_new = Wa[:, mask]
                    if verify_arrays(U_new, V_new, W_new):
                        Ua, Va, Wa = U_new, V_new, W_new
                        current_rank -= 2
                        removed = True
                        break
                if removed:
                    break
            if not removed:
                break
        return Ua, Va, Wa, current_rank

    def try_remove_triples(Ua, Va, Wa, max_iter=80):
        """Try removing triples of columns."""
        Ua = Ua.copy()
        Va = Va.copy()
        Wa = Wa.copy()
        current_rank = Ua.shape[1]

        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            if current_rank <= 2:
                break
            removed = False
            for c1 in range(current_rank - 1, 1, -1):
                if time.time() > deadline - 5.0:
                    break
                for c2 in range(c1 - 1, 0, -1):
                    if time.time() > deadline - 5.0:
                        break
                    for c3 in range(c2 - 1, -1, -1):
                        if time.time() > deadline - 5.0:
                            break
                        mask = [i for i in range(current_rank) if i != c1 and i != c2 and i != c3]
                        U_new = Ua[:, mask]
                        V_new = Va[:, mask]
                        W_new = Wa[:, mask]
                        if verify_arrays(U_new, V_new, W_new):
                            Ua, Va, Wa = U_new, V_new, W_new
                            current_rank -= 3
                            removed = True
                            break
                    if removed:
                        break
                if removed:
                    break
            if not removed:
                break
        return Ua, Va, Wa, current_rank

    def random_local_search(Ua, Va, Wa, budget=30.0):
        """Try random column removals with replacement to explore the space."""
        Ua = Ua.copy()
        Va = Va.copy()
        Wa = Wa.copy()
        current_rank = Ua.shape[1]
        best_rank = current_rank
        best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()

        t_end = start_time + budget
        while time.time() < t_end:
            if current_rank <= 0:
                break

            # Try removing a random column
            col = random.randint(0, current_rank - 1)
            mask = [i for i in range(current_rank) if i != col]
            U_new = Ua[:, mask]
            V_new = Va[:, mask]
            W_new = Wa[:, mask]

            if verify_arrays(U_new, V_new, W_new):
                Ua, Va, Wa = U_new, V_new, W_new
                current_rank -= 1
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
            else:
                # Occasionally try removing 2 or 3 random columns
                if current_rank >= 3 and random.random() < 0.2:
                    c1 = random.randint(0, current_rank - 1)
                    c2 = random.randint(0, current_rank - 2)
                    if c2 >= c1:
                        c2 -= 1
                    c3 = random.randint(0, current_rank - 3)
                    if c3 >= c1:
                        c3 += 1
                    if c3 >= c2:
                        c3 += 1
                    mask = [i for i in range(current_rank) if i != c1 and i != c2 and i != c3]
                    U_new = Ua[:, mask]
                    V_new = Va[:, mask]
                    W_new = Wa[:, mask]
                    if verify_arrays(U_new, V_new, W_new):
                        Ua, Va, Wa = U_new, V_new, W_new
                        current_rank -= 3
                        if current_rank < best_rank:
                            best_rank = current_rank
                            best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
                elif current_rank >= 2 and random.random() < 0.5:
                    c1 = random.randint(0, current_rank - 1)
                    c2 = random.randint(0, current_rank - 2)
                    if c2 >= c1:
                        c2 -= 1
                    mask = [i for i in range(current_rank) if i != c1 and i != c2]
                    U_new = Ua[:, mask]
                    V_new = Va[:, mask]
                    W_new = Wa[:, mask]
                    if verify_arrays(U_new, V_new, W_new):
                        Ua, Va, Wa = U_new, V_new, W_new
                        current_rank -= 2
                        if current_rank < best_rank:
                            best_rank = current_rank
                            best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()

        return best_U, best_V, best_W, best_rank

    def multi_start_pruning(U66, V66, W66, num_starts=5):
        """Try multiple random orderings of column removal."""
        best_U, best_V, best_W = U66.copy(), V66.copy(), W66.copy()
        best_rank = U66.shape[1]

        for s in range(num_starts):
            if time.time() > deadline - 10.0:
                break
            # Shuffle column order
            U_shuf = U66[:, random.sample(range(66), 66)]
            V_shuf = V66[:, random.sample(range(66), 66)]
            W_shuf = W66[:, random.sample(range(66), 66)]

            U_r, V_r, W_r, r = greedy_prune_single(U_shuf, V_shuf, W_shuf, max_iter=100)
            if r < best_rank:
                best_rank = r
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()

        return best_U, best_V, best_W, best_rank

    # Build the base rank-66 decomposition
    U66, V66, W66 = get_rank66()

    # Verify it's correct
    assert verify_arrays(U66, V66, W66), "Base rank-66 decomposition is invalid!"

    best_rank = 66
    best_U, best_V, best_W = U66.copy(), V66.copy(), W66.copy()

    # Phase 1: Greedy single-column pruning (original order)
    if time.time() < deadline - 50.0:
        U1, V1, W1, r1 = greedy_prune_single(U66, V66, W66, max_iter=300)
        if r1 < best_rank:
            best_rank = r1
            best_U, best_V, best_W = U1.copy(), V1.copy(), W1.copy()

    # Phase 2: Multi-start pruning with different orderings
    if time.time() < deadline - 40.0:
        U_ms, V_ms, W_ms, r_ms = multi_start_pruning(U66, V66, W66, num_starts=8)
        if r_ms < best_rank:
            best_rank = r_ms
            best_U, best_V, best_W = U_ms.copy(), V_ms.copy(), W_ms.copy()

    # Phase 3: Try pair removals from the best so far
    if time.time() < deadline - 30.0:
        U2, V2, W2, r2 = try_remove_pairs(best_U, best_V, best_W, max_iter=100)
        if r2 < best_rank:
            best_rank = r2
            best_U, best_V, best_W = U2.copy(), V2.copy(), W2.copy()

    # Phase 4: Try triple removals
    if time.time() < deadline - 25.0:
        U3, V3, W3, r3 = try_remove_triples(best_U, best_V, best_W, max_iter=50)
        if r3 < best_rank:
            best_rank = r3
            best_U, best_V, best_W = U3.copy(), V3.copy(), W3.copy()

    # Phase 5: Random local search for further reduction
    if time.time() < deadline - 15.0:
        U4, V4, W4, r4 = random_local_search(best_U, best_V, best_W, budget=20.0)
        if r4 < best_rank:
            best_rank = r4
            best_U, best_V, best_W = U4.copy(), V4.copy(), W4.copy()

    # Phase 6: Another round of greedy pruning on the result
    if time.time() < deadline - 5.0 and best_rank > 60:
        U5, V5, W5, r5 = greedy_prune_single(best_U, best_V, best_W, max_iter=100)
        if r5 < best_rank:
            best_rank = r5
            best_U, best_V, best_W = U5.copy(), V5.copy(), W5.copy()

    # Final verification
    if not verify_arrays(best_U, best_V, best_W):
        # Fallback to the known-good rank-66
        best_U, best_V, best_W = U66.copy(), V66.copy(), W66.copy()
        best_rank = 66

    # Convert to lists for the interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
