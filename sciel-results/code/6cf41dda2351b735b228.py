import random
import time
import math

def construct():
    deadline = time.time() + 118
    rng = random.Random(42)

    def count_k4(n, A):
        cnt = 0
        for a in range(n):
            for b in range(a + 1, n):
                if not ((A[b] >> a) & 1):
                    continue
                for c in range(b + 1, n):
                    if not ((A[c] >> a) & 1 and (A[c] >> b) & 1):
                        continue
                    for d in range(c + 1, n):
                        if (A[d] >> a) & 1 and (A[d] >> b) & 1 and (A[d] >> c) & 1:
                            cnt += 1
        return cnt

    def has_k4(n, A):
        for a in range(n):
            for b in range(a + 1, n):
                if not ((A[b] >> a) & 1):
                    continue
                for c in range(b + 1, n):
                    if not ((A[c] >> a) & 1 and (A[c] >> b) & 1):
                        continue
                    for d in range(c + 1, n):
                        if (A[d] >> a) & 1 and (A[d] >> b) & 1 and (A[d] >> c) & 1:
                            return True
        return False

    def max_independent_set_size(n, A):
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(A[i] | (1 << i))
        best = [0]

        def bk(P, R_size):
            if not P:
                if R_size > best[0]:
                    best[0] = R_size
                return
            if R_size + bin(P).count("1") <= best[0]:
                return
            if best[0] >= 6:
                return
            # Pivot
            if P:
                u_bit = P & -P
                u = u_bit.bit_length() - 1
                candidates = P & ~comp[u]
            else:
                candidates = P
            c = candidates
            while c:
                low = c & -c
                v = low.bit_length() - 1
                bk(P & comp[v], R_size + 1)
                P = P & ~low
                c = c & ~low

        bk(all_mask, 0)
        return best[0]

    def valid(n, A):
        if has_k4(n, A):
            return False
        if max_independent_set_size(n, A) >= 6:
            return False
        return True

    def to_edges(n, A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (A[j] >> i) & 1:
                    edges.append([i, j])
        return edges

    def paley_graph(q):
        squares = {x * x % q for x in range(1, q)}
        n = q
        A = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and (j - i) % q in squares:
                    A[i] |= (1 << j)
        return n, A

    # Start with Paley(17)
    n0, A0 = paley_graph(17)
    assert valid(n0, A0), "Paley(17) should be valid"
    best_n = n0
    best_A = list(A0)

    def energy(n, A):
        """Penalty: number of K4s + max(0, alpha - 5) * 10"""
        k4_count = count_k4(n, A)
        alpha = max_independent_set_size(n, A)
        ind_penalty = max(0, alpha - 5) * 10
        return k4_count + ind_penalty

    def try_grow(n, A, deadline):
        """Try to add one vertex with a valid neighborhood."""
        new_n = n + 1
        if new_n > 40:
            return None
        new_idx = n

        best_mask = None
        for restart in range(30):
            if time.time() > deadline - 15:
                break
            verts = list(range(n))
            rng.shuffle(verts)
            current_mask = 0
            for v in verts:
                # Check triangle-free: (A[v] & current_mask) must have no edge
                neigh_in_current = A[v] & current_mask
                has_edge = False
                m = neigh_in_current
                while m:
                    low = m & -m
                    u = low.bit_length() - 1
                    m ^= low
                    if A[u] & (neigh_in_current & ~(1 << u)):
                        has_edge = True
                        break
                if has_edge:
                    continue
                current_mask |= (1 << v)

            if current_mask == 0:
                continue

            # Check independent set constraint
            non_neigh = ((1 << n) - 1) & ~current_mask
            if bin(non_neigh).count("1") >= 5:
                # Check if non-neighbors contain an independent set of size 5
                sub_verts = []
                m = non_neigh
                while m:
                    low = m & -m
                    v = low.bit_length() - 1
                    sub_verts.append(v)
                    m ^= low
                k_sub = len(sub_verts)
                if k_sub >= 5:
                    idx_map = {v: i for i, v in enumerate(sub_verts)}
                    sub_A = [0] * k_sub
                    for i in range(k_sub):
                        u = sub_verts[i]
                        for j in range(i + 1, k_sub):
                            v = sub_verts[j]
                            if (A[v] >> u) & 1:
                                sub_A[i] |= (1 << j)
                                sub_A[j] |= (1 << i)
                    # Quick check: does subgraph have independent set of size 5?
                    # Use a simplified BK
                    comp_sub = [0] * k_sub
                    all_sub = (1 << k_sub) - 1
                    for i in range(k_sub):
                        comp_sub[i] = all_sub & ~(sub_A[i] | (1 << i))
                    found = [False]
                    def bk5(P, Rsz):
                        if found[0]:
                            return
                        if Rsz == 5:
                            found[0] = True
                            return
                        if not P:
                            return
                        if Rsz + bin(P).count("1") < 5:
                            return
                        u_bit = P & -P
                        u = u_bit.bit_length() - 1
                        cands = P & ~comp_sub[u]
                        c = cands
                        while c and not found[0]:
                            low = c & -c
                            v = low.bit_length() - 1
                            bk5(P & comp_sub[v], Rsz + 1)
                            P = P & ~low
                            c = c & ~low
                    bk5(all_sub, 0)
                    if found[0]:
                        continue  # Invalid, skip this mask

            # Build new adjacency
            new_A = A + [0]
            for i in range(n):
                if (current_mask >> i) & 1:
                    new_A[i] |= (1 << new_idx)
                    new_A[new_idx] |= (1 << i)
            return new_n, new_A

        return None

    def simulated_annealing(n, A, deadline):
        """Simulated annealing on edge flips to minimize energy."""
        cur_A = list(A)
        cur_energy = energy(n, cur_A)
        best_A_sa = list(cur_A)
        best_energy = cur_energy

        T = 5.0
        T_min = 0.01
        cooling = 0.997

        step = 0
        while time.time() < deadline - 10 and T > T_min:
            step += 1
            # Flip one random edge
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            i, j = min(i, j), max(i, j)

            old_edge = (cur_A[j] >> i) & 1
            cur_A[i] &= ~(1 << j)
            cur_A[j] &= ~(1 << i)
            if not old_edge:
                cur_A[i] |= (1 << j)
                cur_A[j] |= (1 << i)

            new_energy = energy(n, cur_A)
            delta_e = new_energy - cur_energy

            if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                cur_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_A_sa = list(cur_A)
            else:
                # Revert
                cur_A[i] &= ~(1 << j)
                cur_A[j] &= ~(1 << i)
                if old_edge:
                    cur_A[i] |= (1 << j)
                    cur_A[j] |= (1 << i)

            T *= cooling

            # Periodically check if we found a valid graph
            if step % 200 == 0 and best_energy == 0:
                if valid(n, best_A_sa):
                    return best_A_sa

        if best_energy == 0 and valid(n, best_A_sa):
            return best_A_sa
        return None

    # Main loop: alternate between growing and annealing
    while time.time() < deadline - 5:
        cur_n = best_n
        cur_A = list(best_A)

        # Try to grow
        if cur_n < 40:
            result = try_grow(cur_n, cur_A, deadline)
            if result:
                new_n, new_A = result
                if valid(new_n, new_A):
                    best_n = new_n
                    best_A = new_A
                    continue

        # Try simulated annealing to find a better configuration at current n
        if time.time() < deadline - 30:
            sa_result = simulated_annealing(cur_n, cur_A, deadline - 5)
            if sa_result and valid(cur_n, sa_result):
                # Check if this is better (it should be same n but potentially better structure)
                # Try to grow from this new structure
                if cur_n < 40:
                    result = try_grow(cur_n, sa_result, deadline)
                    if result:
                        new_n, new_A = result
                        if valid(new_n, new_A):
                            best_n = new_n
                            best_A = new_A
                            continue
                # Even if we can't grow, keep the better structure
                best_A = sa_result

        # If stuck, try a random restart with a different seed graph
        if time.time() < deadline - 40:
            # Try a random graph with appropriate density
            for attempt in range(3):
                if time.time() > deadline - 20:
                    break
                trial_n = min(best_n + rng.randint(0, 2), 40)
                if trial_n < 10:
                    trial_n = 10
                # Random graph with p ~ 0.4 (typical for K4-free, alpha<6)
                trial_A = [0] * trial_n
                for i in range(trial_n):
                    for j in range(i + 1, trial_n):
                        if rng.random() < 0.4:
                            trial_A[i] |= (1 << j)
                            trial_A[j] |= (1 << i)
                # Run SA on this
                sa_res = simulated_annealing(trial_n, trial_A, deadline - 5)
                if sa_res and valid(trial_n, sa_res):
                    if trial_n > best_n:
                        best_n = trial_n
                        best_A = sa_res

    return {'n': best_n, 'edges': to_edges(best_n, best_A)}
