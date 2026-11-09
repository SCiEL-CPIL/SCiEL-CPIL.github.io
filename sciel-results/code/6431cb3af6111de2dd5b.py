import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = 88.0
    n = 180

    # Precompute triples (a,b,c) such that 1/a = 1/b + 1/c
    # a < b < c
    # (b-a)(c-a) = a^2
    triples = []
    for a in range(1, n + 1):
        a2 = a * a
        d = 1
        while d * d <= a2:
            if a2 % d == 0:
                e = a2 // d
                if d < e:
                    b = a + d
                    c = a + e
                    if c <= n:
                        triples.append((a, b, c))
            d += 1

    # Build conflict lists: for each vertex i, list of (j,k) pairs that form a triple with i
    conflicts = [[] for _ in range(n)]
    for (a, b, c) in triples:
        conflicts[a - 1].append((b - 1, c - 1))
        conflicts[b - 1].append((a - 1, c - 1))
        conflicts[c - 1].append((a - 1, b - 1))

    # Precompute for each vertex: how many other pairs it conflicts with (degree in hypergraph)
    vertex_weight = [len(conflicts[i]) for i in range(n)]

    best_set = None
    best_size = 0

    def count_violations(S):
        """Count number of violated triples in S."""
        count = 0
        for (a, b, c) in triples:
            if S[a] and S[b] and S[c]:
                count += 1
        return count

    def try_add(S, a):
        """Try to add element a (0-indexed) to set S. Returns True if successful."""
        for (u, v) in conflicts[a]:
            if S[u] and S[v]:
                return False
        S[a] = True
        return True

    def greedy_complete(S):
        """Try to add all possible elements to S in order of least conflicts."""
        cur_size = sum(S)
        missing = [i for i in range(n) if not S[i]]
        missing.sort(key=lambda x: vertex_weight[x])
        for a in missing:
            if try_add(S, a):
                cur_size += 1
        return cur_size

    def local_search_improved(S, max_passes=30):
        """Improved local search: try swaps and additions with cost-aware removal."""
        cur_size = sum(S)

        for _ in range(max_passes):
            if time.time() - t0 > DEADLINE - 3:
                break
            improved = False

            # Phase 1: Try to add any missing element
            missing = [i for i in range(n) if not S[i]]
            random.shuffle(missing)
            for a in missing:
                if try_add(S, a):
                    cur_size += 1
                    improved = True

            # Phase 2: Try 1-for-1 swaps with cost-aware removal
            present = [i for i in range(n) if S[i]]
            missing = [i for i in range(n) if not S[i]]

            # Sort present by vertex_weight descending (remove high-conflict vertices first)
            present.sort(key=lambda x: -vertex_weight[x])
            random.shuffle(missing)

            for a in present:
                S[a] = False
                cur_size -= 1
                added_count = 0
                added_elements = []
                for b in missing:
                    if try_add(S, b):
                        added_count += 1
                        added_elements.append(b)
                if added_count >= 1:
                    cur_size += added_count
                    improved = True
                    missing = [x for x in missing if x not in set(added_elements)]
                else:
                    S[a] = True
                    cur_size += 1

            if not improved:
                break

        return cur_size

    def maxsat_annealing(S, t_start=2.0, t_end=0.005, max_iters=8000):
        """
        Max-SAT inspired simulated annealing.
        Energy = -|S| + lambda * (# violated triples)
        We accept moves that reduce energy (improve -|S| + lambda*violations).
        """
        cur_size = sum(S)
        cur_viol = count_violations(S)
        # Energy: we want to minimize violations and maximize size
        # Use: E = violations * 10 - size (violations dominate)
        LAMBDA = 10
        cur_energy = cur_viol * LAMBDA - cur_size
        best_energy = cur_energy
        best_S = S[:]

        T = t_start
        cooling = (t_end / t_start) ** (1.0 / max_iters)

        for it in range(max_iters):
            if time.time() - t0 > DEADLINE - 4:
                break
            T *= cooling

            # Choose a random vertex to flip
            a = random.randint(0, n - 1)
            old_state = S[a]

            if old_state:
                # Removing a
                S[a] = False
                new_size = cur_size - 1
                # Recompute violations: only triples containing a can change
                # A triple was violated if all three were in S. Now a is out, so those become unviolated.
                # Count how many triples containing a had all three in S (before removal)
                removed_viol = 0
                for (u, v) in conflicts[a]:
                    if S[u] and S[v]:
                        removed_viol += 1
                new_viol = cur_viol - removed_viol
                new_energy = new_viol * LAMBDA - new_size
            else:
                # Adding a
                # Check if this creates new violations
                added_viol = 0
                for (u, v) in conflicts[a]:
                    if S[u] and S[v]:
                        added_viol += 1
                if added_viol == 0:
                    S[a] = True
                    new_size = cur_size + 1
                    new_viol = cur_viol
                    new_energy = new_viol * LAMBDA - new_size
                else:
                    # Adding creates violations; still consider it (Max-SAT style)
                    S[a] = True
                    new_size = cur_size + 1
                    new_viol = cur_viol + added_viol
                    new_energy = new_viol * LAMBDA - new_size

            delta = new_energy - cur_energy

            if delta <= 0:
                # Accept improvement
                cur_size = new_size
                cur_viol = new_viol
                cur_energy = new_energy
                if cur_energy < best_energy:
                    best_energy = cur_energy
                    best_S = S[:]
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < math.exp(-delta / max(T, 0.001)):
                    cur_size = new_size
                    cur_viol = new_viol
                    cur_energy = new_energy
                    if cur_energy < best_energy:
                        best_energy = cur_energy
                        best_S = S[:]
                else:
                    # Reject: undo
                    S[a] = old_state

        # Return the best state found
        S[:] = best_S
        return sum(S)

    # Strategy 1: Suffix + Greedy Add
    for K in range(1, n + 1):
        if time.time() - t0 > DEADLINE - 20:
            break

        S = [False] * n
        for i in range(K - 1, n):
            S[i] = True

        valid_suffix = True
        for (a, b, c) in triples:
            if a >= K and b >= K and c >= K:
                valid_suffix = False
                break

        if not valid_suffix:
            continue

        cur_size = n - K + 1
        if cur_size > best_size:
            best_size = cur_size
            best_set = S[:]

        # Greedy add smaller numbers
        for a in range(K - 1, 0, -1):
            if S[a - 1]:
                continue
            can_add = True
            for (u, v) in conflicts[a - 1]:
                if S[u] and S[v]:
                    can_add = False
                    break
            if can_add:
                S[a - 1] = True
                cur_size += 1
                if cur_size > best_size:
                    best_size = cur_size
                    best_set = S[:]

    # Strategy 2: Local search from best set
    if best_set is not None and time.time() - t0 < DEADLINE - 10:
        S2 = best_set[:]
        cur_size = local_search_improved(S2)
        if cur_size > best_size:
            best_size = cur_size
            best_set = S2[:]

    # Strategy 3: Max-SAT annealing from best set
    if best_set is not None and time.time() - t0 < DEADLINE - 8:
        S3 = best_set[:]
        cur_size = maxsat_annealing(S3, t_start=2.0, t_end=0.005, max_iters=6000)
        if cur_size > best_size:
            best_size = cur_size
            best_set = S3[:]

    # Strategy 4: Multiple random restarts with Max-SAT annealing
    if time.time() - t0 < DEADLINE - 15:
        for restart in range(25):
            if time.time() - t0 > DEADLINE - 6:
                break

            K = random.randint(20, 140)
            S = [False] * n
            for i in range(K - 1, n):
                S[i] = True

            valid = True
            for (a, b, c) in triples:
                if a >= K and b >= K and c >= K:
                    valid = False
                    break

            if not valid:
                continue

            cur_size = n - K + 1

            # Greedy add
            for a in range(K - 1, 0, -1):
                if S[a - 1]:
                    continue
                can_add = True
                for (u, v) in conflicts[a - 1]:
                    if S[u] and S[v]:
                        can_add = False
                        break
                if can_add:
                    S[a - 1] = True
                    cur_size += 1

            # Local search
            cur_size = local_search_improved(S)

            # Max-SAT annealing
            cur_size = maxsat_annealing(S, t_start=1.5, t_end=0.01, max_iters=4000)

            if cur_size > best_size:
                best_size = cur_size
                best_set = S[:]

    # Strategy 5: Final polish with Max-SAT annealing on best
    if best_set is not None and time.time() - t0 < DEADLINE - 3:
        S_final = best_set[:]
        cur_size = maxsat_annealing(S_final, t_start=1.0, t_end=0.002, max_iters=5000)
        if cur_size > best_size:
            best_size = cur_size
            best_set = S_final[:]

    if best_set is None:
        best_set = [False] * n
        best_set[n - 1] = True
        best_size = 1

    # Verify and fix: ensure no violations remain
    S_check = best_set[:]
    for (a, b, c) in triples:
        if S_check[a] and S_check[b] and S_check[c]:
            # Remove the element with highest weight (most conflicts)
            weights = [vertex_weight[a], vertex_weight[b], vertex_weight[c]]
            max_idx = a + weights.index(max(weights))
            S_check[max_idx] = False

    out = [i + 1 for i in range(n) if S_check[i]]

    return out
