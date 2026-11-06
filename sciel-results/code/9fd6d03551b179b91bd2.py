import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0   # safety margin within 90s wall-clock

    N = 128
    random.seed(42)

    # ---- Precompute bad triples (0-indexed) and per-element conflict data ----
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                if gcd(a, b) == gcd(b, c) == gcd(a, c):
                    bad_triples.append((a - 1, b - 1, c - 1))

    # conflicts[x] = list of (u, v) such that (u,v,x) is a bad triple
    conflicts = [[] for _ in range(N)]
    for (a, b, c) in bad_triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))

    # conflict_bitset[x] = bitmask of all elements that appear in any conflict pair of x
    conflict_bitset = [0] * N
    for x in range(N):
        m = 0
        for (u, v) in conflicts[x]:
            m |= (1 << u) | (1 << v)
        conflict_bitset[x] = m

    def is_valid_add(mask, x):
        """Fast: adding x is valid iff no conflict pair (u,v) of x is fully inside mask."""
        if (conflict_bitset[x] & mask) == 0:
            return True
        for (u, v) in conflicts[x]:
            if ((mask >> u) & 1) and ((mask >> v) & 1):
                return False
        return True

    degree = [len(conflicts[x]) for x in range(N)]

    # ---- Greedy construction with many orderings ----
    all_elements = list(range(N))

    def greedy(order):
        mask = 0
        size = 0
        for x in order:
            if is_valid_add(mask, x):
                mask |= (1 << x)
                size += 1
        return mask, size

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n ** 0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    primes = [x for x in range(N) if is_prime(x + 1)]
    composites = [x for x in range(N) if not is_prime(x + 1)]

    def is_power_of_2(n):
        return (n & (n - 1)) == 0 and n > 0
    p2 = [x for x in range(N) if is_power_of_2(x + 1)]
    np2 = [x for x in range(N) if not is_power_of_2(x + 1)]

    orders = [
        all_elements[:],
        all_elements[::-1],
        sorted(all_elements, key=lambda x: degree[x]),
        sorted(all_elements, key=lambda x: -degree[x]),
        sorted(all_elements, key=lambda x: (degree[x], -x)),
        sorted(all_elements, key=lambda x: (-degree[x], x)),
        primes + composites,
        composites + primes,
        p2 + np2,
        np2 + p2,
    ]
    # Interleave primes and composites
    mixed = []
    pi = ci = 0
    while pi < len(primes) or ci < len(composites):
        if pi < len(primes):
            mixed.append(primes[pi]); pi += 1
        if ci < len(composites):
            mixed.append(composites[ci]); ci += 1
    orders.append(mixed)

    # Random orders
    for _ in range(40):
        ro = all_elements[:]
        random.shuffle(ro)
        orders.append(ro)

    best_mask = 0
    best_size = 0
    for order in orders:
        mask, size = greedy(order)
        if size > best_size:
            best_size = size
            best_mask = mask

    # ---- Local search with TRUE growth moves ----
    def try_add_one(mask):
        for x in range(N):
            if not ((mask >> x) & 1) and is_valid_add(mask, x):
                return mask | (1 << x)
        return None

    def _valid_candidates(mask):
        return [x for x in range(N)
                if not ((mask >> x) & 1) and is_valid_add(mask, x)]

    def try_swap_1_for_2(mask):
        """Remove 1 element, add 2 -> net +1."""
        in_mask = [x for x in range(N) if (mask >> x) & 1]
        random.shuffle(in_mask)
        for u in in_mask:
            if time.time() > deadline:
                return None
            temp = mask & ~(1 << u)
            vc = _valid_candidates(temp)
            if len(vc) >= 2:
                for i in range(len(vc)):
                    x = vc[i]
                    tx = temp | (1 << x)
                    for j in range(i + 1, len(vc)):
                        y = vc[j]
                        if is_valid_add(tx, y):
                            return tx | (1 << y)
        return None

    def try_swap_2_for_3(mask):
        """Remove 2 elements, add 3 -> net +1."""
        in_mask = [x for x in range(N) if (mask >> x) & 1]
        random.shuffle(in_mask)
        max_pairs = 60
        pc = 0
        for i in range(len(in_mask)):
            if pc >= max_pairs:
                break
            for j in range(i + 1, len(in_mask)):
                if pc >= max_pairs:
                    break
                if time.time() > deadline - 2.0:
                    break
                pc += 1
                temp = mask & ~(1 << in_mask[i]) & ~(1 << in_mask[j])
                vc = _valid_candidates(temp)
                if len(vc) >= 3:
                    max_triples = 300
                    tc = 0
                    for a in range(len(vc)):
                        if tc >= max_triples:
                            break
                        x = vc[a]
                        tx = temp | (1 << x)
                        for b in range(a + 1, len(vc)):
                            if tc >= max_triples:
                                break
                            y = vc[b]
                            txy = tx | (1 << y)
                            for c in range(b + 1, len(vc)):
                                if tc >= max_triples:
                                    break
                                z = vc[c]
                                if is_valid_add(txy, z):
                                    return txy | (1 << z)
                                tc += 1
        return None

    def try_swap_3_for_4(mask):
        """Remove 3 elements, add 4 -> net +1."""
        in_mask = [x for x in range(N) if (mask >> x) & 1]
        random.shuffle(in_mask)
        max_triples = 30
        tc = 0
        for i in range(len(in_mask)):
            if tc >= max_triples:
                break
            for j in range(i + 1, len(in_mask)):
                if tc >= max_triples:
                    break
                for k in range(j + 1, len(in_mask)):
                    if tc >= max_triples:
                        break
                    if time.time() > deadline - 2.0:
                        break
                    tc += 1
                    a, b, c = in_mask[i], in_mask[j], in_mask[k]
                    temp = mask & ~(1 << a) & ~(1 << b) & ~(1 << c)
                    vc = _valid_candidates(temp)
                    if len(vc) >= 4:
                        max_quads = 200
                        qc = 0
                        for p in range(len(vc)):
                            if qc >= max_quads:
                                break
                            x = vc[p]
                            tx = temp | (1 << x)
                            for q in range(p + 1, len(vc)):
                                if qc >= max_quads:
                                    break
                                y = vc[q]
                                txy = tx | (1 << y)
                                for r in range(q + 1, len(vc)):
                                    if qc >= max_quads:
                                        break
                                    z = vc[r]
                                    txz = txy | (1 << z)
                                    for s in range(r + 1, len(vc)):
                                        if qc >= max_quads:
                                            break
                                        w = vc[s]
                                        if is_valid_add(txz, w):
                                            return txz | (1 << w)
                                        qc += 1
        return None

    # Main local search loop
    S_mask = best_mask
    current_size = best_size
    stuck = 0
    no_improve = 0

    while time.time() < deadline - 12.0:
        improved = False

        for move in (try_add_one, try_swap_1_for_2, try_swap_2_for_3, try_swap_3_for_4):
            nm = move(S_mask)
            if nm is not None:
                S_mask = nm
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_mask = S_mask
                    no_improve = 0
                else:
                    no_improve += 1
                stuck = 0
                improved = True
                break
        if improved:
            continue

        # 1-for-1 structure swap (no size change, to escape local minima)
        in_mask = [x for x in range(N) if (S_mask >> x) & 1]
        not_in_mask = [x for x in range(N) if not ((S_mask >> x) & 1)]
        random.shuffle(in_mask)
        swapped = False
        for u in in_mask:
            if time.time() > deadline - 12.0:
                break
            temp = S_mask & ~(1 << u)
            for v in not_in_mask:
                if is_valid_add(temp, v):
                    S_mask = temp | (1 << v)
                    swapped = True
                    break
            if swapped:
                break
        if swapped:
            stuck = 0
            no_improve += 1
        else:
            stuck += 1
            no_improve += 1
            if stuck > 12 or no_improve > 40:
                # Random restart + perturb
                S_mask = best_mask
                current_size = best_size
                stuck = 0
                no_improve = 0
                in_mask = [x for x in range(N) if (S_mask >> x) & 1]
                not_in_mask = [x for x in range(N) if not ((S_mask >> x) & 1)]
                for _ in range(random.randint(1, 3)):
                    if not in_mask or not not_in_mask:
                        break
                    u = random.choice(in_mask)
                    v = random.choice(not_in_mask)
                    temp = S_mask & ~(1 << u)
                    if is_valid_add(temp, v):
                        S_mask = temp | (1 << v)
                        in_mask = [x for x in range(N) if (S_mask >> x) & 1]
                        not_in_mask = [x for x in range(N) if not ((S_mask >> x) & 1)]

    # ---- Branch-and-bound polish (maximum independent set in 2-uniform conflict graph) ----
    # Edge between u,v if they are conflict partners (appear together in a bad triple with some w).
    # A valid set is an independent set in this graph.
    adj = [0] * N
    for (a, b, c) in bad_triples:
        adj[a] |= (1 << b) | (1 << c)
        adj[b] |= (1 << a) | (1 << c)
        adj[c] |= (1 << a) | (1 << b)

    def clique_bound(bits):
        """Greedy clique cover of the induced subgraph on bits -> upper bound on MIS size."""
        count = 0
        rem = bits
        while rem:
            # pick a vertex, form a maximal clique greedily
            lb = rem & (-rem)
            v = lb.bit_length() - 1
            cliq = 1
            cand = rem & adj[v]
            while cand:
                lb2 = cand & (-cand)
                v2 = lb2.bit_length() - 1
                cliq |= (1 << v2)
                cand &= adj[v2]
            count += 1
            rem &= ~cliq
        return count

    best_bnb = best_size

    def bnb(bits, size):
        nonlocal best_bnb
        if time.time() > deadline:
            return
        if bits == 0:
            if size > best_bnb:
                best_bnb = size
            return
        if size + clique_bound(bits) <= best_bnb:
            return
        lb = bits & (-bits)
        v = lb.bit_length() - 1
        # branch include v
        bnb(bits & ~(1 << v) & ~adj[v], size + 1)
        # branch exclude v
        bnb(bits & ~(1 << v), size)

    bnb(best_mask, best_size)

    result = [x + 1 for x in range(N) if (best_mask >> x) & 1]
    return result if result else [1]
