import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    n = 128
    numbers = list(range(1, n + 1))

    def lcm(a, b):
        return a * b // gcd(a, b)

    # Precompute LCM table for all pairs
    lcm_table = {}
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[(i, j)] = lcm(i, j)

    def get_lcm(a, b):
        if a <= b:
            return lcm_table[(a, b)]
        else:
            return lcm_table[(b, a)]

    # Find all forbidden triples (a,b,c) with a<b<c where
    # lcm(a,b)==lcm(a,c)==lcm(b,c)
    forbidden_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = get_lcm(a, b)
            for c in range(b + 1, n + 1):
                if get_lcm(a, c) == lab and get_lcm(b, c) == lab:
                    forbidden_triples.append((a, b, c))

    num_triples = len(forbidden_triples)

    # Build bitmask conflicts: for each element x, a bitmask of triples that contain x
    # Also for each triple, a bitmask of its three elements
    triple_elem_mask = [0] * num_triples
    for idx, (a, b, c) in enumerate(forbidden_triples):
        triple_elem_mask[idx] = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))

    # For each element x, list of triple indices containing x
    elem_triples = [[] for _ in range(n + 1)]
    for idx, (a, b, c) in enumerate(forbidden_triples):
        elem_triples[a].append(idx)
        elem_triples[b].append(idx)
        elem_triples[c].append(idx)

    # Precompute: for each pair (a,b), which triples contain both
    # This lets us quickly check if adding c would create a violation
    # Actually, for WalkSAT we need: given current assignment, how many clauses are violated?
    # A clause (a,b,c) is violated if all three are in the set.

    # For efficient counting: maintain count of elements in each triple
    # When we add/remove an element, update the counts for its triples

    # WalkSAT / Max-SAT local search
    # State: bitmask of selected elements (128 bits fits in a Python int)

    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(n)
    prime_powers = set()
    for p in primes:
        power = p
        while power <= n:
            prime_powers.add(power)
            power *= p

    core_mask = 0
    for x in prime_powers | {1}:
        core_mask |= (1 << (x - 1))

    # Count violations for a given state (bitmask)
    def count_violations(state):
        """Count how many forbidden triples are fully contained in state."""
        count = 0
        for t in range(num_triples):
            if (state & triple_elem_mask[t]) == triple_elem_mask[t]:
                count += 1
        return count

    # For WalkSAT: given state, find which elements are in violated triples
    # and their contribution to violation count
    def get_violation_info(state):
        """Return (total_violations, dict of element -> number of violated triples containing it)"""
        violated_triples = []
        for t in range(num_triples):
            if (state & triple_elem_mask[t]) == triple_elem_mask[t]:
                violated_triples.append(t)

        elem_vcount = {}
        for t in violated_triples:
            a, b, c = forbidden_triples[t]
            elem_vcount[a] = elem_vcount.get(a, 0) + 1
            elem_vcount[b] = elem_vcount.get(b, 0) + 1
            elem_vcount[c] = elem_vcount.get(c, 0) + 1

        return len(violated_triples), elem_vcount, violated_triples

    # Greedy initialization
    def greedy_init(ordering, start_mask=0):
        state = start_mask
        for x in ordering:
            if state & (1 << (x - 1)):
                continue
            # Check if adding x creates any violation
            new_state = state | (1 << (x - 1))
            # Check only triples containing x
            ok = True
            for t in elem_triples[x]:
                if (new_state & triple_elem_mask[t]) == triple_elem_mask[t]:
                    ok = False
                    break
            if ok:
                state = new_state
        return state

    # WalkSAT-style local search
    def walksat_search(init_state, max_steps=20000, p_flip=0.3):
        state = init_state
        best_state = init_state
        best_violations = count_violations(state)
        best_size = bin(state).count('1')

        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break

            violations, elem_vcount, violated_triples = get_violation_info(state)

            if violations == 0:
                # No violations: try to add more elements
                size = bin(state).count('1')
                if size > best_size:
                    best_size = size
                    best_state = state
                # Try greedy add
                improved = True
                while improved:
                    improved = False
                    for x in range(1, n + 1):
                        if state & (1 << (x - 1)):
                            continue
                        new_state = state | (1 << (x - 1))
                        ok = True
                        for t in elem_triples[x]:
                            if (new_state & triple_elem_mask[t]) == triple_elem_mask[t]:
                                ok = False
                                break
                        if ok:
                            state = new_state
                            improved = True
                violations, elem_vcount, violated_triples = get_violation_info(state)
                size = bin(state).count('1')
                if size > best_size:
                    best_size = size
                    best_state = state

            if violations == 0:
                # Try more aggressive: remove one element and greedily re-add
                # Only do this occasionally to save time
                if step % 500 == 0:
                    for x in range(1, n + 1):
                        if not (state & (1 << (x - 1))):
                            continue
                        new_state = state & ~(1 << (x - 1))
                        # Greedy re-add
                        for y in range(1, n + 1):
                            if new_state & (1 << (y - 1)):
                                continue
                            ns2 = new_state | (1 << (y - 1))
                            ok = True
                            for t in elem_triples[y]:
                                if (ns2 & triple_elem_mask[t]) == triple_elem_mask[t]:
                                    ok = False
                                    break
                            if ok:
                                new_state = ns2
                        new_size = bin(new_state).count('1')
                        if new_size > best_size:
                            best_size = new_size
                            best_state = new_state
                            state = new_state
                break

            # Choose a violated triple at random
            t = random.choice(violated_triples)
            a, b, c = forbidden_triples[t]

            # Decide which element to flip
            if random.random() < p_flip:
                # Random flip among the three
                candidates = [a, b, c]
                random.shuffle(candidates)
                flip_elem = candidates[0]
            else:
                # Flip the one with highest violation count (most "responsible")
                flip_elem = max([a, b, c], key=lambda e: elem_vcount.get(e, 0))

            # Flip: if in state, remove; if not in state, add
            if state & (1 << (flip_elem - 1)):
                state &= ~(1 << (flip_elem - 1))
            else:
                state |= (1 << (flip_elem - 1))

            # Check if this improved things
            new_violations = count_violations(state)
            if new_violations < best_violations:
                best_violations = new_violations
                best_state = state
                best_size = bin(state).count('1')

            # If no violations, try greedy add
            if new_violations == 0:
                improved = True
                while improved:
                    improved = False
                    for x in range(1, n + 1):
                        if state & (1 << (x - 1)):
                            continue
                        new_state = state | (1 << (x - 1))
                        ok = True
                        for t2 in elem_triples[x]:
                            if (new_state & triple_elem_mask[t2]) == triple_elem_mask[t2]:
                                ok = False
                                break
                        if ok:
                            state = new_state
                            improved = True
                size = bin(state).count('1')
                if size > best_size:
                    best_size = size
                    best_state = state
                break

        return best_state, best_size

    # Generate initial states
    def num_distinct_primes(x):
        count = 0
        d = 2
        while d * d <= x:
            if x % d == 0:
                count += 1
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            count += 1
        return count

    remaining = [x for x in range(1, n + 1) if not (core_mask & (1 << (x - 1)))]

    # Build various orderings
    orderings = []
    orderings.append(sorted(remaining, key=lambda x: (len(elem_triples[x]), num_distinct_primes(x), x)))
    orderings.append(sorted(remaining, key=lambda x: (len(elem_triples[x]), num_distinct_primes(x), -x)))
    orderings.append(sorted(remaining, key=lambda x: x))
    orderings.append(sorted(remaining, key=lambda x: -x))

    for _ in range(15):
        rand_order = remaining[:]
        random.shuffle(rand_order)
        orderings.append(rand_order)

    best_mask = 0
    best_size = 0

    # Phase 1: Greedy + WalkSAT from multiple starts
    for ordering in orderings:
        if time.time() > deadline - 10.0:
            break

        init = greedy_init(ordering, core_mask)
        init_size = bin(init).count('1')

        if init_size > best_size:
            best_size = init_size
            best_mask = init

        # WalkSAT polish
        ws_mask, ws_size = walksat_search(init, max_steps=15000, p_flip=0.3)
        if ws_size > best_size:
            best_size = ws_size
            best_mask = ws_mask

    # Phase 2: More WalkSAT restarts from the best solution with perturbations
    if time.time() < deadline - 8.0:
        for restart in range(30):
            if time.time() > deadline - 5.0:
                break

            # Perturb: remove k random elements, then WalkSAT
            k = random.randint(3, 10)
            selected = [x for x in range(1, n + 1) if best_mask & (1 << (x - 1))]
            removed = random.sample(selected, min(k, len(selected)))
            perturbed = best_mask
            for x in removed:
                perturbed &= ~(1 << (x - 1))

            # Greedy re-add
            for x in range(1, n + 1):
                if perturbed & (1 << (x - 1)):
                    continue
                new_state = perturbed | (1 << (x - 1))
                ok = True
                for t in elem_triples[x]:
                    if (new_state & triple_elem_mask[t]) == triple_elem_mask[t]:
                        ok = False
                        break
                if ok:
                    perturbed = new_state

            ws_mask, ws_size = walksat_search(perturbed, max_steps=10000, p_flip=0.35)
            if ws_size > best_size:
                best_size = ws_size
                best_mask = ws_mask

    # Phase 3: 2-for-k swap local search on the best solution
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            selected = [x for x in range(1, n + 1) if best_mask & (1 << (x - 1))]
            unselected = [x for x in range(1, n + 1) if not (best_mask & (1 << (x - 1)))]

            # Try removing pairs and greedily re-adding
            for i in range(len(selected)):
                if time.time() > deadline - 3.0:
                    break
                for j in range(i + 1, len(selected)):
                    if time.time() > deadline - 3.0:
                        break

                    new_mask = best_mask
                    new_mask &= ~(1 << (selected[i] - 1))
                    new_mask &= ~(1 << (selected[j] - 1))

                    # Greedy re-add
                    for x in unselected:
                        if new_mask & (1 << (x - 1)):
                            continue
                        ns = new_mask | (1 << (x - 1))
                        ok = True
                        for t in elem_triples[x]:
                            if (ns & triple_elem_mask[t]) == triple_elem_mask[t]:
                                ok = False
                                break
                        if ok:
                            new_mask = ns

                    new_size = bin(new_mask).count('1')
                    if new_size > best_size:
                        best_size = new_size
                        best_mask = new_mask
                        improved = True
                        selected = [x for x in range(1, n + 1) if best_mask & (1 << (x - 1))]
                        unselected = [x for x in range(1, n + 1) if not (best_mask & (1 << (x - 1)))]
                        break
                if improved:
                    break

    # Final validation
    S = [x for x in range(1, n + 1) if best_mask & (1 << (x - 1))]
    valid = True
    for t in range(num_triples):
        a, b, c = forbidden_triples[t]
        if (best_mask & (1 << (a - 1))) and (best_mask & (1 << (b - 1))) and (best_mask & (1 << (c - 1))):
            valid = False
            break

    if not valid:
        # Remove one element from each violated triple
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            for t in range(num_triples):
                a, b, c = forbidden_triples[t]
                if (best_mask & (1 << (a - 1))) and (best_mask & (1 << (b - 1))) and (best_mask & (1 << (c - 1))):
                    # Remove the one with fewest conflicts
                    counts = [len(elem_triples[a]), len(elem_triples[b]), len(elem_triples[c])]
                    remove_idx = counts.index(min(counts))
                    elems = [a, b, c]
                    best_mask &= ~(1 << (elems[remove_idx] - 1))
                    changed = True
                    break

        # Re-validate
        S = [x for x in range(1, n + 1) if best_mask & (1 << (x - 1))]
        for t in range(num_triples):
            a, b, c = forbidden_triples[t]
            if (best_mask & (1 << (a - 1))) and (best_mask & (1 << (b - 1))) and (best_mask & (1 << (c - 1))):
                # Force remove
                best_mask &= ~(1 << (c - 1))
                break

    result = [x for x in range(1, n + 1) if best_mask & (1 << (x - 1))]
    if not result:
        result = [1]
    return result
