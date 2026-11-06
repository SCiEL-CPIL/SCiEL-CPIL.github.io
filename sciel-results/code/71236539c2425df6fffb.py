import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    all_nums = list(range(1, 513))
    best_score = 0
    best_set = None

    # Precompute: for each d in 1..255, the list of pairs (a, b) where
    # a+d=b and both could be in a set (a in 1..511, b in 2..512)
    # Actually we need: for difference d, pairs (a, a+d) with a in 1..512-d
    # A 3AP with difference d exists iff there's some a with a, a+d, a+2d in S.
    # Equivalently, there's a pair (a, a+d) in S such that a+2d is also in S.
    # So score = number of d where exists a: a, a+d, a+2d all in S.

    # For fast delta computation:
    # When we remove x and add y, the score change depends on:
    # - 3APs that were using x (lost)
    # - 3APs that now use y (gained)
    # A 3AP with difference d and start a: {a, a+d, a+2d}
    # x participates in a 3AP if x = a, x = a+d, or x = a+2d for some d, a.

    # Precompute for each element e: list of (d, role) where role in {0,1,2}
    # meaning e is the start, middle, or end of a 3AP with difference d.
    # But we also need the other two elements to check if they're in S.

    # Better: precompute for each d, the list of 3APs (triples)
    # For d=1: (1,2,3), (2,3,4), ..., (510,511,512) -> 510 triples
    # For d=2: (1,3,5), ..., (508,510,512) -> 255 triples
    # Total triples across all d: sum_{d=1}^{255} (512 - 2*d) = 255*512 - 2*(255*256/2) = 130560 - 65280 = 65280

    # For each element e, precompute which (d, triple_index) it belongs to.
    # Actually, let's precompute:
    # triples_by_d[d] = list of (a, a+d, a+2d)
    # For each element e, and each d, determine if e is in any triple for that d,
    # and which other elements are in that triple.

    # For fast delta: when adding y to S (removing x), 
    # new_score = old_score + (number of d where y completes a new 3AP that wasn't there before)
    #            - (number of d where x was completing a 3AP that's now broken)
    # But a 3AP is "present" if all three elements are in S.
    # So: score = sum over d of [1 if exists a with a,a+d,a+2d in S else 0]

    # When we swap x->y:
    # For each d: 
    #   old_present = exists a: a,a+d,a+2d in S
    #   new_present = exists a: a,a+d,a+2d in (S - {x} + {y})
    #   delta_d = new_present - old_present (can be -1, 0, or +1)

    # To compute this fast, for each d we need to know:
    #   - whether the 3AP is currently present (and if so, which one)
    #   - whether removing x breaks it
    #   - whether adding y creates a new one

    # Precompute for each d:
    #   triples_d[d] = list of (a, m, b) where m=a+d, b=a+2d
    #   For each element e, and each d, the list of triples (by index) containing e.

    # Let's build:
    # elem_to_triples[e] = dict mapping d -> list of triple indices in triples_d[d] that contain e

    # Actually, for speed, let's precompute:
    # For each d, and each position (start, mid, end), store the triple.
    # For each element e, for each d, store which triples contain e and at which position.

    # Given the size (512 elements, 255 d values), let's precompute:
    # For each d in 1..255:
    #   For each a in 1..512-2d:
    #     triple = (a, a+d, a+2d)
    #     For each element in triple, record (d, triple) in elem_triples[element]

    # Then for a swap x->y:
    #   For each d where x is in some triple: check if removing x breaks the only 3AP for that d
    #   For each d where y is in some triple: check if adding y creates a new 3AP for that d

    # This is still O(number of d values where x or y appears) per swap.
    # Each element appears in at most sum over d of (1 if e in [1,512-2d] range) triples... 
    # Actually each element e appears in at most ~255 triples total (one per d, at most).

    # Let's precompute elem_appearances: for each e, list of (d, a, pos) where pos=0,1,2
    # meaning e is at position pos in triple (a, a+d, a+2d) for difference d.

    elem_appearances = [[] for _ in range(513)]  # 1-indexed
    for d in range(1, 256):
        max_a = 512 - 2 * d
        for a in range(1, max_a + 1):
            mid = a + d
            end = a + 2 * d
            elem_appearances[a].append((d, a, 0))
            elem_appearances[mid].append((d, a, 1))
            elem_appearances[end].append((d, a, 2))

    def compute_score(S_set, S_bits):
        """Compute score from scratch."""
        count = 0
        for d in range(1, 256):
            if (S_bits & (S_bits >> d) & (S_bits >> (2 * d))) != 0:
                count += 1
        return count

    def score_delta(S_set, S_bits, x, y):
        """
        Compute the change in score when swapping x (removing) for y (adding).
        x is currently in S_set, y is not.
        Returns (delta, new_score)
        """
        old_score = 0
        # We'll compute new score by checking each d
        # For each d, determine if 3AP exists in old S and new S
        delta = 0
        for d in range(1, 256):
            # Check if 3AP exists in old S
            old_has = (S_bits & (S_bits >> d) & (S_bits >> (2 * d))) != 0
            # New bits: remove x, add y
            new_bits = S_bits ^ (1 << x) ^ (1 << y)
            new_has = (new_bits & (new_bits >> d) & (new_bits >> (2 * d))) != 0
            if new_has and not old_has:
                delta += 1
            elif old_has and not new_has:
                delta -= 1
        return delta

    # Faster delta: only check d values where x or y appears
    def score_delta_fast(S_set, S_bits, x, y):
        """
        Compute score change for swapping x->y. Only check relevant d values.
        """
        # Collect all d values where x or y appears in any triple
        relevant_d = set()
        for d, a, pos in elem_appearances[x]:
            relevant_d.add(d)
        for d, a, pos in elem_appearances[y]:
            relevant_d.add(d)

        delta = 0
        new_bits = S_bits ^ (1 << x) ^ (1 << y)
        for d in relevant_d:
            old_has = (S_bits & (S_bits >> d) & (S_bits >> (2 * d))) != 0
            new_has = (new_bits & (new_bits >> d) & (new_bits >> (2 * d))) != 0
            if new_has and not old_has:
                delta += 1
            elif old_has and not old_has:
                pass
            elif old_has and not new_has:
                delta -= 1
        return delta

    # Even faster: precompute for each d, the bit masks for positions 0, 1, 2
    # mask0[d] = bits where element a is at position 0 (start) for difference d
    # mask1[d] = bits where element m is at position 1 (middle)
    # mask2[d] = bits where element b is at position 2 (end)
    # A 3AP with difference d exists iff (S & mask0[d] & mask1[d] & mask2[d]) != 0
    # Wait, that's not quite right. We need: exists a such that a in S, a+d in S, a+2d in S.
    # This is: exists a: (S has bit a) AND (S has bit a+d) AND (S has bit a+2d)
    # = exists a: bit a in S AND bit (a+d) in S AND bit (a+2d) in S
    # Using the shift formulation: (S & (S>>d) & (S>>(2d))) != 0 works because
    # bit a in (S & (S>>d) & (S>>(2d))) means S has bit a, S has bit a+d, S has bit a+2d.

    # For the delta, we only need to check d where x or y could affect the result.
    # x affects d if x is in [1, 512-2d] as start, or [1+d, 512-d] as middle, or [1+2d, 512] as end.
    # i.e., d such that x-2d >= 1 (x is end), x-d >= 1 and x+d <= 512 (x is middle), or x+2d <= 512 (x is start)
    # More simply: d is relevant for x if 1 <= x <= 512 and there exists a valid triple.

    # Let's just use the relevant_d approach but make it faster with precomputed lists.

    # Precompute for each element, the sorted list of d values it appears in
    elem_d_list = [list(set(d for d, a, p in elem_appearances[e])) for e in range(513)]

    def score_delta_v2(S_bits, x, y):
        """Compute score delta for removing x and adding y."""
        # Get all relevant d values
        dx = elem_d_list[x]
        dy = elem_d_list[y]
        # Merge and deduplicate
        relevant = set(dx) | set(dy)
        
        new_bits = S_bits ^ (1 << x) ^ (1 << y)
        delta = 0
        for d in relevant:
            old_has = (S_bits & (S_bits >> d) & (S_bits >> (2 * d))) != 0
            new_has = (new_bits & (new_bits >> d) & (new_bits >> (2 * d))) != 0
            if new_has != old_has:
                delta += 1 if new_has else -1
        return delta

    # Density-aware initialization
    def density_init():
        """Initialize with a density-based heuristic to get close to high-score regions."""
        # Start with a structured set, then randomize
        # Try arithmetic progressions with different steps
        S = set()
        # Pick a base and step to create a dense cluster
        base = random.randint(1, 50)
        step = random.choice([1, 2, 3, 4, 5, 6, 7, 8])
        count = 0
        x = base
        while x <= 512 and count < 32:
            S.add(x)
            x += step
            count += 1
        # Fill remaining randomly
        while len(S) < 32:
            S.add(random.choice(all_nums))
        return S

    def random_init():
        return set(random.sample(all_nums, 32))

    for restart in range(500):
        if time.time() > deadline - 15:
            break

        # Alternate between density and random init
        if restart % 2 == 0:
            S = density_init()
        else:
            S = random_init()

        S_list = list(S)
        S_bits = 0
        for x in S:
            S_bits |= (1 << x)

        cur_score = compute_score(S, S_bits)

        if cur_score > best_score:
            best_score = cur_score
            best_set = set(S)

        # Simulated annealing with single and triple swaps
        T = 8.0
        T_min = 0.5
        cooling = 0.997
        no_improve = 0
        max_no_improve = 8000
        sa_steps = 0

        while time.time() < deadline - 8:
            if no_improve > max_no_improve:
                break

            sa_steps += 1
            r = random.random()
            if r < 0.5:
                # Single swap
                x = random.choice(S_list)
                y = random.choice(all_nums)
                if y in S:
                    continue
                S_bits_new = S_bits ^ (1 << x) ^ (1 << y)
                delta = score_delta_v2(S_bits, x, y)
                new_score = cur_score + delta

                if delta > 0 or random.random() < np.exp(delta / T):
                    S.discard(x)
                    S.add(y)
                    S_list.remove(x)
                    S_list.append(y)
                    S_bits = S_bits_new
                    cur_score = new_score
                    no_improve = 0
                    if cur_score > best_score:
                        best_score = cur_score
                        best_set = set(S)
                else:
                    no_improve += 1

            elif r < 0.8:
                # Double swap
                x1, x2 = random.sample(S_list, 2)
                y1 = random.choice(all_nums)
                while y1 in S:
                    y1 = random.choice(all_nums)
                y2 = random.choice(all_nums)
                while y2 in S or y2 == y1:
                    y2 = random.choice(all_nums)

                S_bits_new = S_bits ^ (1 << x1) ^ (1 << x2) ^ (1 << y1) ^ (1 << y2)
                # Compute delta for double swap
                # Check all relevant d
                relevant = set()
                for e in [x1, x2, y1, y2]:
                    relevant.update(elem_d_list[e])
                
                delta = 0
                for d in relevant:
                    old_has = (S_bits & (S_bits >> d) & (S_bits >> (2 * d))) != 0
                    new_has = (S_bits_new & (S_bits_new >> d) & (S_bits_new >> (2 * d))) != 0
                    if new_has != old_has:
                        delta += 1 if new_has else -1
                new_score = cur_score + delta

                if delta > 0 or random.random() < np.exp(delta / T):
                    S.discard(x1)
                    S.discard(x2)
                    S.add(y1)
                    S.add(y2)
                    S_list.remove(x1)
                    S_list.remove(x2)
                    S_list.append(y1)
                    S_list.append(y2)
                    S_bits = S_bits_new
                    cur_score = new_score
                    no_improve = 0
                    if cur_score > best_score:
                        best_score = cur_score
                        best_set = set(S)
                else:
                    no_improve += 1

            else:
                # Triple swap
                x1, x2, x3 = random.sample(S_list, 3)
                y1 = random.choice(all_nums)
                while y1 in S:
                    y1 = random.choice(all_nums)
                y2 = random.choice(all_nums)
                while y2 in S or y2 == y1:
                    y2 = random.choice(all_nums)
                y3 = random.choice(all_nums)
                while y3 in S or y3 == y1 or y3 == y2:
                    y3 = random.choice(all_nums)

                S_bits_new = S_bits
                for x in [x1, x2, x3]:
                    S_bits_new ^= (1 << x)
                for y in [y1, y2, y3]:
                    S_bits_new ^= (1 << y)

                relevant = set()
                for e in [x1, x2, x3, y1, y2, y3]:
                    relevant.update(elem_d_list[e])

                delta = 0
                for d in relevant:
                    old_has = (S_bits & (S_bits >> d) & (S_bits >> (2 * d))) != 0
                    new_has = (S_bits_new & (S_bits_new >> d) & (S_bits_new >> (2 * d))) != 0
                    if new_has != old_has:
                        delta += 1 if new_has else -1
                new_score = cur_score + delta

                if delta > 0 or random.random() < np.exp(delta / T):
                    for x in [x1, x2, x3]:
                        S.discard(x)
                        S_list.remove(x)
                    for y in [y1, y2, y3]:
                        S.add(y)
                        S_list.append(y)
                    S_bits = S_bits_new
                    cur_score = new_score
                    no_improve = 0
                    if cur_score > best_score:
                        best_score = cur_score
                        best_set = set(S)
                else:
                    no_improve += 1

            T = max(T_min, T * cooling)

        # Polish: strict local search on single swaps
        polish_iter = 0
        while polish_iter < 30:
            if time.time() > deadline - 3:
                break
            best_swap_score = cur_score
            best_swap_x = -1
            best_swap_y = -1

            for i in range(len(S_list)):
                x = S_list[i]
                for y in all_nums:
                    if y in S:
                        continue
                    delta = score_delta_v2(S_bits, x, y)
                    new_score = cur_score + delta
                    if new_score > best_swap_score:
                        best_swap_score = new_score
                        best_swap_x = x
                        best_swap_y = y

            if best_swap_x != -1:
                S.discard(best_swap_x)
                S.add(best_swap_y)
                S_list.remove(best_swap_x)
                S_list.append(best_swap_y)
                S_bits ^= (1 << best_swap_x) ^ (1 << best_swap_y)
                cur_score = best_swap_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_set = set(S)
                polish_iter += 1
            else:
                break

        T = 8.0

    if best_set is None:
        return list(range(1, 33))

    return list(best_set)

def _score_set_bits(bits):
    count = 0
    for d in range(1, 256):
        if (bits & (bits >> d) & (bits >> (2 * d))) != 0:
            count += 1
    return count
