import time
import random

P = {'n': 12, 'm': 40}

def construct():
    deadline = time.time() + 85
    n = P['n']
    m = P['m']
    N = 1 << n

    def is_subset(a, b):
        return (a & b) == a and a != b

    def count_comparable(family):
        count = 0
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                if is_subset(family[i], family[j]) or is_subset(family[j], family[i]):
                    count += 1
        return count

    def local_search(family, deadline):
        current_score = count_comparable(family)
        best_score = current_score
        best_family = family[:]
        no_improve_count = 0
        max_no_improve = 5000

        while time.time() < deadline - 3:
            if no_improve_count >= max_no_improve:
                break

            improved = False
            # Try bit-flip moves on random positions
            for _ in range(200):
                if time.time() > deadline - 3:
                    break
                if improved:
                    break

                i = random.randint(0, m - 1)
                old_mask = family[i]

                # Try flipping one bit
                for bit in range(n):
                    new_mask = old_mask ^ (1 << bit)
                    if new_mask == old_mask:
                        continue
                    if new_mask in family[:i] + family[i+1:]:
                        continue

                    lost = 0
                    gained = 0
                    for j in range(m):
                        if j == i:
                            continue
                        other = family[j]
                        if is_subset(old_mask, other) or is_subset(other, old_mask):
                            lost += 1
                        if is_subset(new_mask, other) or is_subset(other, new_mask):
                            gained += 1

                    delta = gained - lost
                    if delta > 0:
                        family[i] = new_mask
                        current_score += delta
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_family = family[:]
                        break

            if not improved:
                # Try 2-opt swap moves
                for _ in range(50):
                    if time.time() > deadline - 3:
                        break
                    i = random.randint(0, m - 1)
                    j = random.randint(0, m - 1)
                    if i == j:
                        continue
                    mi = family[i]
                    mj = family[j]

                    # Compute delta for swapping i and j
                    # Remove mi from consideration, add mj at i's position
                    # Remove mj from consideration, add mi at j's position
                    lost = 0
                    gained = 0
                    for k in range(m):
                        if k == i or k == j:
                            continue
                        other = family[k]
                        # Pairs involving i: (mi, other) and (mj, other)
                        if is_subset(mi, other) or is_subset(other, mi):
                            lost += 1
                        if is_subset(mj, other) or is_subset(other, mj):
                            gained += 1
                        # Pairs involving j: (mj, other) and (mi, other)
                        if is_subset(mj, other) or is_subset(other, mj):
                            lost += 1
                        if is_subset(mi, other) or is_subset(other, mi):
                            gained += 1
                    # Pair (i,j) itself
                    if is_subset(mi, mj) or is_subset(mj, mi):
                        lost += 1
                        gained += 1  # same after swap

                    delta = gained - lost
                    if delta > 0:
                        family[i], family[j] = family[j], family[i]
                        current_score += delta
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_family = family[:]
                        break

            if not improved:
                no_improve_count += 1
                # Perturb: try a random restart of a few elements
                if no_improve_count % 100 == 0:
                    for _ in range(3):
                        idx = random.randint(0, m - 1)
                        new_mask = random.randint(0, N - 1)
                        if new_mask not in family:
                            old = family[idx]
                            lost = 0
                            gained = 0
                            for j in range(m):
                                if j == idx:
                                    continue
                                other = family[j]
                                if is_subset(old, other) or is_subset(other, old):
                                    lost += 1
                                if is_subset(new_mask, other) or is_subset(other, new_mask):
                                    gained += 1
                            if gained - lost > -5:  # allow small loss for perturbation
                                family[idx] = new_mask
                                current_score += gained - lost
                                break

        return best_family, best_score

    best_overall = None
    best_score_overall = 0

    for restart in range(8):
        if time.time() > deadline - 15:
            break

        random.seed(42 + restart * 7)

        if restart == 0:
            # Build a chain: empty set, then singletons, then pairs, etc.
            # Use a systematic chain that maximizes comparability
            chain = [0]
            # Add all singletons
            for b in range(n):
                chain.append(1 << b)
            # Add some pairs to fill up to m
            if len(chain) < m:
                used = set(chain)
                # Add pairs in a structured way
                for a in range(n):
                    for b in range(a + 1, n):
                        if len(chain) >= m:
                            break
                        pair_mask = (1 << a) | (1 << b)
                        if pair_mask not in used:
                            chain.append(pair_mask)
                            used.add(pair_mask)
                    if len(chain) >= m:
                        break
            # If still need more, add triples
            if len(chain) < m:
                used = set(chain)
                for a in range(n):
                    for b in range(a + 1, n):
                        for c in range(b + 1, n):
                            if len(chain) >= m:
                                break
                            triple_mask = (1 << a) | (1 << b) | (1 << c)
                            if triple_mask not in used:
                                chain.append(triple_mask)
                                used.add(triple_mask)
                        if len(chain) >= m:
                            break
                    if len(chain) >= m:
                        break
            # If still not enough, fill randomly
            if len(chain) < m:
                used = set(chain)
                remaining = [x for x in range(N) if x not in used]
                random.shuffle(remaining)
                chain.extend(remaining[:m - len(chain)])

            family = chain[:m]
        elif restart == 1:
            # Different chain structure: start with full set and go down
            chain = [N - 1]
            cur = N - 1
            for b in range(n - 1, -1, -1):
                cur &= ~(1 << b)
                chain.append(cur)
            if len(chain) < m:
                used = set(chain)
                remaining = [x for x in range(N) if x not in used]
                random.shuffle(remaining)
                chain.extend(remaining[:m - len(chain)])
            family = chain[:m]
        elif restart == 2:
            # Mix of sizes: include empty, full, and various middle sets
            family = [0, N - 1]
            # Add sets of size 1, 2, 3, 4, 5, 6 in a balanced way
            for size in range(1, n):
                if len(family) >= m:
                    break
                # Add a few sets of this size
                count_to_add = min(3, m - len(family))
                used = set(family)
                for _ in range(count_to_add):
                    if len(family) >= m:
                        break
                    while True:
                        bits = random.sample(range(n), size)
                        mask = 0
                        for b in bits:
                            mask |= (1 << b)
                        if mask not in used:
                            family.append(mask)
                            used.add(mask)
                            break
        else:
            # Random initialization with some structure
            family = random.sample(range(N), m)

        fam, score = local_search(family, deadline)

        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam

    if best_overall is None:
        best_overall = list(range(m))

    result = []
    for mask in best_overall:
        result.append([b for b in range(n) if (mask >> b) & 1])

    return result
# EVOLVE-BLOCK-END
