import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55

    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)

    n_total = len(all_masks)
    rng = random.Random(42)

    def is_valid_addition(fam, new_mask):
        n = len(fam)
        if n < 2:
            return True
        inter = [fam[i] & new_mask for i in range(n)]
        groups = defaultdict(list)
        for i, v in enumerate(inter):
            groups[v].append(i)
        for v, indices in groups.items():
            if len(indices) < 2:
                continue
            for idx_a in range(len(indices)):
                for idx_b in range(idx_a + 1, len(indices)):
                    a = indices[idx_a]
                    b = indices[idx_b]
                    if (fam[a] & fam[b]) == v:
                        return False
        return True

    best_family = []
    best_size = 0

    # Multiple greedy trials with different orderings
    for trial in range(5):
        if time.time() > deadline - 30:
            break
        order = list(range(n_total))
        if trial > 0:
            rng.shuffle(order)
        family = []
        for idx in order:
            m = all_masks[idx]
            if is_valid_addition(family, m):
                family.append(m)
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)

    family = list(best_family)
    family_set = set(family)

    # Try to add any remaining sets
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for m in all_masks:
            if m in family_set:
                continue
            if is_valid_addition(family, m):
                family.append(m)
                family_set.add(m)
                improved = True
                if len(family) > best_size:
                    best_size = len(family)
                    best_family = list(family)
                if len(family) >= 250:
                    break
        if len(family) >= 250:
            break

    # Iterated Local Search
    current_family = list(family)
    current_set = set(family)

    for it in range(5000):
        if time.time() > deadline - 2:
            break
        if not current_family:
            break

        remove_idx = rng.randrange(len(current_family))
        removed_mask = current_family[remove_idx]
        current_family.pop(remove_idx)
        current_set.discard(removed_mask)

        added = False
        n_candidates = min(500, n_total)
        candidates = rng.sample(all_masks, n_candidates)
        for new_mask in candidates:
            if new_mask in current_set:
                continue
            if is_valid_addition(current_family, new_mask):
                current_family.append(new_mask)
                current_set.add(new_mask)
                added = True
                break

        if added:
            if len(current_family) > best_size:
                best_size = len(current_family)
                best_family = list(current_family)
        else:
            current_family.insert(remove_idx, removed_mask)
            current_set.add(removed_mask)

        if best_size >= 250:
            break

    result = []
    for mask in best_family:
        elements = [i + 1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elements))

    return result
