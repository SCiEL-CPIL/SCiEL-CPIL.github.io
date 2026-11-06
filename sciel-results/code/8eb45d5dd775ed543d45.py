import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit

    # Precompute all 4-subsets of [1,16] as bitmasks
    all_masks = []
    for comb in combinations(range(1, 17), 4):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_masks.append(mask)

    N = len(all_masks)  # 1820
    mask_to_idx = {m: i for i, m in enumerate(all_masks)}

    def is_sunflower(a, b, c):
        """Check if three 4-element sets (as bitmasks) form a sunflower."""
        ab = a & b
        ac = a & c
        bc = b & c
        return ab == ac == bc

    def can_add_fast(candidate, family_masks):
        """Check if candidate can be added to family without forming a sunflower."""
        for i in range(len(family_masks)):
            a = family_masks[i]
            for j in range(i + 1, len(family_masks)):
                b = family_masks[j]
                if is_sunflower(a, b, candidate):
                    return False
        return True

    def greedy_build_deterministic(deadline):
        """Build family greedily in lexicographic order (proven to score 1.0)."""
        family = []
        for idx in range(N):
            if time.time() > deadline:
                break
            mask = all_masks[idx]
            if can_add_fast(mask, family):
                family.append(mask)
        return family

    def local_search_swap(family, deadline, max_steps=5000):
        """Local search: try to add sets by removing conflicting pairs, then re-adding."""
        family = family[:]
        family_set = set(family)
        best_size = len(family)
        steps_no_improve = 0

        # Precompute which sets conflict with which pairs
        # For efficiency, track blockers for each candidate
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if steps_no_improve > 500:
                break

            # Try to add a random set not in family
            idx = random.randint(0, N - 1)
            c = all_masks[idx]

            if c in family_set:
                steps_no_improve += 1
                continue

            # Check if c can be added
            valid = True
            blocker_pair = None
            for i in range(len(family)):
                a = family[i]
                for j in range(i + 1, len(family)):
                    b = family[j]
                    if is_sunflower(a, b, c):
                        valid = False
                        blocker_pair = (i, j)
                        break
                if not valid:
                    break

            if valid:
                family.append(c)
                family_set.add(c)
                if len(family) > best_size:
                    best_size = len(family)
                    steps_no_improve = 0
                else:
                    steps_no_improve += 1
            else:
                # Try swap: remove one of the blocker sets, add c
                if blocker_pair:
                    i, j = blocker_pair
                    # Try removing i
                    victim = family[i]
                    family.pop(i)
                    family_set.remove(victim)

                    if can_add_fast(c, family):
                        family.append(c)
                        family_set.add(c)
                        # Size unchanged, but structure improved
                        # Try to add more sets now
                        added_more = 0
                        for _ in range(3):  # try a few more additions
                            if time.time() > deadline:
                                break
                            idx2 = random.randint(0, N - 1)
                            c2 = all_masks[idx2]
                            if c2 not in family_set and can_add_fast(c2, family):
                                family.append(c2)
                                family_set.add(c2)
                                added_more += 1
                        if added_more > 0:
                            steps_no_improve = 0
                        else:
                            steps_no_improve += 1
                    else:
                        # Restore
                        family.insert(i, victim)
                        family_set.add(victim)

                        # Try removing j instead
                        victim = family[j]
                        family.pop(j)
                        family_set.remove(victim)

                        if can_add_fast(c, family):
                            family.append(c)
                            family_set.add(c)
                            added_more = 0
                            for _ in range(3):
                                if time.time() > deadline:
                                    break
                                idx2 = random.randint(0, N - 1)
                                c2 = all_masks[idx2]
                                if c2 not in family_set and can_add_fast(c2, family):
                                    family.append(c2)
                                    family_set.add(c2)
                                    added_more += 1
                            if added_more > 0:
                                steps_no_improve = 0
                            else:
                                steps_no_improve += 1
                        else:
                            # Restore
                            family.insert(j, victim)
                            family_set.add(victim)
                            steps_no_improve += 1
                else:
                    steps_no_improve += 1

        return family

    def try_add_all_remaining(family, deadline):
        """Try to add all remaining sets that can be added."""
        family = family[:]
        family_set = set(family)
        added_any = True
        while added_any and time.time() < deadline:
            added_any = False
            for idx in range(N):
                if time.time() > deadline:
                    break
                mask = all_masks[idx]
                if mask not in family_set and can_add_fast(mask, family):
                    family.append(mask)
                    family_set.add(mask)
                    added_any = True
        return family

    best_family = []
    best_size = 0

    # Phase 1: Deterministic greedy (proven to score 1.0)
    family = greedy_build_deterministic(deadline - 10)
    if len(family) > best_size:
        best_size = len(family)
        best_family = family[:]

    # Phase 2: Local search on best family
    if time.time() < deadline - 5:
        improved = local_search_swap(best_family, deadline - 5, max_steps=5000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_family = improved[:]

    # Phase 3: Try to add any remaining sets
    if time.time() < deadline - 2:
        final = try_add_all_remaining(best_family, deadline - 2)
        if len(final) > best_size:
            best_size = len(final)
            best_family = final

    # Convert bitmasks back to sorted lists
    result = []
    for m in best_family:
        s = [i for i in range(1, 17) if (m >> i) & 1]
        result.append(sorted(s))

    return result
