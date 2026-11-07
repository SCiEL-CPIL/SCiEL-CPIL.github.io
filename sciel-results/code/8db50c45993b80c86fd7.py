"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

P = {'n': 12, 'm': 40}

def construct():
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit

    # Precompute bit counts for all masks
    bit_counts = [0] * total_masks
    for mask in range(1, total_masks):
        bit_counts[mask] = bit_counts[mask & (mask - 1)] + 1

    # Precompute comparability: comp[a][b] = 1 if a and b are comparable.
    # Two masks a, b are comparable iff (a & b) == a or (a & b) == b.
    # Stored as a flat bytearray for O(1) lookup.
    comp = bytearray(total_masks * total_masks)
    for a in range(total_masks):
        base = a * total_masks
        for b in range(total_masks):
            if (a & b) == a or (a & b) == b:
                comp[base + b] = 1

    def is_comparable(a, b):
        return comp[a * total_masks + b] == 1

    def calc_score(family):
        """Calculate number of comparable unordered pairs in family."""
        score = 0
        L = len(family)
        for i in range(L):
            fi = family[i]
            base = fi * total_masks
            for j in range(i + 1, L):
                if comp[base + family[j]]:
                    score += 1
        return score

    # Candidate pools by size
    by_size = [[] for _ in range(n + 1)]
    for mask in range(total_masks):
        by_size[bit_counts[mask]].append(mask)

    # Precompute, for each mask, the list of masks comparable with it (for B&B clustering)
    # This is expensive (4096 * avg_degree). We compute lazily / on demand instead.

    random.seed(42)

    # Build a good initial solution using greedy approach
    def build_greedy_initial():
        """Build initial solution by greedily adding most comparable sets."""
        family = []
        used = set()

        # Start with a chain to ensure a good base (13 nested sets)
        chain_masks = [(1 << s) - 1 for s in range(n + 1)]
        for mask in chain_masks:
            family.append(mask)
            used.add(mask)

        # Now greedily add sets that maximize comparability with existing family
        remaining_slots = m - len(family)
        for _ in range(remaining_slots):
            best_mask = None
            best_comp_count = -1
            # Try masks from small and large sizes first (they nest well with the chain)
            for size in [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 5, 6, 7]:
                candidates = [mask for mask in by_size[size] if mask not in used]
                random.shuffle(candidates)
                # Only check first few candidates for speed
                for mask in candidates[:60]:
                    base = mask * total_masks
                    comp_count = 0
                    for f in family:
                        if comp[base + f]:
                            comp_count += 1
                    if comp_count > best_comp_count:
                        best_comp_count = comp_count
                        best_mask = mask
                if best_mask is not None and best_comp_count >= len(family) * 0.7:
                    break

            if best_mask is not None:
                family.append(best_mask)
                used.add(best_mask)
            else:
                # Fallback: add any unused mask
                all_remaining = [mask for mask in range(total_masks) if mask not in used]
                if all_remaining:
                    random.shuffle(all_remaining)
                    family.append(all_remaining[0])
                    used.add(all_remaining[0])

        return family[:m]

    family = build_greedy_initial()
    current_score = calc_score(family)
    best_family = list(family)
    best_score = current_score

    # Local search with improved strategies
    no_improve_count = 0
    max_no_improve = 8000
    restarts = 0
    max_restarts = 6

    family_set = set(family)

    while time.time() < deadline:
        if no_improve_count >= max_no_improve:
            # Try a restart with a different random seed
            restarts += 1
            if restarts > max_restarts:
                break

            random.seed(42 + restarts * 1000)
            family = build_greedy_initial()
            family_set = set(family)
            current_score = calc_score(family)
            if current_score > best_score:
                best_score = current_score
                best_family = list(family)
            no_improve_count = 0
            continue

        # Try random swaps with bias toward promising sizes
        idx = random.randint(0, m - 1)
        old_mask = family[idx]

        # Pick a new mask with bias toward small and large sizes
        r = random.random()
        if r < 0.35:
            new_size = random.randint(0, 3)
        elif r < 0.70:
            new_size = random.randint(9, 12)
        elif r < 0.85:
            new_size = random.randint(4, 8)
        else:
            new_size = random.randint(0, n)

        if not by_size[new_size]:
            continue
        new_mask = random.choice(by_size[new_size])
        if new_mask == old_mask:
            continue
        if new_mask in family_set:
            continue

        # Calculate delta score using precomputed comparability
        delta = 0
        old_base = old_mask * total_masks
        new_base = new_mask * total_masks
        for j in range(m):
            if j == idx:
                continue
            other = family[j]
            if comp[old_base + other]:
                delta -= 1
            if comp[new_base + other]:
                delta += 1

        if delta > 0:
            family[idx] = new_mask
            family_set.remove(old_mask)
            family_set.add(new_mask)
            current_score += delta
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_family = list(family)
        else:
            no_improve_count += 1

    # Branch-and-bound clique-guided refinement:
    # For a set of "anchor" masks, greedily grow a comparable cluster and see if
    # replacing a poorly-connected family member with a cluster member helps.
    if time.time() < deadline - 8:
        family_set = set(best_family)
        # Precompute for each family member its comparable count within the family
        fam_comp_count = [0] * m
        for i in range(m):
            base = best_family[i] * total_masks
            c = 0
            for j in range(m):
                if i != j and comp[base + best_family[j]]:
                    c += 1
            fam_comp_count[i] = c

        # Try each family member as a replacement candidate for the worst-connected ones
        # We attempt to find a mask that is comparable with the maximum number of
        # current family members (a "max comparable" swap), which is a local clique growth.
        improved = True
        while improved and time.time() < deadline - 8:
            improved = False
            # Find the member with the fewest comparabilities (most "lonely")
            min_idx = 0
            min_c = fam_comp_count[0]
            for i in range(1, m):
                if fam_comp_count[i] < min_c:
                    min_c = fam_comp_count[i]
                    min_idx = i
            old_mask = best_family[min_idx]
            old_base = old_mask * total_masks

            # Find the best replacement mask (not in family) that maximizes comparability
            # with the rest of the family (excluding old_mask).
            best_delta = 0
            best_new_mask = None
            for size in range(n + 1):
                if time.time() >= deadline - 8:
                    break
                for new_mask in by_size[size]:
                    if new_mask in family_set or new_mask == old_mask:
                        continue
                    new_base = new_mask * total_masks
                    delta = 0
                    for j in range(m):
                        if j == min_idx:
                            continue
                        other = best_family[j]
                        if comp[old_base + other]:
                            delta -= 1
                        if comp[new_base + other]:
                            delta += 1
                    if delta > best_delta:
                        best_delta = delta
                        best_new_mask = new_mask

            if best_new_mask is not None and best_delta > 0:
                family_set.remove(old_mask)
                family_set.add(best_new_mask)
                best_family[min_idx] = best_new_mask
                best_score += best_delta
                # Recompute comparable counts
                for i in range(m):
                    base = best_family[i] * total_masks
                    c = 0
                    for j in range(m):
                        if i != j and comp[base + best_family[j]]:
                            c += 1
                    fam_comp_count[i] = c
                improved = True

    # Final high-precision polish: for each position, try ALL candidate masks
    # (the full 4096) and keep the best single-element replacement.
    if time.time() < deadline - 3:
        family_set = set(best_family)
        for idx in range(m):
            if time.time() >= deadline - 3:
                break
            old_mask = best_family[idx]
            old_base = old_mask * total_masks
            best_delta = 0
            best_new_mask = None
            for new_mask in range(total_masks):
                if new_mask in family_set or new_mask == old_mask:
                    continue
                new_base = new_mask * total_masks
                delta = 0
                for j in range(m):
                    if j == idx:
                        continue
                    other = best_family[j]
                    if comp[old_base + other]:
                        delta -= 1
                    if comp[new_base + other]:
                        delta += 1
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = new_mask
            if best_new_mask is not None and best_delta > 0:
                family_set.remove(old_mask)
                family_set.add(best_new_mask)
                best_family[idx] = best_new_mask
                best_score += best_delta

    # Convert best_family to lists of integers
    result = []
    for mask in best_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)

    return result
# EVOLVE-BLOCK-END
