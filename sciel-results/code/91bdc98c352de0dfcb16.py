import random
import time
from itertools import combinations

def construct():
    deadline = time.time() + 78  # Safety margin before 90s kill

    # Precompute all 4-subsets as bitmasks
    all_sets = []
    mask_to_idx = {}
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
        mask_to_idx[mask] = len(all_sets) - 1

    n_sets = len(all_sets)  # 1820
    all_mask = (1 << 16) - 1

    # Precompute partners for fast blocked update
    # partners[i][j] is a list of indices k such that sets i, j, k form a sunflower
    # We compute this lazily and cache.
    partner_cache = {}

    def get_partners(i, j):
        key = (i, j) if i < j else (j, i)
        if key in partner_cache:
            return partner_cache[key]

        mask_i = all_sets[i]
        mask_j = all_sets[j]
        core = mask_i & mask_j
        sym = mask_i ^ mask_j
        avail = all_mask ^ (mask_i | mask_j)

        k_needed = 4 - bin(core).count('1')
        if k_needed < 0 or k_needed > bin(avail).count('1'):
            partner_cache[key] = []
            return []

        # Generate all subsets of 'avail' of size k_needed
        bits = [b for b in range(16) if avail & (1 << b)]
        partners = []
        for combo in combinations(bits, k_needed):
            add_mask = 0
            for b in combo:
                add_mask |= (1 << b)
            full_mask = core | add_mask
            idx = mask_to_idx.get(full_mask)
            if idx is not None:
                partners.append(idx)

        partner_cache[key] = partners
        return partners

    def greedy_min_degree(seed):
        """
        Greedy algorithm that selects the next set with the minimum 'degree'
        (number of conflicts) with the current family.
        """
        rng = random.Random(seed)
        family = []
        family_set = set()
        blocked = [0] * n_sets
        available = set(range(n_sets))
        
        # To make it randomized, we can pick among the top few minimum degree nodes
        # or just pick the global minimum. Let's try global minimum first for determinism
        # in the heuristic, but maybe add some randomness in tie-breaking.
        
        for _ in range(n_sets):
            if not available:
                break
            
            # Find the node with minimum blocked count
            min_blocked = float('inf')
            candidates = []
            
            # Optimization: We can't just scan all available every time if it's slow.
            # But n_sets=1820 is small enough.
            for idx in available:
                if blocked[idx] < min_blocked:
                    min_blocked = blocked[idx]
                    candidates = [idx]
                elif blocked[idx] == min_blocked:
                    candidates.append(idx)
            
            if not candidates:
                break
                
            # Pick randomly from candidates to break ties and add diversity
            idx = rng.choice(candidates)
            
            family.append(idx)
            family_set.add(idx)
            available.remove(idx)
            
            # Update blocked counts
            for j in family[:-1]:
                partners = get_partners(idx, j)
                for k in partners:
                    if k in available: # Only care about available nodes
                        blocked[k] += 1
                    elif k in family_set:
                        # This would mean a sunflower was formed, which shouldn't happen
                        # if our logic is correct, but blocked counts for family members
                        # aren't strictly needed for the 'available' set logic.
                        pass

        return family

    def greedy_randomized(seed):
        """
        Standard random greedy for comparison/fallback.
        """
        rng = random.Random(seed)
        family = []
        family_set = set()
        blocked = [0] * n_sets

        indices = list(range(n_sets))
        rng.shuffle(indices)

        for idx in indices:
            if blocked[idx] == 0 and idx not in family_set:
                # Add idx
                family.append(idx)
                family_set.add(idx)

                # Update blocked
                for j in family[:-1]:  # All previously added
                    partners = get_partners(idx, j)
                    for k in partners:
                        blocked[k] += 1

        return family

    def local_search(family, rng, time_limit):
        family_set = set(family)
        blocked = [0] * n_sets

        # Initialize blocked
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                partners = get_partners(family[i], family[j])
                for k in partners:
                    blocked[k] += 1

        improved = True
        while improved and time.time() < time_limit:
            improved = False

            # Try to add
            available = [i for i in range(n_sets) if blocked[i] == 0 and i not in family_set]
            if available:
                # Pick random
                idx = rng.choice(available)
                family.append(idx)
                family_set.add(idx)
                for j in family[:-1]:
                    partners = get_partners(idx, j)
                    for k in partners:
                        blocked[k] += 1
                improved = True
                continue

            # Try swap: remove 1, add 2
            if len(family) == 0:
                break
            rem_idx = rng.choice(family)

            # Remove rem_idx
            family.remove(rem_idx)
            family_set.remove(rem_idx)
            for j in family:
                partners = get_partners(rem_idx, j)
                for k in partners:
                    blocked[k] -= 1

            # Try to add 2
            added = []
            available = [i for i in range(n_sets) if blocked[i] == 0 and i not in family_set]
            if len(available) >= 2:
                # Pick first
                idx1 = rng.choice(available)
                family.append(idx1)
                family_set.add(idx1)
                for j in family[:-1]:
                    partners = get_partners(idx1, j)
                    for k in partners:
                        blocked[k] += 1

                # Update available for second
                available2 = [i for i in range(n_sets) if blocked[i] == 0 and i not in family_set]
                if available2:
                    idx2 = rng.choice(available2)
                    family.append(idx2)
                    family_set.add(idx2)
                    for j in family[:-1]:
                        partners = get_partners(idx2, j)
                        for k in partners:
                            blocked[k] += 1

                    added = [idx1, idx2]
                    improved = True
                    continue

            # Revert removal
            family.append(rem_idx)
            family_set.add(rem_idx)
            for j in family[:-1]:
                partners = get_partners(rem_idx, j)
                for k in partners:
                    blocked[k] += 1

        return family

    best_family = []
    best_size = 0

    # Try multiple restarts with both strategies
    # Alternate between min-degree and random greedy
    for seed in range(200):
        if time.time() > deadline:
            break
        
        # Use min-degree greedy for even seeds, random for odd?
        # Or just try min-degree first as it's the structural change.
        if seed % 2 == 0:
            fam = greedy_min_degree(seed)
        else:
            fam = greedy_randomized(seed)
            
        if len(fam) > best_size:
            best_family = fam[:]
            best_size = len(fam)

        # Local search with a small time budget per restart
        remaining = deadline - time.time()
        if remaining > 1.0:
            rng = random.Random(seed)
            ts_limit = time.time() + min(1.5, remaining * 0.8)
            fam = local_search(fam, rng, ts_limit)
            if len(fam) > best_size:
                best_family = fam[:]
                best_size = len(fam)

    return [sorted([i + 1 for i in range(16) if all_sets[idx] & (1 << i)]) for idx in best_family]
