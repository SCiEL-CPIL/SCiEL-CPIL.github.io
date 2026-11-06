"""Triple systems on 16 points without K(2,2,2)."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin for 120s limit

    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute all K(2,2,2) configurations
    # A K(2,2,2) is formed by three disjoint pairs (a,a'), (b,b'), (c,c')
    # The 8 transversal triples are: {a or a', b or b', c or c'} for all 8 combos
    # We need to find all sets of 6 points that can be partitioned into 3 pairs
    # such that all 8 transversal triples exist.

    # For each set of 6 points, check if there's a partition into 3 pairs
    # where all 8 transversal triples are present.
    # Actually, we need: for the 6 points {p1,...,p6}, partition into pairs
    # (x1,x2),(y1,y2),(z1,z2) such that all 8 triples {xi,yj,zk} exist.

    # Precompute: for each set of 6 points, which triples are "transversal"
    # for some pairing.

    # Better approach: for each 6-set, enumerate all 15 ways to partition into 3 pairs.
    # For each partition, the 8 transversal triples are determined.
    # A K(2,2,2) is "present" if all 8 of those triples are in our system.

    # Let's precompute all (6-set, partition) -> list of 8 triples
    # Then a K(2,2,2) is violated if any such set of 8 triples are all present.

    # Number of 6-sets: C(16,6) = 8008
    # Number of pairings per 6-set: 15
    # Total K(2,2,2) configs: 8008 * 15 = 120120
    # Each has 8 triples.

    # For efficiency, let's precompute for each triple, which K(2,2,2) configs it belongs to.
    # Then when adding a triple, we can check if it completes any K(2,2,2).

    # Actually, let's use a different approach:
    # For each 6-set S and each pairing P of S into 3 pairs, define the 8 transversal triples.
    # A K(2,2,2) is present iff all 8 are in our triple set.

    # Let's precompute: for each 6-set, for each pairing, the 8 triples.
    # Store as: k222_configs = list of (tuple_of_8_triple_indices)

    # But 120120 * 8 = ~1M entries, manageable.

    # Alternative: For each triple t, find all 6-sets containing t where t is a transversal triple
    # for some pairing. Then when we add t, check if all other 7 in any such config are present.

    # Let me precompute: for each triple, the list of (config_id, other_7_triples)
    # where config_id identifies the specific K(2,2,2) and other_7 are the other 7 triples.

    # Actually simpler: precompute all K(2,2,2) configs as frozensets of 8 triple indices.
    # Then use a counter: for each config, count how many of its 8 triples are present.
    # When adding a triple, increment counters for all configs containing it.
    # If any counter reaches 8, the addition is invalid.

    # Let's precompute config_triples: for each triple index, list of config indices containing it.
    # And configs: list of frozensets of 8 triple indices.

    # First, generate all pairings of a 6-set.
    def get_pairings(six_points):
        """Return all 15 ways to partition 6 points into 3 unordered pairs."""
        p = list(six_points)
        pairings = []
        # Fix p[0] paired with p[1..5], then partition remaining 4
        for i in range(1, 6):
            pair1 = (p[0], p[i])
            remaining = [p[j] for j in range(6) if j != 0 and j != i]
            # Partition 4 points into 2 pairs: 3 ways
            r = remaining
            pairings.append([pair1, (r[0], r[1]), (r[2], r[3])])
            pairings.append([pair1, (r[0], r[2]), (r[1], r[3])])
            pairings.append([pair1, (r[0], r[3]), (r[1], r[2])])
        return pairings

    # Generate all K(2,2,2) configs
    configs = []  # list of frozensets of 8 triple indices
    config_for_triple = [[] for _ in range(num_triples)]  # for each triple, which configs contain it

    for six_set in combinations(range(n), 6):
        for pairing in get_pairings(six_set):
            # pairing is [(x1,x2), (y1,y2), (z1,z2)]
            triples_in_config = []
            for a in pairing[0]:
                for b in pairing[1]:
                    for c in pairing[2]:
                        t = tuple(sorted((a, b, c)))
                        triples_in_config.append(triple_to_idx[t])
            config = frozenset(triples_in_config)
            config_idx = len(configs)
            configs.append(config)
            for t_idx in triples_in_config:
                config_for_triple[t_idx].append(config_idx)

    num_configs = len(configs)
    config_count = [0] * num_configs  # how many of the 8 triples are present

    # Greedy: start with all triples through point 0 (valid seed)
    seed_triples = set()
    for a in range(1, n):
        for b in range(a + 1, n):
            t = (0, a, b)
            seed_triples.add(t)

    # Initialize config counts
    for t in seed_triples:
        t_idx = triple_to_idx[t]
        for c_idx in config_for_triple[t_idx]:
            config_count[c_idx] += 1

    # Remaining triples to consider
    remaining = []
    for t in all_triples:
        if t not in seed_triples:
            remaining.append(t)

    # Greedy add in random order with multiple restarts
    best_triples = set(seed_triples)
    best_count = len(seed_triples)

    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break

        random.shuffle(remaining)
        current_triples = set(seed_triples)
        current_config_count = config_count[:]  # copy

        for t in remaining:
            if time.time() > deadline - 5:
                break
            t_idx = triple_to_idx[t]
            # Check if adding t would complete any K(2,2,2)
            valid = True
            for c_idx in config_for_triple[t_idx]:
                if current_config_count[c_idx] == 7:
                    valid = False
                    break
            if valid:
                current_triples.add(t)
                for c_idx in config_for_triple[t_idx]:
                    current_config_count[c_idx] += 1

        if len(current_triples) > best_count:
            best_count = len(current_triples)
            best_triples = set(current_triples)

    # Local search: try to swap out one triple for two others
    if time.time() < deadline - 15:
        current = list(best_triples)
        current_set = set(best_triples)
        current_config_count = [0] * num_configs
        for t in current_set:
            t_idx = triple_to_idx[t]
            for c_idx in config_for_triple[t_idx]:
                current_config_count[c_idx] += 1

        # Try removing one triple and adding up to 2 others
        removed_triples = []
        for t in list(current_set):
            if time.time() > deadline - 10:
                break
            t_idx = triple_to_idx[t]
            # Temporarily remove t
            for c_idx in config_for_triple[t_idx]:
                current_config_count[c_idx] -= 1
            current_set.discard(t)
            removed_triples.append(t)

            # Try to add triples that were previously blocked
            added = 0
            for t2 in remaining:
                if added >= 2:
                    break
                if time.time() > deadline - 10:
                    break
                if t2 in current_set:
                    continue
                t2_idx = triple_to_idx[t2]
                valid = True
                for c_idx in config_for_triple[t2_idx]:
                    if current_config_count[c_idx] == 7:
                        valid = False
                        break
                if valid:
                    current_set.add(t2)
                    for c_idx in config_for_triple[t2_idx]:
                        current_config_count[c_idx] += 1
                    added += 1

            # Restore t if we didn't gain net
            if added <= 1:
                current_set.add(t)
                for c_idx in config_for_triple[t_idx]:
                    current_config_count[c_idx] += 1
                removed_triples.pop()

        if len(current_set) > best_count:
            best_count = len(current_set)
            best_triples = set(current_set)

    return [list(t) for t in best_triples]
