"""Bitmask-accelerated local search for equal-LCM-triple-free subset of [1,128].

We maximise |S| such that no three distinct elements a,b,c in S satisfy
lcm(a,b)=lcm(a,c)=lcm(b,c).  The search is a greedy initialisation followed by
repeated first-improvement hill-climbing on a small set of deterministic
perturbations, with a hard wall-clock deadline.
"""
import time
import random
from math import gcd


def construct():
    deadline = time.time() + 82
    N = 128
    FULL_MASK = (1 << N) - 1

    # ---- Precompute LCM table -------------------------------------------------
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[(i, j)] = i * j // gcd(i, j)

    # ---- Find all bad triples and store as bitmasks ---------------------------
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[(a, b)]
            for c in range(b + 1, N + 1):
                if lcm_table[(a, c)] == lab and lcm_table[(b, c)] == lab:
                    bad_triples.append((1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)))

    num_bad = len(bad_triples)

    # For each node, list of (triple_index, others_mask)
    triples_by_node = [[] for _ in range(N)]
    for idx, mask in enumerate(bad_triples):
        m = mask
        while m:
            lsb = m & (-m)
            node = lsb.bit_length() - 1
            triples_by_node[node].append((idx, mask & ~(1 << node)))
            m ^= lsb

    # ---- Helper functions -----------------------------------------------------

    def count_violations(mask):
        """Count number of bad triples fully contained in mask."""
        cnt = 0
        for tm in bad_triples:
            if (mask & tm) == tm:
                cnt += 1
        return cnt

    def delta_add(node, mask):
        """Number of new violations if node is added to mask."""
        cnt = 0
        for _, others in triples_by_node[node]:
            if (mask & others) == others:
                cnt += 1
        return cnt

    def delta_remove(node, mask):
        """Number of violations broken if node is removed from mask."""
        cnt = 0
        for _, others in triples_by_node[node]:
            if (mask & others) == others:
                cnt += 1
        return cnt

    def try_greedy_add(mask, order=None):
        """Add any node that is safe (zero new violations), iterating until fixpoint."""
        if order is None:
            order = list(range(N))
        changed = True
        while changed:
            changed = False
            for node in order:
                if not (mask >> node) & 1:
                    if delta_add(node, mask) == 0:
                        mask |= (1 << node)
                        changed = True
        return mask

    def first_improvement_hill_climb(mask, time_limit, max_iters=200000):
        """
        First-improvement hill-climbing:
        - try to add a random unselected node; accept if it doesn't increase violations
          (i.e., delta_add == 0)
        - if stuck (no add possible), try to remove a random selected node to break
          a violation, then re-add greedily.
        This is a local search that maintains validity (E=0) as an invariant whenever
        possible, and only temporarily leaves E>0 to escape local optima.
        """
        current_E = count_violations(mask)
        best_mask = mask
        best_size = mask.bit_count()
        iters = 0

        while time.time() < time_limit and iters < max_iters:
            iters += 1

            if current_E == 0:
                # Try to add a random unselected node
                unselected = FULL_MASK & ~mask
                if unselected:
                    # Pick a random bit from unselected
                    # Efficient: use random to pick an index
                    # But we need a random bit position. Let's pick random node and check.
                    # For speed, we'll try a few random nodes.
                    improved = False
                    for _ in range(5):
                        node = random.randrange(N)
                        if not (mask >> node) & 1:
                            if delta_add(node, mask) == 0:
                                mask |= (1 << node)
                                improved = True
                                sz = mask.bit_count()
                                if sz > best_size:
                                    best_size = sz
                                    best_mask = mask
                                break
                    if not improved:
                        # Try all nodes in a shuffled order
                        order = list(range(N))
                        random.shuffle(order)
                        for node in order:
                            if not (mask >> node) & 1:
                                if delta_add(node, mask) == 0:
                                    mask |= (1 << node)
                                    improved = True
                                    sz = mask.bit_count()
                                    if sz > best_size:
                                        best_size = sz
                                        best_mask = mask
                                    break
                        if not improved:
                            # Stuck: remove a random node that's in at least one violation
                            # But E==0 so no violations. Instead, try a random swap.
                            # Remove a random node, then greedily add others.
                            if mask:
                                lsb = mask & (-mask)
                                node = lsb.bit_length() - 1
                                mask &= ~(1 << node)
                                mask = try_greedy_add(mask)
                else:
                    # All nodes selected
                    break
            else:
                # E > 0: try to reduce violations
                # Remove a node that participates in the most violations
                # Find a node in a violation
                # Pick a random bad triple that's fully in mask, remove one of its nodes
                # For efficiency, just pick a random node and see if removal helps
                improved = False
                for _ in range(5):
                    if mask == 0:
                        break
                    lsb = mask & (-mask)
                    node = lsb.bit_length() - 1
                    d = delta_remove(node, mask)
                    if d > 0:
                        mask &= ~(1 << node)
                        current_E -= d
                        improved = True
                        break
                if not improved:
                    # Try all nodes
                    for node in range(N):
                        if (mask >> node) & 1:
                            d = delta_remove(node, mask)
                            if d > 0:
                                mask &= ~(1 << node)
                                current_E -= d
                                improved = True
                                break
                if not improved:
                    # Shouldn't happen if E>0, but just in case
                    mask = try_greedy_add(mask)
                    current_E = count_violations(mask)

            # Periodically recompute E if it got out of sync
            if iters % 1000 == 0:
                current_E = count_violations(mask)

            # If valid, try greedy add
            if current_E == 0:
                mask = try_greedy_add(mask)
                sz = mask.bit_count()
                if sz > best_size:
                    best_size = sz
                    best_mask = mask

        # Final greedy add
        mask = try_greedy_add(mask)
        sz = mask.bit_count()
        if sz > best_size:
            best_size = sz
            best_mask = mask

        return best_mask

    # ---- Search ----------------------------------------------------------------

    best_mask = 0
    best_size = 0

    def update_best(mask):
        nonlocal best_mask, best_size
        sz = mask.bit_count()
        if sz > best_size:
            best_size = sz
            best_mask = mask

    random.seed(12345)

    # --- Phase 1: Multiple greedy initialisations with different orders ---
    # Order by fewest conflicts first
    order_fewest = sorted(range(N), key=lambda x: len(triples_by_node[x]))
    S = try_greedy_add(0, order_fewest)
    update_best(S)

    # Order by most conflicts first (counter-intuitive but sometimes works)
    order_most = sorted(range(N), key=lambda x: -len(triples_by_node[x]))
    S = try_greedy_add(0, order_most)
    update_best(S)

    # Random orders
    for _ in range(20):
        if time.time() > deadline - 15:
            break
        order = list(range(N))
        random.shuffle(order)
        S = try_greedy_add(0, order)
        update_best(S)

    # --- Phase 2: Hill-climb from best and random starts ---
    time_limit = deadline - 5

    # Hill-climb from best
    S = first_improvement_hill_climb(best_mask, time_limit)
    update_best(S)

    # Hill-climb from several random starts
    for _ in range(30):
        if time.time() > time_limit:
            break
        # Random start: select random subset
        size = random.randint(40, 80)
        sample = random.sample(range(N), size)
        S_start = 0
        for node in sample:
            S_start |= (1 << node)
        S = first_improvement_hill_climb(S_start, time_limit, max_iters=50000)
        update_best(S)

    # --- Phase 3: Final polish ---
    if time.time() < deadline - 2:
        S_best = best_mask
        S_best = try_greedy_add(S_best)
        update_best(S_best)

    # --- Phase 4: Try removing one node and re-adding to escape local optima ---
    if time.time() < deadline - 2:
        for node in range(N):
            if time.time() > deadline - 2:
                break
            if (best_mask >> node) & 1:
                S = best_mask & ~(1 << node)
                S = try_greedy_add(S)
                update_best(S)

    if best_mask == 0:
        best_mask = 1  # element 1

    # Convert bitmask to list
    result = []
    m = best_mask
    while m:
        lsb = m & (-m)
        node = lsb.bit_length() - 1
        result.append(node + 1)
        m ^= lsb

    return result
