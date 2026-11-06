import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin (evaluator kills at 150s)

    N = 16
    FULL = (1 << N) - 1

    def compute_alpha(triples):
        """
        Fast independence number computation.
        For each mask, check if it contains any triple.
        alpha = max popcount of mask with no triple contained.
        """
        if not triples:
            return N

        # Precompute triple masks
        triple_masks = []
        for t in triples:
            m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            triple_masks.append(m)

        # For each mask, check if any triple is a subset
        # A triple m is contained in mask iff (m & mask) == m
        # We want the largest mask with NO triple contained.

        # Build a "bad" set: mask is bad if it contains any triple
        # Instead of iterating submasks, we can use:
        # bad[mask] = 1 if any tm in triple_masks has (tm & mask) == tm

        # Efficient approach: for each triple mask tm, mark all supersets as bad
        # But that's what the old code did. Let's try a different approach:
        # Iterate masks from high popcount down, check if independent.

        # Precompute popcount
        pop = [0] * (1 << N)
        for i in range(1, 1 << N):
            pop[i] = pop[i >> 1] + (i & 1)

        # Check from high to low popcount
        for k in range(N, 0, -1):
            # Check all masks of popcount k
            # Generate masks of popcount k efficiently
            # Use Gosper's hack or just iterate
            found = False
            for mask in range(1 << N):
                if pop[mask] != k:
                    continue
                # Check if mask contains any triple
                independent = True
                for tm in triple_masks:
                    if (tm & mask) == tm:
                        independent = False
                        break
                if independent:
                    return k
        return 0

    def fast_alpha(triples):
        """
        Even faster: precompute a bytearray where bad[mask]=1 if mask contains any triple.
        Use the submask enumeration but only once per triple.
        """
        if not triples:
            return N

        bad = bytearray(1 << N)
        for t in triples:
            m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            rest = FULL ^ m
            sub = rest
            while True:
                bad[m | sub] = 1
                if sub == 0:
                    break
                sub = (sub - 1) & rest

        # Find largest k with some independent mask
        # Group by popcount
        pop = [0] * (1 << N)
        for i in range(1, 1 << N):
            pop[i] = pop[i >> 1] + (i & 1)

        for k in range(N, 0, -1):
            for mask in range(1 << N):
                if pop[mask] == k and not bad[mask]:
                    return k
        return 0

    def is_linear(triples):
        used_pairs = set()
        for t in triples:
            pairs = [(min(t[0],t[1]), max(t[0],t[1])), (min(t[0],t[2]), max(t[0],t[2])), (min(t[1],t[2]), max(t[1],t[2]))]
            for p in pairs:
                if p in used_pairs:
                    return False
                used_pairs.add(p)
        return True

    def greedy_build(seed_val):
        rng = random.Random(seed_val)
        triples = []
        used_pairs = set()
        all_triples = list(combinations(range(N), 3))
        rng.shuffle(all_triples)
        for t in all_triples:
            p01 = (min(t[0],t[1]), max(t[0],t[1]))
            p02 = (min(t[0],t[2]), max(t[0],t[2]))
            p12 = (min(t[1],t[2]), max(t[1],t[2]))
            if p01 not in used_pairs and p02 not in used_pairs and p12 not in used_pairs:
                triples.append(list(t))
                used_pairs.add(p01)
                used_pairs.add(p02)
                used_pairs.add(p12)
        return triples

    def local_search(triples, seed_val, max_iters=5000):
        """
        Local search with fast alpha computation.
        Try swap moves: remove one triple, add another.
        Also try drop moves if they don't increase alpha.
        """
        rng = random.Random(seed_val)
        triples = [list(t) for t in triples]
        current_alpha = fast_alpha(triples)

        used_pairs = set()
        for t in triples:
            used_pairs.add((min(t[0],t[1]), max(t[0],t[1])))
            used_pairs.add((min(t[0],t[2]), max(t[0],t[2])))
            used_pairs.add((min(t[1],t[2]), max(t[1],t[2])))

        all_triples_list = list(combinations(range(N), 3))

        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            rng.shuffle(triples)
            for idx, t in enumerate(triples):
                t_pairs = [(min(t[0],t[1]), max(t[0],t[1])), (min(t[0],t[2]), max(t[0],t[2])), (min(t[1],t[2]), max(t[1],t[2]))]
                for p in t_pairs:
                    used_pairs.discard(p)

                # Find candidates
                candidates = []
                for nt in all_triples_list:
                    p01 = (min(nt[0],nt[1]), max(nt[0],nt[1]))
                    p02 = (min(nt[0],nt[2]), max(nt[0],nt[2]))
                    p12 = (min(nt[1],nt[2]), max(nt[1],nt[2]))
                    if p01 not in used_pairs and p02 not in used_pairs and p12 not in used_pairs:
                        candidates.append(nt)

                if not candidates:
                    for p in t_pairs:
                        used_pairs.add(p)
                    continue

                rng.shuffle(candidates)
                found_better = False
                for nt in candidates[:30]:
                    nt_pairs = [(min(nt[0],nt[1]), max(nt[0],nt[1])), (min(nt[0],nt[2]), max(nt[0],nt[2])), (min(nt[1],nt[2]), max(nt[1],nt[2]))]
                    for p in nt_pairs:
                        used_pairs.add(p)
                    triples[idx] = list(nt)
                    new_alpha = fast_alpha(triples)
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        found_better = True
                        improved = True
                        break
                    else:
                        for p in nt_pairs:
                            used_pairs.discard(p)
                        triples[idx] = list(t)

                if not found_better:
                    for p in t_pairs:
                        used_pairs.add(p)

        return triples, current_alpha

    best_triples = None
    best_alpha = N + 1

    num_restarts = 200

    for i in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break

        triples = greedy_build(i * 101 + 7)
        if not triples:
            continue

        alpha = fast_alpha(triples)

        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [list(t) for t in triples]

        if best_alpha <= 5:
            break

    # Local search polish
    if best_triples is not None and best_alpha > 5:
        if time.time() - start_time < DEADLINE - 30:
            polished, polished_alpha = local_search(best_triples, 42, max_iters=3000)
            if polished_alpha < best_alpha:
                best_alpha = polished_alpha
                best_triples = polished

        if time.time() - start_time < DEADLINE - 15:
            for i in range(30):
                if time.time() - start_time > DEADLINE - 5:
                    break
                triples = greedy_build(10000 + i * 137)
                if not triples:
                    continue
                alpha = fast_alpha(triples)
                if alpha <= best_alpha:
                    polished, polished_alpha = local_search(triples, 10000 + i * 137, max_iters=2000)
                    if polished_alpha < best_alpha:
                        best_alpha = polished_alpha
                        best_triples = polished
                    if best_alpha <= 5:
                        break

    if best_triples is None:
        return [[3*i, 3*i+1, 3*i+2] for i in range(5)]

    return best_triples
