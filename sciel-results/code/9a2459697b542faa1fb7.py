import random
import time
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    N = 16
    FULL_MASK = (1 << N) - 1
    ALL_TRIPLES = list(combinations(range(N), 3))
    TRIPLE_MASKS = [(1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in ALL_TRIPLES]

    # Precompute: for each triple index, its mask
    # For fast alpha computation, we'll use a precomputed "contains triple" table
    # contains[s] = True if subset s (bitmask) contains at least one triple from the system

    def compute_alpha(triple_indices):
        """
        Compute independence number for a set of triples given by their indices.
        Uses a bytearray 'contains' where contains[s] = 1 if s contains any triple.
        Then alpha = max popcount(s) where contains[s] == 0.
        """
        if not triple_indices:
            return N

        # Build contains table
        # For each triple mask, mark all its supersets
        contains = bytearray(1 << N)
        for idx in triple_indices:
            mask = TRIPLE_MASKS[idx]
            complement = (~mask) & FULL_MASK
            # Iterate all subsets of complement
            sub = complement
            while True:
                contains[mask | sub] = 1
                if sub == 0:
                    break
                sub = (sub - 1) & complement

        # Find max popcount among independent subsets
        # We can iterate from high popcount down
        # Precompute popcount for all 2^16 values
        # Actually, let's just iterate all and track max
        best = 0
        # Use a list of (popcount, subset) sorted by popcount descending? No, too slow.
        # Just iterate all 65536 subsets
        # But we can be smarter: iterate by popcount from N down to 1
        # and stop at first independent subset
        
        # Precompute popcount
        # For speed, use bin(s).count('1') or a lookup
        # Let's use a precomputed popcount array
        # Actually for 65536 entries, computing on the fly is fine if we're careful
        
        # Better approach: iterate subsets, skip if contains[s]
        # Track best
        for s in range(1 << N):
            if contains[s]:
                continue
            pc = bin(s).count('1')
            if pc > best:
                best = pc
                if best == N:
                    break
        return best

    def compute_alpha_fast(triple_indices):
        """
        Optimized alpha computation.
        Key insight: we only need to check subsets that are NOT supersets of any triple.
        Use a different strategy: for each subset size k from N down to 1,
        check if there exists a k-subset with no triple.
        But that's expensive.
        
        Alternative: use the fact that the independence number is at most 
        N - (number of points that are "forced" to be excluded).
        
        Actually, let's use a smarter enumeration:
        - Start with all 16 points
        - For each triple, at least one point must be excluded
        - This is a hitting set / vertex cover type problem
        
        For small N=16, the bytearray approach is fast enough.
        Let's optimize the superset marking.
        """
        if not triple_indices:
            return N

        contains = bytearray(1 << N)
        for idx in triple_indices:
            mask = TRIPLE_MASKS[idx]
            complement = (~mask) & FULL_MASK
            sub = complement
            while True:
                contains[mask | sub] = 1
                if sub == 0:
                    break
                sub = (sub - 1) & complement

        best = 0
        # Iterate subsets in order of decreasing popcount for early termination
        # Group subsets by popcount
        # Precompute lists of subsets by popcount
        # For 16 points, popcount ranges from 0 to 16
        # subsets_by_pc[k] = list of subsets with popcount k
        # But building this every time is expensive.
        
        # Instead, just iterate all and track max
        # With 65536 iterations and simple checks, this is ~0.01s in Python
        for s in range(1 << N):
            if contains[s]:
                continue
            pc = bin(s).count('1')
            if pc > best:
                best = pc
        return best

    def is_valid_addition(triple_idx, used_pairs):
        """Check if adding triple_idx would violate linearity."""
        t = ALL_TRIPLES[triple_idx]
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        for p in pairs:
            if p in used_pairs or (p[1], p[0]) in used_pairs:
                return False
        return True

    def get_pairs(triple_idx):
        t = ALL_TRIPLES[triple_idx]
        return [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]

    def greedy_construct(seed_val):
        """Greedy construction with given seed."""
        rng = random.Random(seed_val)
        indices = list(range(len(ALL_TRIPLES)))
        rng.shuffle(indices)

        used_pairs = set()
        triple_indices = []

        for idx in indices:
            if is_valid_addition(idx, used_pairs):
                for p in get_pairs(idx):
                    used_pairs.add(p)
                    used_pairs.add((p[1], p[0]))
                triple_indices.append(idx)

        return triple_indices

    def local_search(triple_indices, max_iterations=2000):
        """
        Local search to minimize independence number.
        Try swaps: remove one triple, add another.
        """
        triple_indices = list(triple_indices)
        used_pairs = set()
        for idx in triple_indices:
            for p in get_pairs(idx):
                used_pairs.add(p)
                used_pairs.add((p[1], p[0]))

        current_alpha = compute_alpha_fast(triple_indices)

        improved = True
        iters = 0
        rng = random.Random(42)

        while improved and iters < max_iterations:
            improved = False
            iters += 1

            # Try random swaps
            for _ in range(10):
                if not triple_indices:
                    break
                old_idx = rng.randrange(len(triple_indices))
                
                # Remove old triple
                for p in get_pairs(old_idx):
                    used_pairs.discard(p)
                    used_pairs.discard((p[1], p[0]))
                triple_indices.pop(old_idx)

                # Find valid candidates to add
                candidates = []
                for idx in range(len(ALL_TRIPLES)):
                    if idx in triple_indices:
                        continue
                    if is_valid_addition(idx, used_pairs):
                        candidates.append(idx)

                if candidates:
                    new_idx = rng.choice(candidates)
                    for p in get_pairs(new_idx):
                        used_pairs.add(p)
                        used_pairs.add((p[1], p[0]))
                    triple_indices.append(new_idx)

                    new_alpha = compute_alpha_fast(triple_indices)
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        improved = True
                    else:
                        # Revert
                        for p in get_pairs(new_idx):
                            used_pairs.discard(p)
                            used_pairs.discard((p[1], p[0]))
                        triple_indices.pop()
                        # Restore old
                        for p in get_pairs(old_idx):
                            used_pairs.add(p)
                            used_pairs.add((p[1], p[0]))
                        triple_indices.append(old_idx)
                else:
                    # Restore old
                    for p in get_pairs(old_idx):
                        used_pairs.add(p)
                        used_pairs.add((p[1], p[0]))
                    triple_indices.append(old_idx)

        return triple_indices, current_alpha

    def add_triple_if_beneficial(triple_indices, used_pairs, max_adds=5):
        """Try to add triples that reduce independence number."""
        triple_indices = list(triple_indices)
        used_pairs = set(used_pairs)
        current_alpha = compute_alpha_fast(triple_indices)

        for _ in range(max_adds):
            candidates = []
            for idx in range(len(ALL_TRIPLES)):
                if idx in triple_indices:
                    continue
                if is_valid_addition(idx, used_pairs):
                    candidates.append(idx)

            if not candidates:
                break

            best_candidate = None
            best_alpha = current_alpha
            for idx in candidates[:20]:  # Try a subset
                for p in get_pairs(idx):
                    used_pairs.add(p)
                    used_pairs.add((p[1], p[0]))
                triple_indices.append(idx)

                new_alpha = compute_alpha_fast(triple_indices)
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_candidate = idx
                else:
                    # Revert
                    for p in get_pairs(idx):
                        used_pairs.discard(p)
                        used_pairs.discard((p[1], p[0]))
                    triple_indices.pop()

            if best_candidate is not None:
                # Re-add the best candidate
                for p in get_pairs(best_candidate):
                    used_pairs.add(p)
                    used_pairs.add((p[1], p[0]))
                triple_indices.append(best_candidate)
                current_alpha = best_alpha
            else:
                break

        return triple_indices, used_pairs, current_alpha

    best_triples = None
    best_indep = N + 1
    best_triple_indices = None

    # Phase 1: Greedy constructions
    num_greedy = 300
    for i in range(num_greedy):
        if time.time() - start_time > DEADLINE:
            break

        triple_indices = greedy_construct(i * 104729 + 13)
        if not triple_indices:
            continue

        alpha = compute_alpha_fast(triple_indices)

        if alpha < best_indep:
            best_indep = alpha
            best_triple_indices = list(triple_indices)
            best_triples = [list(ALL_TRIPLES[idx]) for idx in triple_indices]

        if best_indep <= 4:
            break

    # Phase 2: Local search on best solutions
    if best_triple_indices is not None and time.time() - start_time < DEADLINE:
        improved_indices, improved_alpha = local_search(best_triple_indices, max_iterations=3000)
        if improved_alpha < best_indep:
            best_indep = improved_alpha
            best_triple_indices = improved_indices
            best_triples = [list(ALL_TRIPLES[idx]) for idx in improved_indices]

    # Phase 3: More greedy + local search
    for i in range(num_greedy, num_greedy + 500):
        if time.time() - start_time > DEADLINE:
            break

        triple_indices = greedy_construct(i * 104729 + 13)
        if not triple_indices:
            continue

        # Try to improve by adding triples
        used_pairs = set()
        for idx in triple_indices:
            for p in get_pairs(idx):
                used_pairs.add(p)
                used_pairs.add((p[1], p[0]))

        triple_indices, used_pairs, alpha = add_triple_if_beneficial(triple_indices, used_pairs)

        if alpha < best_indep:
            best_indep = alpha
            best_triple_indices = list(triple_indices)
            best_triples = [list(ALL_TRIPLES[idx]) for idx in triple_indices]

            # Run local search on this promising solution
            if time.time() - start_time < DEADLINE:
                improved_indices, improved_alpha = local_search(best_triple_indices, max_iterations=2000)
                if improved_alpha < best_indep:
                    best_indep = improved_alpha
                    best_triple_indices = improved_indices
                    best_triples = [list(ALL_TRIPLES[idx]) for idx in improved_indices]

        if best_indep <= 4:
            break

    if best_triples is None:
        # Fallback: return a simple valid system
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

    return best_triples
