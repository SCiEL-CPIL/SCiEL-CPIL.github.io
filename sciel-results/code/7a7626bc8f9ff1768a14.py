import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    n = 16
    all_points = list(range(n))

    # Precompute all C(16,3) = 560 possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    
    # For each triple, precompute the 3 pairs it uses (as frozensets or sorted tuples)
    triple_pairs = {}
    for t in all_triples:
        pairs = (t[0], t[1]), (t[0], t[2]), (t[1], t[2])
        triple_pairs[t] = pairs

    # Index: pair -> list of triples containing that pair
    pair_to_triples = defaultdict(list)
    for t in all_triples:
        for p in triple_pairs[t]:
            pair_to_triples[p].append(t)

    # Bitmask helpers for independence check
    # Each triple is a 16-bit mask
    triple_masks = {}
    for t in all_triples:
        m = 0
        for x in t:
            m |= (1 << x)
        triple_masks[t] = m

    def is_linear(triple_set):
        """Check if no pair is shared by two triples."""
        seen_pairs = set()
        for t in triple_set:
            for p in triple_pairs[t]:
                if p in seen_pairs:
                    return False
                seen_pairs.add(p)
        return True

    def compute_alpha(triple_set, timeout=2.0):
        """
        Compute the independence number (largest set containing no triple) 
        using branch-and-bound with bitmasks.
        Returns (alpha, best_set) or (None, None) on timeout.
        """
        if not triple_set:
            return n, set(range(n))

        # Build adjacency: for each point, which triples contain it
        point_triples = defaultdict(list)
        for t in triple_set:
            for x in t:
                point_triples[x].append(t)

        # For fast "does adding point p create a triple" check:
        # A set S is independent iff no triple is fully contained in S.
        # We can check incrementally.

        best_alpha = 0
        best_set = set()
        deadline_local = start_time + timeout
        node_count = [0]

        # Sort points by degree (number of triples containing them) descending
        # Points in more triples are "more constrained"
        degrees = {x: len(point_triples[x]) for x in range(n)}
        order = sorted(range(n), key=lambda x: -degrees[x])

        # Reverse order for backtracking: we decide points in this order
        # Use a bitmask for current set
        def backtrack(idx, current_mask, current_size):
            nonlocal best_alpha, best_set
            node_count[0] += 1
            if node_count[0] % 5000 == 0 and time.time() > deadline_local:
                raise TimeoutError()

            # Upper bound: current_size + remaining points
            remaining = n - idx
            if current_size + remaining <= best_alpha:
                return

            if idx == n:
                if current_size > best_alpha:
                    best_alpha = current_size
                    best_set = set(i for i in range(n) if current_mask & (1 << i))
                return

            p = order[idx]
            bit = 1 << p

            # Option 1: don't include p
            backtrack(idx + 1, current_mask, current_size)

            # Option 2: include p (if it doesn't complete a triple)
            new_mask = current_mask | bit
            # Check: does adding p complete any triple?
            # A triple is completed if the other two points are in current_mask
            valid = True
            for t in point_triples[p]:
                other_mask = triple_masks[t] & ~bit
                if (current_mask & other_mask) == other_mask:
                    valid = False
                    break
            if valid:
                backtrack(idx + 1, new_mask, current_size + 1)

        try:
            backtrack(0, 0, 0)
            return best_alpha, best_set
        except TimeoutError:
            return None, None

    def quick_alpha_upper_bound(triple_set):
        """
        Fast greedy upper bound on alpha.
        Greedily build a maximal independent set; the size is a lower bound on alpha.
        For an upper bound, we use a simple heuristic: 
        alpha <= n - (number of triples * 2) / 3 ... not tight.
        Actually, let's just use the greedy lower bound as a proxy for ranking,
        and use exact alpha only for the best candidates.
        """
        # Greedy: repeatedly add the point that blocks the fewest remaining triples
        used = set()
        remaining_triples = set(triple_set)
        current_set = set()
        
        for _ in range(n):
            best_point = None
            best_score = -1
            for p in range(n):
                if p in used:
                    continue
                # Count how many remaining triples would be "hit" by adding p
                # Actually, we want to add p if it doesn't complete any triple
                # Check if p can be added
                can_add = True
                for t in remaining_triples:
                    if p in t:
                        others = [x for x in t if x != p]
                        if all(o in current_set for o in others):
                            can_add = False
                            break
                if not can_add:
                    continue
                # Score: prefer points that are in fewer remaining triples (less constraining)
                count = sum(1 for t in remaining_triples if p in t)
                if count < best_score or best_point is None:
                    best_score = count
                    best_point = p
            if best_point is None:
                break
            current_set.add(best_point)
            used.add(best_point)
            # Remove triples that are now "blocked" (contain best_point and another in current_set)
            # Actually, we just need to track which triples are still "available"
            # A triple is available if none of its points are in current_set... no.
            # A triple is "completed" if all 3 points are in current_set.
            # We just need to ensure no triple is fully in current_set.
            # The greedy above already checks this.
        
        return len(current_set)

    # --- Main: Simulated Annealing ---
    
    # Initial solution: sparse random linear packing
    def random_linear_packing(num_triples, seed=0):
        rng = random.Random(seed)
        triples = []
        used_pairs = set()
        candidates = all_triples[:]
        rng.shuffle(candidates)
        for t in candidates:
            if len(triples) >= num_triples:
                break
            ok = True
            for p in triple_pairs[t]:
                if p in used_pairs:
                    ok = False
                    break
            if ok:
                triples.append(t)
                for p in triple_pairs[t]:
                    used_pairs.add(p)
        return triples

    # Start with a moderate number of triples
    # Max possible for linear 3-uniform on 16 points: floor(16*15/6) = 40
    # But we want sparse to allow removal
    initial_count = 20
    current = random_linear_packing(initial_count, seed=42)
    
    # Verify linearity
    assert is_linear(current), "Initial packing not linear"
    
    current_alpha, _ = compute_alpha(current, timeout=3.0)
    if current_alpha is None:
        current_alpha = quick_alpha_upper_bound(current)
    
    best = current[:]
    best_alpha = current_alpha
    best_set = None
    
    T = 5.0  # initial temperature
    cooling = 0.995
    T_min = 0.1
    
    rng = random.Random(123)
    
    while time.time() < deadline and T > T_min:
        # Generate a neighbor
        action = rng.random()
        new_triples = current[:]
        
        if action < 0.35:
            # Add a random triple
            candidates = [t for t in all_triples if t not in new_triples]
            if candidates:
                t = rng.choice(candidates)
                # Check linearity
                ok = True
                for p in triple_pairs[t]:
                    for existing in new_triples:
                        if p in triple_pairs[existing]:
                            ok = False
                            break
                        if not ok:
                            break
                if ok:
                    new_triples.append(t)
        elif action < 0.70:
            # Remove a random triple
            if new_triples:
                idx = rng.randrange(len(new_triples))
                new_triples.pop(idx)
        elif action < 0.85:
            # Swap: remove one, add one
            if new_triples:
                removed = new_triples.pop(rng.randrange(len(new_triples)))
                candidates = [t for t in all_triples if t not in new_triples and t != removed]
                if candidates:
                    t = rng.choice(candidates)
                    ok = True
                    for p in triple_pairs[t]:
                        for existing in new_triples:
                            if p in triple_pairs[existing]:
                                ok = False
                                break
                            if not ok:
                                break
                    if ok:
                        new_triples.append(t)
                    else:
                        new_triples.append(removed)  # put it back
        else:
            # 2-opt: remove two, add two
            if len(new_triples) >= 2:
                idx1 = rng.randrange(len(new_triples))
                idx2 = rng.randrange(len(new_triples))
                if idx1 == idx2:
                    idx2 = (idx2 + 1) % len(new_triples)
                r1 = new_triples.pop(idx1)
                r2 = new_triples.pop(idx2 if idx2 < idx1 else idx2 - 1)
                candidates = [t for t in all_triples if t not in new_triples and t != r1 and t != r2]
                if len(candidates) >= 2:
                    picks = rng.sample(candidates, 2)
                    ok = True
                    for t in picks:
                        for p in triple_pairs[t]:
                            for existing in new_triples:
                                if p in triple_pairs[existing]:
                                    ok = False
                                    break
                                if not ok:
                                    break
                        if not ok:
                            break
                    # Also check picks don't conflict with each other
                    if ok:
                        for p1 in triple_pairs[picks[0]]:
                            for p2 in triple_pairs[picks[1]]:
                                if p1 == p2:
                                    ok = False
                                    break
                            if not ok:
                                break
                    if ok:
                        new_triples.extend(picks)
                    else:
                        new_triples.extend([r1, r2])
                else:
                    new_triples.extend([r1, r2])

        if new_triples == current:
            continue
        
        # Evaluate
        new_alpha, _ = compute_alpha(new_triples, timeout=2.0)
        if new_alpha is None:
            # Timeout: use quick bound as proxy
            new_alpha = quick_alpha_upper_bound(new_triples)
        
        delta = new_alpha - current_alpha
        
        if delta <= 0 or rng.random() < pow(2.718281828, -delta / max(T, 0.01)):
            current = new_triples
            current_alpha = new_alpha
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best = new_triples[:]
        
        T *= cooling
        
        # Periodic local search: try to remove triples that are "redundant" for independence
        if time.time() < deadline and len(current) > 5:
            # Try removing each triple and see if alpha stays the same or improves
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                candidate = current[:]
                candidate.pop(i)
                a, _ = compute_alpha(candidate, timeout=1.5)
                if a is None:
                    a = quick_alpha_upper_bound(candidate)
                if a <= current_alpha:
                    current = candidate
                    current_alpha = a
                    if a < best_alpha:
                        best_alpha = a
                        best = current[:]
                    break  # restart the loop

    # Final: verify best is valid
    if not is_linear(best):
        # Fallback
        best = random_linear_packing(15, seed=99)
    
    return [list(t) for t in best]
