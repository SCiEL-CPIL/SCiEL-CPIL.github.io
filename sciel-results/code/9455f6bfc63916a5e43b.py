<<<<<<< SEARCH
def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561

    # Precompute balls: balls[w] is the list of words within distance 1 of w
    # There are 1 + 2*n = 17 such words
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    weights = [3 ** (n - 1 - i) for i in range(n)]

    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1

    # Precompute a lookup: for each word, which codewords cover it
    # This is the transpose of balls, but we'll build it efficiently
    # Actually, for the 2-exchange we need: given a set of uncovered words, find codewords that cover all of them
    # We can precompute for each word, the list of codewords within distance 1
    # Since balls is symmetric (if w is in balls[c], then c is in balls[w]), balls itself serves this purpose

    best_size = total
    best_set = None

    def verify_and_update(selected):
        """Verify coverage and update best if improved"""
        nonlocal best_size, best_set
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1
        if np.all(cover_count > 0):
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = list(selected)
                return True
        return False

    def greedy_construct(seed, deadline_check):
        """Diversity-biased greedy construction"""
        np.random.seed(seed)
        random.seed(seed)

        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)

        while np.any(uncovered_mask):
            if time.time() > deadline_check:
                return None

            # Count how many uncovered words each candidate covers
            scores = uncovered_mask[balls].sum(axis=1)
            max_score = np.max(scores)
            if max_score == 0:
                break

            candidates = np.where(scores == max_score)[0]

            # Diversity bias: prefer candidates that are far from already selected
            if selected and len(candidates) > 1:
                # Compute minimum distance to existing selected
                min_dists = np.full(len(candidates), 999)
                for ci, cand in enumerate(candidates):
                    min_d = min(_hamming(cand, s) for s in selected[-20:])  # Only check recent 20
                    min_dists[ci] = min_d
                # Weight towards higher minimum distance
                # Use softmax-like selection
                max_d = np.max(min_dists)
                weights_arr = np.exp((min_dists - max_d) / 2.0)
                weights_arr /= weights_arr.sum()
                best_idx = candidates[np.random.choice(len(candidates), p=weights_arr)]
            else:
                best_idx = candidates[np.random.randint(len(candidates))]

            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)

        if np.any(uncovered_mask):
            return None

        return selected, cover_count

    def _hamming(a, b):
        """Compute Hamming distance between two word indices"""
        d = 0
        for i in range(n):
            if ((a // weights[i]) % 3) != ((b // weights[i]) % 3):
                d += 1
        return d

    def local_search(selected, cover_count, deadline_check):
        """Apply removal, 1-swap, and 2-exchange local search"""
        # Phase 1: Remove redundant codewords
        changed = True
        while changed and time.time() < deadline_check:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected

        # Phase 2: 1-Swap (Replace one codeword with another to enable more removals)
        improved = True
        while improved and time.time() < deadline_check - 3.0:
            improved = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue

                critical_mask = (cover_count[balls[c]] == 1)
                critical_words = balls[c][critical_mask]

                if len(critical_words) == 0:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue

                # Find candidates that cover all critical words
                candidates = set(balls[critical_words[0]].tolist())
                for cw in critical_words[1:]:
                    candidates.intersection_update(balls[cw].tolist())
                    if not candidates:
                        break

                candidates.discard(c)

                if candidates:
                    c_prime = random.choice(list(candidates))
                    cover_count[balls[c]] -= 1
                    cover_count[balls[c_prime]] += 1
                    new_selected.append(c_prime)
                    improved = True
                else:
                    new_selected.append(c)

            selected = new_selected

        # Phase 3: 2-Exchange (Remove 2, Add 1)
        if len(selected) > 486 and time.time() < deadline_check - 5.0:
            improved_2ex = True
            while improved_2ex and time.time() < deadline_check - 3.0:
                improved_2ex = False

                # Try to find a pair that can be replaced by one
                num_trials = min(3000, len(selected) * (len(selected) - 1) // 2)

                for _ in range(num_trials):
                    if time.time() > deadline_check - 3.0:
                        break

                    idx1 = random.randint(0, len(selected) - 1)
                    idx2 = random.randint(0, len(selected) - 1)
                    if idx1 == idx2:
                        continue

                    c1 = selected[idx1]
                    c2 = selected[idx2]

                    # Temporarily remove both
                    cover_count[balls[c1]] -= 1
                    cover_count[balls[c2]] -= 1

                    uncovered_after = (cover_count == 0)

                    if np.any(uncovered_after):
                        uncovered_words = np.where(uncovered_after)[0]

                        # Find c3 that covers all uncovered words
                        if len(uncovered_words) <= 5:  # Only try if few uncovered
                            candidates = set(balls[uncovered_words[0]].tolist())
                            for uw in uncovered_words[1:]:
                                candidates.intersection_update(balls[uw].tolist())
                                if not candidates:
                                    break

                            candidates.discard(c1)
                            candidates.discard(c2)

                            if candidates:
                                c3 = random.choice(list(candidates))
                                cover_count[balls[c3]] += 1
                                # Rebuild selected without c1, c2, add c3
                                new_selected = []
                                for c in selected:
                                    if c != c1 and c != c2:
                                        new_selected.append(c)
                                new_selected.append(c3)
                                selected = new_selected
                                improved_2ex = True
                                break
                            else:
                                # Revert
                                cover_count[balls[c1]] += 1
                                cover_count[balls[c2]] += 1
                        else:
                            # Too many uncovered, revert
                            cover_count[balls[c1]] += 1
                            cover_count[balls[c2]] += 1
                    else:
                        # Both can be removed entirely
                        new_selected = []
                        for c in selected:
                            if c != c1 and c != c2:
                                new_selected.append(c)
                        selected = new_selected
                        improved_2ex = True
                        break

                    if not improved_2ex:
                        cover_count[balls[c1]] += 1
                        cover_count[balls[c2]] += 1

        # Final removal pass
        changed = True
        while changed and time.time() < deadline_check:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected

        return selected, cover_count

    # Main loop: multiple iterations
    iteration = 0
    while time.time() < deadline - 10.0:
        iteration += 1
        deadline_check = deadline - 5.0

        # Construct
        result = greedy_construct(iteration * 7 + 13, deadline_check)
        if result is None:
            continue

        selected, cover_count = result

        # Local search
        selected, cover_count = local_search(selected, cover_count, deadline_check)

        # Verify and update
        verify_and_update(selected)

        if best_size <= 486:
            break

    # If we found a good solution, return it
    if best_set is not None:
        result = []
        for w in best_set:
            digits = []
            temp = w
            for _ in range(n):
                digits.append(temp % 3)
                temp //= 3
            digits.reverse()
            result.append(digits)
        return result

    # Fallback: Simple greedy
    selected = []
    cover_count = np.zeros(total, dtype=np.int32)
    uncovered_mask = np.ones(total, dtype=bool)

    while np.any(uncovered_mask):
        scores = uncovered_mask[balls].sum(axis=1)
        max_score = np.max(scores)
        if max_score == 0:
            break
        candidates = np.where(scores == max_score)[0]
        best_idx = candidates[0]
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        uncovered_mask = (cover_count == 0)

    # Quick removal pass
    changed = True
    while changed:
        changed = False
        new_selected = []
        for c in selected:
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected

    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    return result
=======
def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561

    # Precompute balls: balls[w] is the list of words within distance 1 of w
    # There are 1 + 2*n = 17 such words
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    weights = [3 ** (n - 1 - i) for i in range(n)]

    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1

    # Precompute frozenset of ball neighbors for fast intersection
    ball_frozensets = [frozenset(balls[w].tolist()) for w in range(total)]

    best_size = total
    best_set = None

    def verify_and_update(selected):
        """Verify coverage and update best if improved"""
        nonlocal best_size, best_set
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1
        if np.all(cover_count > 0):
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = list(selected)
                return True
        return False

    def greedy_construct(seed, deadline_check):
        """Diversity-biased greedy construction"""
        np.random.seed(seed)
        random.seed(seed)

        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)

        while np.any(uncovered_mask):
            if time.time() > deadline_check:
                return None

            # Count how many uncovered words each candidate covers
            scores = uncovered_mask[balls].sum(axis=1)
            max_score = np.max(scores)
            if max_score == 0:
                break

            candidates = np.where(scores == max_score)[0]

            # Diversity bias: prefer candidates that are far from already selected
            if selected and len(candidates) > 1:
                # Compute minimum distance to existing selected
                min_dists = np.full(len(candidates), 999)
                for ci, cand in enumerate(candidates):
                    min_d = min(_hamming(cand, s) for s in selected[-20:])  # Only check recent 20
                    min_dists[ci] = min_d
                # Weight towards higher minimum distance
                # Use softmax-like selection
                max_d = np.max(min_dists)
                weights_arr = np.exp((min_dists - max_d) / 2.0)
                weights_arr /= weights_arr.sum()
                best_idx = candidates[np.random.choice(len(candidates), p=weights_arr)]
            else:
                best_idx = candidates[np.random.randint(len(candidates))]

            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)

        if np.any(uncovered_mask):
            return None

        return selected, cover_count

    def _hamming(a, b):
        """Compute Hamming distance between two word indices"""
        d = 0
        for i in range(n):
            if ((a // weights[i]) % 3) != ((b // weights[i]) % 3):
                d += 1
        return d

    def compute_objective(selected, cover_count, lam=10.0):
        """
        WalkSAT-inspired weighted objective:
        minimize (#codewords + lam * #uncovered_words)
        This is a smooth surrogate that guides both coverage and minimization.
        """
        uncovered_count = int(np.sum(cover_count == 0))
        return len(selected) + lam * uncovered_count

    def local_search_wsa(selected, cover_count, deadline_check):
        """
        WalkSAT-inspired local search with weighted clause violations.
        Uses a unified objective: minimize #codewords + lam * #uncovered.
        Moves: remove, swap, 2-exchange — accepted if objective improves.
        """
        lam = 10.0  # Weight for uncovered words (penalty)
        current_obj = compute_objective(selected, cover_count, lam)
        
        # Track selected as a set for O(1) membership
        selected_set = set(selected)
        
        # Phase 1: Greedy removal (remove any codeword that doesn't cause uncovered)
        changed = True
        while changed and time.time() < deadline_check:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            if changed:
                selected = new_selected
                selected_set = set(selected)
                current_obj = compute_objective(selected, cover_count, lam)

        # Phase 2: WalkSAT-style stochastic local search
        # Mix of deterministic improvement and random restarts
        max_no_improve = 500
        no_improve = 0
        best_local_obj = current_obj
        best_local_selected = list(selected)
        best_local_cover = cover_count.copy()

        while time.time() < deadline_check - 5.0 and no_improve < max_no_improve:
            if time.time() > deadline_check - 5.0:
                break

            # Choose a random move type
            move_type = random.random()
            
            if move_type < 0.4:
                # Try removal: pick a random codeword and see if removing it helps
                if len(selected) <= 1:
                    no_improve += 1
                    continue
                idx = random.randint(0, len(selected) - 1)
                c = selected[idx]
                
                # Check if removal causes uncovered
                cover_count[balls[c]] -= 1
                new_obj = compute_objective(selected, cover_count, lam)
                
                if new_obj < current_obj:
                    # Accept: remove c
                    selected = [x for x in selected if x != c]
                    selected_set.discard(c)
                    current_obj = new_obj
                    no_improve = 0
                    if current_obj < best_local_obj:
                        best_local_obj = current_obj
                        best_local_selected = list(selected)
                        best_local_cover = cover_count.copy()
                else:
                    # Reject: restore
                    cover_count[balls[c]] += 1
                    no_improve += 1

            elif move_type < 0.7:
                # Try 1-swap: replace one codeword with another
                if len(selected) <= 1:
                    no_improve += 1
                    continue
                idx = random.randint(0, len(selected) - 1)
                c = selected[idx]
                
                # Find candidates that cover all words uniquely covered by c
                critical_mask = (cover_count[balls[c]] == 1)
                critical_words = balls[c][critical_mask]
                
                if len(critical_words) == 0:
                    # Can just remove
                    cover_count[balls[c]] -= 1
                    new_obj = compute_objective(selected, cover_count, lam)
                    if new_obj < current_obj:
                        selected = [x for x in selected if x != c]
                        selected_set.discard(c)
                        current_obj = new_obj
                        no_improve = 0
                        if current_obj < best_local_obj:
                            best_local_obj = current_obj
                            best_local_selected = list(selected)
                            best_local_cover = cover_count.copy()
                    else:
                        cover_count[balls[c]] += 1
                        no_improve += 1
                    continue

                # Find intersection of balls of critical words
                candidates = ball_frozensets[critical_words[0]]
                for cw in critical_words[1:]:
                    candidates = candidates & ball_frozensets[cw]
                    if not candidates:
                        break

                candidates.discard(c)

                if candidates:
                    c_prime = random.choice(list(candidates))
                    # Perform swap
                    cover_count[balls[c]] -= 1
                    if c_prime not in selected_set:
                        cover_count[balls[c_prime]] += 1
                        new_obj = compute_objective(selected, cover_count, lam)
                    else:
                        new_obj = current_obj + 1  # Adding a duplicate is bad

                    if new_obj < current_obj:
                        # Accept swap
                        selected = [x for x in selected if x != c]
                        selected_set.discard(c)
                        if c_prime not in selected_set:
                            selected.append(c_prime)
                            selected_set.add(c_prime)
                        current_obj = new_obj
                        no_improve = 0
                        if current_obj < best_local_obj:
                            best_local_obj = current_obj
                            best_local_selected = list(selected)
                            best_local_cover = cover_count.copy()
                    else:
                        # Revert
                        cover_count[balls[c]] += 1
                        if c_prime not in selected_set:
                            cover_count[balls[c_prime]] -= 1
                        no_improve += 1
                else:
                    no_improve += 1

            else:
                # Try 2-exchange: remove 2, add 1
                if len(selected) <= 2:
                    no_improve += 1
                    continue
                
                idx1 = random.randint(0, len(selected) - 1)
                idx2 = random.randint(0, len(selected) - 1)
                if idx1 == idx2:
                    no_improve += 1
                    continue

                c1 = selected[idx1]
                c2 = selected[idx2]

                # Temporarily remove both
                cover_count[balls[c1]] -= 1
                cover_count[balls[c2]] -= 1

                uncovered_after = (cover_count == 0)

                if np.any(uncovered_after):
                    uncovered_words = np.where(uncovered_after)[0]

                    if len(uncovered_words) <= 6:
                        # Find c3 that covers all uncovered words
                        candidates = ball_frozensets[uncovered_words[0]]
                        for uw in uncovered_words[1:]:
                            candidates = candidates & ball_frozensets[uw]
                            if not candidates:
                                break

                        candidates.discard(c1)
                        candidates.discard(c2)

                        if candidates:
                            c3 = random.choice(list(candidates))
                            if c3 not in selected_set:
                                cover_count[balls[c3]] += 1
                                new_obj = compute_objective(selected, cover_count, lam)
                            else:
                                new_obj = current_obj + 1

                            if new_obj < current_obj:
                                # Accept: remove c1, c2, add c3
                                selected = [x for x in selected if x != c1 and x != c2]
                                selected_set.discard(c1)
                                selected_set.discard(c2)
                                if c3 not in selected_set:
                                    selected.append(c3)
                                    selected_set.add(c3)
                                current_obj = new_obj
                                no_improve = 0
                                if current_obj < best_local_obj:
                                    best_local_obj = current_obj
                                    best_local_selected = list(selected)
                                    best_local_cover = cover_count.copy()
                            else:
                                # Revert
                                cover_count[balls[c1]] += 1
                                cover_count[balls[c2]] += 1
                                if c3 not in selected_set:
                                    cover_count[balls[c3]] -= 1
                                no_improve += 1
                        else:
                            cover_count[balls[c1]] += 1
                            cover_count[balls[c2]] += 1
                            no_improve += 1
                    else:
                        cover_count[balls[c1]] += 1
                        cover_count[balls[c2]] += 1
                        no_improve += 1
                else:
                    # Both can be removed
                    selected = [x for x in selected if x != c1 and x != c2]
                    selected_set.discard(c1)
                    selected_set.discard(c2)
                    current_obj = compute_objective(selected, cover_count, lam)
                    no_improve = 0
                    if current_obj < best_local_obj:
                        best_local_obj = current_obj
                        best_local_selected = list(selected)
                        best_local_cover = cover_count.copy()

            # Periodic restart to escape local minima
            if no_improve % 100 == 0 and no_improve > 0:
                # Soft restart: keep best found so far, perturb slightly
                if random.random() < 0.3:
                    # Add a random codeword to break symmetry
                    if len(selected) < total:
                        random_c = random.randint(0, total - 1)
                        if random_c not in selected_set:
                            selected.append(random_c)
                            selected_set.add(random_c)
                            cover_count[balls[random_c]] += 1
                            current_obj = compute_objective(selected, cover_count, lam)
                            no_improve = 0

        # Restore best local solution
        selected = best_local_selected
        cover_count = best_local_cover

        # Final polish: aggressive removal pass
        changed = True
        while changed and time.time() < deadline_check:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected

        return selected, cover_count

    # Main loop: multiple iterations
    iteration = 0
    while time.time() < deadline - 10.0:
        iteration += 1
        deadline_check = deadline - 5.0

        # Construct
        result = greedy_construct(iteration * 7 + 13, deadline_check)
        if result is None:
            continue

        selected, cover_count = result

        # Local search
        selected, cover_count = local_search_wsa(selected, cover_count, deadline_check)

        # Verify and update
        verify_and_update(selected)

        if best_size <= 486:
            break

    # If we found a good solution, return it
    if best_set is not None:
        result = []
        for w in best_set:
            digits = []
            temp = w
            for _ in range(n):
                digits.append(temp % 3)
                temp //= 3
            digits.reverse()
            result.append(digits)
        return result

    # Fallback: Simple greedy
    selected = []
    cover_count = np.zeros(total, dtype=np.int32)
    uncovered_mask = np.ones(total, dtype=bool)

    while np.any(uncovered_mask):
        scores = uncovered_mask[balls].sum(axis=1)
        max_score = np.max(scores)
        if max_score == 0:
            break
        candidates = np.where(scores == max_score)[0]
        best_idx = candidates[0]
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        uncovered_mask = (cover_count == 0)

    # Quick removal pass
    changed = True
    while changed:
        changed = False
        new_selected = []
        for c in selected:
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected

    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    return result
>>>>>>> REPLACE
