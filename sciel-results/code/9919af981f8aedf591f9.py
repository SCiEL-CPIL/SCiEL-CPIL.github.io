<<<<<<< SEARCH
def _local_search(initial, max_steps=2000):
    """Local search: add/remove/swap elements to improve score."""
    best_U = list(initial)
    best_score = _score_set(best_U)
    current_U = list(best_U)
    current_score = best_score
    stall_count = 0

    for step in range(max_steps):
        if _elapsed() > _DEADLINE:
            break
        max_val = max(current_U) if current_U else 0
        improved = False
        cur_set = set(current_U)

        candidates = []
        
        # Values that extend the range
        for multiplier in [2, 3, 5, 7, 11]:
            for delta in range(1, 30):
                candidates.append(max_val * multiplier + delta)
        
        # Prime-like steps
        for a in current_U:
            for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
                candidates.append(a + p)
                candidates.append(a + p * 2)
                candidates.append(a + p * 3)
        
        # Interior values
        if max_val > 10:
            step_size = max(1, max_val // 80)
            for v in range(1, max_val + 1, step_size):
                if v not in cur_set:
                    candidates.append(v)
        
        # Random exploration
        for _ in range(40):
            candidates.append(random.randint(1, max_val * 5))
        
        candidates = list(set([v for v in candidates if v > 0 and v not in cur_set]))
        random.shuffle(candidates)

        for new_val in candidates[:40]:
            if new_val <= 0 or new_val in cur_set:
                continue
            new_U = list(current_U) + [new_val]
            new_U.sort()
            new_score = _score_set(new_U)
            if new_score > current_score + 1e-10:
                current_U = new_U
                current_score = new_score
                cur_set = set(current_U)
                if current_score > best_score:
                    best_U = list(current_U)
                    best_score = current_score
                improved = True
                stall_count = 0
                break

        if not improved:
            if len(current_U) > 2:
                remove_candidates = list(range(1, len(current_U)))
                random.shuffle(remove_candidates)
                for idx in remove_candidates[:20]:
                    new_U = [x for i, x in enumerate(current_U) if i != idx]
                    new_score = _score_set(new_U)
                    if new_score > current_score + 1e-10:
                        current_U = new_U
                        current_score = new_score
                        cur_set = set(current_U)
                        if current_score > best_score:
                            best_U = list(current_U)
                            best_score = current_score
                        improved = True
                        stall_count = 0
                        break

        if not improved:
            stall_count += 1
            if stall_count > 50:
                if len(current_U) > 2:
                    idx = random.randint(1, len(current_U) - 1)
                    max_val = max(current_U)
                    new_val = random.randint(1, max_val * 5)
                    if new_val not in cur_set:
                        current_U[idx] = new_val
                        current_U.sort()
                        cur_set = set(current_U)
                        current_score = _score_set(current_U)
                        stall_count = 0
                else:
                    break
            elif step > 200:
                break

    return best_U, best_score
=======
def _local_search(initial, max_steps=2000):
    """Local search: add/remove/swap elements to improve score."""
    best_U = list(initial)
    best_score = _score_set(best_U)
    current_U = list(best_U)
    current_score = best_score
    stall_count = 0

    for step in range(max_steps):
        if _elapsed() > _DEADLINE:
            break
        max_val = max(current_U) if current_U else 0
        improved = False
        cur_set = set(current_U)

        candidates = []
        
        # Values that extend the range
        for multiplier in [2, 3, 5, 7, 11, 13, 17]:
            for delta in range(1, 20):
                candidates.append(max_val * multiplier + delta)
        
        # Pairwise differences and sums (structural candidates)
        for i, a in enumerate(current_U):
            for b in current_U[i+1:]:
                candidates.append(a + b)
                candidates.append(abs(a - b))
                candidates.append(a * 2 + b)
                candidates.append(a + b * 2)
        
        # Prime-like steps from each element
        for a in current_U:
            for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]:
                candidates.append(a + p)
                candidates.append(a + p * 2)
        
        # Interior values (denser sampling)
        if max_val > 10:
            step_size = max(1, max_val // 120)
            for v in range(1, max_val + 1, step_size):
                if v not in cur_set:
                    candidates.append(v)
        
        # Random exploration
        for _ in range(30):
            candidates.append(random.randint(1, max_val * 3))
        for _ in range(10):
            candidates.append(random.randint(1, max_val * 10))
        
        candidates = list(set([v for v in candidates if v > 0 and v not in cur_set]))
        random.shuffle(candidates)

        for new_val in candidates[:50]:
            if new_val <= 0 or new_val in cur_set:
                continue
            new_U = list(current_U) + [new_val]
            new_U.sort()
            new_score = _score_set(new_U)
            if new_score > current_score + 1e-12:
                current_U = new_U
                current_score = new_score
                cur_set = set(current_U)
                if current_score > best_score:
                    best_U = list(current_U)
                    best_score = current_score
                improved = True
                stall_count = 0
                break

        if not improved:
            if len(current_U) > 2:
                remove_candidates = list(range(1, len(current_U)))
                random.shuffle(remove_candidates)
                for idx in remove_candidates[:25]:
                    new_U = [x for i, x in enumerate(current_U) if i != idx]
                    new_score = _score_set(new_U)
                    if new_score > current_score + 1e-12:
                        current_U = new_U
                        current_score = new_score
                        cur_set = set(current_U)
                        if current_score > best_score:
                            best_U = list(current_U)
                            best_score = current_score
                        improved = True
                        stall_count = 0
                        break

        if not improved:
            stall_count += 1
            if stall_count > 40:
                if len(current_U) > 2:
                    idx = random.randint(1, len(current_U) - 1)
                    max_val = max(current_U)
                    # Try several random replacements
                    for _ in range(10):
                        new_val = random.randint(1, max_val * 5)
                        if new_val not in cur_set:
                            old_val = current_U[idx]
                            current_U[idx] = new_val
                            current_U.sort()
                            cur_set = set(current_U)
                            new_score = _score_set(current_U)
                            if new_score >= current_score - 0.01:
                                current_score = new_score
                                break
                            else:
                                current_U[idx] = old_val
                                current_U.sort()
                                cur_set = set(current_U)
                    stall_count = 0
                else:
                    break
            elif step > 300:
                break

    return best_U, best_score
>>>>>>> REPLACE

