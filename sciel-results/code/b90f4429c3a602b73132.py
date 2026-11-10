<<<<<<< SEARCH
def construct():
    start = _now()
    deadline = start + 150.0 - DEADLINE_MARGIN
    rng = random.Random(12345)

    n = 16
    best_triples = None
    best_alpha = n + 1

    def consider(triples):
        nonlocal best_triples, best_alpha
        if not _is_linear(triples):
            return
        a = _independence_number(triples, n)
        if a < best_alpha:
            best_alpha = a
            best_triples = [list(t) for t in triples]

    # Phase 1: many random greedy packings
    attempts = 0
    while _now() < deadline:
        attempts += 1
        triples = _random_greedy_packing(rng, n)
        consider(triples)
        # local search: try to break independent sets
        for _ in range(200):
            if _now() > deadline:
                break
            iset = _greedy_maximal_independent_set(triples, n, rng)
            if len(iset) <= best_alpha:
                # can't improve via this set
                continue
            new_t = _try_add_triple_breaking_iset(triples, iset, rng, n)
            if new_t is not None:
                triples.append(new_t)
                consider(triples)
            else:
                # try removing a random triple then adding
                removed, _ = _remove_triple_redundant(triples, rng, n)
                if removed is not None:
                    new_t = _try_add_triple_breaking_iset(triples, iset, rng, n)
                    if new_t is not None:
                        triples.append(new_t)
                        consider(triples)
                    else:
                        triples.append(removed)
                break
        if attempts >= 200:
            break

    # Phase 2: targeted improvement on best
    if best_triples is not None and _now() < deadline:
        triples = [list(t) for t in best_triples]
        for _ in range(500):
            if _now() > deadline:
                break
            iset = _greedy_maximal_independent_set(triples, n, rng)
            if len(iset) <= best_alpha:
                break
            new_t = _try_add_triple_breaking_iset(triples, iset, rng, n)
            if new_t is not None:
                triples.append(new_t)
                consider(triples)
            else:
                # try a few random removals then add
                improved = False
                for _ in range(10):
                    removed, _ = _remove_triple_redundant(triples, rng, n)
                    if removed is None:
                        break
                    new_t = _try_add_triple_breaking_iset(triples, iset, rng, n)
                    if new_t is not None:
                        triples.append(new_t)
                        consider(triples)
                        improved = True
                        break
                    else:
                        triples.append(removed)
                if not improved:
                    break

    # ensure we always return a valid system
    if best_triples is None:
        # fallback: seed
        best_triples = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    return best_triples
=======
def construct():
    start = _now()
    deadline = start + 150.0 - DEADLINE_MARGIN
    rng = random.Random(12345)
    n = 16
    
    # Initial solution
    best_triples = _random_greedy_packing(rng, n)
    best_alpha = _independence_number(best_triples, n)
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.9995
    current_triples = [list(t) for t in best_triples]
    current_alpha = best_alpha
    
    all_triples = list(combinations(range(n), 3))
    
    while _now() < deadline:
        # Generate a move
        move_type = rng.random()
        new_triples = [list(t) for t in current_triples]
        
        if move_type < 0.3:
            # Add a triple
            new_t = None
            for _ in range(20):
                cand = list(rng.sample(all_triples, 1))[0]
                ok = True
                for ex in current_triples:
                    if len(set(cand) & set(ex)) > 1:
                        ok = False
                        break
                if ok:
                    new_t = cand
                    break
            if new_t:
                new_triples.append(new_t)
            else:
                continue
        elif move_type < 0.6:
            # Remove a triple
            if current_triples:
                idx = rng.randrange(len(current_triples))
                new_triples.pop(idx)
            else:
                continue
        else:
            # Swap a triple
            if not current_triples:
                continue
            idx = rng.randrange(len(current_triples))
            old_t = current_triples[idx]
            new_triples.pop(idx)
            
            new_t = None
            for _ in range(20):
                cand = list(rng.sample(all_triples, 1))[0]
                ok = True
                for ex in new_triples:
                    if len(set(cand) & set(ex)) > 1:
                        ok = False
                        break
                if ok:
                    new_t = cand
                    break
            
            if new_t:
                new_triples.append(new_t)
            else:
                continue

        new_alpha = _independence_number(new_triples, n)
        
        # Acceptance criterion
        if new_alpha < current_alpha:
            current_triples = new_triples
            current_alpha = new_alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = [list(t) for t in current_triples]
        else:
            delta = new_alpha - current_alpha
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current_triples = new_triples
                current_alpha = new_alpha
        
        T *= cooling
        if T < 0.1:
            T = 0.1

    return best_triples
>>>>>>> REPLACE
