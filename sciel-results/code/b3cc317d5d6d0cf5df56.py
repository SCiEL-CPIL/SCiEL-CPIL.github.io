<<<<<<< SEARCH
def construct():
    """Return two length-N lists of destination states in 0..N-1.

    Must define a complete binary deterministic automaton that synchronizes.
    Score is the exact shortest reset-word length, maximized.
    """
    best_thresh = -1
    best_a = None
    best_b = None
    rng = random.Random(12345)

    # A known-good fallback (cycle + sink) guarantees synchronization.
    fallback_a = [(i + 1) % N for i in range(N)]
    fallback_b = [0] * N
    fb_thr = _threshold(fallback_a, fallback_b)
    if fb_thr > best_thresh:
        best_thresh = fb_thr
        best_a = fallback_a[:]
        best_b = fallback_b[:]

    # ---- Structured search over a = cycle, b = perturbed monotone ----
    a_cycle = [(i + 1) % N for i in range(N)]
    
    # Construction A: b = [0,1,2,...,10,10]
    b_mono = list(range(N-1)) + [N-2]
    thr = _threshold(a_cycle, b_mono)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_mono[:]

    # Construction B: a = cycle, b = identity except b(0) = 1
    b_id = list(range(N))
    b_id[0] = 1
    thr = _threshold(a_cycle, b_id)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_id[:]

    # Construction C: a = cycle, b = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    b_shift = [0, 0] + list(range(1, N-1))
    thr = _threshold(a_cycle, b_shift)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_shift[:]

    # Construction D: Try all "one defect" perturbations of the monotone b
    base_b = list(range(N-1)) + [N-2]
    for i in range(N):
        for new_val in range(N):
            if new_val == base_b[i]:
                continue
            if _elapsed() > _DEADLINE:
                break
            b_test = base_b[:]
            b_test[i] = new_val
            thr = _threshold(a_cycle, b_test)
            if thr > best_thresh:
                best_thresh = thr
                best_a = a_cycle[:]
                best_b = b_test[:]

    # Construction E: Random search with a = cycle (increased trials)
    for trial in range(1000):
        if _elapsed() > _DEADLINE:
            break
        b_rand = [rng.randint(0, N-1) for _ in range(N)]
        thr = _threshold(a_cycle, b_rand)
        if thr > best_thresh:
            best_thresh = thr
            best_a = a_cycle[:]
            best_b = b_rand[:]

    # Construction F: Random permutations for a, random b (increased trials)
    for trial in range(500):
        if _elapsed() > _DEADLINE:
            break
        a_perm = list(range(N))
        rng.shuffle(a_perm)
        b_rand = [rng.randint(0, N-1) for _ in range(N)]
        thr = _threshold(a_perm, b_rand)
        if thr > best_thresh:
            best_thresh = thr
            best_a = a_perm[:]
            best_b = b_rand[:]

    # Construction G: a = cycle in reverse direction
    a_rev = [(i - 1) % N for i in range(N)]
    b_mono2 = list(range(N-1)) + [N-2]
    thr = _threshold(a_rev, b_mono2)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_rev[:]
        best_b = b_mono2[:]

    # Construction H: Try a "bottleneck" construction
    for sink in range(N):
        if _elapsed() > _DEADLINE:
            break
        b_sink = [(i + 1) % N for i in range(N)]
        b_sink[sink] = sink
        thr = _threshold(a_cycle, b_sink)
        if thr > best_thresh:
            best_thresh = thr
            best_a = a_cycle[:]
            best_b = b_sink[:]

    # Construction I: Two-step bottleneck
    for i in range(N):
        if _elapsed() > _DEADLINE:
            break
        b_two = [(i + 1) % N for i in range(N)]
        b_two[i] = (i + 2) % N
        thr = _threshold(a_cycle, b_two)
        if thr > best_thresh:
            best_thresh = thr
            best_a = a_cycle[:]
            best_b = b_two[:]

    # Construction J: a = cycle, b = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    b_three = [0, 0, 0] + list(range(1, N-1))
    thr = _threshold(a_cycle, b_three)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_three[:]

    # Construction K: a = cycle, b = [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    b_dup = [0, 1, 1] + list(range(2, N-1))
    thr = _threshold(a_cycle, b_dup)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_dup[:]

    # Construction L: a = cycle, b = "fold" pattern [0,1,2,...,5,5,4,3,2,1,0]
    b_fold = [i for i in range(N//2)] + [N//2-1-i for i in range(N//2)]
    if len(b_fold) < N:
        b_fold.extend([0] * (N - len(b_fold)))
    thr = _threshold(a_cycle, b_fold)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_fold[:]

    # Construction M: a = cycle, b = "two peaks" [0,1,2,3,4,5,5,4,3,2,1,0]
    b_twopeaks = [min(i, 11-i) for i in range(N)]
    thr = _threshold(a_cycle, b_twopeaks)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_twopeaks[:]

    # Construction N: a = cycle, b = "staircase with gap" [0,1,2,3,4,4,5,6,7,8,9,10]
    b_gap = [0,1,2,3,4,4,5,6,7,8,9,10]
    thr = _threshold(a_cycle, b_gap)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_gap[:]

    # Construction O: a = cycle, b = "zigzag" [0,5,1,6,2,7,3,8,4,9,5,10]
    b_zigzag = [i if i % 2 == 0 else i + 5 for i in range(N)]
    b_zigzag = [x % N for x in b_zigzag]
    thr = _threshold(a_cycle, b_zigzag)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_zigzag[:]

    # Construction P: a = cycle, b = "reverse half" [0,1,2,3,4,5,10,9,8,7,6,5]
    b_revhalf = list(range(6)) + list(range(10, 4, -1))
    thr = _threshold(a_cycle, b_revhalf)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_revhalf[:]

    # Construction Q: a = cycle, b = "three clusters" [0,0,1,1,2,2,3,3,4,4,5,5]
    b_clusters = [i // 2 for i in range(N)]
    thr = _threshold(a_cycle, b_clusters)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_clusters[:]

    # Construction R: a = cycle, b = "four clusters" [0,0,0,1,1,1,2,2,2,3,3,3]
    b_fourclust = [i // 3 for i in range(N)]
    thr = _threshold(a_cycle, b_fourclust)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_fourclust[:]

    # ---- Genetic Algorithm for global search ----
    if best_a is not None and best_b is not None and _elapsed() < _DEADLINE - 10:
        pop_size = 60
        # Initialize population with best solution + random perturbations
        population = []
        population.append((best_a[:], best_b[:], best_thresh))
        
        for _ in range(pop_size - 1):
            a_gen = list(range(N))
            rng.shuffle(a_gen)
            b_gen = [rng.randint(0, N-1) for _ in range(N)]
            thr_gen = _threshold(a_gen, b_gen)
            if thr_gen > 0:
                population.append((a_gen, b_gen, thr_gen))
            else:
                population.append((a_gen, b_gen, 0))
        
        # Sort by fitness
        population.sort(key=lambda x: x[2], reverse=True)
        
        for gen in range(50):
            if _elapsed() > _DEADLINE:
                break
            
            new_population = []
            
            # Elitism: keep top 2
            new_population.append(population[0])
            if len(population) > 1:
                new_population.append(population[1])
            
            # Generate offspring
            while len(new_population) < pop_size:
                # Tournament selection
                def tournament():
                    cands = [population[rng.randrange(len(population))] for _ in range(3)]
                    return max(cands, key=lambda x: x[2])
                
                parent1 = tournament()
                parent2 = tournament()
                
                # Crossover: random mix of a and b from parents
                child_a = []
                child_b = []
                for i in range(N):
                    if rng.random() < 0.5:
                        child_a.append(parent1[0][i])
                        child_b.append(parent1[1][i])
                    else:
                        child_a.append(parent2[0][i])
                        child_b.append(parent2[1][i])
                
                # Mutation
                if rng.random() < 0.3:
                    idx = rng.randrange(N)
                    child_a[idx] = rng.randrange(N)
                if rng.random() < 0.3:
                    idx = rng.randrange(N)
                    child_b[idx] = rng.randrange(N)
                
                thr_child = _threshold(child_a, child_b)
                new_population.append((child_a, child_b, thr_child if thr_child > 0 else 0))
            
            population = new_population
            population.sort(key=lambda x: x[2], reverse=True)
            
            # Update best
            if population[0][2] > best_thresh:
                best_thresh = population[0][2]
                best_a = population[0][0][:]
                best_b = population[0][1][:]
    
    # ---- Local search: perturb best found automaton (increased iterations) ----
    if best_a is not None and best_b is not None:
        current_a = best_a[:]
        current_b = best_b[:]
        current_thr = best_thresh
        for _ in range(2000):
            if _elapsed() > _DEADLINE:
                break
            # Randomly perturb one transition
            new_a = current_a[:]
            new_b = current_b[:]
            if rng.random() < 0.5:
                idx = rng.randrange(N)
                new_a[idx] = rng.randrange(N)
            else:
                idx = rng.randrange(N)
                new_b[idx] = rng.randrange(N)
            thr = _threshold(new_a, new_b)
            if thr > current_thr:
                current_a = new_a
                current_b = new_b
                current_thr = thr
                if current_thr > best_thresh:
                    best_thresh = current_thr
                    best_a = current_a[:]
                    best_b = current_b[:]
            elif thr > 0 and rng.random() < 0.15:
                # Accept worse solutions occasionally (simulated annealing-like)
                current_a = new_a
                current_b = new_b
                current_thr = thr

    # ---- Second local search phase with different random seed ----
    if best_a is not None and best_b is not None and _elapsed() < _DEADLINE - 5:
        rng2 = random.Random(99999)
        current_a = best_a[:]
        current_b = best_b[:]
        current_thr = best_thresh
        for _ in range(1000):
            if _elapsed() > _DEADLINE:
                break
            new_a = current_a[:]
            new_b = current_b[:]
            if rng2.random() < 0.5:
                idx = rng2.randrange(N)
                new_a[idx] = rng2.randrange(N)
            else:
                idx = rng2.randrange(N)
                new_b[idx] = rng2.randrange(N)
            thr = _threshold(new_a, new_b)
            if thr > current_thr:
                current_a = new_a
                current_b = new_b
                current_thr = thr
                if current_thr > best_thresh:
                    best_thresh = current_thr
                    best_a = current_a[:]
                    best_b = current_b[:]
            elif thr > 0 and rng2.random() < 0.15:
                current_a = new_a
                current_b = new_b
                current_thr = thr

    # Fallback: ensure we have a valid synchronizing automaton
    if best_a is None or best_b is None or best_thresh <= 0:
        best_a = fallback_a
        best_b = fallback_b
        best_thresh = _threshold(best_a, best_b)

    return best_a, best_b
=======
def _steepest_ascent(a, b, current_thr, max_iters=200):
    """Steepest ascent hill climbing: at each step, try all single-position
    changes in a and b, pick the best improvement. Repeat until no improvement
    or max_iters reached. Returns (best_a, best_b, best_thr)."""
    cur_a = a[:]
    cur_b = b[:]
    cur_thr = current_thr

    for _ in range(max_iters):
        if _elapsed() > _DEADLINE - 2:
            break
        best_neighbor_thr = cur_thr
        best_neighbor_a = None
        best_neighbor_b = None

        # Try all single changes in a
        for i in range(N):
            for val in range(N):
                if val == cur_a[i]:
                    continue
                if _elapsed() > _DEADLINE - 2:
                    break
                new_a = cur_a[:]
                new_a[i] = val
                thr = _threshold(new_a, cur_b)
                if thr > best_neighbor_thr:
                    best_neighbor_thr = thr
                    best_neighbor_a = new_a
                    best_neighbor_b = cur_b[:]
            if _elapsed() > _DEADLINE - 2:
                break

        # Try all single changes in b
        for i in range(N):
            for val in range(N):
                if val == cur_b[i]:
                    continue
                if _elapsed() > _DEADLINE - 2:
                    break
                new_b = cur_b[:]
                new_b[i] = val
                thr = _threshold(cur_a, new_b)
                if thr > best_neighbor_thr:
                    best_neighbor_thr = thr
                    best_neighbor_a = cur_a[:]
                    best_neighbor_b = new_b
            if _elapsed() > _DEADLINE - 2:
                break

        if best_neighbor_a is None:
            # No improvement found, stop
            break
        cur_a = best_neighbor_a
        cur_b = best_neighbor_b
        cur_thr = best_neighbor_thr

    return cur_a, cur_b, cur_thr


def _steepest_ascent_2step(a, b, current_thr, max_iters=100):
    """Steepest ascent allowing up to 2 simultaneous position changes per step.
    This explores a larger neighborhood but at higher cost per step."""
    cur_a = a[:]
    cur_b = b[:]
    cur_thr = current_thr

    for _ in range(max_iters):
        if _elapsed() > _DEADLINE - 2:
            break
        best_neighbor_thr = cur_thr
        best_neighbor_a = None
        best_neighbor_b = None

        # Try all pairs of single changes (one in a, one in b, or two in same)
        # To keep cost manageable, try: (a[i], b[j]) for all i,j
        for i in range(N):
            for val_a in range(N):
                if val_a == cur_a[i]:
                    continue
                if _elapsed() > _DEADLINE - 2:
                    break
                for j in range(N):
                    for val_b in range(N):
                        if val_b == cur_b[j]:
                            continue
                        if _elapsed() > _DEADLINE - 2:
                            break
                        new_a = cur_a[:]
                        new_a[i] = val_a
                        new_b = cur_b[:]
                        new_b[j] = val_b
                        thr = _threshold(new_a, new_b)
                        if thr > best_neighbor_thr:
                            best_neighbor_thr = thr
                            best_neighbor_a = new_a
                            best_neighbor_b = new_b
                    if _elapsed() > _DEADLINE - 2:
                        break
                if _elapsed() > _DEADLINE - 2:
                    break
            if _elapsed() > _DEADLINE - 2:
                break

        if best_neighbor_a is None:
            break
        cur_a = best_neighbor_a
        cur_b = best_neighbor_b
        cur_thr = best_neighbor_thr

    return cur_a, cur_b, cur_thr


def construct():
    """Return two length-N lists of destination states in 0..N-1.

    Must define a complete binary deterministic automaton that synchronizes.
    Score is the exact shortest reset-word length, maximized.
    """
    best_thresh = -1
    best_a = None
    best_b = None
    rng = random.Random(12345)

    # A known-good fallback (cycle + sink) guarantees synchronization.
    fallback_a = [(i + 1) % N for i in range(N)]
    fallback_b = [0] * N
    fb_thr = _threshold(fallback_a, fallback_b)
    if fb_thr > best_thresh:
        best_thresh = fb_thr
        best_a = fallback_a[:]
        best_b = fallback_b[:]

    a_cycle = [(i + 1) % N for i in range(N)]

    # ---- Phase 1: Evaluate a diverse set of structured constructions ----
    constructions = []

    # Monotone with duplicate at end
    constructions.append((a_cycle[:], list(range(N-1)) + [N-2]))
    # Identity with one defect
    b_id = list(range(N))
    b_id[0] = 1
    constructions.append((a_cycle[:], b_id[:]))
    # Shift with duplicate at start
    constructions.append((a_cycle[:], [0, 0] + list(range(1, N-1))))
    # Three zeros at start
    constructions.append((a_cycle[:], [0, 0, 0] + list(range(1, N-1))))
    # Duplicate in middle
    constructions.append((a_cycle[:], [0, 1, 1] + list(range(2, N-1))))
    # Fold pattern
    b_fold = [i for i in range(N//2)] + [N//2-1-i for i in range(N//2)]
    if len(b_fold) < N:
        b_fold.extend([0] * (N - len(b_fold)))
    constructions.append((a_cycle[:], b_fold[:]))
    # Two peaks
    constructions.append((a_cycle[:], [min(i, 11-i) for i in range(N)]))
    # Staircase with gap
    constructions.append((a_cycle[:], [0,1,2,3,4,4,5,6,7,8,9,10]))
    # Zigzag
    b_zigzag = [(i if i % 2 == 0 else i + 5) % N for i in range(N)]
    constructions.append((a_cycle[:], b_zigzag[:]))
    # Reverse half
    constructions.append((a_cycle[:], list(range(6)) + list(range(10, 4, -1))))
    # Three clusters
    constructions.append((a_cycle[:], [i // 2 for i in range(N)]))
    # Four clusters
    constructions.append((a_cycle[:], [i // 3 for i in range(N)]))
    # Reverse cycle + monotone
    a_rev = [(i - 1) % N for i in range(N)]
    constructions.append((a_rev[:], list(range(N-1)) + [N-2]))
    # Bottleneck constructions
    for sink in range(N):
        b_sink = [(i + 1) % N for i in range(N)]
        b_sink[sink] = sink
        constructions.append((a_cycle[:], b_sink[:]))

    # Evaluate all constructions
    for a_c, b_c in constructions:
        if _elapsed() > _DEADLINE - 5:
            break
        thr = _threshold(a_c, b_c)
        if thr > best_thresh:
            best_thresh = thr
            best_a = a_c[:]
            best_b = b_c[:]

    # ---- Phase 2: Random starts + steepest ascent ----
    # Use the best found so far as one starting point, plus random ones
    rng = random.Random(12345)

    # Start from best
    if best_a is not None and best_b is not None:
        sa, sb, st = _steepest_ascent(best_a, best_b, best_thresh, max_iters=30)
        if st > best_thresh:
            best_thresh = st
            best_a = sa[:]
            best_b = sb[:]

    # Multiple random restarts with steepest ascent
    for restart in range(30):
        if _elapsed() > _DEADLINE - 5:
            break

        # Generate a random synchronizing automaton
        # Strategy: a = random permutation (or cycle), b = random
        # Ensure synchronization by construction: make b have a sink-like property
        a_start = list(range(N))
        rng.shuffle(a_start)
        b_start = [rng.randint(0, N-1) for _ in range(N)]
        # Ensure b is not a permutation (to help synchronization)
        # With high probability random b will work; if not, steepest ascent handles it
        thr_start = _threshold(a_start, b_start)

        if thr_start < 0:
            # Not synchronizing, try a different construction
            # Make b have a fixed point to help
            b_start[rng.randrange(N)] = rng.randrange(N)
            thr_start = _threshold(a_start, b_start)
            if thr_start < 0:
                continue

        # Steepest ascent from this point
        sa, sb, st = _steepest_ascent(a_start, b_start, thr_start, max_iters=50)
        if st > best_thresh:
            best_thresh = st
            best_a = sa[:]
            best_b = sb[:]

    # ---- Phase 3: More targeted random starts with cycle a ----
    for restart in range(50):
        if _elapsed() > _DEADLINE - 5:
            break
        a_start = a_cycle[:]
        b_start = [rng.randint(0, N-1) for _ in range(N)]
        thr_start = _threshold(a_start, b_start)
        if thr_start < 0:
            continue
        sa, sb, st = _steepest_ascent(a_start, b_start, thr_start, max_iters=40)
        if st > best_thresh:
            best_thresh = st
            best_a = sa[:]
            best_b = sb[:]

    # ---- Phase 4: Fine-tune with 2-step steepest ascent on best ----
    if best_a is not None and best_b is not None and _elapsed() < _DEADLINE - 10:
        sa, sb, st = _steepest_ascent_2step(best_a, best_b, best_thresh, max_iters=30)
        if st > best_thresh:
            best_thresh = st
            best_a = sa[:]
            best_b = sb[:]

    # ---- Phase 5: Another round of 1-step steepest ascent from best ----
    if best_a is not None and best_b is not None and _elapsed() < _DEADLINE - 5:
        sa, sb, st = _steepest_ascent(best_a, best_b, best_thresh, max_iters=50)
        if st > best_thresh:
            best_thresh = st
            best_a = sa[:]
            best_b = sb[:]

    # ---- Phase 6: Additional random restarts if time permits ----
    if _elapsed() < _DEADLINE - 10:
        rng2 = random.Random(99999)
        for restart in range(40):
            if _elapsed() > _DEADLINE - 5:
                break
            # Mix of strategies
            strategy = rng2.random()
            if strategy < 0.4:
                a_start = a_cycle[:]
            elif strategy < 0.7:
                a_start = list(range(N))
                rng2.shuffle(a_start)
            else:
                a_start = [(i + rng2.randint(1, N-1)) % N for i in range(N)]
                # Ensure it's a valid function (it is, by construction)
            b_start = [rng2.randint(0, N-1) for _ in range(N)]
            thr_start = _threshold(a_start, b_start)
            if thr_start < 0:
                continue
            sa, sb, st = _steepest_ascent(a_start, b_start, thr_start, max_iters=30)
            if st > best_thresh:
                best_thresh = st
                best_a = sa[:]
                best_b = sb[:]

    # Fallback: ensure we have a valid synchronizing automaton
    if best_a is None or best_b is None or best_thresh <= 0:
        best_a = fallback_a
        best_b = fallback_b
        best_thresh = _threshold(best_a, best_b)

    return best_a, best_b
>>>>>>> REPLACE

