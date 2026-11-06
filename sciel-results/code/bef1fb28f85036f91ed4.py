<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    random.seed(42)

    best_a = None
    best_b = None
    best_score = 0

    # Phase 1: Evaluate structured candidates
    structures = _generate_ping_pong_structures()
    random.shuffle(structures)

    for a_struct, b_struct in structures:
        if time.time() > deadline - 5.0:
            break

        score = compute_threshold(a_struct, b_struct)
        if score > best_score:
            best_score = score
            best_a = a_struct[:]
            best_b = b_struct[:]
            
            # If we hit the theoretical max (121) or a very high score, we can be more aggressive
            # but for now we continue to ensure we don't miss a slightly better one if 112 isn't max.
            # However, if best_score is very high, we can reduce time spent on lower candidates.

    # Phase 2: Local search refinement from best structured candidate
    if best_a is not None and time.time() < deadline - 3.0:
        refined_a, refined_b, refined_score = _refine_automaton(
            best_a, best_b, start_time, deadline - 3.0
        )
        if refined_score > best_score:
            best_score = refined_score
            best_a = refined_a
            best_b = refined_b

    # Phase 3: Try random starts with refinement
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        # If we already have a high score, use less time on random starts
        num_random_starts = min(5, int(remaining_time / 5))
        for _ in range(num_random_starts):
            if time.time() > deadline - 3.0:
                break

            a_rand = [random.randint(0, N - 1) for _ in range(N)]
            b_rand = [random.randint(0, N - 1) for _ in range(N)]

            score = compute_threshold(a_rand, b_rand)
            if score == 0:
                continue

            if score > best_score:
                best_score = score
                best_a = a_rand[:]
                best_b = b_rand[:]

            if score >= best_score - 5 and time.time() < deadline - 3.0:
                refined_a, refined_b, refined_score = _refine_automaton(
                    a_rand, b_rand, start_time, deadline - 3.0
                )
                if refined_score > best_score:
                    best_score = refined_score
                    best_a = refined_a
                    best_b = refined_b

    # Fallback: ensure we return a valid synchronizing automaton
    if best_score == 0 or best_a is None:
        a = [(i + 1) % N for i in range(N)]
        b = [(i - 1) % N for i in range(N)]
        return [a, b]

    return [best_a, best_b]
=======
def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    random.seed(42)

    best_a = None
    best_b = None
    best_score = 0

    # Phase 1: Evaluate structured candidates
    structures = _generate_ping_pong_structures()
    random.shuffle(structures)

    # Evaluate a subset of structures to find a good starting point for SA
    # We don't need to evaluate all if we have time for SA
    num_structs_to_eval = 50
    for i, (a_struct, b_struct) in enumerate(structures):
        if i >= num_structs_to_eval or time.time() > deadline - 10.0:
            break

        score = compute_threshold(a_struct, b_struct)
        if score > best_score:
            best_score = score
            best_a = a_struct[:]
            best_b = b_struct[:]
            
    # If we found a good structure, use it as seed for SA
    # If not, pick a random one or the first ping-pong
    if best_a is None:
        best_a = structures[0][0][:]
        best_b = structures[0][1][:]
        best_score = compute_threshold(best_a, best_b)
        if best_score == 0:
             # Fallback to a known synchronizing one if the first one fails
             a_known = [(i + 1) % N for i in range(N)]
             b_known = [0] + [(i - 1) for i in range(1, N)]
             best_a = a_known
             best_b = b_known
             best_score = compute_threshold(best_a, best_b)
             if best_score == 0:
                 # This should never happen for the standard one
                 return [a_known, b_known]

    # Phase 2: Simulated Annealing from the best structured candidate
    # We can run SA multiple times from different seeds if time permits
    # But let's start with the best one found
    
    if time.time() < deadline - 2.0:
        # Run SA for a specific duration
        sa_deadline = deadline - 2.0
        refined_a, refined_b, refined_score = _simulated_annealing(
            best_a, best_b, start_time, sa_deadline
        )
        if refined_score > best_score:
            best_score = refined_score
            best_a = refined_a
            best_b = refined_b

    # Phase 3: If time remains, try another SA run from a different seed
    # Pick a random structure from the list
    if time.time() < deadline - 2.0 and len(structures) > 1:
        seed_idx = random.randint(1, min(len(structures)-1, 10))
        a_seed = structures[seed_idx][0][:]
        b_seed = structures[seed_idx][1][:]
        seed_score = compute_threshold(a_seed, b_seed)
        
        if seed_score > 0:
            # Run SA from this seed
            # Limit time for this second run
            sa2_deadline = deadline - 2.0
            # To avoid spending too much time, we can just run it until deadline
            # But we should prioritize the first run.
            # Let's just run it if we have > 5s left
            if time.time() < deadline - 5.0:
                refined_a2, refined_b2, refined_score2 = _simulated_annealing(
                    a_seed, b_seed, start_time, sa2_deadline
                )
                if refined_score2 > best_score:
                    best_score = refined_score2
                    best_a = refined_a2
                    best_b = refined_b2

    # Fallback: ensure we return a valid synchronizing automaton
    if best_score == 0 or best_a is None:
        a = [(i + 1) % N for i in range(N)]
        b = [0] + [(i - 1) for i in range(1, N)]
        return [a, b]

    return [best_a, best_b]
>>>>>>> REPLACE
