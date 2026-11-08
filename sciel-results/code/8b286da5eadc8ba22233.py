def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_overall_score = 0.0
    best_overall_U = [0, 1]
    
    # 1. Exhaustive search for small optimal patterns
    # This is likely to find the global optimum for small M
    M_small = 20
    score_small, U_small = exhaustive_small_search(M_small)
    if score_small > best_overall_score:
        best_overall_score = score_small
        best_overall_U = U_small
        
    # 2. Geometric search (refined)
    # Use smaller M_max because large M reduces score for same ratio
    M_max_geo = 30
    bases = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ratios = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.5, 3.0]
    
    for b in bases:
        for r in ratios:
            if time.time() > deadline - 10:
                break
            U_gen = generate_geometric_set(b, r, M_max_geo)
            if 0 not in U_gen:
                U_gen.append(0)
            U_gen.sort()
            
            # Quick score check before refinement
            s = compute_score_fast(U_gen)
            if s > best_overall_score:
                best_overall_score = s
                best_overall_U = U_gen
            
            # Refine
            U_refined, score_refined = local_refinement(U_gen, M_max_geo, deadline - 5)
            if score_refined > best_overall_score:
                best_overall_score = score_refined
                best_overall_U = U_refined
                
    # 3. Random dense sets (small M)
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        M_rand = np.random.randint(5, 25)
        size = np.random.randint(3, M_rand // 2 + 1)
        U_rand = [0]
        others = np.random.choice(range(1, M_rand + 1), size=size, replace=False)
        U_rand.extend(others)
        U_rand.sort()
        
        U_refined, score_refined = local_refinement(U_rand, M_rand, deadline - 5)
        
        if score_refined > best_overall_score:
            best_overall_score = score_refined
            best_overall_U = U_refined
            
    if 0 not in best_overall_U:
        best_overall_U.append(0)
        
    return sorted(best_overall_U)
