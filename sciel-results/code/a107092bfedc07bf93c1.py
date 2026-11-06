<<<<<<< SEARCH
    # Phase 1: DPLL
    result = dpll_with_unit_prop(0, time_budget=30)
    unsat_count = count_unsatisfied(result)
    if unsat_count == 0:
        return result

    best_assignment = result
    best_unsat = unsat_count

    # Phase 2: WalkSAT from DPLL result
    if time.time() < deadline - 15:
        result = walksat(0, init_assignment=best_assignment, noise=0.3, max_steps=500000)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 2b: Local DPLL refinement after WalkSAT
    if time.time() < deadline - 12 and best_unsat > 0:
        result = local_dpll_refine(best_assignment, time_budget=5.0)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 2c: Targeted repair on best assignment
    if time.time() < deadline - 10 and best_unsat > 0:
        result = targeted_repair(best_assignment, max_rounds=5000)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 3: More DPLL restarts with local refinement
    for seed in range(1, 20):
        if time.time() > deadline - 8:
            break
        result = dpll_with_unit_prop(seed, time_budget=5)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result
            # Local DPLL refinement on improved solution
            if time.time() < deadline - 6 and best_unsat > 0:
                refined = local_dpll_refine(best_assignment, time_budget=3.0)
                refined_unsat = count_unsatisfied(refined)
                if refined_unsat < best_unsat:
                    best_unsat = refined_unsat
                    best_assignment = refined

    # Phase 4: Multiple WalkSAT restarts with local DPLL refinement
    for seed in range(1, 50):
        if time.time() > deadline - 5:
            break
        if seed % 2 == 0:
            result = walksat(seed, init_assignment=best_assignment, noise=0.2, max_steps=300000)
        else:
            result = walksat(seed, init_assignment=None, noise=0.3, max_steps=300000)
        
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result
            # Local DPLL refinement on improved solution
            if time.time() < deadline - 4 and best_unsat > 0:
                refined = local_dpll_refine(best_assignment, time_budget=2.0)
                refined_unsat = count_unsatisfied(refined)
                if refined_unsat < best_unsat:
                    best_unsat = refined_unsat
                    best_assignment = refined

    # Phase 5: Final targeted repair polish
    if time.time() < deadline - 2 and best_unsat > 0:
        result = targeted_repair(best_assignment, max_rounds=10000)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 5b: Final local DPLL refinement
    if time.time() < deadline - 1.5 and best_unsat > 0:
        result = local_dpll_refine(best_assignment, time_budget=2.0)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 6: Final WalkSAT polish
    if time.time() < deadline - 1 and best_unsat > 0:
        result = walksat(999, init_assignment=best_assignment, noise=0.1, max_steps=200000)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 7: Final local DPLL polish
    if time.time() < deadline - 0.5 and best_unsat > 0:
        result = local_dpll_refine(best_assignment, time_budget=1.0)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    return best_assignment
=======
    # Phase 1: DPLL with unit propagation
    result = dpll_with_unit_prop(0, time_budget=30)
    unsat_count = count_unsatisfied(result)
    if unsat_count == 0:
        return result

    best_assignment = result
    best_unsat = unsat_count

    # Phase 2: Vectorized WalkSAT from DPLL result
    if time.time() < deadline - 15:
        result = vectorized_walksat(0, init_assignment=best_assignment, noise=0.3, max_steps=1000000)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 2b: Local DPLL refinement after WalkSAT
    if time.time() < deadline - 12 and best_unsat > 0:
        result = local_dpll_refine(best_assignment, time_budget=5.0)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 2c: Targeted repair on best assignment
    if time.time() < deadline - 10 and best_unsat > 0:
        result = targeted_repair(best_assignment, max_rounds=5000)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 3: More DPLL restarts with local refinement
    for seed in range(1, 20):
        if time.time() > deadline - 8:
            break
        result = dpll_with_unit_prop(seed, time_budget=5)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result
            # Local DPLL refinement on improved solution
            if time.time() < deadline - 6 and best_unsat > 0:
                refined = local_dpll_refine(best_assignment, time_budget=3.0)
                refined_unsat = count_unsatisfied(refined)
                if refined_unsat < best_unsat:
                    best_unsat = refined_unsat
                    best_assignment = refined

    # Phase 4: Multiple vectorized WalkSAT restarts with local DPLL refinement
    for seed in range(1, 50):
        if time.time() > deadline - 5:
            break
        if seed % 2 == 0:
            result = vectorized_walksat(seed, init_assignment=best_assignment, noise=0.2, max_steps=500000)
        else:
            result = vectorized_walksat(seed, init_assignment=None, noise=0.3, max_steps=500000)
        
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result
            # Local DPLL refinement on improved solution
            if time.time() < deadline - 4 and best_unsat > 0:
                refined = local_dpll_refine(best_assignment, time_budget=2.0)
                refined_unsat = count_unsatisfied(refined)
                if refined_unsat < best_unsat:
                    best_unsat = refined_unsat
                    best_assignment = refined

    # Phase 5: Final targeted repair polish
    if time.time() < deadline - 2 and best_unsat > 0:
        result = targeted_repair(best_assignment, max_rounds=10000)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 5b: Final local DPLL refinement
    if time.time() < deadline - 1.5 and best_unsat > 0:
        result = local_dpll_refine(best_assignment, time_budget=2.0)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 6: Final vectorized WalkSAT polish
    if time.time() < deadline - 1 and best_unsat > 0:
        result = vectorized_walksat(999, init_assignment=best_assignment, noise=0.1, max_steps=300000)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 7: Final local DPLL polish
    if time.time() < deadline - 0.5 and best_unsat > 0:
        result = local_dpll_refine(best_assignment, time_budget=1.0)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    return best_assignment
>>>>>>> REPLACE
