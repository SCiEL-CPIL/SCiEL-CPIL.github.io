"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285.0  # safety margin before the 330s kill

    # Pre-compute clauses as 0-based (idx, pos) tuples
    clauses_idx = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    m = len(clauses_idx)
    
    # Build variable -> list of (clause_id, position_in_clause) for O(1) mask updates
    var_to_clause_pos = [[] for _ in range(n)]
    for ci, c in enumerate(clauses_idx):
        for k, (idx, pos) in enumerate(c):
            var_to_clause_pos[idx].append((ci, k))
    
    # Precompute for each variable: which clauses it appears in and at what position
    # This lets us update masks in O(degree) without searching

    def compute_masks(assignment):
        masks = [0] * m
        for ci, c in enumerate(clauses_idx):
            mask = 0
            for k, (idx, pos) in enumerate(c):
                if assignment[idx] == pos:
                    mask |= (1 << k)
            masks[ci] = mask
        return masks

    def get_unsat(masks):
        return [ci for ci in range(m) if masks[ci] == 0]

    rng = random.Random(12345)

    best_assignment = None
    best_unsat = m + 1
    run_count = 0

    max_flips_per_run = 8000

    while time.time() < deadline:
        # For first run, random init; subsequent runs, perturb best if available
        if best_assignment is not None and run_count > 0:
            # Start from best assignment with some random perturbations
            assignment = best_assignment[:]
            # Perturb a few random variables to escape local minimum
            n_perturb = max(1, n // 10)
            for _ in range(n_perturb):
                vi = rng.randrange(n)
                assignment[vi] = not assignment[vi]
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        run_count += 1
        masks = compute_masks(assignment)
        unsat_list = get_unsat(masks)
        unsat_set = set(unsat_list)

        if not unsat_set:
            return assignment

        if len(unsat_set) < best_unsat:
            best_unsat = len(unsat_set)
            best_assignment = assignment[:]

        noise = 0.3
        last_unsat_count = len(unsat_set)
        no_improve = 0
        flip_count = 0

        while flip_count < max_flips_per_run:
            if time.time() > deadline:
                break
            if not unsat_set:
                return assignment

            flip_count += 1
            cur_unsat = len(unsat_set)
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

            # Pick an unsatisfied clause (weighted by how unsatisfied - always 0 for 3-SAT, so uniform)
            ci = rng.choice(unsat_list)
            c = clauses_idx[ci]

            # Score each literal position by how many other clauses it would help/hurt
            # Greedy: prefer flipping a literal that is currently false in this clause
            # and whose flip satisfies the most other clauses
            best_score = -1
            best_k = 0
            for k in range(3):
                idx, pos = c[k]
                # Is this literal currently satisfied?
                lit_satisfied = (assignment[idx] == pos)
                # If we flip this variable, which clauses change?
                # For each clause containing idx, check if flip helps or hurts
                score = 0
                for (cj, k2) in var_to_clause_pos[idx]:
                    if cj == ci:
                        continue
                    cc = clauses_idx[cj]
                    _, pos2 = cc[k2]
                    # Current state of this literal in clause cj
                    cur_satisfied = (assignment[idx] == pos2)
                    new_val = not assignment[idx]
                    new_satisfied = (new_val == pos2)
                    if new_satisfied and not cur_satisfied:
                        score += 1  # This flip satisfies another clause
                    elif not new_satisfied and cur_satisfied:
                        score -= 1  # This flip breaks another clause
                    # Check if clause cj was already satisfied (mask != 0)
                    # If mask == 0, it's unsat, so satisfying one more literal doesn't make it sat unless all are sat
                    # Simplified: just count net effect on literal satisfaction
                # Bonus if this literal is currently false in our chosen clause (flipping makes it true)
                if not lit_satisfied:
                    score += 2  # Strong preference for fixing the chosen clause
                
                if score > best_score:
                    best_score = score
                    best_k = k

            # Apply noise: with probability noise, pick a random literal instead
            if rng.random() < noise:
                k = rng.randrange(3)
            else:
                k = best_k

            idx, pos = c[k]
            new_val = not assignment[idx]
            assignment[idx] = new_val

            # Incrementally update masks using precomputed positions
            for (cj, k2) in var_to_clause_pos[idx]:
                bit = 1 << k2
                if new_val == clauses_idx[cj][k2][1]:
                    masks[cj] |= bit
                else:
                    masks[cj] &= ~bit
                if masks[cj] == 0:
                    unsat_set.add(cj)
                else:
                    unsat_set.discard(cj)

            # Rebuild unsat_list periodically for efficient choice
            if flip_count % 100 == 0:
                unsat_list = list(unsat_set)

            # Adaptive noise
            cur_unsat = len(unsat_set)
            if cur_unsat < last_unsat_count:
                no_improve = 0
                last_unsat_count = cur_unsat
            else:
                no_improve += 1

            if no_improve >= 200:
                noise = min(0.8, noise + 0.1)
                no_improve = 0
            elif noise > 0.3 and no_improve == 0:
                noise = max(0.3, noise - 0.05)

        # End of run

    # Final polish: focused local search from best assignment
    if best_assignment is not None and best_unsat > 0:
        remaining_time = deadline - time.time()
        if remaining_time > 3.0:
            assignment = best_assignment[:]
            masks = compute_masks(assignment)
            unsat_set = set(get_unsat(masks))
            unsat_list = list(unsat_set)
            
            if unsat_set:
                noise = 0.1
                last_unsat_count = len(unsat_set)
                no_improve = 0
                polish_flips = 0
                max_polish = 5000
                
                while polish_flips < max_polish:
                    if time.time() > deadline:
                        break
                    if not unsat_set:
                        return assignment
                    
                    polish_flips += 1
                    cur_unsat = len(unsat_set)
                    if cur_unsat < best_unsat:
                        best_unsat = cur_unsat
                        best_assignment = assignment[:]

                    ci = rng.choice(unsat_list)
                    c = clauses_idx[ci]

                    # Same scoring heuristic but lower noise
                    best_score = -1
                    best_k = 0
                    for k in range(3):
                        idx, pos = c[k]
                        lit_satisfied = (assignment[idx] == pos)
                        score = 0
                        for (cj, k2) in var_to_clause_pos[idx]:
                            if cj == ci:
                                continue
                            cc = clauses_idx[cj]
                            _, pos2 = cc[k2]
                            cur_satisfied = (assignment[idx] == pos2)
                            new_val = not assignment[idx]
                            new_satisfied = (new_val == pos2)
                            if new_satisfied and not cur_satisfied:
                                score += 1
                            elif not new_satisfied and cur_satisfied:
                                score -= 1
                        if not lit_satisfied:
                            score += 2
                        if score > best_score:
                            best_score = score
                            best_k = k

                    if rng.random() < noise:
                        k = rng.randrange(3)
                    else:
                        k = best_k

                    idx, pos = c[k]
                    new_val = not assignment[idx]
                    assignment[idx] = new_val

                    for (cj, k2) in var_to_clause_pos[idx]:
                        bit = 1 << k2
                        if new_val == clauses_idx[cj][k2][1]:
                            masks[cj] |= bit
                        else:
                            masks[cj] &= ~bit
                        if masks[cj] == 0:
                            unsat_set.add(cj)
                        else:
                            unsat_set.discard(cj)

                    if polish_flips % 100 == 0:
                        unsat_list = list(unsat_set)

                    cur_unsat = len(unsat_set)
                    if cur_unsat < last_unsat_count:
                        no_improve = 0
                        last_unsat_count = cur_unsat
                    else:
                        no_improve += 1

                    if no_improve >= 150:
                        noise = min(0.5, noise + 0.1)
                        no_improve = 0

    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
