<<<<<<< SEARCH
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
=======
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
    
    # Bit-vector representation:
    # For each variable i and polarity p (0=False, 1=True):
    #   sat_mask[i][p] = bitmask of clauses where literal (i, p) appears
    # A clause is satisfied if any of its 3 literals' sat_masks have a bit set at that clause position
    # Actually, let's think differently:
    #   For each variable i, precompute:
    #     pos_mask[i] = bitmask of clause positions where positive literal i appears
    #     neg_mask[i] = bitmask of clause positions where negative literal i appears
    #   Then for assignment, satisfied_clauses_mask = OR over all i of (pos_mask[i] if assignment[i] else neg_mask[i])
    #   A clause j is satisfied iff bit j is set in satisfied_clauses_mask

    # Build pos_mask and neg_mask
    pos_mask = [0] * n
    neg_mask = [0] * n
    for ci, c in enumerate(clauses_idx):
        for k, (idx, is_pos) in enumerate(c):
            bit = 1 << ci
            if is_pos:
                pos_mask[idx] |= bit
            else:
                neg_mask[idx] |= bit

    def compute_sat_mask(assignment):
        """Return bitmask of satisfied clauses."""
        sat = 0
        for i in range(n):
            if assignment[i]:
                sat |= pos_mask[i]
            else:
                sat |= neg_mask[i]
        return sat

    def get_unsat_list(sat_mask):
        """Return list of unsatisfied clause indices."""
        unsat = []
        for ci in range(m):
            if not (sat_mask >> ci) & 1:
                unsat.append(ci)
        return unsat

    rng = random.Random(12345)

    best_assignment = None
    best_unsat = m + 1
    run_count = 0

    max_flips_per_run = 20000  # Increased since bit-vector makes each flip cheaper

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
        sat_mask = compute_sat_mask(assignment)
        unsat_list = get_unsat_list(sat_mask)
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

            # Pick an unsatisfied clause
            ci = rng.choice(unsat_list)
            c = clauses_idx[ci]

            # Score each literal position by how many other clauses it would help/hurt
            best_score = -1
            best_k = 0
            for k in range(3):
                idx, pos = c[k]
                lit_satisfied = (assignment[idx] == pos)
                
                # Compute net effect of flipping variable idx
                # If idx is currently True, flipping makes it False
                # Clauses that had idx as positive literal lose that satisfaction
                # Clauses that had idx as negative literal gain that satisfaction
                if assignment[idx]:
                    # Flipping True -> False
                    # Lose: clauses where pos_mask[idx] has bit set AND clause is currently satisfied only by this literal? 
                    # Simplified: count net change in satisfied clauses
                    # Clauses that become unsatisfied: those where idx was the ONLY satisfying literal
                    # Clauses that become satisfied: those where idx was negative and all other literals were already satisfied
                    # This is complex. Use the same heuristic as before but faster.
                    
                    # Net effect: for each clause containing idx:
                    # If clause had idx as positive (satisfied by idx=True):
                    #   After flip, this literal is no longer satisfied
                    #   Clause becomes unsat if no other literal is satisfied
                    # If clause had idx as negative (not satisfied by idx=True):
                    #   After flip, this literal becomes satisfied
                    #   Clause becomes sat if all other literals are also satisfied
                    score = 0
                    for (cj, k2) in var_to_clause_pos[idx]:
                        cc = clauses_idx[cj]
                        _, pos2 = cc[k2]
                        cur_satisfied = (assignment[idx] == pos2)
                        new_val = not assignment[idx]
                        new_satisfied = (new_val == pos2)
                        if new_satisfied and not cur_satisfied:
                            score += 1
                        elif not new_satisfied and cur_satisfied:
                            score -= 1
                else:
                    # Flipping False -> True
                    score = 0
                    for (cj, k2) in var_to_clause_pos[idx]:
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

            # Apply noise
            if rng.random() < noise:
                k = rng.randrange(3)
            else:
                k = best_k

            idx, pos = c[k]
            new_val = not assignment[idx]
            assignment[idx] = new_val

            # Update sat_mask using bit-vector representation
            # If new_val is True, we add pos_mask[idx] and remove neg_mask[idx] contribution
            # If new_val is False, we add neg_mask[idx] and remove pos_mask[idx] contribution
            # But we can't simply OR/AND because other variables may also satisfy those clauses
            # We need to recompute or track per-clause satisfaction counts
            
            # Alternative: maintain a count array for how many literals satisfy each clause
            # Let's use that approach instead for correctness
            
            # Actually, let's stick with the incremental mask approach but make it faster
            # Rebuild sat_mask from scratch every flip? No, too slow.
            # Use the old mask-based approach but with precomputed positions
            
            # Incrementally update using var_to_clause_pos
            # We'll maintain a separate 'masks' array for per-clause literal satisfaction masks
            # This is what the original code did. Let's keep that but optimize the scoring.
            
            # For now, let's just use the original incremental approach
            # The speedup will come from more flips per second due to simpler logic
            
            # Actually, let's properly implement the count-based approach:
            # sat_count[ci] = number of literals in clause ci that are satisfied
            # A clause is unsat iff sat_count[ci] == 0
            
            # Let me restructure to use sat_count
            pass  # Placeholder - will implement below

        # End of run

    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I'm overcomplicating this. Let me implement a cleaner version with proper sat_count tracking:

