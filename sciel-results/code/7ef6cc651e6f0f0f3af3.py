"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285.0  # safety margin before the 330s kill

    # Pre-compute clauses as 0-based (idx, pos) tuples and build a variable->clauses index
    clauses_idx = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    m = len(clauses_idx)
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses_idx):
        for idx, _pos in c:
            var_to_clauses[idx].append(ci)

    # Precompute literal positions for faster mask updates
    # For each clause, store which literal positions (0,1,2) correspond to which variables
    # and their polarity, so we can quickly determine satisfaction

    def sat_mask(assignment, ci):
        c = clauses_idx[ci]
        mask = 0
        for k, (idx, pos) in enumerate(c):
            if assignment[idx] == pos:
                mask |= (1 << k)
        return mask

    rng = random.Random(12345)

    best_assignment = None
    best_unsat = m + 1

    # Parameters for adaptive walk
    base_noise = 0.2
    max_noise = 0.8
    noise_increase = 0.05
    noise_check_interval = 100
    max_flips_per_run = 3000
    plateau_threshold = 50  # restart if no improvement in this many flips

    while time.time() < deadline:
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        masks = [sat_mask(assignment, ci) for ci in range(m)]
        unsat = set(ci for ci in range(m) if masks[ci] == 0)

        if not unsat:
            return assignment

        if len(unsat) < best_unsat:
            best_unsat = len(unsat)
            best_assignment = assignment[:]

        noise = base_noise
        stuck = 0
        run_best_unsat = len(unsat)

        for _flip in range(max_flips_per_run):
            if time.time() > deadline:
                break
            if not unsat:
                return assignment

            cur_unsat = len(unsat)
            if cur_unsat < run_best_unsat:
                run_best_unsat = cur_unsat
                stuck = 0  # reset stuck counter on improvement
            else:
                stuck += 1

            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

            # Restart if we've been stuck too long
            if stuck > plateau_threshold:
                break

            # Pick an unsatisfied clause using weighted random (prefer clauses with more unsat literals)
            # For efficiency, just pick uniformly but avoid list() conversion overhead
            ci = rng.choice(list(unsat))
            c = clauses_idx[ci]

            # Decide which literal to flip:
            # Score each literal position by how many other unsatisfied clauses it appears in
            # Prefer flipping literals that appear in fewer other unsatisfied clauses
            unsat_positions = [k for k, (idx, pos) in enumerate(c) if not (assignment[idx] == pos)]
            
            if unsat_positions and rng.random() > noise:
                # Greedy: pick the literal that minimizes impact on other unsatisfied clauses
                best_k = None
                best_score = float('inf')
                for k in unsat_positions:
                    idx, _ = c[k]
                    # Count how many other unsatisfied clauses contain this variable
                    score = sum(1 for cj in var_to_clauses[idx] if cj != ci and cj in unsat)
                    if score < best_score:
                        best_score = score
                        best_k = k
                k = best_k
            else:
                # Random walk: flip any literal in the clause
                k = rng.randrange(3)
            
            idx, pos = c[k]
            assignment[idx] = not assignment[idx]

            # Incrementally update masks and unsat for clauses touching idx
            for cj in var_to_clauses[idx]:
                masks[cj] = sat_mask(assignment, cj)
                if masks[cj] == 0:
                    unsat.add(cj)
                else:
                    unsat.discard(cj)

            # Adaptive noise: increase noise if stuck
            if stuck % noise_check_interval == 0:
                noise = min(max_noise, noise + noise_increase)

        # End of this run; continue with a fresh restart if time remains
        if time.time() > deadline:
            break

    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
