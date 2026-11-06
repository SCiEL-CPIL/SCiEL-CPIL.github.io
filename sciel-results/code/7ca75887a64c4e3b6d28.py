"""Rewritten SAT solver with bit-vector acceleration and hybrid search strategy."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285  # safety margin before 330s kill

    m = len(clauses)
    if m == 0:
        return [True] * n

    # Build clause data
    cnf = [tuple(c) for c in clauses]

    # Build variable-to-clause index
    var_to_clauses = [[] for _ in range(n)]
    clause_lit_pos = []
    for ci, c in enumerate(cnf):
        positions = []
        for lit in c:
            vi = abs(lit) - 1
            var_to_clauses[vi].append(ci)
            positions.append((vi, 1 if lit > 0 else -1))
        clause_lit_pos.append(positions)

    # Precompute bitmasks for each variable's positive and negative literal occurrences
    # pos_mask[vi] has bit ci set if clause ci contains literal +vi
    # neg_mask[vi] has bit ci set if clause ci contains literal -vi
    pos_mask = [0] * n
    neg_mask = [0] * n
    for ci, c in enumerate(cnf):
        bit = 1 << ci
        for lit in c:
            vi = abs(lit) - 1
            if lit > 0:
                pos_mask[vi] |= bit
            else:
                neg_mask[vi] |= bit

    def is_satisfied(assignment):
        for c in cnf:
            sat = False
            for lit in c:
                val = assignment[abs(lit) - 1]
                if val == (lit > 0):
                    sat = True
                    break
            if not sat:
                return False
        return True

    def count_unsat(assignment):
        count = 0
        for c in cnf:
            sat = False
            for lit in c:
                val = assignment[abs(lit) - 1]
                if val == (lit > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def count_unsat_fast(assignment):
        """Count unsatisfied clauses using bitmasks."""
        # A clause is satisfied if any of its literals is true.
        # For each clause, check if any literal is satisfied.
        # We can compute the set of satisfied clauses as the union of
        # pos_mask[vi] for True variables and neg_mask[vi] for False variables.
        sat_mask = 0
        for vi in range(n):
            if assignment[vi]:
                sat_mask |= pos_mask[vi]
            else:
                sat_mask |= neg_mask[vi]
        # Total clauses mask
        total_mask = (1 << m) - 1
        unsat_mask = total_mask & ~sat_mask
        return bin(unsat_mask).count('1')

    def get_unsat_clauses_fast(assignment):
        """Return list of unsatisfied clause indices using bitmasks."""
        sat_mask = 0
        for vi in range(n):
            if assignment[vi]:
                sat_mask |= pos_mask[vi]
            else:
                sat_mask |= neg_mask[vi]
        total_mask = (1 << m) - 1
        unsat_mask = total_mask & ~sat_mask
        result = []
        ci = 0
        while unsat_mask:
            if unsat_mask & 1:
                result.append(ci)
            unsat_mask >>= 1
            ci += 1
        return result

    # Unit propagation and pure literal elimination
    def preprocess(cnf_list):
        assignment = [None] * n
        cnf_work = [list(c) for c in cnf_list]
        changed = True
        while changed:
            changed = False
            new_cnf = []
            for c in cnf_work:
                sat = False
                for lit in c:
                    vi = abs(lit) - 1
                    if assignment[vi] is not None and assignment[vi] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    new_cnf.append(c)
            cnf_work = new_cnf

            for i, c in enumerate(cnf_work):
                if len(c) == 1:
                    lit = c[0]
                    vi = abs(lit) - 1
                    if assignment[vi] is None:
                        assignment[vi] = (lit > 0)
                        changed = True
                    elif assignment[vi] != (lit > 0):
                        return None, None, assignment
                    else:
                        cnf_work[i] = []
                        changed = True
                elif len(c) == 0:
                    return None, None, assignment

            lit_counts = [[0, 0] for _ in range(n)]
            for c in cnf_work:
                seen = set()
                for lit in c:
                    vi = abs(lit) - 1
                    if vi in seen:
                        continue
                    seen.add(vi)
                    if lit > 0:
                        lit_counts[vi][0] += 1
                    else:
                        lit_counts[vi][1] += 1

            for vi in range(n):
                if assignment[vi] is not None:
                    continue
                if lit_counts[vi][0] == 0 and lit_counts[vi][1] > 0:
                    assignment[vi] = False
                    changed = True
                elif lit_counts[vi][1] == 0 and lit_counts[vi][0] > 0:
                    assignment[vi] = True
                    changed = True

        if all(a is not None for a in assignment):
            final_assign = [bool(a) for a in assignment]
            if is_satisfied(final_assign):
                return final_assign, None, final_assign
            return None, None, assignment

        return None, cnf_work, assignment

    solved, reduced_cnf, partial_assign = preprocess(cnf)
    if solved is not None:
        return solved

    def make_full_assignment(partial):
        rng = random.Random()
        assign = []
        for i in range(n):
            if partial[i] is not None:
                assign.append(bool(partial[i]))
            else:
                assign.append(rng.random() < 0.5)
        return assign

    if reduced_cnf is not None:
        cnf = [tuple(c) for c in reduced_cnf if len(c) > 0]
        m = len(cnf)
        var_to_clauses = [[] for _ in range(n)]
        clause_lit_pos = []
        for ci, c in enumerate(cnf):
            positions = []
            for lit in c:
                vi = abs(lit) - 1
                var_to_clauses[vi].append(ci)
                positions.append((vi, 1 if lit > 0 else -1))
            clause_lit_pos.append(positions)
        pos_mask = [0] * n
        neg_mask = [0] * n
        for ci, c in enumerate(cnf):
            bit = 1 << ci
            for lit in c:
                vi = abs(lit) - 1
                if lit > 0:
                    pos_mask[vi] |= bit
                else:
                    neg_mask[vi] |= bit
        start_assign = make_full_assignment(partial_assign)
    else:
        start_assign = [random.Random(0).random() < 0.5 for _ in range(n)]

    if is_satisfied(start_assign):
        return start_assign

    best_assign = start_assign[:]
    best_unsat = count_unsat_fast(start_assign)

    def greedy_flip(assign, max_steps, seed):
        """Greedy local search: flip variable that maximizes net gain in satisfied clauses."""
        if time.time() > deadline:
            return False, assign, count_unsat_fast(assign)

        rng = random.Random(seed)
        cur = assign[:]
        cur_unsat = count_unsat_fast(cur)
        min_unsat = cur_unsat
        min_assign = cur[:]

        for step in range(max_steps):
            if time.time() > deadline:
                break
            if cur_unsat == 0:
                return True, cur, 0

            # Find best flip: try each variable, compute delta in unsat count
            best_delta = 0
            best_vi = -1
            candidates = []

            for vi in range(n):
                # Compute delta: how many clauses become satisfied/unsatisfied
                if cur[vi]:
                    # Flipping to False: lose pos_mask[vi] satisfaction, gain neg_mask[vi]
                    # Clauses currently satisfied only by this literal will become unsat
                    # Clauses currently unsat but satisfied by neg literal will become sat
                    lose_mask = pos_mask[vi]
                    gain_mask = neg_mask[vi]
                else:
                    lose_mask = neg_mask[vi]
                    gain_mask = pos_mask[vi]

                # We need to know which clauses are currently unsat
                # This is expensive to compute each time, so we approximate
                # by counting: delta = (clauses that gain satisfaction) - (clauses that lose satisfaction)
                # But we need to be careful about clauses that have other satisfying literals

                # Simpler approach: just count using the fast method on a trial flip
                # But that's O(n) per variable, O(n^2) per step - too slow for large n

                # Instead, use the bitmask approach more cleverly:
                # A clause becomes unsatisfied if it was only satisfied by the literal we're flipping
                # A clause becomes satisfied if it was unsatisfied and the new literal satisfies it

                # For efficiency, let's just do a limited search: pick a random subset of variables
                if step % 100 < 50:  # only do full greedy every other 100 steps
                    trial = cur[:]
                    trial[vi] = not cur[vi]
                    trial_unsat = count_unsat_fast(trial)
                    delta = trial_unsat - cur_unsat
                    if delta < best_delta:
                        best_delta = delta
                        best_vi = vi
                    elif delta == best_delta and delta < 0:
                        candidates.append(vi)

            if best_vi == -1 and candidates:
                best_vi = rng.choice(candidates)
            elif best_vi == -1:
                # No improving flip found; do a random flip to escape
                best_vi = rng.randrange(n)
                cur[best_vi] = not cur[best_vi]
                cur_unsat = count_unsat_fast(cur)
                if cur_unsat < min_unsat:
                    min_unsat = cur_unsat
                    min_assign = cur[:]
                continue

            cur[best_vi] = not cur[best_vi]
            cur_unsat = count_unsat_fast(cur)

            if cur_unsat < min_unsat:
                min_unsat = cur_unsat
                min_assign = cur[:]

        return min_unsat == 0, (cur if min_unsat == 0 else min_assign), min_unsat

    def walksat_fast(assign, max_steps, p_bias, seed):
        """WalkSAT with bitmask-based unsat clause selection."""
        if time.time() > deadline:
            return False, assign, count_unsat_fast(assign)

        rng = random.Random(seed)
        cur = assign[:]
        cur_unsat = count_unsat_fast(cur)
        min_unsat = cur_unsat
        min_assign = cur[:]

        for step in range(max_steps):
            if time.time() > deadline:
                break
            if cur_unsat == 0:
                return True, cur, 0

            # Get unsat clauses
            unsat_clauses = get_unsat_clauses_fast(cur)
            if not unsat_clauses:
                return True, cur, 0

            # Pick a random unsat clause
            ci = rng.choice(unsat_clauses)
            clause = cnf[ci]

            # Pick a literal in the clause to flip
            # Weight by degree (lower degree = less likely to hurt other clauses)
            weights = []
            for lit in clause:
                vi = abs(lit) - 1
                deg = len(var_to_clauses[vi])
                w = 1.0 / (1 + deg)
                weights.append(w)

            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen_idx = 0
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen_idx = i
                    break

            literal = clause[chosen_idx]
            vi = abs(literal) - 1

            # With probability p_bias, flip to satisfy the clause
            if rng.random() < p_bias:
                cur[vi] = (literal > 0)
            else:
                cur[vi] = not cur[vi]

            # Update unsat count
            cur_unsat = count_unsat_fast(cur)

            if cur_unsat < min_unsat:
                min_unsat = cur_unsat
                min_assign = cur[:]

        return min_unsat == 0, (cur if min_unsat == 0 else min_assign), min_unsat

    # Multi-restart strategy
    attempt = 0
    seeds = [42, 123, 7, 999, 5555, 12345, 67890, 24680, 13579, 98765, 31415, 27182, 16180, 14142, 17320]
    p_bias_values = [0.8, 0.75, 0.7, 0.85, 0.6, 0.9, 0.65, 0.78, 0.55, 0.82, 0.72, 0.88, 0.68, 0.76, 0.62]

    while time.time() < deadline - 10:
        seed = seeds[attempt % len(seeds)]
        p_bias = p_bias_values[attempt % len(p_bias_values)]
        steps = min(3000 + attempt * 800, 40000)

        # Alternate between WalkSAT and greedy
        if attempt % 3 == 0:
            success, result, min_unsat = walksat_fast(best_assign, steps, p_bias, seed)
        else:
            success, result, min_unsat = greedy_flip(best_assign, steps // 2, seed)

        if success:
            return result

        if min_unsat < best_unsat:
            best_unsat = min_unsat
            best_assign = result[:]

        attempt += 1

        if attempt % 3 == 0:
            fresh = [random.Random(attempt * 1000).random() < 0.5 for _ in range(n)]
            fu = count_unsat_fast(fresh)
            if fu < best_unsat:
                best_assign = fresh
                best_unsat = fu

    # Final polish
    if is_satisfied(best_assign):
        return best_assign

    if best_unsat <= 5:
        for _ in range(5000):
            if time.time() > deadline - 2:
                break
            unsat_clauses = get_unsat_clauses_fast(best_assign)
            if not unsat_clauses:
                return best_assign

            ci = unsat_clauses[0]
            clause = cnf[ci]
            for lit in clause:
                vi = abs(lit) - 1
                old_val = best_assign[vi]
                best_assign[vi] = not old_val
                u = count_unsat_fast(best_assign)
                if u == 0:
                    return best_assign
                if u < best_unsat:
                    best_unsat = u
                else:
                    best_assign[vi] = old_val

    rng = random.Random(0)
    for _ in range(2000):
        if time.time() > deadline - 2:
            break
        vi = rng.randrange(n)
        best_assign[vi] = not best_assign[vi]
        if is_satisfied(best_assign):
            return best_assign
        u = count_unsat_fast(best_assign)
        if u > best_unsat:
            best_assign[vi] = not best_assign[vi]

    return best_assign

# EVOLVE-BLOCK-END
