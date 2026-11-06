"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    deadline = time.time() + 315.0

    # Preprocess clauses into variable indices and sign arrays.
    # For each clause: vars are 0-based variable indices, signs are True if positive.
    cvars = []
    csigns = []
    for cl in clauses:
        vs = [abs(l) - 1 for l in cl]
        ss = [l > 0 for l in cl]
        cvars.append(vs)
        csigns.append(ss)

    # Build variable -> clause index lists.
    var_clauses = [[] for _ in range(n)]
    for ci, vs in enumerate(cvars):
        for v in vs:
            var_clauses[v].append(ci)

    rng = random.Random(0)

    def check_all():
        for ci in range(m):
            vs = cvars[ci]
            ss = csigns[ci]
            for k in range(len(vs)):
                if assignment[vs[k]] == ss[k]:
                    break
            else:
                return False
        return True

    def unsat_clauses():
        out = []
        for ci in range(m):
            vs = cvars[ci]
            ss = csigns[ci]
            ok = False
            for k in range(len(vs)):
                if assignment[vs[k]] == ss[k]:
                    ok = True
                    break
            if not ok:
                out.append(ci)
        return out

    best_assignment = None
    best_unsat = m + 1
    max_restarts = 200
    max_steps_per_restart = 20000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Compute initial unsat counts for each clause: 0 means satisfied.
        unsat_count = [0] * m
        for ci in range(m):
            vs = cvars[ci]
            ss = csigns[ci]
            cnt = 0
            for k in range(len(vs)):
                if assignment[vs[k]] == ss[k]:
                    cnt += 1
            if cnt == 0:
                unsat_count[ci] = 1
            else:
                unsat_count[ci] = 0

        total_unsat = sum(unsat_count)
        if total_unsat == 0:
            return assignment

        if total_unsat < best_unsat:
            best_unsat = total_unsat
            best_assignment = assignment[:]

        no_improve_steps = 0
        improve_threshold = 500

        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break

            # Pick a random unsatisfied clause.
            # To avoid building the full list every step, we can sample.
            # For simplicity and correctness, build list if not too big; otherwise sample.
            # But for 150 vars and planted 3-SAT, m is likely moderate.
            # We'll build the list for now; if performance is an issue, optimize later.
            unsat_list = [ci for ci in range(m) if unsat_count[ci] > 0]
            if not unsat_list:
                return assignment

            if len(unsat_list) < best_unsat:
                best_unsat = len(unsat_list)
                best_assignment = assignment[:]
                no_improve_steps = 0
            else:
                no_improve_steps += 1

            if no_improve_steps > improve_threshold:
                # Restart this attempt
                break

            ci = rng.choice(unsat_list)
            vs = cvars[ci]
            ss = csigns[ci]

            # Compute flip scores: for each variable in clause, count how many other
            # unsat clauses would be satisfied by flipping it.
            best_var = -1
            best_score = -1
            best_rand = -1.0

            for v in vs:
                score = 0
                for cj in var_clauses[v]:
                    if unsat_count[cj] == 0:
                        continue
                    # Check if flipping v would satisfy clause cj
                    # Current value: assignment[v]
                    # Clause cj is unsat, so no literal is true.
                    # If we flip v, it becomes not assignment[v].
                    # The literal in cj for v: sign = csigns[cj][k] where cvars[cj][k]==v
                    # But we don't have direct mapping. We need to check.
                    # For 3-SAT, we can just check all literals in cj.
                    # This is O(3) per clause.
                    vs_cj = cvars[cj]
                    ss_cj = csigns[cj]
                    for k in range(len(vs_cj)):
                        if vs_cj[k] == v:
                            # After flip, assignment[v] becomes not assignment[v]
                            # The literal is satisfied if not assignment[v] == ss_cj[k]
                            # i.e., assignment[v] != ss_cj[k]
                            if assignment[v] != ss_cj[k]:
                                score += 1
                            break
                rand_val = rng.random()
                if score > best_score or (score == best_score and rand_val > best_rand):
                    best_score = score
                    best_rand = rand_val
                    best_var = v

            if best_var == -1:
                # Should not happen
                break

            # Flip best_var and update unsat counts for all clauses containing best_var
            old_val = assignment[best_var]
            new_val = not old_val
            assignment[best_var] = new_val

            for cj in var_clauses[best_var]:
                # Determine if this clause was satisfied before and after flip
                # We know it's unsat if unsat_count[cj] > 0
                # But we need to know if the literal in cj for best_var is now true
                vs_cj = cvars[cj]
                ss_cj = csigns[cj]
                # Find index of best_var in cj
                idx = -1
                for k in range(len(vs_cj)):
                    if vs_cj[k] == best_var:
                        idx = k
                        break
                if idx == -1:
                    continue

                # After flip, the literal is true if new_val == ss_cj[idx]
                literal_now_true = (new_val == ss_cj[idx])
                literal_was_true = (old_val == ss_cj[idx])

                if unsat_count[cj] > 0:
                    # Clause was unsat
                    if literal_now_true:
                        # Now satisfied
                        unsat_count[cj] = 0
                        total_unsat -= 1
                    else:
                        # Still unsat, but other literals might have changed? No, only this var flipped.
                        # Since it was unsat, no literal was true. After flip, this literal is not true,
                        # so still unsat. Count unchanged.
                        pass
                else:
                    # Clause was sat
                    if literal_was_true and not literal_now_true:
                        # We just made the only true literal false? Not necessarily, but since it was sat,
                        # at least one literal was true. If this was the only true literal, now it's unsat.
                        # We need to check if any other literal is true.
                        any_true = False
                        for k in range(len(vs_cj)):
                            if vs_cj[k] == best_var:
                                continue
                            if assignment[vs_cj[k]] == ss_cj[k]:
                                any_true = True
                                break
                        if not any_true:
                            unsat_count[cj] = 1
                            total_unsat += 1
                    # If this literal was not the only true one, or was not true, still sat.

            if total_unsat == 0:
                return assignment

            if total_unsat < best_unsat:
                best_unsat = total_unsat
                best_assignment = assignment[:]
                no_improve_steps = 0

    # Return best found
    if best_assignment is not None:
        # Verify best_assignment
        if check_all():
            return best_assignment
    # Fallback: return last assignment or random
    if 'assignment' in locals():
        return assignment
    return [False] * n
# EVOLVE-BLOCK-END
