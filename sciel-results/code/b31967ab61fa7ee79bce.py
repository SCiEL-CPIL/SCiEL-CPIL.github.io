import time
import numpy as np
from collections import deque


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [True] * n

    start_time = time.time()
    deadline = start_time + 320.0   # safety margin before 330s kill

    # Convert to 0-based
    lits_0 = [[l - 1 for l in c] for c in clauses]
    m = len(lits_0)

    # ---------- CDCL SAT Solver ----------
    class CDCL:
        def __init__(self, n_vars, clauses):
            self.n = n_vars
            self.clauses = clauses  # list of lists of 0-based signed literals
            self.m = len(clauses)
            self.assign = np.full(n_vars, -1, dtype=np.int8)  # -1 unassigned, 0 False, 1 True
            self.level = np.zeros(n_vars, dtype=np.int32)
            self.reason = np.full(n_vars, -1, dtype=np.int32)  # clause index or -1
            self.trail = []  # list of (var, level)
            self.trail_pos = {}  # var -> position in trail
            self.lcl = 0  # literal count level
            self.learned = []  # learned clauses
            self.watches = [[] for _ in range(2 * n_vars)]  # watches[lit] = list of clause indices
            self.decision_level = 0
            self.var_order = list(range(n_vars))
            self.rng = np.random.default_rng(42)
            self.conflicts = 0
            self.solved = False
            self.solution = None

            # Initialize watches
            for ci, clause in enumerate(self.clauses):
                for lit in clause:
                    self.watches[lit + self.n].append(ci)
                # Also watch the negation
                for lit in clause:
                    self.watches[-lit].append(ci)

        def lit_val(self, lit):
            v = abs(lit)
            val = self.assign[v]
            if val == -1:
                return -1  # unassigned
            if lit > 0:
                return val
            else:
                return 1 - val

        def is_true(self, lit):
            return self.lit_val(lit) == 1

        def is_false(self, lit):
            return self.lit_val(lit) == 0

        def enqueue(self, lit, level):
            v = abs(lit)
            if self.assign[v] != -1:
                return
            val = 1 if lit > 0 else 0
            self.assign[v] = val
            self.level[v] = level
            self.trail.append((v, level))
            self.trail_pos[v] = len(self.trail) - 1
            self.lcl += 1

        def propagate(self):
            """Watched literal unit propagation. Returns False on conflict, True otherwise."""
            while self.lcl < len(self.trail):
                lit = self.trail[self.lcl][0]
                # Check if this literal's complement is in watches
                comp = -lit
                watches = self.watches[comp + self.n]
                i = 0
                while i < len(watches):
                    ci = watches[i]
                    clause = self.clauses[ci] if ci < self.m else self.learned[ci - self.m]

                    # Remove this watch if it's satisfied
                    if self.is_true(lit):
                        # Find another literal to watch
                        found = False
                        for j, c_lit in enumerate(clause):
                            if c_lit == comp:
                                continue
                            if not self.is_false(c_lit):
                                # Swap watch
                                watches[i] = watches[-1]
                                watches.pop()
                                self.watches[c_lit + self.n].append(ci)
                                found = True
                                break
                        if not found:
                            # Conflict
                            return False
                        continue

                    # Check if clause has another non-false literal
                    has_unassigned = False
                    for c_lit in clause:
                        if c_lit == comp:
                            continue
                        if self.is_false(c_lit):
                            continue
                        if self.lit_val(c_lit) == 1:
                            # Clause satisfied, remove this watch
                            watches[i] = watches[-1]
                            watches.pop()
                            break
                        else:
                            has_unassigned = True
                    else:
                        if has_unassigned:
                            # Unit clause, propagate
                            for c_lit in clause:
                                if c_lit != comp and self.lit_val(c_lit) == -1:
                                    self.enqueue(c_lit, self.decision_level)
                                    self.reason[abs(c_lit)] = ci
                                    break
                            continue
                        else:
                            # All literals false, conflict
                            return False

                    i += 1

            return True

        def analyze(self, conflict_clause):
            """Analyze conflict, return learned clause and backtrack level."""
            # Simplify: use 1-UIP
            learned = []
            seen = set()
            bt_level = 0

            # Get literals from conflict clause
            clause = self.clauses[conflict_clause] if conflict_clause < self.m else self.learned[conflict_clause - self.m]

            # Build learned clause via 1-UIP
            learned_lits = []
            p = len(self.trail) - 1
            counter = 0

            while True:
                lit = self.trail[p][0]
                v = abs(lit)
                if v not in seen:
                    seen.add(v)
                    learned_lits.append(lit)
                    counter += 1
                    if self.level[v] == self.decision_level:
                        counter -= 1
                        if counter == 0:
                            break

                # Get reason clause
                reason_idx = self.reason[v]
                if reason_idx == -1:
                    break
                reason = self.clauses[reason_idx] if reason_idx < self.m else self.learned[reason_idx - self.m]
                for r_lit in reason:
                    if r_lit == lit:
                        continue
                    rv = abs(r_lit)
                    if rv not in seen and self.level[rv] >= 0:
                        seen.add(rv)
                        learned_lits.append(r_lit)
                        if self.level[rv] == self.decision_level:
                            counter += 1
                p -= 1

            # Find backtrack level
            if len(learned_lits) > 1:
                levels = [self.level[abs(l)] for l in learned_lits[1:]]
                bt_level = max(levels) if levels else 0
            else:
                bt_level = 0

            # Negate learned literals
            learned_clause = [-l for l in learned_lits]

            # Add to learned clauses
            self.learned.append(learned_clause)
            for lit in learned_clause:
                self.watches[lit + self.n].append(self.m + len(self.learned) - 1)
                self.watches[-lit].append(self.m + len(self.learned) - 1)

            return learned_clause, bt_level

        def backtrack(self, level):
            """Backtrack to given decision level."""
            while len(self.trail) > 0 and self.trail[-1][1] > level:
                v, _ = self.trail.pop()
                self.trail_pos.pop(v, None)
                self.assign[v] = -1
                self.reason[v] = -1
                self.lcl = min(self.lcl, len(self.trail))

        def pick_variable(self):
            """Pick an unassigned variable for decision."""
            unassigned = [v for v in self.var_order if self.assign[v] == -1]
            if not unassigned:
                return None
            # Simple: pick first unassigned
            return unassigned[0]

        def search(self, time_limit):
            """Main CDCL search loop."""
            max_conflicts = 100000
            conflict_count = 0

            while True:
                if time.time() > time_limit:
                    break

                # Propagate
                if not self.propagate():
                    # Conflict
                    conflict_count += 1
                    if conflict_count > max_conflicts:
                        break

                    # Find conflict clause
                    # Simplified: find a clause that is falsified
                    conflict_ci = -1
                    for ci in range(self.m):
                        clause = self.clauses[ci]
                        all_false = True
                        any_false = False
                        for lit in clause:
                            lv = self.lit_val(lit)
                            if lv == 1:
                                all_false = False
                                break
                            if lv == 0:
                                any_false = True
                        if all_false and any_false:
                            conflict_ci = ci
                            break

                    if conflict_ci == -1:
                        # Check learned clauses
                        for li in range(len(self.learned)):
                            clause = self.learned[li]
                            all_false = True
                            any_false = False
                            for lit in clause:
                                lv = self.lit_val(lit)
                                if lv == 1:
                                    all_false = False
                                    break
                                if lv == 0:
                                    any_false = True
                            if all_false and any_false:
                                conflict_ci = self.m + li
                                break

                    if conflict_ci == -1:
                        break

                    # Analyze
                    learned, bt_level = self.analyze(conflict_ci)

                    # Backtrack
                    self.backtrack(bt_level)

                    # Enqueue learned clause's literal (if unit)
                    if len(learned) == 1:
                        self.enqueue(learned[0], self.decision_level)
                    else:
                        # Add as watch and continue
                        pass

                else:
                    # No conflict, check if solved
                    all_assigned = all(self.assign[v] != -1 for v in range(self.n))
                    if all_assigned:
                        # Check if all clauses satisfied
                        for ci in range(self.m):
                            clause = self.clauses[ci]
                            if not any(self.is_true(lit) for lit in clause):
                                break
                        else:
                            self.solved = True
                            self.solution = self.assign.copy()
                            return True
                        # Also check learned
                        for li in range(len(self.learned)):
                            clause = self.learned[li]
                            if not any(self.is_true(lit) for lit in clause):
                                break
                        else:
                            self.solved = True
                            self.solution = self.assign.copy()
                            return True

                    # Pick next variable
                    v = self.pick_variable()
                    if v is None:
                        # All assigned but not satisfied? Shouldn't happen
                        break

                    self.decision_level += 1
                    # Random polarity
                    val = self.rng.integers(2)
                    lit = v if val == 1 else -v - 1  # -1-based for negative
                    # Wait, lit should be signed 0-based: positive = v, negative = -v-1? No.
                    # In our convention: lit > 0 means variable abs(lit) is True
                    # lit < 0 means variable abs(lit) is False
                    # So for variable v: True = v+1? No, we use 0-based vars.
                    # Let me fix: lit for var v, value True = v (if v>0) but v=0 is special.
                    # Better: use lit = v+1 for True, lit = -(v+1) for False? 
                    # Actually let's keep it simple: lit = v for True (v>=0), lit = -v-1 for False
                    # But then abs(lit) for -v-1 is v+1 which is wrong.
                    # Let me redefine: use 1-based internally for CDCL.
                    # Actually, let's just use: positive lit = var index (0-based), negative = -var-1
                    # Then abs(negative) = var+1, which is off by one.
                    # Simpler: use lit = var+1 for positive, lit = -(var+1) for negative.
                    # Then abs(lit) - 1 = var index.

                    # Let me restart with cleaner convention.
                    break

            return self.solved

    # Given the complexity, let's use a simpler but effective approach:
    # Use a proper CDCL with 1-based literal convention

    class CDCL2:
        def __init__(self, n_vars, clauses):
            self.n = n_vars
            self.clauses = clauses  # 1-based signed literals: lit = var+1 or -(var+1)
            self.m = len(clauses)
            self.assign = np.full(n_vars, -1, dtype=np.int8)
            self.level = np.zeros(n_vars, dtype=np.int32)
            self.reason = np.full(n_vars, -1, dtype=np.int32)
            self.trail = []
            self.trail_pos = {}
            self.lcl = 0
            self.learned = []
            self.watches = [[] for _ in range(2 * n_vars + 2)]  # index by lit directly
            self.decision_level = 0
            self.rng = np.random.default_rng(42)
            self.solved = False
            self.solution = None

            # Convert clauses to 1-based
            self.clauses_1based = [[l + 1 if l >= 0 else l for l in c] for c in clauses]
            # Actually, our input is 1-based already from the task description
            # "clauses as signed 1-based literals"
            # So lits_0 was wrong. Let me re-read.
            # The task says: "instance has n and clauses as signed 1-based literals"
            # So clauses are already 1-based. My lits_0 conversion was wrong.

        def lit_to_var(self, lit):
            return abs(lit) - 1

        def lit_val(self, lit):
            v = abs(lit) - 1
            val = self.assign[v]
            if val == -1:
                return -1
            if lit > 0:
                return val
            else:
                return 1 - val

        def is_true(self, lit):
            return self.lit_val(lit) == 1

        def is_false(self, lit):
            return self.lit_val(lit) == 0

        def enqueue(self, lit, level):
            v = abs(lit) - 1
            if self.assign[v] != -1:
                return
            val = 1 if lit > 0 else 0
            self.assign[v] = val
            self.level[v] = level
            self.trail.append((v, level))
            self.trail_pos[v] = len(self.trail) - 1
            self.lcl += 1

        def watch_index(self, lit):
            return lit  # use lit as index, but we need to handle negative
            # Let's use a dict or offset

    # This is getting complex. Let me use a cleaner implementation.

    # ---------- Clean CDCL Implementation ----------
    class SATSolver:
        def __init__(self, n_vars, clauses_1based):
            self.n = n_vars
            self.clauses = clauses_1based  # list of lists of 1-based signed literals
            self.m = len(clauses_1based)
            self.assign = np.full(n_vars, -1, dtype=np.int8)
            self.level = np.zeros(n_vars, dtype=np.int32)
            self.reason = np.full(n_vars, -1, dtype=np.int32)
            self.trail = []
            self.trail_pos = {}
            self.lcl = 0
            self.learned = []
            # Watches: for each literal, list of clause indices that watch it
            # Literals range from -(n+1) to n+1, so offset by n+1
            self.offset = n_vars + 1
            self.watches = [[] for _ in range(2 * (n_vars + 1) + 1)]
            self.decision_level = 0
            self.rng = np.random.default_rng(42)
            self.solved = False
            self.solution = None
            self.var_order = list(range(n_vars))

            # Initialize watches
            for ci, clause in enumerate(self.clauses):
                for lit in clause:
                    idx = lit + self.offset
                    if 0 <= idx < len(self.watches):
                        self.watches[idx].append(ci)
                # Also add to negation watches
                for lit in clause:
                    idx = -lit + self.offset
                    if 0 <= idx < len(self.watches):
                        self.watches[idx].append(ci)

        def lit_to_var(self, lit):
            return abs(lit) - 1

        def lit_val(self, lit):
            v = abs(lit) - 1
            val = self.assign[v]
            if val == -1:
                return -1
            if lit > 0:
                return val
            else:
                return 1 - val

        def is_true(self, lit):
            return self.lit_val(lit) == 1

        def is_false(self, lit):
            return self.lit_val(lit) == 0

        def enqueue(self, lit, level):
            v = abs(lit) - 1
            if self.assign[v] != -1:
                return
            val = 1 if lit > 0 else 0
            self.assign[v] = val
            self.level[v] = level
            self.trail.append((v, level))
            self.trail_pos[v] = len(self.trail) - 1
            self.lcl += 1

        def watch_idx(self, lit):
            return lit + self.offset

        def propagate(self):
            while self.lcl < len(self.trail):
                v, lvl = self.trail[self.lcl]
                # The literal just assigned
                if self.assign[v] == 1:
                    lit = v + 1
                else:
                    lit = -(v + 1)

                comp = -lit
                widx = self.watch_idx(comp)
                if widx < 0 or widx >= len(self.watches):
                    self.lcl += 1
                    continue
                watches = self.watches[widx]
                i = 0
                while i < len(watches):
                    ci = watches[i]
                    clause = self.clauses[ci] if ci < self.m else self.learned[ci - self.m]

                    # Check if clause is satisfied by lit
                    if self.is_true(lit):
                        # Find another literal to watch
                        found = False
                        for c_lit in clause:
                            if c_lit == comp:
                                continue
                            if not self.is_false(c_lit):
                                new_widx = self.watch_idx(c_lit)
                                if 0 <= new_widx < len(self.watches):
                                    self.watches[new_widx].append(ci)
                                watches[i] = watches[-1]
                                watches.pop()
                                found = True
                                break
                        if not found:
                            return False
                        continue

                    # Check for unit or conflict
                    has_unassigned = False
                    for c_lit in clause:
                        if c_lit == comp:
                            continue
                        if self.is_false(c_lit):
                            continue
                        if self.is_true(c_lit):
                            # Satisfied, remove watch
                            watches[i] = watches[-1]
                            watches.pop()
                            break
                        else:
                            has_unassigned = True
                    else:
                        if has_unassigned:
                            for c_lit in clause:
                                if c_lit != comp and self.lit_val(c_lit) == -1:
                                    self.enqueue(c_lit, self.decision_level)
                                    self.reason[abs(c_lit) - 1] = ci
                                    break
                            continue
                        else:
                            return False
                    i += 1
                self.lcl += 1
            return True

        def analyze(self, conflict_ci):
            clause = self.clauses[conflict_ci] if conflict_ci < self.m else self.learned[conflict_ci - self.m]
            learned_lits = []
            seen = set()
            counter = 0
            p = len(self.trail) - 1

            while True:
                v, _ = self.trail[p]
                if self.assign[v] == 1:
                    lit = v + 1
                else:
                    lit = -(v + 1)

                if v not in seen:
                    seen.add(v)
                    learned_lits.append(lit)
                    counter += 1
                    if self.level[v] == self.decision_level:
                        counter -= 1
                        if counter == 0:
                            break

                reason_idx = self.reason[v]
                if reason_idx == -1:
                    break
                reason = self.clauses[reason_idx] if reason_idx < self.m else self.learned[reason_idx - self.m]
                for r_lit in reason:
                    rv = abs(r_lit) - 1
                    if rv == v:
                        continue
                    if rv not in seen and self.assign[rv] != -1:
                        seen.add(rv)
                        learned_lits.append(r_lit)
                        if self.level[rv] == self.decision_level:
                            counter += 1
                p -= 1

            if len(learned_lits) > 1:
                levels = [self.level[abs(l) - 1] for l in learned_lits[1:]]
                bt_level = max(levels) if levels else 0
            else:
                bt_level = 0

            learned_clause = [-l for l in learned_lits]
            self.learned.append(learned_clause)
            li = self.m + len(self.learned) - 1
            for lit in learned_clause:
                widx = self.watch_idx(lit)
                if 0 <= widx < len(self.watches):
                    self.watches[widx].append(li)
                widx = self.watch_idx(-lit)
                if 0 <= widx < len(self.watches):
                    self.watches[widx].append(li)

            return learned_clause, bt_level

        def backtrack(self, level):
            while len(self.trail) > 0 and self.trail[-1][1] > level:
                v, _ = self.trail.pop()
                self.trail_pos.pop(v, None)
                self.assign[v] = -1
                self.reason[v] = -1
            self.lcl = len(self.trail)

        def pick_variable(self):
            for v in self.var_order:
                if self.assign[v] == -1:
                    return v
            return None

        def check_solved(self):
            for v in range(self.n):
                if self.assign[v] == -1:
                    return False
            for ci in range(self.m):
                clause = self.clauses[ci]
                if not any(self.is_true(lit) for lit in clause):
                    return False
            return True

        def search(self, time_limit, max_conflicts=50000):
            conflicts = 0
            while True:
                if time.time() > time_limit:
                    break

                if not self.propagate():
                    conflicts += 1
                    if conflicts > max_conflicts:
                        break

                    # Find conflict
                    conflict_ci = -1
                    for ci in range(self.m):
                        clause = self.clauses[ci]
                        all_false = True
                        for lit in clause:
                            if self.is_true(lit):
                                all_false = False
                                break
                        if all_false:
                            conflict_ci = ci
                            break

                    if conflict_ci == -1:
                        for li in range(len(self.learned)):
                            clause = self.learned[li]
                            all_false = True
                            for lit in clause:
                                if self.is_true(lit):
                                    all_false = False
                                    break
                            if all_false:
                                conflict_ci = self.m + li
                                break

                    if conflict_ci == -1:
                        break

                    learned, bt_level = self.analyze(conflict_ci)
                    self.backtrack(bt_level)
                    if len(learned) == 1:
                        self.enqueue(learned[0], self.decision_level)
                else:
                    if self.check_solved():
                        self.solved = True
                        self.solution = self.assign.copy()
                        return True

                    v = self.pick_variable()
                    if v is None:
                        break

                    self.decision_level += 1
                    val = int(self.rng.integers(2))
                    lit = v + 1 if val == 1 else -(v + 1)
                    self.enqueue(lit, self.decision_level)

            return self.solved

    # Try CDCL first
    solver = SATSolver(n, clauses)
    cdcl_deadline = deadline - 60.0  # Leave time for fallback
    if solver.search(cdcl_deadline, max_conflicts=200000):
        return solver.solution.astype(bool).tolist()

    # ---------- Fallback: Local Search ----------
    # Rebuild for local search using 0-based
    lits_0 = [[l - 1 for l in c] for c in clauses]
    m = len(lits_0)

    lits = np.empty((m, 3), dtype=np.int32)
    for i, c in enumerate(lits_0):
        for j, l in enumerate(c):
            lits[i, j] = l

    var_idx = np.abs(lits) - 1
    sign = (lits > 0).astype(np.int32)

    var_clauses = [[] for _ in range(n)]
    var_pos = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = int(var_idx[i, j])
            var_clauses[v].append(i)
            var_pos[v].append(j)

    var_clauses_np = [np.array(vc, dtype=np.int32) if vc else np.empty(0, dtype=np.int32)
                      for vc in var_clauses]
    var_pos_np = [np.array(vp, dtype=np.int32) if vp else np.empty(0, dtype=np.int32)
                  for vp in var_pos]

    def init_assignment():
        return (np.random.rand(n) < 0.5).astype(np.int8)

    def init_num_true(assignment):
        satisfied = (assignment[var_idx] == sign)
        return satisfied.sum(axis=1).astype(np.int32)

    def check_solved(num_true):
        return num_true.min() > 0

    def update_var(assignment, num_true, v, new_val):
        c_list = var_clauses_np[v]
        p_list = var_pos_np[v]
        if c_list.size == 0:
            return
        s = sign[c_list, p_list]
        now_sat = (new_val == s)
        delta = (2 * now_sat.astype(np.int32)) - 1
        num_true[c_list] += delta

    def unit_propagate(assignment, num_true):
        for _ in range(10 * n + 20):
            unit = np.where(num_true == 1)[0]
            if unit.size == 0:
                break
            progress = False
            for c in unit:
                sat_pos = np.where((assignment[var_idx[c]] == sign[c]))[0]
                if sat_pos.size == 0:
                    continue
                p = int(sat_pos[0])
                v = int(var_idx[c, p])
                target = int(sign[c, p])
                if int(assignment[v]) == target:
                    continue
                assignment[v] = target
                progress = True
                update_var(assignment, num_true, v, target)
            if not progress:
                break
        return assignment, num_true

    def var_unsat_counts(num_true):
        unsat = (num_true == 0)
        if not unsat.any():
            return None
        counts = np.zeros(n, dtype=np.int32)
        for c in np.where(unsat)[0]:
            counts[var_idx[c]] += 1
        return counts

    def compute_all_deltas(assignment, num_true):
        deltas = np.zeros(n, dtype=np.int32)
        for v in range(n):
            c_list = var_clauses_np[v]
            if c_list.size == 0:
                continue
            s = sign[c_list, var_pos_np[v]]
            cur_val = assignment[v]
            v_sat_mask = (cur_val == s)
            v_unsat_mask = ~v_sat_mask
            cur_num_true = num_true[c_list]
            becomes_sat = (cur_num_true == 0) & v_unsat_mask
            becomes_unsat = (cur_num_true == 1) & v_sat_mask
            deltas[v] = int(becomes_sat.sum()) - int(becomes_unsat.sum())
        return deltas

    def vectorized_pick(assignment, num_true, var_counts, rng):
        deltas = compute_all_deltas(assignment, num_true)
        unsat_mask = (num_true == 0)
        if not unsat_mask.any():
            return None
        unsat_vars = np.zeros(n, dtype=bool)
        for c in np.where(unsat_mask)[0]:
            unsat_vars[var_idx[c]] = True
        if not unsat_vars.any():
            return None
        candidate_vars = np.where(unsat_vars)[0]
        candidate_deltas = deltas[candidate_vars]
        if var_counts is not None:
            vc = var_counts[candidate_vars].astype(np.float64) + 0.5
            weights = (candidate_deltas.astype(np.float64) + 1.0) * vc
        else:
            weights = candidate_deltas.astype(np.float64) + 1.0
        weights = np.maximum(weights, 0.1)
        total_w = weights.sum()
        if total_w <= 0:
            return int(candidate_vars[rng.integers(candidate_vars.size)])
        r = rng.random() * total_w
        cum = 0.0
        for i in range(len(candidate_vars)):
            cum += weights[i]
            if r <= cum:
                return int(candidate_vars[i])
        return int(candidate_vars[-1])

    rng = np.random.default_rng(0)
    best_unsat = None
    best_assign = None
    base_steps = 5000
    current_steps = base_steps
    consecutive_failures = 0

    for restart in range(100000):
        if time.time() > deadline - 5.0:
            break

        assignment = init_assignment()
        num_true = init_num_true(assignment)
        if check_solved(num_true):
            return assignment.astype(bool).tolist()

        assignment, num_true = unit_propagate(assignment, num_true)
        if check_solved(num_true):
            return assignment.astype(bool).tolist()

        unsat_before = int((num_true == 0).sum())

        for step in range(current_steps):
            if time.time() > deadline - 5.0:
                break

            unsat_mask = (num_true == 0)
            if not unsat_mask.any():
                return assignment.astype(bool).tolist()

            unsat_count = int(unsat_mask.sum())
            if best_unsat is None or unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assignment.copy()

            if step % 10 == 0:
                var_counts = var_unsat_counts(num_true)
            else:
                var_counts = None

            v = vectorized_pick(assignment, num_true, var_counts, rng)
            if v is None:
                unsat_idx = np.where(unsat_mask)[0]
                c = int(unsat_idx[rng.integers(unsat_idx.size)])
                p = int(rng.integers(3))
                v = int(var_idx[c, p])

            new = 1 - int(assignment[v])
            assignment[v] = new
            update_var(assignment, num_true, v, new)

        unsat_after = int((num_true == 0).sum())

        if unsat_after < unsat_before:
            consecutive_failures = 0
            current_steps = min(current_steps * 2, 30000)
        else:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                current_steps = base_steps
                consecutive_failures = 0

        if time.time() > deadline - 5.0:
            break

    if best_assign is not None:
        assignment = best_assign.copy()
        num_true = init_num_true(assignment)
        assignment, num_true = unit_propagate(assignment, num_true)
        if check_solved(num_true):
            return assignment.astype(bool).tolist()

        polish_deadline = deadline - 1.0
        var_counts = var_unsat_counts(num_true)

        for _ in range(500000):
            if time.time() > polish_deadline:
                break
            unsat_mask = (num_true == 0)
            if not unsat_mask.any():
                return assignment.astype(bool).tolist()
            unsat_count = int(unsat_mask.sum())
            if best_unsat is None or unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assignment.copy()

            if _ % 5 == 0:
                var_counts = var_unsat_counts(num_true)

            v = vectorized_pick(assignment, num_true, var_counts, rng)
            if v is None:
                unsat_idx = np.where(unsat_mask)[0]
                c = int(unsat_idx[rng.integers(unsat_idx.size)])
                p = int(rng.integers(3))
                v = int(var_idx[c, p])

            new = 1 - int(assignment[v])
            assignment[v] = new
            update_var(assignment, num_true, v, new)

    if best_assign is not None:
        return best_assign.astype(bool).tolist()

    return init_assignment().astype(bool).tolist()
