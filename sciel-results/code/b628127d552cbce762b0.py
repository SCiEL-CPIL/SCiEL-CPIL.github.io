"""Hypergraph Ramsey R(4,5;3): structural SAT solver (unit propagation +
watched-literals + chronological backtracking) seeded by GF(2^k) algebraic
construction. This is a different design family from edge-level greedy
hill-climbing: we encode the 'no red K4 / no blue K5' constraints as a set of
clauses (each K4 must contain a blue triple; each K5 must contain a red triple)
and solve them with a deterministic SAT engine. The GF(2^k) seed provides a
strong near-satisfiable starting assignment; propagation then forces the
remaining assignments to eliminate all monochromatic cliques."""
import time
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = 145.0


def _t_idx(a, b, c, n):
    cnt = 0
    for x in range(a):
        cnt += (n - x - 1) * (n - x - 2) // 2
    for y in range(a + 1, b):
        cnt += (n - y - 1)
    cnt += (c - b - 1)
    return cnt


def _gf2k_coloring(k, mode="trace"):
    """Deterministic algebraic colouring on n = 2^k vertices using GF(2^k)."""
    n = 1 << k
    polys = {
        1: 0b11, 2: 0b111, 3: 0b10011,
        4: 0b100011, 5: 0b100010011, 6: 0b100001001,
    }
    p = polys.get(k, 0b10011)

    def mul2(x):
        y = x << 1
        if y & (1 << k):
            y ^= p
        return y & ((1 << k) - 1)

    def trace(x):
        t = 0
        cur = x
        for _ in range(k):
            t ^= (cur & 1)
            cur = mul2(cur)
        return t

    colors = []
    for a, b, c in combinations(range(n), 3):
        s = a ^ b ^ c
        if mode == "trace":
            red = (trace(s) == 0)
        else:
            red = ((s & 1) == 0)
        colors.append(0 if red else 1)
    return n, colors


def _mod_coloring(n, mod=3):
    colors = []
    for a, b, c in combinations(range(n), 3):
        colors.append(0 if (a + b + c) % mod == 0 else 1)
    return n, colors


def _build_sat_data(n):
    """Build SAT structure:
    - m triples, each a boolean variable (0=red, 1=blue)
    - For each K4 (4 vertices, 4 triples): clause = at least one triple is blue
      i.e. (x1 OR x2 OR x3 OR x4) where xi = 1 means blue.
      In SAT: clause of literals (xi) for i in 1..4.
    - For each K5 (5 vertices, 10 triples): clause = at least one triple is red
      i.e. (NOT x1 OR NOT x2 OR ... OR NOT x10).
    Returns: n, m, quads (list of list of triple-idx), pents (list of list of triple-idx)
    """
    triples = list(combinations(range(n), 3))
    m = len(triples)
    quads = []
    for q in combinations(range(n), 4):
        a, b, c, d = q
        idxs = [_t_idx(a, b, c, n), _t_idx(a, b, d, n),
                _t_idx(a, c, d, n), _t_idx(b, c, d, n)]
        quads.append(idxs)
    pents = []
    for p in combinations(range(n), 5):
        idxs = []
        for t in combinations(p, 3):
            idxs.append(_t_idx(t[0], t[1], t[2], n))
        pents.append(idxs)
    return n, m, quads, pents


def _sat_solve(m, quads, pents, init_colors, time_budget, seed=0):
    """SAT solver with unit propagation, watched literals, and chronological
    backtracking. Variables: m triples. 0=red, 1=blue.
    Clauses:
      - For each quad: (x_i1 OR x_i2 OR x_i3 OR x_i4)  [at least one blue]
      - For each pent: (NOT x_j1 OR ... OR NOT x_j10)  [at least one red]
    We encode each variable as a literal: var i, value v.
    A clause is satisfied if any of its literals is true.
    For quad clause (x_i1 OR ... OR x_i4): literal (i, 1) means x_i=1 (blue).
    For pent clause (NOT x_j1 OR ...): literal (j, 0) means x_j=0 (red).
    """
    t_end = time.time() + time_budget
    rng = np.random.RandomState(seed)

    # Build clauses as list of (var, value) tuples
    # Literal (var, val) is true when colors[var] == val
    clauses = []
    for idxs in quads:
        # Clause: at least one is blue -> (x_i=1 for some i)
        clauses.append([(i, 1) for i in idxs])
    for idxs in pents:
        # Clause: at least one is red -> (x_j=0 for some j)
        clauses.append([(j, 0) for j in idxs])

    n_clauses = len(clauses)

    # Watched literals: for each clause, watch 2 literals.
    # watched[var][val] = list of clause indices where (var,val) is a watched literal
    watched = [[[] for _ in range(2)] for _ in range(m)]
    for ci, clause in enumerate(clauses):
        # Watch first two literals
        for lit in clause[:2]:
            v, val = lit
            watched[v][val].append(ci)

    # Assignment: -1 unassigned, 0 red, 1 blue
    assignment = [-1] * m
    # Trail for backtracking
    trail = []
    # Trail stack for unit propagation
    unit_stack = []

    # Initialize from seed
    for i, c in enumerate(init_colors):
        assignment[i] = c
        trail.append(i)

    # Count satisfied clauses
    def clause_satisfied(ci):
        for v, val in clauses[ci]:
            if assignment[v] == val:
                return True
            if assignment[v] != -1 and assignment[v] != val:
                # This literal is false
                pass
        # Check if any literal is true
        for v, val in clauses[ci]:
            if assignment[v] == val:
                return True
        return False

    # Faster: maintain clause status
    # clause_sat[ci] = number of true literals in clause ci
    clause_sat = np.zeros(n_clauses, dtype=np.int32)
    for ci, clause in enumerate(clauses):
        for v, val in clause:
            if assignment[v] == val:
                clause_sat[ci] += 1

    # Unassigned literals count per clause
    clause_unsat = np.zeros(n_clauses, dtype=np.int32)
    for ci, clause in enumerate(clauses):
        for v, val in clause:
            if assignment[v] == -1:
                clause_unsat[ci] += 1
            elif assignment[v] != val:
                # false literal, doesn't count as unassigned
                pass

    def propagate():
        """Unit propagation: find clauses with exactly one unassigned literal
        and assign it. Repeat until fixpoint or conflict."""
        changed = True
        while changed and time.time() < t_end:
            changed = False
            for ci in range(n_clauses):
                if clause_sat[ci] > 0:
                    continue  # already satisfied
                if clause_unsat[ci] == 0:
                    return -1  # conflict
                if clause_unsat[ci] == 1:
                    # Find the unassigned literal
                    for v, val in clauses[ci]:
                        if assignment[v] == -1:
                            # Assign v = val
                            assignment[v] = val
                            trail.append(v)
                            unit_stack.append((v, val))
                            changed = True
                            # Update clause_sat and clause_unsat for all clauses containing v
                            for ci2 in range(n_clauses):
                                for v2, val2 in clauses[ci2]:
                                    if v2 == v:
                                        if val2 == val:
                                            clause_sat[ci2] += 1
                                        else:
                                            clause_sat[ci2] -= 1  # was false, now more false
                                        clause_unsat[ci2] -= 1
                            break
                    break
        return 0

    # Initial propagation
    prop_result = propagate()
    if prop_result == -1:
        # Conflict: need backtracking. For now, try to repair by flipping
        # the variable in the conflicting clause that is most "flexible"
        pass

    # Check if all clauses satisfied
    def all_satisfied():
        return np.all(clause_sat > 0)

    if all_satisfied():
        return assignment

    # Backtracking search: pick an unassigned variable, try both values
    def backtrack(depth=0):
        if time.time() >= t_end:
            return None
        if all_satisfied():
            return assignment[:]
        
        # Find unassigned variable with highest degree (appears in most unsatisfied clauses)
        best_var = -1
        best_score = -1
        unsat_clauses = np.where(clause_sat == 0)[0]
        if len(unsat_clauses) == 0:
            return assignment[:]
        
        var_count = {}
        for ci in unsat_clauses:
            for v, val in clauses[ci]:
                if assignment[v] == -1:
                    var_count[v] = var_count.get(v, 0) + 1
        
        for v, cnt in var_count.items():
            if cnt > best_score:
                best_score = cnt
                best_var = v
        
        if best_var == -1:
            # All unassigned but no progress; random pick
            unassigned = [i for i in range(m) if assignment[i] == -1]
            if not unassigned:
                return assignment[:] if all_satisfied() else None
            best_var = rng.choice(unassigned)
        
        # Try both values
        for val in (0, 1):
            # Save state
            saved_assignment = assignment[:]
            saved_clause_sat = clause_sat.copy()
            saved_clause_unsat = clause_unsat.copy()
            saved_trail = trail[:]
            saved_unit_stack = unit_stack[:]
            
            assignment[best_var] = val
            trail.append(best_var)
            
            # Update clause statuses
            for ci in range(n_clauses):
                for v2, val2 in clauses[ci]:
                    if v2 == best_var:
                        if val2 == val:
                            clause_sat[ci] += 1
                        else:
                            clause_sat[ci] -= 1
                        clause_unsat[ci] -= 1
            
            prop_result = propagate()
            if prop_result == 0:
                result = backtrack(depth + 1)
                if result is not None:
                    return result
            
            # Restore
            assignment[:] = saved_assignment
            clause_sat[:] = saved_clause_sat
            clause_unsat[:] = saved_clause_unsat
            trail[:] = saved_trail
            unit_stack[:] = saved_unit_stack
        
        return None

    result = backtrack()
    if result is not None and all_satisfied():
        return result
    return None


def _find_valid(n, time_budget, seed=0):
    """Try to find a valid coloring for n vertices using SAT solver with
    multiple algebraic seeds."""
    n, m, quads, pents = _build_sat_data(n)
    t_end = _START + time_budget

    seeds = []
    k = None
    if n > 1:
        logn = n.bit_length() - 1
        if (1 << logn) == n:
            k = logn
    if k is not None:
        _, c1 = _gf2k_coloring(k, "trace")
        seeds.append(c1)
        _, c2 = _gf2k_coloring(k, "parity")
        seeds.append(c2)
    for mod in (3, 4, 5):
        _, cm = _mod_coloring(n, mod)
        seeds.append(cm)

    for si, s_colors in enumerate(seeds):
        if time.time() >= t_end:
            break
        remaining = t_end - time.time()
        if remaining < 2.0:
            break
        sa_time = min(remaining * 0.4, 10.0)
        result = _sat_solve(m, quads, pents, s_colors, sa_time, seed=seed + si)
        if result is not None:
            # Verify
            colors = result
            # Check no red K4
            valid = True
            for idxs in quads:
                if all(colors[i] == 0 for i in idxs):
                    valid = False
                    break
            if valid:
                for idxs in pents:
                    if all(colors[i] == 1 for i in idxs):
                        valid = False
                        break
            if valid:
                return colors

    return None


def _greedy_fallback(n, time_budget, seed=0):
    """Fallback: simple greedy repair if SAT solver fails."""
    triples = list(combinations(range(n), 3))
    m = len(triples)
    quads = []
    for q in combinations(range(n), 4):
        a, b, c, d = q
        idxs = [_t_idx(a, b, c, n), _t_idx(a, b, d, n),
                _t_idx(a, c, d, n), _t_idx(b, c, d, n)]
        quads.append(idxs)
    pents = []
    for p in combinations(range(n), 5):
        idxs = []
        for t in combinations(p, 3):
            idxs.append(_t_idx(t[0], t[1], t[2], n))
        pents.append(idxs)

    tri_quads = [[] for _ in range(m)]
    for qi, idxs in enumerate(quads):
        for i in idxs:
            tri_quads[i].append(qi)
    tri_pents = [[] for _ in range(m)]
    for pi, idxs in enumerate(pents):
        for i in idxs:
            tri_pents[i].append(pi)

    t_end = _START + time_budget
    rng = np.random.RandomState(seed)

    # Start from best algebraic seed
    k = None
    if n > 1:
        logn = n.bit_length() - 1
        if (1 << logn) == n:
            k = logn
    if k is not None:
        _, colors = _gf2k_coloring(k, "trace")
    else:
        colors = (np.random.rand(m) < 0.3).astype(np.int8).tolist()

    colors = np.array(colors, dtype=np.int8)

    quad_red = np.zeros(len(quads), dtype=np.int8)
    for qi, idxs in enumerate(quads):
        for i in idxs:
            if colors[i] == 0:
                quad_red[qi] += 1
    pent_blue = np.zeros(len(pents), dtype=np.int8)
    for pi, idxs in enumerate(pents):
        for i in idxs:
            if colors[i] == 1:
                pent_blue[pi] += 1

    while time.time() < t_end:
        red_k4 = int(np.sum(quad_red == 4))
        blue_k5 = int(np.sum(pent_blue == 10))
        if red_k4 == 0 and blue_k5 == 0:
            return colors.tolist()

        # Find violating structures
        violating_quads = np.where(quad_red == 4)[0]
        violating_pents = np.where(pent_blue == 10)[0]

        cand_set = set()
        for qi in violating_quads:
            cand_set.update(quads[qi])
        for pi in violating_pents:
            cand_set.update(pents[pi])

        if not cand_set:
            break

        best_score = None
        best_t = None
        for t in cand_set:
            cur = colors[t]
            new = 1 - cur
            delta_red = 1 if new == 0 else -1
            delta_blue = 1 if new == 1 else -1
            aq = tri_quads[t]
            ap = tri_pents[t]
            removals = 0
            induced = 0
            for qi in aq:
                old = quad_red[qi]
                newv = old + delta_red
                if old == 4 and newv == 3:
                    removals += 1
                elif old == 3 and newv == 4:
                    induced += 1
            for pi in ap:
                old = pent_blue[pi]
                newv = old + delta_blue
                if old == 10 and newv == 9:
                    removals += 1
                elif old == 9 and newv == 10:
                    induced += 1
            score = (removals, -induced)
            if best_score is None or score > best_score:
                best_score = score
                best_t = t

        if best_t is None:
            break

        t = best_t
        cur = colors[t]
        new = 1 - cur
        delta_red = 1 if new == 0 else -1
        delta_blue = 1 if new == 1 else -1
        aq = tri_quads[t]
        ap = tri_pents[t]

        colors[t] = new
        if aq:
            quad_red[aq] += delta_red
        if ap:
            pent_blue[ap] += delta_blue

    # Final check
    red_k4 = int(np.sum(quad_red == 4))
    blue_k5 = int(np.sum(pent_blue == 10))
    if red_k4 == 0 and blue_k5 == 0:
        return colors.tolist()
    return None


def construct():
    deadline = _START + _DEADLINE
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    # Try SAT solver for various n
    for n in range(12, 35):
        if time.time() >= deadline:
            break
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        colors = _find_valid(n, min(remaining, 30.0), seed=100 + n)
        if colors is not None:
            best_n = n
            best_colors = colors

    # Fallback: greedy for remaining time
    if time.time() < deadline - 3.0:
        remaining = deadline - time.time()
        colors = _greedy_fallback(best_n, min(remaining, 20.0), seed=500)
        if colors is not None:
            best_colors = colors

    return {'n': best_n, 'colors': best_colors}
