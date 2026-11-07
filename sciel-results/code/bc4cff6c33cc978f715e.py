import sys
import time

def construct():
    TOTAL_BUDGET = 145.0
    start = time.time()
    sys.setrecursionlimit(1000000)

    def instance(splits, seed, palette=5):
        rng = __import__('random').Random(seed)
        faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
        edges = {(min(a,b),max(a,b)) for f in faces for a,b in zip(f, f[1:]+f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n+4))
            n += 4
            for i in range(4):
                a,b = face[i], face[(i+1)%4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i+1)%4]), max(inner[i], inner[(i+1)%4])))
                faces.append([a, b, inner[(i+1)%4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_sadwsat(n, adj, lists, time_limit):
        """
        Solve list coloring as SAT with:
        - Exactly-one-of for each vertex
        - At-most-one (pairwise) for each edge
        - List constraints (only allow colors in list)
        
        Uses DPLL with unit propagation, MRV, and chronological backtracking.
        """
        # Build SAT clauses
        # For vertex v with colors [c1, c2, c3], variable index for (v, c)
        var_index = {}
        var_count = 0
        for v in range(n):
            for c in lists[v]:
                var_index[(v, c)] = var_count
                var_count += 1
        
        # SAT: 1 per vertex (at least one) + at-most-one (pairwise NOT) per vertex
        # Edge constraints: NOT(v,c) OR NOT(u,c) for shared colors
        clauses = []
        
        # At least one per vertex
        for v in range(n):
            vs = [var_index[(v, c)] for c in lists[v]]
            clauses.append(vs)
        
        # At most one per vertex (pairwise)
        for v in range(n):
            ls = lists[v]
            for i in range(len(ls)):
                for j in range(i+1, len(ls)):
                    clauses.append([-var_index[(v, ls[i])], -var_index[(v, ls[j])]])
        
        # Edge constraints: for each edge (u,v), for each color c in both lists:
        # NOT(u,c) OR NOT(v,c)
        for u, v in [(min(a,b), max(a,b)) for a, b in adj]:
            common = set(lists[u]) & set(lists[v])
            for c in common:
                clauses.append([-var_index[(u, c)], -var_index[(v, c)]])
        
        # DPLL solver
        assignment = [None] * var_count  # None=unassigned, True/False
        clause_status = []  # for each clause: 'ok' if not falsified, 'unit' if one unassigned
        
        def propagate(clause_idx, var, val):
            """
            Unit propagation: assign var=val, propagate consequences.
            Returns (success, trail) where trail is list of (var, val) for undo.
            """
            trail = [(var, val)]
            assignment[var] = val
            queue = [(clause_idx,)]
            
            while queue:
                ci = queue.pop()
                clause = clauses[ci]
                satisfied = False
                unassigned = None
                unassigned_count = 0
                for lit in clause:
                    v = abs(lit) - 1
                    if assignment[v] is not None:
                        if (assignment[v] and lit > 0) or (not assignment[v] and lit < 0):
                            satisfied = True
                            break
                    else:
                        unassigned = lit
                        unassigned_count += 1
                if satisfied:
                    continue
                if unassigned_count == 0:
                    return False, trail
                if unassigned_count == 1:
                    # Unit: must assign
                    lit = unassigned
                    v = abs(lit) - 1
                    new_val = lit > 0
                    if assignment[v] is not None and assignment[v] != new_val:
                        return False, trail
                    if assignment[v] is None:
                        assignment[v] = new_val
                        trail.append((v, new_val))
                        # Propagate this new assignment to all clauses containing v
                        for ci2 in range(len(clauses)):
                            for lit2 in clauses[ci2]:
                                if abs(lit2) - 1 == v:
                                    queue.append(ci2)
                                    break
        
        # Simpler DPLL:
        def dpll(deadline):
            # Check for conflicts and units
            while True:
                # Conflict check
                conflict = None
                # Find unit clause
                unit_lit = None
                for ci, clause in enumerate(clauses):
                    sat = False
                    unassigned = []
                    for lit in clause:
                        v = abs(lit) - 1
                        if assignment[v] is not None:
                            if (assignment[v] and lit > 0) or (not assignment[v] and lit < 0):
                                sat = True
                                break
                        else:
                            unassigned.append(lit)
                    if not sat:
                        if len(unassigned) == 0:
                            conflict = ci
                            break
                        elif len(unassigned) == 1:
                            unit_lit = unassigned[0]
                            break
                if conflict is not None:
                    return False
                if unit_lit is not None:
                    v = abs(unit_lit) - 1
                    val = unit_lit > 0
                    assignment[v] = val
                    if time.time() > deadline:
                        return False
                    continue
                break
            
            # If all assigned, check solution
            if all(a is not None for a in assignment):
                return True
            
            # MRV: pick unassigned variable
            unassigned_vars = [v for v in range(var_count) if assignment[v] is None]
            if not unassigned_vars:
                return True
            
            # Simple heuristic: pick first unassigned
            v = unassigned_vars[0]
            
            # Try True first
            assignment[v] = True
            if dpll(deadline):
                return True
            assignment[v] = False
            if dpll(deadline):
                return True
            assignment[v] = None
            return False
        
        deadline = time.time() + time_limit
        if dpll(deadline):
            # Extract coloring
            color = [-1] * n
            for v in range(n):
                for c in lists[v]:
                    idx = var_index[(v, c)]
                    if assignment[idx] is True:
                        color[v] = c
                        break
            return color
        return None

    def solve_backtracking(n, adj, lists, time_limit):
        """Classic backtracking with MRV and forward checking."""
        domains = [list(lists[v]) for v in range(n)]
        color = [-1] * n
        deadline = time.time() + time_limit
        state = {'ok': False, 'count': 0}
        
        def forward_check(v, c, undo):
            for u in adj[v]:
                if color[u] != -1:
                    if color[u] == c:
                        # Already assigned same color - shouldn't happen if we check
                        return False
                    continue
                if c in domains[u]:
                    domains[u].remove(c)
                    undo.append((u, c))
                    if not domains[u]:
                        # Undo
                        for (uu, cc) in undo:
                            domains[uu].append(cc)
                        return False
            return True
        
        def bt():
            if all(c != -1 for c in color):
                state['ok'] = True
                return True
            state['count'] += 1
            if (state['count'] & 511) == 0 and time.time() > deadline:
                return False
            
            # MRV
            best_v = -1
            best_len = 999
            for v in range(n):
                if color[v] != -1:
                    continue
                ln = len(domains[v])
                if ln == 0:
                    return False
                if ln < best_len:
                    best_len = ln
                    best_v = v
                    if ln == 1:
                        break
            if best_v == -1:
                return False
            
            for c in list(domains[best_v]):
                undo = []
                if not forward_check(best_v, c, undo):
                    continue
                color[best_v] = c
                if bt():
                    return True
                color[best_v] = -1
                for (u, cc) in undo:
                    domains[u].append(cc)
            return False
        
        if bt() and state['ok']:
            return [int(c) for c in color]
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        remaining = TOTAL_BUDGET - (time.time() - start)
        n, edges, lists = instance(splits, seed)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        solved = False
        if remaining > 3.0:
            tl = max(1.0, min(20.0, remaining * 0.45))
            res = solve_backtracking(n, adj, lists, tl)
            if res is not None:
                out.append([int(c) for c in res])
                solved = True
        
        if not solved:
            if remaining > 1.0:
                tl = max(0.5, remaining * 0.3)
                res = solve_sadwsat(n, adj, lists, tl)
                if res is not None:
                    out.append([int(c) for c in res])
                    solved = True
        
        if not solved:
            # Fallback greedy
            order = sorted(range(n), key=lambda v: -len(adj[v]))
            color = [-1] * n
            for v in order:
                taken = {color[u] for u in adj[v] if color[u] != -1}
                c = -1
                for cc in lists[v]:
                    if cc not in taken:
                        c = cc
                        break
                if c == -1:
                    c = lists[v][0]
                color[v] = c
            out.append([int(c) for c in color])
    
    # Validate output
    lengths = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, _, _ = instance(splits, seed)
        lengths.append(n)
    
    for i in range(len(out)):
        if not isinstance(out[i], list):
            out[i] = list(out[i])
        if len(out[i]) != lengths[i]:
            if len(out[i]) < lengths[i]:
                out[i] = out[i] + [0] * (lengths[i] - len(out[i]))
            else:
                out[i] = out[i][:lengths[i]]
        out[i] = [int(x) for x in out[i]]
    
    return out
