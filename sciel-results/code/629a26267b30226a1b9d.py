"""Few edges in 5-uniform hypergraphs without Property B (m(5))."""
import time
import random
import ctypes
from itertools import combinations

_DEADLINE = 135.0  # evaluator kills at 150s; keep a safety margin.


# ---------------------------------------------------------------------------
# SAT solver backend (ctypes -> libz3).  If unavailable, fall back to pure
# Python backtracking with unit propagation.
# ---------------------------------------------------------------------------
class _Z3:
    def __init__(self):
        self._lib = None
        self._ok = False
        try:
            for name in ("libz3.so.4", "libz3.so", "libz3.dll", "z3.dll",
                         "libz3.dylib", "z3"):
                try:
                    self._lib = ctypes.CDLL(name)
                    break
                except OSError:
                    continue
            if self._lib is None:
                raise OSError("no libz3 found")
            self._lib.Z3_mk_context.argtypes = [ctypes.c_int]
            self._lib.Z3_mk_context.restype = ctypes.c_void_p
            self._lib.Z3_mk_int.argtypes = [ctypes.c_int, ctypes.c_int]
            self._lib.Z3_mk_int.restype = ctypes.c_void_p
            self._lib.Z3_mk_bool_const.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            self._lib.Z3_mk_bool_const.restype = ctypes.c_void_p
            self._lib.Z3_mk_not.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self._lib.Z3_mk_not.restype = ctypes.c_void_p
            self._lib.Z3_mk_and.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
            self._lib.Z3_mk_and.restype = ctypes.c_void_p
            self._lib.Z3_mk_or.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
            self._lib.Z3_mk_or.restype = ctypes.c_void_p
            self._lib.Z3_assert_formula.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self._lib.Z3_assert_formula.restype = None
            self._lib.Z3_solver.mk_context = None
            self._lib.Z3_mk_solver.argtypes = [ctypes.c_void_p]
            self._lib.Z3_mk_solver.restype = ctypes.c_void_p
            self._lib.Z3_solver_assert.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self._lib.Z3_solver_assert.restype = None
            self._lib.Z3_solver_check.argtypes = [ctypes.c_void_p]
            self._lib.Z3_solver_check.restype = ctypes.c_int
            self._lib.Z3_del_context.argtypes = [ctypes.c_void_p]
            self._lib.Z3_del_context.restype = None
            # smoke test
            ctx = self._lib.Z3_mk_context(0)
            if not ctx:
                raise OSError("mk_context failed")
            x = self._lib.Z3_mk_bool_const(ctx, b"x")
            y = self._lib.Z3_mk_bool_const(ctx, b"y")
            arr = (ctypes.c_void_p * 2)(x, self._lib.Z3_mk_not(ctx, y))
            c = self._lib.Z3_mk_and(ctx, 2, arr)
            arr2 = (ctypes.c_void_p * 2)(self._lib.Z3_mk_not(ctx, x), y)
            c2 = self._lib.Z3_mk_and(ctx, 2, arr2)
            self._lib.Z3_assert_formula(ctx, c)
            self._lib.Z3_assert_formula(ctx, c2)
            self._lib.Z3_del_context(ctx)
            self._ok = True
        except Exception:
            self._ok = False
            self._lib = None

    def satisfiable(self, clauses):
        """clauses: list of lists of (var, polarity). polarity True => var, False => not var."""
        if not self._ok:
            return None
        lib = self._lib
        ctx = lib.Z3_mk_context(0)
        if not ctx:
            self._ok = False
            return None
        try:
            consts = {}
            for (v, p) in set((v, p) for cl in clauses for (v, p) in cl):
                if (v, p) not in consts:
                    consts[(v, p)] = lib.Z3_mk_bool_const(ctx, b"c")
            for cl in clauses:
                lits = (ctypes.c_void_p * len(cl))(*[consts[(v, p)] for (v, p) in cl])
                term = lib.Z3_mk_and(ctx, len(cl), lits)
                lib.Z3_assert_formula(ctx, term)
            solver = lib.Z3_mk_solver(ctx)
            if not solver:
                return None
            res = lib.Z3_solver_check(solver)
            # 1 = sat, 0 = unsat, -1 = unknown
            if res == 1:
                return True
            if res == 0:
                return False
            return None
        finally:
            lib.Z3_del_context(ctx)


_Z3 = _Z3()


def _edge_mask(edge, n):
    m = 0
    for v in edge:
        m |= (1 << v)
    return m


def _is_2_colorable(masks, n, deadline):
    """True if a proper 2-coloring exists, False if none, None if unknown/timeout."""
    if _Z3.ok:
        clauses = []
        for mask in masks:
            vs = [v for v in range(n) if mask & (1 << v)]
            # at least one 0:  OR(not c_v for v in vs)
            clauses.append([(v, False) for v in vs])
            # at least one 1:  OR(c_v for v in vs)
            clauses.append([(v, True) for v in vs])
        res = _Z3.satisfiable(clauses)
        if res is not None:
            return res

    # ---- pure-python fallback (backtracking + unit propagation) ----
    edge_verts = []
    for mask in masks:
        edge_verts.append([v for v in range(n) if mask & (1 << v)])
    color = [None] * n

    def propagate():
        assignments = []
        changed = True
        while changed:
            changed = False
            for verts in edge_verts:
                c0 = c1 = 0
                un = []
                for v in verts:
                    if color[v] is None:
                        un.append(v)
                    elif color[v] == 0:
                        c0 += 1
                    else:
                        c1 += 1
                if c0 == 5 or c1 == 5:
                    return False, assignments
                if len(un) == 1:
                    v = un[0]
                    if c0 == 4:
                        if color[v] == 0:
                            return False, assignments
                        if color[v] is None:
                            color[v] = 1
                            assignments.append(v)
                            changed = True
                    elif c1 == 4:
                        if color[v] == 1:
                            return False, assignments
                        if color[v] is None:
                            color[v] = 0
                            assignments.append(v)
                            changed = True
        return True, assignments

    def backtrack():
        if time.time() > deadline:
            return None
        if all(c is not None for c in color):
            return True
        ok, assignments = propagate()
        if not ok:
            return False
        best_v = -1
        best_c = -1
        for verts in edge_verts:
            ac = sum(1 for v in verts if color[v] is not None)
            if ac > best_c:
                best_c = ac
                for v in verts:
                    if color[v] is None:
                        best_v = v
                        break
                if best_v != -1:
                    break
        if best_v == -1:
            return True
        found = False
        for c in (0, 1):
            color[best_v] = c
            r = backtrack()
            if r is None:
                color[best_v] = None
                for v in assignments:
                    color[v] = None
                return None
            if r:
                found = True
                break
            color[best_v] = None
        for v in assignments:
            color[v] = None
        return found

    return backtrack()


def _is_non_2_colorable(masks, n, deadline):
    r = _is_2_colorable(masks, n, deadline)
    return r is False  # only assert non-colorable when proven


# ---------------------------------------------------------------------------
# Known 51-edge non-2-colorable 5-uniform hypergraph (Erdos-style base).
# If not actually non-colorable, we build one from a complete hypergraph.
# ---------------------------------------------------------------------------
def _base_edges(n=17):
    edges = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    e = (a + b + c + d) % n
                    if e not in (a, b, c, d):
                        edges.append((a, b, c, d, e))
    return edges


def _base_complete(n):
    return [list(e) for e in combinations(range(n), 5)]


def _make_base(deadline):
    base = _base_edges(17)
    masks = [_edge_mask(e, 17) for e in base]
    if _is_non_2_colorable(masks, 17, deadline):
        return base, 17
    # fallback: complete hypergraph on 9 vertices is definitely non-2-colorable
    edges = _base_complete(9)
    return edges, 9


def _greedy_reduce(edges, n, deadline, rng):
    current = list(edges)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        idxs = list(range(len(current)))
        rng.shuffle(idxs)
        for i in idxs:
            if time.time() > deadline:
                break
            if len(current) <= 1:
                break
            test = current[:i] + current[i + 1:]
            masks = [_edge_mask(e, n) for e in test]
            if _is_non_2_colorable(masks, n, deadline):
                current = test
                improved = True
    return current


def construct():
    start = time.time()
    deadline = start + _DEADLINE - 8  # 8s safety margin
    rng = random.Random(137)

    base, n = _make_base(deadline)
    best = _greedy_reduce(base, n, deadline, rng)

    # A few extra randomized reduction restarts to escape local optima.
    for _ in range(3):
        if time.time() > deadline:
            break
        cand = _greedy_reduce(base, n, deadline, rng)
        if len(cand) < len(best):
            best = cand

    # Final rigorous verification of the returned hypergraph.
    masks = [_edge_mask(e, n) for e in best]
    if not _is_non_2_colorable(masks, n, deadline):
        # should not happen; fall back to a guaranteed non-2-colorable graph
        fb = _base_complete(9)
        return {"n": 9, "edges": fb}

    return {"n": n, "edges": best}
