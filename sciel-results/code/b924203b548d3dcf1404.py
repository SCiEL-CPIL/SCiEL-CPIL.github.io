"""Structured Z4-symmetric LP approach for rank-minimal 3x4x5 matmul."""
DIMENSIONS = [3, 4, 5]

from fractions import Fraction
import itertools


def construct():
    n, m, p = 3, 4, 5
    Nm, Mp, Pn = n * m, m * p, p * n  # 12, 20, 15

    # Build tensor: T[(i,j),(j,k),(k,i)] = 1
    tensor = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[(i * m + j, j * p + k, k * n + i)] = Fraction(1)

    # Z4 structure: orbits under x -> x+2 mod 4: {0,2} (even), {1,3} (odd)
    # Z5 structure: orbit {0} and orbit {1,2,3,4}
    #
    # Ansatz: rank-1 terms indexed by (i, j_orbit, k_orbit, k_offset)
    # For a given (i, j_o, k_o, k_off), the term has:
    #   U[a,r] = u_i  (depends only on i)
    #   V[b,r] = v_{j_o}(b's j-component, k_o, k_off)
    #   W[c,r] = w_{k_o}(c's k-component, i, k_off)
    #
    # This gives a constrained set of rank-1 tensors. We find linear
    # combinations that sum to the matmul tensor.

    # Define term templates
    # j_orbit: 0 = {0,2}, 1 = {1,3}
    # k_orbit: 0 = {0}, 1 = {1,2,3,4}
    # k_off: 0 for orbit 0, 0..3 for orbit 1 (representing k=1,2,3,4)

    # For each template, define the (U_col, V_col, W_col) as functions of (i, k_off)
    # The j-index in V is fixed by the orbit; we sum over j in orbit
    # The k-index in W is fixed by orbit+offset

    def make_term(i, j_orbit, k_orbit, k_off):
        """
        Return (U_col, V_col, W_col) for a structured rank-1 term.
        U_col: length 12, V_col: length 20, W_col: length 15
        """
        U = [Fraction(0)] * Nm
        V = [Fraction(0)] * Mp
        W = [Fraction(0)] * Pn

        # U: nonzero at positions (i, j) for j in j_orbit
        for j in range(m):
            if (j % 2 == 0) == (j_orbit == 0):
                U[i * m + j] = Fraction(1)

        # V: nonzero at positions (j, k) for j in j_orbit, k = k_orbit_value
        k_val = 0 if k_orbit == 0 else (k_off + 1)  # k_off 0..3 -> k 1..4
        for j in range(m):
            if (j % 2 == 0) == (j_orbit == 0):
                V[j * p + k_val] = Fraction(1)

        # W: nonzero at positions (k, i') for k = k_val, all i'
        for i2 in range(n):
            W[k_val * n + i2] = Fraction(1)

        return U, V, W

    # Enumerate all template terms
    templates = []
    for i in range(n):
        for j_o in range(2):
            for k_o in range(2):
                if k_o == 0:
                    k_offs = [0]
                else:
                    k_offs = range(4)
                for k_off in k_offs:
                    templates.append((i, j_o, k_o, k_off))

    T_count = len(templates)

    # Precompute tensor products for each template
    # tp[r, a, b, c] = U[r,a] * V[r,b] * W[r,c]
    # We only need tp at positions where tensor is nonzero (60 positions)
    # But the system must hold at ALL (a,b,c) in the 12x20x15 space
    # However, the matmul tensor is zero except at 60 positions,
    # so we need: sum_r w_r * tp_r[a,b,c] = T[a,b,c] for all a,b,c
    # This is 12*20*15 = 3600 equations.

    # Precompute for each template r, the list of (a,b,c, value) where U*V*W != 0
    tp_sparse = []
    for idx, (i, j_o, k_o, k_off) in enumerate(templates):
        U, V, W = make_term(i, j_o, k_o, k_off)
        entries = []
        for a in range(Nm):
            if U[a] == 0:
                continue
            for b in range(Mp):
                if V[b] == 0:
                    continue
                for c in range(Pn):
                    if W[c] == 0:
                        continue
                    val = U[a] * V[b] * W[c]
                    entries.append((a, b, c, val))
        tp_sparse.append(entries)

    # Collect all equations: for each (a,b,c), sum over r of w_r * U_r[a]*V_r[b]*W_r[c] = T[a,b,c]
    # Build: A[eq, var] where A[eq, r] = U_r[a]*V_r[b]*W_r[c], rhs[eq] = T[a,b,c]
    # This is a 3600 x 34 linear system.

    # But we can be smarter: many (a,b,c) have T=0 and may not constrain much.
    # Let's build the full system but use sparse representation.

    # Map (a,b,c) -> equation index
    eq_index = {}
    all_positions = []
    for a in range(Nm):
        for b in range(Mp):
            for c in range(Pn):
                eq_index[(a, b, c)] = len(all_positions)
                all_positions.append((a, b, c))
    num_eqs = len(all_positions)  # 3600

    # Build sparse matrix A as list of (eq, var, val)
    A_sparse = []
    for r, entries in enumerate(tp_sparse):
        for (a, b, c, val) in entries:
            eq = eq_index[(a, b, c)]
            A_sparse.append((eq, r, val))

    # RHS
    rhs = [Fraction(0)] * num_eqs
    for (a, b, c), tval in tensor.items():
        eq = eq_index[(a, b, c)]
        rhs[eq] = tval

    # Now solve A * w = rhs where A is num_eqs x T_count
    # Use Gaussian elimination over Q with row reduction
    # T_count = 34 unknowns, num_eqs = 3600

    # Build augmented matrix in row-major sparse form
    # For efficiency, use a dict: A[eq] = {var: val}
    A_rows = [dict() for _ in range(num_eqs)]
    for (eq, var, val) in A_sparse:
        A_rows[eq][var] = A_rows[eq].get(var, Fraction(0)) + val

    # Gaussian elimination
    # We want to find w such that A*w = rhs
    # Use standard GE with partial pivoting (by absolute value of Fraction)

    # Build full matrix as list of lists (3600 x 35) - might be large but ok
    # Actually, let's do sparse GE

    # Sort rows by number of nonzeros, do elimination
    # Better: use the fact that we only have 34 variables
    # Pick 34 independent rows and solve

    # Find a basis of rows
    # Start with all rows, reduce to RREF

    # For a 34-variable system, pick rows greedily
    # Row = (coeff_dict, rhs_val)
    rows = []
    for eq in range(num_eqs):
        if A_rows[eq]:
            rows.append((dict(A_rows[eq]), rhs[eq]))

    # Reduce: for each variable, find a row where it's the leading term
    pivot_rows = []
    used_vars = set()
    for var in range(T_count):
        # Find a row where var is present and no previously used var is present
        best_row = None
        best_idx = None
        for idx, (coeffs, r) in enumerate(rows):
            if var in coeffs:
                # Check that no used_var is in coeffs
                conflict = False
                for uv in used_vars:
                    if uv in coeffs:
                        conflict = True
                        break
                if not conflict:
                    best_row = coeffs
                    best_idx = idx
                    break
        if best_row is not None:
            # Normalize: divide row by best_row[var]
            pivot_val = best_row[var]
            norm_row = {k: v / pivot_val for k, v in best_row.items()}
            norm_rhs = rows[best_idx][1] / pivot_val
            pivot_rows.append((norm_row, norm_rhs, var))
            used_vars.add(var)
            # Eliminate var from all other rows
            for idx2 in range(len(rows)):
                if idx2 == best_idx:
                    continue
                if var in rows[idx2][0]:
                    factor = rows[idx2][0][var]
                    new_coeffs = dict(rows[idx2][0])
                    for k, v in norm_row.items():
                        new_coeffs[k] = new_coeffs.get(k, Fraction(0)) + factor * (-v)
                        if new_coeffs[k] == 0:
                            del new_coeffs[k]
                    new_rhs = rows[idx2][1] - factor * norm_rhs
                    rows[idx2] = (new_coeffs, new_rhs)
            rows[best_idx] = (norm_row, norm_rhs)

    num_pivots = len(pivot_rows)

    if num_pivots < T_count:
        # Underdetermined or inconsistent - check consistency
        # For now, if we have at least as many pivots as we need, use a particular solution
        pass

    # Extract solution from pivot rows
    # Each pivot row: sum_k norm_row[k] * w[k] = norm_rhs
    # Since we eliminated, each pivot row has exactly one leading var
    # But other vars may still be present
    # For a particular solution, set free vars to 0

    # Determine which vars are pivots
    pivot_var_map = {}
    for pr, prhs, pvar in pivot_rows:
        pivot_var_map[pvar] = (pr, prhs)

    # Free vars = all vars not in pivot_var_map
    free_vars = set(range(T_count)) - set(pivot_var_map.keys())

    # Set free vars to 0, solve for pivot vars
    # Each pivot equation: pvar_coeff * w[pvar] + sum(other in row) = rhs
    # Since we reduced, the row for pvar has pvar with coeff 1
    # But other pivot vars may appear

    # Back-substitute
    w_solution = {var: Fraction(0) for var in free_vars}
    # Sort pivot rows by var index descending for back-substitution
    pivot_sorted = sorted(pivot_rows, key=lambda x: x[2], reverse=True)
    for norm_row, norm_rhs, pvar in pivot_sorted:
        total = norm_rhs
        for k, v in norm_row.items():
            if k == pvar:
                continue
            if k in w_solution:
                total -= v * w_solution[k]
            else:
                # k is another pivot var not yet solved - shouldn't happen in proper RREF
                # but if it does, we have a problem
                total -= v * Fraction(0)
        w_solution[pvar] = total

    w = [w_solution[var] for var in range(T_count)]

    # Verify the solution
    def verify(w_vec):
        for (a, b, c), tval in tensor.items():
            s = Fraction(0)
            for r in range(T_count):
                if w_vec[r] == 0:
                    continue
                U, V, W = make_term(*templates[r])
                s += w_vec[r] * U[a] * V[b] * W[c]
            if s != tval:
                return False
        # Check zeros
        for pos in all_positions:
            a, b, c = pos
            if pos in tensor:
                continue
            s = Fraction(0)
            for r in range(T_count):
                if w_vec[r] == 0:
                    continue
                U, V, W = make_term(*templates[r])
                s += w_vec[r] * U[a] * V[b] * W[c]
            if s != 0:
                return False
        return True

    if verify(w):
        # Build U, V, W matrices
        rank = T_count
        U_mat = [[Fraction(0)] * rank for _ in range(Nm)]
        V_mat = [[Fraction(0)] * rank for _ in range(Mp)]
        W_mat = [[Fraction(0)] * rank for _ in range(Pn)]

        for r in range(T_count):
            U, V, W = make_term(*templates[r])
            wr = w[r]
            if wr == 0:
                continue
            for a in range(Nm):
                if U[a] != 0:
                    U_mat[a][r] = U[a] * wr
            for b in range(Mp):
                if V[b] != 0:
                    V_mat[b][r] = V[b] * wr
            for c in range(Pn):
                if W[c] != 0:
                    W_mat[c][r] = W[c] * wr

        # Convert to int or float
        def to_num(fr):
            if fr.denominator == 1:
                return int(fr)
            else:
                return float(fr)

        U_out = [[to_num(fr) for fr in row] for row in U_mat]
        V_out = [[to_num(fr) for fr in row] for row in V_mat]
        W_out = [[to_num(fr) for fr in row] for row in W_mat]

        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        # Fallback: schoolbook rank 60
        rank = n * m * p
        U = [[0] * rank for _ in range(Nm)]
        V = [[0] * rank for _ in range(Mp)]
        W = [[0] * rank for _ in range(Pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
