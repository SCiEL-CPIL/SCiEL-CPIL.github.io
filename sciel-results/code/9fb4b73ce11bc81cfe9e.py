"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Target triples (a,b,c) that must evaluate to 1
    target = [(i * m + j, j * p + k, k * n + i)
              for i in range(n) for j in range(m) for k in range(p)]
    ta = np.array([t[0] for t in target], dtype=np.int64)
    tb = np.array([t[1] for t in target], dtype=np.int64)
    tc = np.array([t[2] for t in target], dtype=np.int64)
    n_targets = len(target)

    def make_schoolbook():
        R = n * m * p
        U = np.zeros((nm, R), dtype=int)
        V = np.zeros((mp, R), dtype=int)
        W = np.zeros((pn, R), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def verify(U, V, W, R):
        # Check if entries are integer or half-integer
        # The problem statement says "integer/half-integer entries".
        # Usually this means 2*entry is integer.
        # Let's assume standard integer check first, but allow 0.5.
        # The verification equation is sum(U*V*W) == 1.
        vals = (U[ta, :R] * V[tb, :R] * W[tc, :R]).sum(axis=1)
        return bool(np.allclose(vals, 1.0, atol=1e-6))

    def check_half_int(A):
        # Check if all entries are integer or half-integer
        # 2*A should be integer
        return bool(np.allclose(2 * A, np.round(2 * A), atol=1e-6))

    def supports(U, V, col):
        us = tuple(np.nonzero(U[:, col])[0])
        vs = tuple(np.nonzero(V[:, col])[0])
        return us, vs

    def remove_column(U, V, W, R, c):
        idx = np.arange(R)
        mask = idx != c
        return U[:, mask].copy(), V[:, mask].copy(), W[:, mask].copy(), R - 1

    def merge_group(U, V, W, R, members):
        """Merge all columns in 'members' (sharing same U,V support) into one."""
        keep = members[0]
        drop = [c for c in members if c != keep]
        # New W column = sum of W columns for all members
        new_w = sum(W[:, c] for c in members)
        # Build new matrices without dropped columns
        cols_to_keep = [c for c in range(R) if c not in drop]
        Wnew = np.zeros((pn, R - len(drop)), dtype=int)
        Unew = U[:, cols_to_keep].copy()
        Vnew = V[:, cols_to_keep].copy()
        for new_c, old_c in enumerate(cols_to_keep):
            Wnew[:, new_c] = W[:, old_c]
        pos = cols_to_keep.index(keep)
        Wnew[:, pos] = new_w
        return Unew, Vnew, Wnew, R - len(drop)

    def split_column(U, V, W, R, c, w1, w2):
        """Split column c into two columns with W values w1 and w2 (w1+w2 = W[:,c])."""
        # Add a new column after c
        Unew = np.zeros((nm, R + 1), dtype=int)
        Vnew = np.zeros((mp, R + 1), dtype=int)
        Wnew = np.zeros((pn, R + 1), dtype=int)
        for i in range(R):
            if i < c:
                Unew[:, i] = U[:, i]
                Vnew[:, i] = V[:, i]
                Wnew[:, i] = W[:, i]
            elif i == c:
                Unew[:, i] = U[:, i]
                Vnew[:, i] = V[:, i]
                Wnew[:, i] = w1
            else:
                Unew[:, i + 1] = U[:, i]
                Vnew[:, i + 1] = V[:, i]
                Wnew[:, i + 1] = W[:, i]
        # Insert new column at c+1
        Unew[:, c + 1] = U[:, c]
        Vnew[:, c + 1] = V[:, c]
        Wnew[:, c + 1] = w2
        return Unew, Vnew, Wnew, R + 1

    def greedy_search(U, V, W, R, rng, deadline):
        """Iteratively apply removals and merges until stuck."""
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            # Try removals
            cols = list(range(R))
            rng.shuffle(cols)
            for c in cols:
                if time.time() > deadline:
                    return U, V, W, R
                Ut, Vt, Wt, Rt = remove_column(U, V, W, R, c)
                if verify(Ut, Vt, Wt, Rt):
                    U, V, W, R = Ut, Vt, Wt, Rt
                    improved = True
                    break

            if not improved:
                # Try group merges
                sup = {}
                groups = {}
                for c in range(R):
                    s = supports(U, V, c)
                    sup[c] = s
                    groups.setdefault(s, []).append(c)
                merged_any = False
                for key, members in groups.items():
                    if time.time() > deadline:
                        return U, V, W, R
                    if len(members) < 2:
                        continue
                    Ut, Vt, Wt, Rt = merge_group(U, V, W, R, members)
                    if verify(Ut, Vt, Wt, Rt):
                        U, V, W, R = Ut, Vt, Wt, Rt
                        merged_any = True
                        break
                if merged_any:
                    improved = True

            if not improved and R > 1:
                # Try splits to escape local minima
                cols = list(range(R))
                rng.shuffle(cols)
                for c in cols:
                    if time.time() > deadline:
                        return U, V, W, R
                    wc = W[:, c]
                    nz = np.nonzero(wc)[0]
                    if len(nz) < 2:
                        continue
                    # Try splitting: assign one nonzero element to w2
                    for idx in nz:
                        if time.time() > deadline:
                            return U, V, W, R
                        w2 = np.zeros(pn, dtype=int)
                        w2[idx] = wc[idx]
                        w1 = wc - w2
                        Ut, Vt, Wt, Rt = split_column(U, V, W, R, c, w1, w2)
                        if verify(Ut, Vt, Wt, Rt):
                            U, V, W, R = Ut, Vt, Wt, Rt
                            improved = True
                            break
                    if improved:
                        break

        return U, V, W, R

    def try_random_construction(R, rng, deadline):
        """Try to construct a decomposition of rank R using random U, V and solving for W."""
        if time.time() > deadline:
            return None
        
        # We need to solve A w = 1
        # A has shape (n_targets, R * pn)
        # w is flattened W
        # Let's build A
        # A[t, r * pn + idx] = U[ta[t], r] * V[tb[t], r] if tc[t] == idx else 0?
        # No, the equation is sum_r U[ta[t], r] V[tb[t], r] W[tc[t], r] = 1
        # So for a fixed t, the variables are W[tc[t], r] for r in 0..R-1.
        # Coefficient for W[tc[t], r] is U[ta[t], r] * V[tb[t], r].
        
        # This system is coupled across t because W[tc[t], r] appears in multiple equations?
        # No, W[k, i, r] appears in equations where tc[t] = k*n+i.
        # So the system is block diagonal if we group by (k,i)?
        # For a fixed (k,i), the equations are for all j.
        # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all j.
        # This is a system for W[k*n+i, :] of size R.
        # Number of equations for this block: m (which is 2).
        # So for each (k,i), we have a 2xR system for W[k*n+i, :].
        
        # We can solve for W independently for each (k,i) block!
        # This makes construction much easier.
        
        # Generate random U, V
        U = np.zeros((nm, R), dtype=int)
        V = np.zeros((mp, R), dtype=int)
        
        # To ensure solvability, we might need specific structures.
        # Let's try random 0/1 entries with low density.
        density = 0.5
        U = (rng.random((nm, R)) < density).astype(int)
        V = (rng.random((mp, R)) < density).astype(int)
        
        # Solve for W
        W = np.zeros((pn, R), dtype=float)
        
        for k in range(p):
            for i in range(n):
                # Equations for j in range(m)
                # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
                A_block = np.zeros((m, R), dtype=float)
                for j in range(m):
                    A_block[j, :] = U[i*m+j, :] * V[j*p+k, :]
                
                b_block = np.ones(m, dtype=float)
                
                # Solve A_block w = b_block
                # Since m=2 and R is likely > 2, it's underdetermined.
                # We want a solution with integer/half-integer entries.
                # Let's use lstsq to get min norm solution.
                w_sol, residuals, rank, sv = np.linalg.lstsq(A_block, b_block, rcond=None)
                
                # Check if w_sol is close to half-integers
                if not np.allclose(2 * w_sol, np.round(2 * w_sol), atol=1e-5):
                    # Try a different random U, V? Or just fail this block.
                    # If we fail, the whole construction fails.
                    return None
                
                W[k*n+i, :] = w_sol
        
        # Verify the whole thing
        if verify(U, V, W, R) and check_half_int(W):
            # Convert to int if possible, or keep float?
            # The interface expects integer/half-integer.
            # The return type is list of lists.
            # We should probably return integers if they are integers, or floats if half-integers.
            # But the seed program returns int.
            # Let's check if we can scale to integers.
            # If all W are half-integers, 2W are integers.
            # But U, V are integers.
            # The product U*V*W will be half-integers.
            # The sum must be 1.
            # So this is valid.
            
            # However, the evaluator might expect integers?
            # "integer/half-integer entries"
            # So floats are allowed if they are x.0 or x.5.
            
            return U, V, W, R
        return None

    U, V, W = make_schoolbook()
    R = U.shape[1]
    best_U, best_V, best_W, best_R = U.copy(), V.copy(), W.copy(), R

    # Try random constructions for lower ranks
    # Lower bound is max(nm, mp, pn) = 26.
    # Let's try ranks 30 to 50.
    for R_try in range(30, 51):
        if time.time() > deadline:
            break
        for attempt in range(5):
            if time.time() > deadline:
                break
            rng = random.Random(12345 + R_try * 7 + attempt)
            res = try_random_construction(R_try, rng, deadline)
            if res is not None:
                U_c, V_c, W_c, R_c = res
                # Polish with greedy search
                U_c, V_c, W_c, R_c = greedy_search(U_c, V_c, W_c, R_c, rng, deadline)
                if R_c < best_R:
                    best_U, best_V, best_W, best_R = U_c.copy(), V_c.copy(), W_c.copy(), R_c
                break # Found a valid one for this rank, move to next lower rank? 
                      # Actually, if we found one for R_try, we should try to reduce it further.
                      # The greedy search above does that.
                      # So we can break the attempt loop for this R_try.

    # Also run the original greedy search from schoolbook as a fallback
    if time.time() < deadline:
        rng = random.Random(99999)
        U, V, W = make_schoolbook()
        R = U.shape[1]
        U, V, W, R = greedy_search(U, V, W, R, rng, deadline)
        if R < best_R:
            best_U, best_V, best_W, best_R = U.copy(), V.copy(), W.copy(), R

    # Convert to lists of lists
    def to_list(A):
        return A.tolist()

    return {"U": to_list(best_U[:, :best_R]),
            "V": to_list(best_V[:, :best_R]),
            "W": to_list(best_W[:, :best_R])}
# EVOLVE-BLOCK-END
