"""Combined program: algebraic-structured ALS with zero-constraint enforcement."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _verify(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify exact decomposition including zero constraints."""
    nm, mp, pn = n * m, m * p, p * n
    # Build set of valid (a, b, c) indices
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i * m + j, j * p + k, k * n + i))
    
    # Check valid entries equal 1
    for (a, b, c) in valid:
        s = 0.0
        for r in range(rank):
            s += U[a, r] * V[b, r] * W[c, r]
        if abs(s - 1.0) > tol:
            return False
    
    # Check invalid entries equal 0 (sample for efficiency, but for correctness check all)
    # For small dimensions, check all
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in valid:
                    s = 0.0
                    for r in range(rank):
                        s += U[a, r] * V[b, r] * W[c, r]
                    if abs(s) > tol:
                        return False
    return True

def _snap_to_half_int(x):
    x2 = x * 2.0
    r = round(x2)
    return r / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook solution as guaranteed fallback
    rank_school = n * m * p
    U_sch = np.zeros((nm, rank_school))
    V_sch = np.zeros((mp, rank_school))
    W_sch = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i * m + j, col] = 1.0
                V_sch[j * p + k, col] = 1.0
                W_sch[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_sch, V_sch, W_sch
    best_rank = rank_school
    
    # Lower bound
    lb = max(n * m, m * p, p * n)
    
    # Precompute valid indices
    valid_indices = []
    valid_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a, b, c = i * m + j, j * p + k, k * n + i
                valid_indices.append((a, b, c))
                valid_set.add((a, b, c))
    
    # Invalid indices (for zero constraints)
    invalid_indices = []
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in valid_set:
                    invalid_indices.append((a, b, c))
    
    # Try ranks from lower bound upward
    for rank in range(lb, best_rank + 1):
        if time.time() > deadline - 15:
            break
        
        if rank >= best_rank:
            continue
        
        found = False
        # Multiple random seeds for robustness
        for seed in range(50):
            if time.time() > deadline - 5:
                break
            
            rng = np.random.RandomState(rank * 100000 + seed)
            
            # Structured initialization: use schoolbook-like patterns
            # Initialize with random but structured values
            U = rng.randn(nm, rank) * 0.5
            V = rng.randn(mp, rank) * 0.5
            W = rng.randn(pn, rank) * 0.5
            
            # ALS iterations with zero-constraint awareness
            for it in range(80):
                if time.time() > deadline - 3:
                    break
                
                # Update U: for each row a, solve for U[a,:]
                # Constraints: for each valid (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] contributes to 1
                # For each invalid (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = 0
                # This is a linear system in U[a,:] if V and W are fixed
                for a in range(nm):
                    # Gather all equations involving U[a,:]
                    # Valid: for each (a,b,c) in valid where first index is a
                    # Invalid: for each (a,b,c) in invalid where first index is a
                    rows = []
                    rhs = []
                    for (va, vb, vc) in valid_indices:
                        if va == a:
                            row = np.array([V[vb, r] * W[vc, r] for r in range(rank)])
                            rows.append(row)
                            rhs.append(1.0)
                    for (ia, ib, ic) in invalid_indices:
                        if ia == a:
                            row = np.array([V[ib, r] * W[ic, r] for r in range(rank)])
                            rows.append(row)
                            rhs.append(0.0)
                    
                    if rows:
                        A = np.array(rows)
                        b_vec = np.array(rhs)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                            U[a, :] = sol
                        except:
                            pass
                
                # Update V similarly
                for b in range(mp):
                    rows = []
                    rhs = []
                    for (va, vb, vc) in valid_indices:
                        if vb == b:
                            row = np.array([U[va, r] * W[vc, r] for r in range(rank)])
                            rows.append(row)
                            rhs.append(1.0)
                    for (ia, ib, ic) in invalid_indices:
                        if ib == b:
                            row = np.array([U[ia, r] * W[ic, r] for r in range(rank)])
                            rows.append(row)
                            rhs.append(0.0)
                    
                    if rows:
                        A = np.array(rows)
                        b_vec = np.array(rhs)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                            V[b, :] = sol
                        except:
                            pass
                
                # Update W similarly
                for c in range(pn):
                    rows = []
                    rhs = []
                    for (va, vb, vc) in valid_indices:
                        if vc == c:
                            row = np.array([U[va, r] * V[vb, r] for r in range(rank)])
                            rows.append(row)
                            rhs.append(1.0)
                    for (ia, ib, ic) in invalid_indices:
                        if ic == c:
                            row = np.array([U[ia, r] * V[ib, r] for r in range(rank)])
                            rows.append(row)
                            rhs.append(0.0)
                    
                    if rows:
                        A = np.array(rows)
                        b_vec = np.array(rhs)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                            W[c, :] = sol
                        except:
                            pass
                
                # Check convergence every 5 iterations
                if it % 5 == 4:
                    # Quick check on valid entries
                    max_err = 0.0
                    for (a, b, c) in valid_indices:
                        s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                        err = abs(s - 1.0)
                        if err > max_err:
                            max_err = err
                    # Quick check on a sample of invalid entries
                    if len(invalid_indices) > 100:
                        sample = invalid_indices[::10]
                    else:
                        sample = invalid_indices
                    for (a, b, c) in sample:
                        s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                        err = abs(s)
                        if err > max_err:
                            max_err = err
                    
                    if max_err < 1e-6:
                        # Full verification
                        if _verify(U, V, W, n, m, p, rank):
                            # Try to snap to half-integers
                            U_snap = np.array([[_snap_to_half_int(U[i, r]) for r in range(rank)] for i in range(nm)])
                            V_snap = np.array([[_snap_to_half_int(V[i, r]) for r in range(rank)] for i in range(mp)])
                            W_snap = np.array([[_snap_to_half_int(W[i, r]) for r in range(rank)] for i in range(pn)])
                            
                            if _verify(U_snap, V_snap, W_snap, n, m, p, rank):
                                best_U, best_V, best_W = U_snap, V_snap, W_snap
                                best_rank = rank
                            else:
                                best_U, best_V, best_W = U, V, W
                                best_rank = rank
                            found = True
                            break
            
            if found:
                break
    
    # Convert to list format
    def _to_list(arr):
        return [[float(x) for x in row] for row in np.asarray(arr)]
    
    return {"U": _to_list(best_U), "V": _to_list(best_V), "W": _to_list(best_W)}
# EVOLVE-BLOCK-END
