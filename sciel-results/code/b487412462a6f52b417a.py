"""Fresh design: search for a rank-minimal exact tensor decomposition of 3x5x7 matmul.
Uses Strassen-style block recurrences + local rank-reduction (term merging/cancellation)."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [3, 5, 7]
_DEADLINE = 150.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _fracs(x):
    # convert a matrix/list-of-lists of numbers to Fraction, handling ints/floats/Fraction
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, float):
        return Fraction(x).limit_denominator(10**6)
    if hasattr(x, "__iter__"):
        return [_fracs(r) for r in x]
    return x

def _matmul(A, B):
    # A: (n x m), B: (m x p) -> (n x p), over Fractions
    n = len(A); m = len(A[0]); p = len(B[0])
    C = [[Fraction(0) for _ in range(p)] for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for k in range(m):
            aik = Ai[k]
            if aik:
                Bk = B[k]
                for j in range(p):
                    Ci[j] += aik * Bk[j]
    return C

def _matmul_check(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matmul.
    U: (n*m x rank), V: (m*p x rank), W: (p*n x rank).
    For each output (i,k): sum_j A[i,j] B[j,k] must equal sum_c U[i*m+j,c]*V[j*p+k,c]... 
    Actually the tensor: T[i,j,k] (A[i,j]*B[j,k] contributes to C[i,k]).
    Decomposition: C[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_l V[j*p+l,c]*B[j,l])? 
    Standard: C[i,k] = sum_c U[i*m+?, ...]. Let's use the conventional form:
    C[i,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) * (sum_l V[j*p+l,c] B[j,l])? No.
    The bilinear form: C[i,k] = sum_c ( (U[:,c]^T A)_i * (V[:,c]^T B)_k )
    where (U[:,c]^T A)_i = sum_j U[i*m+j,c]*A[i,j] and (V[:,c]^T B)_k = sum_l V[j*p+l,c]*B[j,l].
    We verify by checking that for all (i,j,k), the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    That coefficient = U[i*m+j,c]*V[j*p+k,c] summed over c must equal 1 for all i,j,k.
    """
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c]
                # W[k*n+i][c] should relate; in the standard trace convention the product
                # U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] summed over c = 1 for all i,j,k.
    # Use the 3-way convention: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _random_rank1(n, m, p, rng):
    """Generate a random rank-1 tensor (trilinear form) with small integer entries."""
    U = [0 if rng.random() < 0.5 else (1 if rng.random() < 0.5 else -1) for _ in range(n*m)]
    V = [0 if rng.random() < 0.5 else (1 if rng.random() < 0.5 else -1) for _ in range(m*p)]
    W = [0 if rng.random() < 0.5 else (1 if rng.random() < 0.5 else -1) for _ in range(p*n)]
    return U, V, W

def _tensor_to_dict(U, V, W, n, m, p):
    """Compute the trilinear tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]."""
    r = len(U[0])
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 0:
                    T[(i,j,k)] = s
    return T

def _target_tensor(n, m, p):
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i,j,k)] = Fraction(1)
    return T

def _try_reduce(U, V, W, n, m, p, rng, max_iters=2000):
    """Try to reduce rank by merging/canceling terms. Returns (U,V,W) possibly lower rank.
    Uses float-based search for speed, and exact verification for correctness."""
    import numpy as np
    
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    r = len(U[0])
    
    # Convert to numpy float arrays for fast tensor computation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def _col_tensor_np(c):
        # Returns flat array of shape (n*m*p)
        # U_np[i*m+j, c], V_np[j*p+k, c], W_np[k*n+i, c]
        # We need to sum over c? No, this is for a single column c.
        # The tensor contribution of column c is the outer product of the vectors?
        # No, the tensor T[i,j,k] for a single rank-1 term is u_i * v_j * w_k?
        # Wait, the decomposition is C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) (sum_l V[j,l,c] B[j,l])?
        # No, the standard form is:
        # C[i,k] = sum_c U[i, *, c] A[i, :] * V[:, k, c]^T B[:, k]?
        # Let's stick to the trilinear form:
        # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        # For a single column c, the contribution to T[i,j,k] is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
        # This is NOT a rank-1 tensor in the (i,j,k) space in the simple outer product sense
        # unless we view it as a 3-way array.
        # Actually, for a fixed c, the term is U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i].
        # This IS a rank-1 3-tensor (in the sense that it factors into 3 vectors).
        # So the sum of two such terms is rank-1 if their sum also factors.
        
        # To compute the flat tensor for column c:
        # T_flat[idx] where idx = (i*m+j)*p + k? No, the indices are coupled.
        # Let's just compute the full 3D array for the pair.
        pass

    def _check_rank1_merge(c1, c2):
        """Check if the sum of column c1 and c2 tensors is rank-1.
        Returns (u_vec, v_vec, w_vec) as Fractions if yes, else None."""
        
        # Compute the sum tensor T_sum[i,j,k]
        # T_sum[i,j,k] = U[i*m+j, c1]V[j*p+k, c1]W[k*n+i, c1] + U[i*m+j, c2]V[j*p+k, c2]W[k*n+i, c2]
        
        # We can check if this 3-tensor is rank-1.
        # A 3-tensor X is rank 1 if there exist vectors u, v, w such that X[i,j,k] = u[i]v[j]w[k].
        # Note: Our indices are flattened.
        # u index: i*m+j (size n*m)
        # v index: j*p+k (size m*p)
        # w index: k*n+i (size p*n)
        # The tensor is defined on the "fiber product" of these indices?
        # No, the tensor is defined on (i,j,k) with constraints?
        # Actually, the entries are T[i,j,k] for all i,j,k.
        # The "rank" here refers to the tensor rank of the 3-way array T[i,j,k].
        # Wait, the standard definition of rank-1 for a 3-tensor T[i,j,k] is T[i,j,k] = u_i v_j w_k.
        # In our case, the "vectors" are indexed by (i,j), (j,k), (k,i).
        # This is a "slice" of the full tensor.
        # Specifically, we are looking for u_{i,j}, v_{j,k}, w_{k,i} such that
        # T[i,j,k] = u_{i,j} v_{j,k} w_{k,i}.
        
        # Let's extract the values.
        # To speed up, we can use a random subset of (i,j,k) to test.
        # If it fails the random test, it's not rank 1.
        # If it passes, we do the full exact check.
        
        # Generate random indices
        indices = []
        for _ in range(50):
            i = rng.randint(0, n-1)
            j = rng.randint(0, m-1)
            k = rng.randint(0, p-1)
            indices.append((i,j,k))
            
        # Compute values at these indices for the sum
        vals = []
        for (i,j,k) in indices:
            val1 = U_np[i*m+j, c1] * V_np[j*p+k, c1] * W_np[k*n+i, c1]
            val2 = U_np[i*m+j, c2] * V_np[j*p+k, c2] * W_np[k*n+i, c2]
            vals.append(val1 + val2)
            
        # Check for consistency with a rank-1 model
        # We need to find u, v, w.
        # Pick an anchor (i0, j0, k0) with non-zero val.
        anchor_idx = None
        for idx, val in enumerate(vals):
            if abs(val) > 1e-9:
                anchor_idx = idx
                break
        if anchor_idx is None:
            # All zero? Then the sum is zero, which is rank 0 (or can be removed).
            # We can just remove one column? No, if sum is 0, we can replace both with nothing?
            # If T1 + T2 = 0, then T1 = -T2. They are linearly dependent.
            # We can keep one and negate it? No, the rank is the number of terms.
            # If T1 + T2 = 0, we can drop both? No, the total tensor must remain the same.
            # If T1 + T2 = 0, then removing both doesn't change the sum. So rank decreases by 2.
            # But we need to return a valid decomposition.
            # If we drop both, we must ensure the rest is correct.
            # The function returns a new U,V,W.
            # If we detect this, we can return a special signal or handle it outside.
            # For now, let's assume we want to merge into 1 term. If sum is 0, we can't merge into 1 non-zero term.
            # We can just return None and let the caller handle "remove both" separately?
            # Or, we can return a zero vector? No.
            # Let's just return None for now, and handle "remove zero sum" separately.
            return None
            
        i0, j0, k0 = indices[anchor_idx]
        val0 = vals[anchor_idx]
        
        # Assume w[k0*n+i0] = 1 (gauge).
        # Then u[i0*m+j0] * v[j0*p+k0] = val0.
        # We can set u[i0*m+j0] = 1, v[j0*p+k0] = val0.
        
        u = {}
        v = {}
        w = {}
        u[i0*m+j0] = 1.0
        v[j0*p+k0] = val0
        w[k0*n+i0] = 1.0
        
        # Propagate to other indices
        # For each (i,j,k), we have u[i*m+j] * v[j*p+k] * w[k*n+i] = val
        # If two are known, solve the third.
        
        # We will do this for all 50 random points.
        # If we find a contradiction, return None.
        
        # To be robust, we should use more points or all points if n*m*p is small.
        # n*m*p = 105. It's small enough to check all points in the exact phase.
        # For the float phase, 50 points is a good filter.
        
        for idx, (i,j,k) in enumerate(indices):
            val = vals[idx]
            ui = u.get(i*m+j)
            vj = v.get(j*p+k)
            wk = w.get(k*n+i)
            
            # Count knowns
            knowns = 0
            if ui is not None: knowns += 1
            if vj is not None: knowns += 1
            if wk is not None: knowns += 1
            
            if knowns == 3:
                # Verify
                if abs(ui * vj * wk - val) > 1e-5 * max(1, abs(val)):
                    return None
            elif knowns == 2:
                # Solve for the missing one
                if ui is None:
                    if vj != 0 and wk != 0:
                        u[i*m+j] = val / (vj * wk)
                    else:
                        # If vj or wk is 0, then val must be 0.
                        if abs(val) > 1e-9:
                            return None
                        # u can be anything, set to 0?
                        u[i*m+j] = 0.0
                elif vj is None:
                    if ui != 0 and wk != 0:
                        v[j*p+k] = val / (ui * wk)
                    else:
                        if abs(val) > 1e-9:
                            return None
                        v[j*p+k] = 0.0
                elif wk is None:
                    if ui != 0 and vj != 0:
                        w[k*n+i] = val / (ui * vj)
                    else:
                        if abs(val) > 1e-9:
                            return None
                        w[k*n+i] = 0.0
            elif knowns == 1:
                # Cannot solve yet, skip
                pass
            else:
                # No knowns, skip
                pass
        
        # If we passed the random test, do the EXACT check on all (i,j,k)
        # and construct the exact Fractions.
        
        # We need to determine u, v, w for all indices exactly.
        # We can use the float values as guides, but we must solve exactly.
        # Actually, if the tensor is exactly rank 1, the ratios will be rational.
        # We can reconstruct the exact vectors by picking a pivot and propagating with Fractions.
        
        # Let's collect all (i,j,k) and their exact values.
        # We already have U, V, W as Fractions.
        
        # We need to solve for u_exact, v_exact, w_exact.
        # This is a system of linear equations? No, it's multiplicative.
        # T[i,j,k] = u[i,j] v[j,k] w[k,i].
        
        # We can fix w[k0*n+i0] = 1.
        # Then u[i0*m+j0] * v[j0*p+k0] = T[i0,j0,k0].
        
        # Let's try to propagate exactly.
        u_exact = {}
        v_exact = {}
        w_exact = {}
        
        # Use the float solution to pick a non-zero pivot?
        # The float solution gave us a consistent set.
        # We can just use the first non-zero entry from the float solution as the anchor.
        
        # Find an anchor (i,j,k) where the float value is non-zero.
        anchor = None
        for (i,j,k), val in zip(indices, vals):
            if abs(val) > 1e-9:
                anchor = (i,j,k)
                break
        if anchor is None:
            return None # Should have been caught above
            
        ai, aj, ak = anchor
        # Exact value
        val_exact = U[ai*m+aj][c1]*V[aj*p+ak][c1]*W[ak*n+ai][c1] + U[ai*m+aj][c2]*V[aj*p+ak][c2]*W[ak*n+ai][c2]
        
        if val_exact == 0:
            # Contradiction? Float said non-zero, exact is zero?
            # Maybe due to floating point error?
            # Let's check if all values are zero.
            all_zero = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val1 = U[i*m+j][c1]*V[j*p+k][c1]*W[k*n+i][c1]
                        val2 = U[i*m+j][c2]*V[j*p+k][c2]*W[k*n+i][c2]
                        if val1 + val2 != 0:
                            all_zero = False
                            break
                    if not all_zero: break
                if not all_zero: break
            if all_zero:
                # Return a special marker? Or just None?
                # If sum is 0, we can remove both columns.
                # Let's return None and handle "remove both" in the main loop.
                return None
            # Otherwise, it's a numerical issue. Return None to be safe.
            return None
            
        # Set gauge
        u_exact[ai*m+aj] = Fraction(1)
        v_exact[aj*p+ak] = val_exact
        w_exact[ak*n+ai] = Fraction(1)
        
        # Propagate to all (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U[i*m+j][c1]*V[j*p+k][c1]*W[k*n+i][c1] + U[i*m+j][c2]*V[j*p+k][c2]*W[k*n+i][c2]
                    ui = u_exact.get(i*m+j)
                    vj = v_exact.get(j*p+k)
                    wk = w_exact.get(k*n+i)
                    
                    knowns = 0
                    if ui is not None: knowns += 1
                    if vj is not None: knowns += 1
                    if wk is not None: knowns += 1
                    
                    if knowns == 3:
                        if ui * vj * wk != val:
                            return None
                    elif knowns == 2:
                        if ui is None:
                            if vj == 0 or wk == 0:
                                if val != 0: return None
                                u_exact[i*m+j] = Fraction(0)
                            else:
                                u_exact[i*m+j] = val / (vj * wk)
                        elif vj is None:
                            if ui == 0 or wk == 0:
                                if val != 0: return None
                                v_exact[j*p+k] = Fraction(0)
                            else:
                                v_exact[j*p+k] = val / (ui * wk)
                        elif wk is None:
                            if ui == 0 or vj == 0:
                                if val != 0: return None
                                w_exact[k*n+i] = Fraction(0)
                            else:
                                w_exact[k*n+i] = val / (ui * vj)
                    elif knowns == 1:
                        # Skip, will be determined later?
                        # Actually, if the tensor is rank 1, the propagation should fill everything.
                        # If not, it's not rank 1.
                        pass
                    else:
                        pass
                        
        # Final verification
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U[i*m+j][c1]*V[j*p+k][c1]*W[k*n+i][c1] + U[i*m+j][c2]*V[j*p+k][c2]*W[k*n+i][c2]
                    ui = u_exact.get(i*m+j, Fraction(0))
                    vj = v_exact.get(j*p+k, Fraction(0))
                    wk = w_exact.get(k*n+i, Fraction(0))
                    if ui * vj * wk != val:
                        return None
                        
        # Build full vectors
        uvec = [u_exact.get(i, Fraction(0)) for i in range(n*m)]
        vvec = [v_exact.get(i, Fraction(0)) for i in range(m*p)]
        wvec = [w_exact.get(i, Fraction(0)) for i in range(p*n)]
        
        return (uvec, vvec, wvec)

    # Greedy: repeatedly try to merge pairs of columns
    for _ in range(max_iters):
        if _time_left() < 5:
            break
        r = len(U[0])
        if r <= 1:
            break
        merged = False
        cols = list(range(r))
        rng.shuffle(cols)
        for a in range(len(cols)):
            for b in range(a+1, len(cols)):
                c1 = cols[a]; c2 = cols[b]
                merged_vec = _check_rank1_merge(c1, c2)
                if merged_vec is not None:
                    uvec, vvec, wvec = merged_vec
                    # replace columns c1,c2 with the merged single column, drop one
                    newU = []
                    newV = []
                    newW = []
                    for i in range(n*m):
                        newU.append([U[i][c] for c in range(r) if c not in (c1, c2)] + [uvec[i]])
                    for i in range(m*p):
                        newV.append([V[i][c] for c in range(r) if c not in (c1, c2)] + [vvec[i]])
                    for i in range(p*n):
                        newW.append([W[i][c] for c in range(r) if c not in (c1, c2)] + [wvec[i]])
                    U, V, W = newU, newV, newW
                    # Update numpy arrays
                    U_np = np.array(U, dtype=float)
                    V_np = np.array(V, dtype=float)
                    W_np = np.array(W, dtype=float)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
    return U, V, W

def _strassen_2x2x2():
    """Return U,V,W (as lists) for the 2x2x2 matmul via Strassen (rank 7).
    U[i*m+j,c] with m=2, V[j*p+k,c] with p=2, W[k*n+i,c] with n=2.
    Strassen:
    M1 = (A11-A22)(B22-B11)
    M2 = (A11+A22)B11
    M3 = A22(B21-B11)
    M4 = (A21-A11)B22
    M5 = (A11+A12)B22
    M6 = (A21-A12)(B11+B21)
    M7 = (A12-A22)(B12+B22)
    C11 = M1+M4-M5+M7
    C12 = M3+M5
    C21 = M2+M4
    C22 = M1-M2+M3+M6
    Here A is 2x2, B is 2x2, C is 2x2.
    n=m=p=2.
    U[i*m+j,c] = U[i*2+j, c], c in 0..6.
    V[j*p+k,c] = V[j*2+k, c].
    W[k*n+i,c] = W[k*2+i, c].
    The bilinear form: C[i,k] = sum_c (sum_j U[i*2+j,c] A[i,j]) * (sum_l V[j*2+l,c] B[j,l]).
    So U[i*2+j,c] multiplies A[i,j], V[j*2+l,c] multiplies B[j,l].
    For M1: A terms: A11(+1), A22(-1) => U[0*2+0,c0]=+1, U[1*2+1,c0]=-1.
    B terms: B22(+1), B11(-1) => V[1*2+1,c0]=+1, V[0*2+0,c0]=-1.
    W for M1: C11(+1), C22(+1) => W[0*2+0,c0]=+1, W[1*2+1,c0]=+1.
    Let's define the 7 M's as (c0..c6):
    M1: A: (0,0)+1,(1,1)-1 ; B: (1,1)+1,(0,0)-1 ; C: (0,0)+1,(1,1)+1
    M2: A: (0,0)+1,(1,1)+1 ; B: (0,0)+1 ; C: (1,1)+1,(1,0)+1
    M3: A: (1,1)+1 ; B: (1,0)+1,(0,0)-1 ; C: (0,1)+1,(1,1)+1
    M4: A: (1,0)+1,(0,0)-1 ; B: (1,1)+1 ; C: (0,0)+1,(1,0)+1
    M5: A: (0,0)+1,(0,1)+1 ; B: (1,1)+1 ; C: (0,0)-1,(0,1)+1
    M6: A: (1,0)+1,(0,1)-1 ; B: (0,0)+1,(1,0)+1 ; C: (1,1)+1
    M7: A: (0,1)+1,(1,1)-1 ; B: (0,1)+1,(1,1)+1 ; C: (0,0)+1
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1
    U[0][0] += 1; U[3][0] += -1
    V[3][0] += 1; V[0][0] += -1
    W[0][0] += 1; W[3][0] += 1
    # M2
    U[0][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[3][1] += 1; W[2][1] += 1
    # M3
    U[3][2] += 1
    V[2][2] += 1; V[0][2] += -1
    W[1][2] += 1; W[3][2] += 1
    # M4
    U[2][3] += 1; U[0][3] += -1
    V[3][3] += 1
    W[0][3] += 1; W[2][3] += 1
    # M5
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] += -1; W[1][4] += 1
    # M6
    U[2][5] += 1; U[1][5] += -1
    V[0][5] += 1; V[2][5] += 1
    W[3][5] += 1
    # M7
    U[1][6] += 1; U[3][6] += -1
    V[1][6] += 1; V[3][6] += 1
    W[0][6] += 1
    return U, V, W

def _strassen_2x2x3():
    """2x2 (A) x 2x3 (B) -> 2x3 (C). Use Strassen-style: split B into two 2x2? 
    Actually B is 2x3, can't directly. Use 2x2x2 for first 2 columns and add 2x1x1 for 3rd? 
    Simpler: treat as 2x2x2 (cols 0,1) + 2x1x1 (col 2). Rank = 7 + 2 = 9.
    But better: use a 2x2x3 with rank maybe 8. Let's just do 7+2=9 for simplicity, or try to do better.
    For now, block: A 2x2, B 2x3, C 2x3.
    U: (n*m)=2*2=4 rows, V: (m*p)=2*3=6 rows, W: (p*n)=3*2=6 rows.
    """
    n, m, p = 2, 2, 3
    U2, V2, W2 = _strassen_2x2x2()  # for B cols 0,1 (p=2)
    # For col 2 (B[:,2]), it's a 2x1x1: A (2x2) * B_col (2x1) -> C_col (2x1). Rank 2.
    # U for 2x2x1: U[i*2+j, c] A[i,j]; V[j*1+0, c] B[j,0]; W[0*2+i, c] C[i,0].
    # Rank-2: c0: A[0,0]B[0,0]->C[0,0]; c1: A[1,1]B[1,0]->C[1,0].
    U = [[0]*9 for _ in range(4)]
    V = [[0]*9 for _ in range(6)]
    W = [[0]*9 for _ in range(6)]
    # copy strassen for cols 0,1 (p index 0,1)
    for i in range(4):
        for c in range(7):
            U[i][c] = U2[i][c]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V[j*2+k][c] = V2[j*2+k][c]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W[k*2+i][c] = W2[k*2+i][c]
    # add 2x2x1 for col 2 (B col index 2)
    # V row for j, col 2: V[j*3+2, c]
    # W row for k=2, i: W[2*2+i, c]
    U[0][7] += 1  # A[0,0]
    V[0*3+2][7] += 1  # B[0,2]
    W[2*2+0][7] += 1  # C[0,2]
    U[3][8] += 1  # A[1,1]
    V[1*3+2][8] += 1  # B[1,2]
    W[2*2+1][8] += 1  # C[1,2]
    return U, V, W

def _combine_blocks(block_specs, n, m, p):
    """block_specs: list of (bi, bj, bk, bU, bV, bW) where bi,bj,bk are starting indices
    and bU is (bn*bm x r), bV is (bm*bp x r), bW is (bp*bn x r).
    Returns U (n*m x R), V (m*p x R), W (p*n x R)."""
    R = sum(b[3].__len__() and len(b[3][0]) for b in block_specs)
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    off = 0
    for (bi, bj, bk, bU, bV, bW) in block_specs:
        r = len(bU[0])
        bn = len(bU) // len(bU[0])  # bn*bm rows, bm = len(bV[0])? 
        # bU rows = bn*bm, bV rows = bm*bp, bW rows = bp*bn
        # determine bm from bV[0] length = bm*bp, and bU row count = bn*bm
        # We'll infer: bm = len(bV[0]) // (len(bW)//len(bW[0])) ... let's just use given shapes.
        bm = len(bV[0]) // (len(bW) // len(bW[0]))  # bp = len(bW)//len(bW[0])? bW rows = bp*bn
        # Actually simpler: bU has bn*bm rows, bV has bm*bp rows, bW has bp*bn rows.
        # bm = bU_rows / bn; but we don't have bn directly. Use: bm = gcd-like.
        # We'll pass bn, bm, bp explicitly instead.
        pass
    return U, V, W

def _combine_blocks2(specs, n, m, p):
    """specs: list of (bi, bj, bk, bn, bm, bp, bU, bV, bW)."""
    R = sum(len(b[6][0]) for b in specs)
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    off = 0
    for (bi, bj, bk, bn, bm, bp, bU, bV, bW) in specs:
        r = len(bU[0])
        for ii in range(bn):
            for jj in range(bm):
                for c in range(r):
                    U[(bi+ii)*m + (bj+jj)][off+c] = bU[ii*bm+jj][c]
        for jj in range(bm):
            for kk in range(bp):
                for c in range(r):
                    V[(bj+jj)*p + (bk+kk)][off+c] = bV[jj*bp+kk][c]
        for kk in range(bp):
            for ii in range(bn):
                for c in range(r):
                    W[(bk+kk)*n + (bi+ii)][off+c] = bW[kk*bn+ii][c]
        off += r
    return U, V, W

def _build_tiling(n, m, p):
    """Build a tiling of the n x m x p tensor using 2x2x2 (Strassen) and 2x2x3 / 1x1x1 blocks.
    We tile A (n x m) and B (m x p). The middle dimension m must be shared.
    Strategy: partition i into 2's + 1's, k into 2's + 1's, and for each (i-block, k-block)
    use a 2x2x2 Strassen over a 2x2 sub-block of j? No—j is shared across i-blocks.
    Better: the tensor T[i,j,k] = A[i,j]*B[j,k]. We can decompose by fixing j-blocks.
    For each j-block of size 2, and i-block of size 2, k-block of size 2, use Strassen 2x2x2.
    But j-blocks overlap in i and k. The clean way: 
    C[i,k] = sum_j A[i,j] B[j,k]. Group j into pairs. For each j-pair (j0,j1), 
    C[i,k] += A[i,j0]B[j0,k] + A[i,j1]B[j1,k].
    For fixed j-pair, this is a (n x 2) * (2 x p) product -> n x p.
    That's a 2-column slice. We can compute n x p from (n x 2)*(2 x p) with rank n*p (schoolbook) 
    or better. For n=3,p=7: rank 21 per j-pair, 2 j-pairs + 1 j-single = 2*21 + 21 = 63? No that's worse.
    Hmm. Let's instead tile i and k with 2x2 Strassen blocks and handle j fully.
    For a 2x2x2 block (i0,i1) x (j0,j1) x (k0,k1): rank 7.
    We need to cover all (i,j,k). Tile i into {0,1},{2}, k into {0,1},{2,3},{4,5},{6}, 
    and for each (i-block, k-block) we need all j. The 2x2x2 only covers 2 j's.
    So for (i-block of 2, k-block of 2), we cover j in pairs: 2 j-pairs + 1 j-single = 
    2*(2x2x2 Strassen rank 7) + (2x1x2 rank 2) = 14+2 = 16 per (i-block,k-block) of size 2x2.
    Number of (i-block,k-block) pairs: i-blocks {0,1},{2} (2 blocks), k-blocks {0,1},{2,3},{4,5},{6} (4 blocks) = 8 pairs.
    But the j-pairing must be consistent? No, j is independent per (i-block,k-block) since T[i,j,k] factorizes.
    Total rank = 8 * 16 = 128. Worse than 105. Not good.
    Alternative: the factorization T[i,j,k]=A[i,j]B[j,k] means we can separate. 
    The schoolbook is n*m*p = 105. Strassen only helps if we can share j across i,k blocks, 
    which we can't directly because A[i,j] and B[j,k] are separate.
    Actually the best known: use the fact that rank of matmul tensor for (n,m,p) is at most 
    n*p*rank_2x2... no. Let me reconsider.
    The tensor is the "matrix multiplication tensor" MM_{n,p,m} (A n x m, B m x p).
    Known upper bounds use Strassen: rank <= something. For 3x5x7, 
    a simple bound: pad to 4x8x8 and use Strassen recursively? 
    Let's just do: tile into 2x2x2 blocks where possible and 1x1x1 elsewhere, 
    with j fully covered per (i,k) block.
    For (i-block, k-block), the sub-tensor is (bn x m) * (m x bp) -> bn x bp.
    Rank of that sub-tensor (schoolbook) = bn*m*bp. Strassen doesn't reduce because m is the full shared dim.
    So tiling i,k doesn't help. The only Strassen benefit is when ALL of n,m,p are 2.
    Therefore, to use Strassen, we must restrict to 2x2x2 sub-blocks in ALL three dims, 
    and the rest is schoolbook. 
    Count: number of 2x2x2 blocks = floor(n/2)*floor(m/2)*floor(p/2) = 1*2*3 = 6 blocks, each rank 7 instead of 8: save 6.
    Remaining: the "fringe" where i=2 (1 row), or j=4 (1 col), or k=6 (1 col) not covered by 2x2x2.
    Covered by 2x2x2: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}. 
    Uncovered: i=2 (all j,k): 1*5*7=35; plus i in {0,1}, j=4, k in {0..5}: 2*1*6=12; 
    plus i in {0,1}, j in {0..3}, k=6: 2*4*1=8; plus i in {0,1}, j=4, k=6: 2*1*1=2.
    Wait, let me just do: total = 105. 2x2x2 blocks cover i<2, j<4, k<6: that's 2*4*6=48 entries, 
    as 6 blocks of 8 = 48. Rank for these = 6*7 = 42 (instead of 48). Save 6.
    Remaining 105-48 = 57 entries handled schoolbook rank 57.
    Total rank = 42 + 57 = 99. That matches the measured 99! 
    So the construction: 6 Strassen 2x2x2 blocks + schoolbook for the rest.
    """
    n, m, p = 3, 5, 7
    specs = []
    # 2x2x2 Strassen blocks: i0 in {0}, j0 in {0,2}, k0 in {0,2,4}
    S2 = _strassen_2x2x2()
    for i0 in [0]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                specs.append((i0, j0, k0, 2, 2, 2, S2[0], S2[1], S2[2]))
    # Schoolbook for remaining (i,j,k) not in any 2x2x2 block
    # A (i,j,k) is covered by 2x2x2 if i<2 and j<4 and k<6.
    covered = set()
    for (i0, j0, k0, bn, bm, bp, _, _, _) in specs:
        for ii in range(bn):
            for jj in range(bm):
                for kk in range(bp):
                    covered.add((i0+ii, j0+jj, k0+kk))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    # schoolbook 1x1x1
                    U1 = [[1]]
                    V1 = [[1]]
                    W1 = [[1]]
                    specs.append((i, j, k, 1, 1, 1, U1, V1, W1))
    return specs

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    rng = random.Random(12345)
    # Build initial tiling
    specs = _build_tiling(n, m, p)
    U, V, W = _combine_blocks2(specs, n, m, p)
    # Verify
    target = _target_tensor(n, m, p)
    T = _tensor_to_dict(U, V, W, n, m, p)
    if T != target:
        # fallback to schoolbook
        rank = n*m*p
        Us = [[0]*rank for _ in range(n*m)]
        Vs = [[0]*rank for _ in range(m*p)]
        Ws = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i*m+j][col] = 1
                    Vs[j*p+k][col] = 1
                    Ws[k*n+i][col] = 1
                    col += 1
        return {"U": Us, "V": Vs, "W": Ws}
    best_rank = len(U[0])
    best = (U, V, W)
    # Try to reduce rank via merging
    U2, V2, W2 = _try_reduce(U, V, W, n, m, p, rng, max_iters=3000)
    T2 = _tensor_to_dict(U2, V2, W2, n, m, p)
    if T2 == target and len(U2[0]) < best_rank:
        best_rank = len(U2[0])
        best = (U2, V2, W2)
    # Try a few random restarts of reduction with different seeds
    for seed in range(1, 12):
        if _time_left() < 10:
            break
        rng2 = random.Random(seed * 999)
        U3, V3, W3 = _try_reduce(U, V, W, n, m, p, rng2, max_iters=1500)
        T3 = _tensor_to_dict(U3, V3, W3, n, m, p)
        if T3 == target and len(U3[0]) < best_rank:
            best_rank = len(U3[0])
            best = (U3, V3, W3)
    # Final safety: if time is low, return best
    if _time_left() < 3:
        pass
    U, V, W = best
    # Convert to plain ints/floats (Fractions -> int if integer else float)
    def _to_plain(M):
        out = []
        for row in M:
            nr = []
            for x in row:
                if isinstance(x, Fraction):
                    if x.denominator == 1:
                        nr.append(x.numerator)
                    else:
                        nr.append(float(x))
                else:
                    nr.append(x)
            out.append(nr)
        return out
    return {"U": _to_plain(U), "V": _to_plain(V), "W": _to_plain(W)}
