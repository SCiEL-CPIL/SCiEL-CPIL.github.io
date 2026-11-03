import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    Uses block decomposition with Strassen 2x2x2 (rank 7) and schoolbook 2x2x1 (rank 4),
    then attempts local search to reduce rank by merging/canceling terms.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    nm, mp, pn = n * m, m * p, p * n
    
    start_time = time.time()
    time_limit = 140  # Leave margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    def verify_solution(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        R = U.shape[1]
        # Build the target tensor: T[a*m+b, c*p+d, k*n+i] = 1 if b==c, a==i, d==k else 0
        # Check: sum_mu U[a*m+b, mu] * V[c*p+d, mu] * W[k*n+i, mu] == T[a*m+b, c*p+d, k*n+i]
        for i in range(n):
            for k in range(p):
                for a in range(n):
                    for b in range(m):
                        for c in range(m):
                            for d in range(p):
                                val = 0.0
                                for mu in range(R):
                                    val += U[a*m+b, mu] * V[c*p+d, mu] * W[k*n+i, mu]
                                if b == c and a == i and d == k:
                                    if abs(val - 1.0) > 1e-9:
                                        return False
                                else:
                                    if abs(val - 0.0) > 1e-9:
                                        return False
        return True

    def snap_to_half_integers(U, V, W):
        U_snapped = np.round(U * 2) / 2.0
        V_snapped = np.round(V * 2) / 2.0
        W_snapped = np.round(W * 2) / 2.0
        return U_snapped, V_snapped, W_snapped

    def get_solution_rank36():
        """Build rank-36 solution using 4 Strassen 2x2x2 + 2 schoolbook 2x2x1 blocks."""
        rank = 36
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            # M1: (A00+A11)(B00+B11)
            U[i0*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M2: (A10+A11)B00
            U[(i0+1)*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + (i0+1), col] += 1; W[(k0+1)*n + (i0+1), col] -= 1
            col += 1
            # M3: A00(B10-B11)
            U[i0*m + j0, col] += 1
            V[j0*p + (k0+1), col] += 1; V[(j0+1)*p + (k0+1), col] -= 1
            W[(k0+1)*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M4: A11(B01-B00)
            U[(i0+1)*m + (j0+1), col] += 1
            V[(j0+1)*p + k0, col] += 1; V[j0*p + k0, col] -= 1
            W[k0*n + i0, col] += 1; W[k0*n + (i0+1), col] += 1
            col += 1
            # M5: (A00+A01)B11
            U[i0*m + j0, col] += 1; U[i0*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] -= 1; W[(k0+1)*n + i0, col] += 1
            col += 1
            # M6: (A10-A00)(B00+B01)
            U[(i0+1)*m + j0, col] += 1; U[i0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1; V[j0*p + (k0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M7: (A01-A11)(B10+B11)
            U[i0*m + (j0+1), col] += 1; U[(i0+1)*m + (j0+1), col] -= 1
            V[(j0+1)*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            col += 1
            return col
        
        def add_schoolbook_2x2x1(i0, j0, k0):
            nonlocal col
            for di in range(2):
                for dj in range(2):
                    U[(i0 + di)*m + (j0 + dj), col] += 1
                    V[(j0 + dj)*p + k0, col] += 1
                    W[k0*n + (i0 + di), col] += 1
                    col += 1
            return col
        
        add_strassen_222(0, 0, 0)
        add_strassen_222(0, 2, 0)
        add_strassen_222(0, 0, 2)
        add_strassen_222(0, 2, 2)
        add_schoolbook_2x2x1(0, 0, 4)
        add_schoolbook_2x2x1(0, 2, 4)
        
        return U, V, W

    # Build initial rank-36 solution
    U36, V36, W36 = get_solution_rank36()
    
    if not verify_solution(U36, V36, W36):
        # Fallback: pure schoolbook
        rank = nm * mp
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U36.copy(), V36.copy(), W36.copy()
    best_rank = 36
    
    # Try to reduce rank by attempting to remove columns (terms) one at a time
    # and solving for the remaining to maintain correctness
    def try_reduce_rank(U, V, W, max_attempts=200):
        """Try to reduce rank by removing one column at a time and adjusting others."""
        nonlocal best_U, best_V, best_W, best_rank
        U_cur = U.copy()
        V_cur = V.copy()
        W_cur = W.copy()
        rank_cur = U_cur.shape[1]
        
        attempts = 0
        improved = True
        while improved and attempts < max_attempts and elapsed() < time_limit:
            improved = False
            for col_idx in range(rank_cur):
                if elapsed() > time_limit:
                    break
                attempts += 1
                # Try removing column col_idx
                U_test = np.delete(U_cur, col_idx, axis=1)
                V_test = np.delete(V_cur, col_idx, axis=1)
                W_test = np.delete(W_cur, col_idx, axis=1)
                
                # Check if the remaining terms still satisfy the tensor equation
                # We need: sum_{mu != col_idx} U[a*m+b, mu] * V[c*p+d, mu] * W[k*n+i, mu] == T
                # This is a linear system in the remaining terms.
                # For each (a,b,c,d,k,i), we need the sum to equal T.
                # The contribution from removed col is: U[a*m+b, col_idx] * V[c*p+d, col_idx] * W[k*n+i, col_idx]
                # So we need the remaining sum to equal T - removed_contribution
                # This is generally not solvable by just removing, so we check if it's already 0
                # or if the removed term was redundant.
                
                # Simpler approach: check if the removed column is a linear combination
                # that doesn't affect the tensor. Try setting it to zero and see if we can
                # compensate by adjusting other columns.
                
                # Actually, let's try a different approach: for each pair of columns,
                # try to merge them if their outer products are proportional.
                pass
            
            # Try merging pairs of columns
            for c1 in range(rank_cur):
                if elapsed() > time_limit:
                    break
                for c2 in range(c1+1, rank_cur):
                    if elapsed() > time_limit:
                        break
                    # Check if column c2 is a scalar multiple of column c1 in all three matrices
                    # If U[:,c2] = alpha * U[:,c1], V[:,c2] = alpha * V[:,c1], W[:,c2] = alpha * W[:,c1]
                    # then we can merge them.
                    U1, U2 = U_cur[:, c1], U_cur[:, c2]
                    V1, V2 = V_cur[:, c1], V_cur[:, c2]
                    W1, W2 = W_cur[:, c1], W_cur[:, c2]
                    
                    # Check proportionality
                    nz1 = np.nonzero(U1)[0]
                    nz2 = np.nonzero(U2)[0]
                    if len(nz1) == 0 or len(nz2) == 0:
                        continue
                    
                    # Try to find alpha such that U2 = alpha * U1
                    alphas_u = []
                    for idx in range(len(nz1)):
                        i1 = nz1[idx]
                        if U2[i1] != 0:
                            alphas_u.append(U2[i1] / U1[i1])
                    
                    if not alphas_u:
                        continue
                    
                    alpha = alphas_u[0]
                    valid = all(abs(a - alpha) < 1e-10 for a in alphas_u)
                    if not valid:
                        continue
                    if np.max(np.abs(U2 - alpha * U1)) > 1e-10:
                        continue
                    
                    # Check V
                    nz1v = np.nonzero(V1)[0]
                    if len(nz1v) == 0:
                        continue
                    alphas_v = []
                    for idx in range(len(nz1v)):
                        i1 = nz1v[idx]
                        if V2[i1] != 0:
                            alphas_v.append(V2[i1] / V1[i1])
                    if not alphas_v:
                        continue
                    alpha_v = alphas_v[0]
                    if abs(alpha_v - alpha) > 1e-10 or np.max(np.abs(V2 - alpha * V1)) > 1e-10:
                        continue
                    
                    # Check W
                    nz1w = np.nonzero(W1)[0]
                    if len(nz1w) == 0:
                        continue
                    alphas_w = []
                    for idx in range(len(nz1w)):
                        i1 = nz1w[idx]
                        if W2[i1] != 0:
                            alphas_w.append(W2[i1] / W1[i1])
                    if not alphas_w:
                        continue
                    alpha_w = alphas_w[0]
                    if abs(alpha_w - alpha) > 1e-10 or np.max(np.abs(W2 - alpha * W1)) > 1e-10:
                        continue
                    
                    # Merge: add column c2 to c1
                    U_cur[:, c1] += U_cur[:, c2]
                    V_cur[:, c1] += V_cur[:, c2]
                    W_cur[:, c1] += W_cur[:, c2]
                    U_cur = np.delete(U_cur, c2, axis=1)
                    V_cur = np.delete(V_cur, c2, axis=1)
                    W_cur = np.delete(W_cur, c2, axis=1)
                    rank_cur -= 1
                    improved = True
                    break
                if improved:
                    break
        
        if rank_cur < best_rank and verify_solution(U_cur, V_cur, W_cur):
            best_U, best_V, best_W = U_cur, V_cur, W_cur
            best_rank = rank_cur
        
        return best_U, best_V, best_W, best_rank
    
    # Try to reduce rank
    if elapsed() < time_limit:
        best_U, best_V, best_W, best_rank = try_reduce_rank(U36, V36, W36)
    
    # Snap to half-integers if possible
    U_snap, V_snap, W_snap = snap_to_half_integers(best_U, best_V, best_W)
    if verify_solution(U_snap, V_snap, W_snap):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
