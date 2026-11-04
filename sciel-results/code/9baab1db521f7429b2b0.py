"""
QAP Solver: Spectral Relaxation with Soft-Assignment Guidance and Vectorized 2-Opt.

Strategy:
1. Spectral Initialization: Compute leading eigenvectors of the flow matrix A and 
   distance matrix B. Align them to create a soft-assignment matrix X (n x n).
2. Soft Optimization: Refine X using gradient descent steps to minimize the 
   relaxed objective ||A X B X^T - A X^T B X||_F^2 (or similar alignment metric),
   projecting to the Birkhoff polytope (doubly stochastic) occasionally.
3. Hard Assignment: Use the soft matrix X to generate a high-quality initial permutation.
4. Vectorized 2-Opt: Perform local search using vectorized delta calculations for 
   all possible swaps, selecting the best improving move each step.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _spectral_init(a, b, k=50):
    """
    Generate initial permutation using spectral embedding.
    We align the top eigenvectors of A and B.
    """
    n = len(a)
    
    # Symmetrize A for eigendecomposition stability (QAP objective is invariant to symmetrization of A if B is symmetric, 
    # but B is usually symmetric. If not, we should be careful. 
    # Standard QAP: min sum a_ij b_pi(i) pi(j).
    # If we symmetrize A: A' = (A + A.T)/2, the objective value for any permutation is the same?
    # sum a_ij b_pi(i) pi(j) + sum a_ji b_pi(j) pi(i) = sum (a_ij + a_ji) b_pi(i) pi(j) because b is symmetric.
    # So we can safely symmetrize A.
    a_sym = (a + a.T) / 2.0
    
    # Get top k eigenvectors of A
    # Using np.linalg.eigh for symmetric matrix
    eigvals_a, eigvecs_a = np.linalg.eigh(a_sym)
    # Sort by eigenvalue descending
    idx_a = np.argsort(eigvals_a)[::-1]
    U_a = eigvecs_a[:, idx_a[:k]]
    
    # Get top k eigenvectors of B
    b_sym = (b + b.T) / 2.0
    eigvals_b, eigvecs_b = np.linalg.eigh(b_sym)
    idx_b = np.argsort(eigvals_b)[::-1]
    U_b = eigvecs_b[:, idx_b[:k]]
    
    # We want to align U_a and U_b.
    # The soft assignment X should satisfy X U_a ~ U_b (or similar).
    # Let's construct X = U_a @ U_b.T ? No, dimensions.
    # U_a is n x k, U_b is n x k.
    # We want X (n x n) such that it maps the "space" of A to "space" of B.
    # A common heuristic: X = U_a @ U_b.T is n x n.
    # But this might not be a good permutation matrix.
    # Let's try to maximize trace(X^T A X B) ? No.
    
    # Alternative: 
    # Project facilities onto U_a, locations onto U_b.
    # We want facility i (row i of U_a) to be close to location j (row j of U_b) in the embedding space?
    # Actually, if we define embedding of facility i as u_i (row of U_a) and location j as v_j (row of U_b),
    # we want to minimize sum ||u_i - v_j||^2 ? No, that's matching.
    
    # Let's use the standard "Spectral Bisection" style initialization for QAP.
    # Minimize sum a_ij b_pi(i) pi(j).
    # This is equivalent to maximizing sum a_ij b_pi(i) pi(j) if we negate? No.
    
    # Let's just use the Soft Assignment approach directly.
    # Start with X = U_a @ U_b.T
    X = U_a @ U_b.T
    
    # Normalize X to be roughly doubly stochastic?
    # For now, just use it to generate a hard assignment via Hungarian algorithm on -X (maximize similarity)
    # Or simply take argmax row-wise? That's not a permutation.
    
    # Better: Use X to compute a cost matrix for Hungarian.
    # The objective is sum a_ij b_pi(i) pi(j).
    # This is hard to decompose into a linear assignment directly.
    
    # Let's try a different spectral init:
    # Sort facilities by their projection on the principal eigenvector of A.
    # Sort locations by their projection on the principal eigenvector of B.
    # Match them in order.
    
    # Principal eigenvector of A
    v_a = eigvecs_a[:, idx_a[0]]
    # Principal eigenvector of B
    v_b = eigvecs_b[:, idx_b[0]]
    
    # Ensure same sign
    if np.dot(v_a, v_b) < 0:
        v_b = -v_b
        
    # Project facilities
    fac_proj = v_a
    # Project locations
    loc_proj = v_b
    
    # Sort facilities by projection descending
    fac_order = np.argsort(-fac_proj)
    # Sort locations by projection descending
    loc_order = np.argsort(-loc_proj)
    
    p = np.empty(n, dtype=np.int64)
    for i, f in enumerate(fac_order):
        p[f] = loc_order[i]
        
    return p

def _soft_assignment_optimize(a, b, X, steps=50, lr=0.01):
    """
    Refine soft assignment X using gradient descent.
    Objective: Minimize F(X) = ||A X B X^T - A X^T B X||_F^2 ? 
    Actually, the QAP objective is Tr(X^T A X B).
    We want to MINIMIZE this.
    Gradient of Tr(X^T A X B) w.r.t X is 2 A X B + 2 A^T X B^T (if symmetric, 4 A X B).
    Wait, d/dX Tr(X^T A X B) = A^T X B^T + A X B.
    If A, B symmetric: 2 A X B.
    We want to minimize, so we subtract gradient.
    But X must be a permutation matrix (0/1). We relax to [0,1] and project.
    """
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    for _ in range(steps):
        # Gradient: 2 * A @ X @ B
        grad = 2.0 * (a_sym @ X @ b_sym)
        
        # Update
        X = X - lr * grad
        
        # Project to Birkhoff Polytope (Doubly Stochastic)
        # Simple projection: Iterative scaling (Sinkhorn-Knopp)
        # Or just clip and normalize rows/cols
        X = np.clip(X, 0, None)
        
        # Normalize rows to sum to 1
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        X = X / row_sums
        
        # Normalize cols to sum to 1
        col_sums = X.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        X = X / col_sums
        
        # Alternate a few times for better projection
        for _ in range(5):
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            X = X / row_sums
            col_sums = X.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1
            X = X / col_sums
            
    return X

def _hard_assignment_from_soft(X):
    """
    Convert soft assignment matrix X to a hard permutation.
    Use Hungarian algorithm to maximize sum X[i, p[i]].
    """
    n = len(X)
    # We want to maximize sum X[i, p[i]], so minimize -X
    row_ind, col_ind = linear_sum_assignment(-X)
    p = np.empty(n, dtype=np.int64)
    for i, j in zip(row_ind, col_ind):
        p[i] = j
    return p

def _vectorized_2opt(a, b, p, time_limit, start_time, seed=42):
    """
    Local search using vectorized delta calculation.
    p[i] = location of facility i.
    """
    n = len(a)
    a_sym = a + a.T
    b_sym = b + b.T
    
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Precompute row sums of b_sym?
    # Delta for swapping locations of facility i and j:
    # Let x = p[i], y = p[j].
    # New cost - Old cost = 
    # Sum_{k != i,j} (a[i,k] + a[k,i]) * (b[x, p[k]] - b[y, p[k]]) + 
    # (a[i,j] + a[j,i]) * (b[x,y] - b[y,x]) ... wait b is symmetric so b[x,y]=b[y,x].
    # Actually, the terms involving i and j themselves need care.
    
    # Standard delta formula for swapping positions i and j in permutation q (q[pos]=fac):
    # Here p[fac]=pos.
    # Let's stick to the position swap delta but map it.
    # If we swap locations x and y, it means facility i moves to y and facility j moves to x.
    # This is equivalent to swapping the values p[i] and p[j].
    
    # Delta = Sum_{k} (a[i,k] + a[k,i]) * (b[y, p[k]] - b[x, p[k]]) 
    #         - (a[i,j] + a[j,i]) * (b[y, p[j]] - b[x, p[j]])  # Correction for j being at p[j]
    #         - (a[j,i] + a[i,j]) * (b[x, p[i]] - b[y, p[i]])  # Correction for i being at p[i]
    #         + (a[i,j] + a[j,i]) * (b[x, y] - b[y, x])         # Cross term?
    
    # Let's derive carefully.
    # Cost = Sum_{u,v} a[u,v] b[p[u], p[v]]
    # Change p[i] <-> p[j]. Let x=p[i], y=p[j].
    # Terms involving i or j change.
    # 1. u=i, v=i: a[i,i] b[x,x] -> a[i,i] b[y,y]. Change: a[i,i](b[y,y]-b[x,x])
    # 2. u=j, v=j: a[j,j] b[y,y] -> a[j,j] b[x,x]. Change: a[j,j](b[x,x]-b[y,y])
    # 3. u=i, v=j: a[i,j] b[x,y] -> a[i,j] b[y,x]. Change: a[i,j](b[y,x]-b[x,y]) = 0 if b sym.
    # 4. u=j, v=i: a[j,i] b[y,x] -> a[j,i] b[x,y]. Change: a[j,i](b[x,y]-b[y,x]) = 0 if b sym.
    # 5. u=i, v=k (k!=i,j): a[i,k] b[x, p[k]] -> a[i,k] b[y, p[k]]. Change: a[i,k](b[y, p[k]] - b[x, p[k]])
    # 6. u=k, v=i (k!=i,j): a[k,i] b[p[k], x] -> a[k,i] b[p[k], y]. Change: a[k,i](b[p[k], y] - b[p[k], x])
    #    Since b is symmetric, b[p[k], y] = b[y, p[k]] and b[p[k], x] = b[x, p[k]].
    #    So 5 and 6 combine to: (a[i,k] + a[k,i]) (b[y, p[k]] - b[x, p[k]])
    # 7. u=j, v=k (k!=i,j): a[j,k] b[y, p[k]] -> a[j,k] b[x, p[k]]. Change: a[j,k](b[x, p[k]] - b[y, p[k]])
    # 8. u=k, v=j (k!=i,j): a[k,j] b[p[k], y] -> a[k,j] b[p[k], x]. Change: a[k,j](b[p[k], x] - b[p[k], y])
    #    Combine 7 and 8: (a[j,k] + a[k,j]) (b[x, p[k]] - b[y, p[k]])
    
    # Total Delta = 
    # a[i,i](b[y,y]-b[x,x]) + a[j,j](b[x,x]-b[y,y]) 
    # + Sum_{k!=i,j} (a[i,k] + a[k,i] - a[j,k] - a[k,j]) (b[y, p[k]] - b[x, p[k]])
    
    # Let A_sym = a + a.T.
    # Delta = (a[i,i]-a[j,j])(b[y,y]-b[x,x]) + Sum_{k!=i,j} (A_sym[i,k] - A_sym[j,k]) (b[y, p[k]] - b[x, p[k]])
    
    # Note: b[y,y]-b[x,x] is constant for a pair (i,j) given x,y.
    # We can vectorize the sum over k.
    
    # Let's precompute B_diff for all pairs? No, x,y depend on i,j.
    # We can compute the sum term efficiently.
    # Sum_{k} (A_sym[i,k] - A_sym[j,k]) (b[y, p[k]] - b[x, p[k]])
    # = Sum_k A_sym[i,k] b[y, p[k]] - Sum_k A_sym[i,k] b[x, p[k]] - Sum_k A_sym[j,k] b[y, p[k]] + Sum_k A_sym[j,k] b[x, p[k]]
    
    # Let M[k] = p[k].
    # Term 1: Sum_k A_sym[i,k] b[y, M[k]]
    # This looks like a matrix multiplication.
    # Let B_perm = b[np.ix_(p, p)] ? No.
    # We need b[y, p[k]] for all k. This is row y of b, indexed by p.
    # b_y_p = b[y, p]
    # Sum_k A_sym[i,k] b_y_p[k] = dot(A_sym[i,:], b[y, p])
    
    # So Delta(i,j) = (a[i,i]-a[j,j])(b[y,y]-b[x,x]) + dot(A_sym[i,:], b[y,p]) - dot(A_sym[i,:], b[x,p]) - dot(A_sym[j,:], b[y,p]) + dot(A_sym[j,:], b[x,p])
    
    # We can precompute D[i, l] = dot(A_sym[i,:], b[l, p]) for all i, l.
    # D = A_sym @ b[:, p] ? 
    # (A_sym @ b[:, p])[i, l] = Sum_k A_sym[i,k] b[l, p[k]]. Yes.
    # Let's call this matrix P. P[i, l] = Sum_k A_sym[i,k] b[l, p[k]].
    
    # Then Delta(i,j) = (a[i,i]-a[j,j])(b[y,y]-b[x,x]) + P[i, y] - P[i, x] - P[j, y] + P[j, x]
    
    # This is O(1) per pair once P is computed!
    # Computing P is O(n^3) initially, but we only need to update it when p changes.
    # When we swap p[i] and p[j] (x and y), how does P change?
    # P[k, l] = Sum_m A_sym[k,m] b[l, p[m]].
    # Only terms where m=i or m=j change.
    # p[i] becomes y, p[j] becomes x.
    # P[k, l] changes by: A_sym[k,i] (b[l, y] - b[l, x]) + A_sym[k,j] (b[l, x] - b[l, y])
    # = (A_sym[k,i] - A_sym[k,j]) (b[l, y] - b[l, x])
    
    # So we can update P in O(n^2) per swap.
    # And then finding the best swap is O(n^2) by scanning all pairs using the O(1) formula.
    
    # Total per iteration: O(n^2) update + O(n^2) scan = O(n^2).
    # n=256, n^2 ~ 65k. 115s / 65k ops ~ 1.7M iterations possible if vectorized well.
    
    # Let's implement this.
    
    P = a_sym @ b[:, p]
    
    # Precompute diag_a = a[i,i]
    diag_a = np.diag(a)
    diag_b = np.diag(b)
    
    no_improve = 0
    max_no_improve = 100
    
    while time.time() - start_time < time_limit - 3:
        # Find best swap (i, j)
        # Delta(i,j) = (diag_a[i]-diag_a[j])(b[y,y]-b[x,x]) + P[i,y] - P[i,x] - P[j,y] + P[j,x]
        # where x=p[i], y=p[j]
        
        # We want to minimize Delta.
        # Let's vectorize the calculation of Delta for all i,j.
        # This is tricky because x,y depend on i,j.
        # We can loop over i, and for each i, vectorize over j.
        
        best_delta = np.inf
        best_i = -1
        best_j = -1
        
        for i in range(n):
            x = p[i]
            # For all j != i
            # y = p[j]
            # term1 = (diag_a[i] - diag_a[j]) * (b[y,y] - b[x,x])
            # term2 = P[i, y] - P[i, x] - P[j, y] + P[j, x]
            
            # Vectorize over j
            js = np.arange(n)
            js = js[js != i]
            ys = p[js]
            
            # term1
            da_diff = diag_a[i] - diag_a[js]
            bb_diff = b[ys, ys] - b[x, x]
            t1 = da_diff * bb_diff
            
            # term2
            # P[i, ys] is vector of P[i, y_j]
            pi_y = P[i, ys]
            pi_x = P[i, x]
            pj_y = P[js, ys] # P[j, y] for each j
            pj_x = P[js, x] # P[j, x] for each j
            
            t2 = pi_y - pi_x - pj_y + pj_x
            
            deltas = t1 + t2
            
            min_idx = np.argmin(deltas)
            min_val = deltas[min_idx]
            
            if min_val < best_delta:
                best_delta = min_val
                best_j = js[min_idx]
                best_i = i
                
        if best_i == -1 or best_delta >= 0:
            # No improvement or no candidate
            if no_improve > max_no_improve:
                # Random restart/perturbation
                break
            no_improve += 1
            # Try a random move to escape local minimum?
            # For now, just break and let multi-start handle it.
            # Or do a random swap.
            if random.random() < 0.1:
                best_i = random.randint(0, n-1)
                best_j = random.randint(0, n-1)
                if best_i == best_j:
                    best_j = (best_j + 1) % n
                best_delta = _qap_cost(a, b, p) # Recalc? No, just estimate.
                # Actually, if we accept non-improving moves, we need to be careful.
                # Let's stick to strict improvement for stability, and rely on multi-start.
                # If no improvement, break.
                break
            else:
                break
        else:
            no_improve = 0
            
        # Perform swap
        x = p[best_i]
        y = p[best_j]
        p[best_i], p[best_j] = y, x
        
        # Update P
        # P[k, l] += (A_sym[k, best_i] - A_sym[k, best_j]) * (b[l, y] - b[l, x])
        diff_a = a_sym[:, best_i] - a_sym[:, best_j]
        diff_b = b[:, y] - b[:, x]
        P += np.outer(diff_a, diff_b)
        
        current_cost += best_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0
    
    best_p = None
    best_cost = np.inf
    
    # Multi-start
    # 1. Spectral
    p1 = _spectral_init(a, b)
    # 2. Random
    p2 = np.random.permutation(n)
    
    # 3. Greedy Rank
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    fac_order = np.argsort(-a_row_sums)
    loc_order = np.argsort(b_row_sums)
    p3 = np.empty(n, dtype=np.int64)
    p3[fac_order] = loc_order
    
    inits = [p1, p2, p3]
    
    # Allocate time
    time_per_start = (time_limit - 10) / len(inits)
    
    for idx, p_init in enumerate(inits):
        remaining = time_limit - (time.time() - start_time)
        if remaining < 10:
            break
        
        alloc_time = min(time_per_start, remaining - 5)
        if alloc_time < 5:
            break
            
        p_best, cost_best = _vectorized_2opt(a, b, p_init.copy(), alloc_time, start_time, seed=idx)
        
        if cost_best < best_cost:
            best_cost = cost_best
            best_p = p_best
            
    if best_p is None:
        best_p = p1
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
