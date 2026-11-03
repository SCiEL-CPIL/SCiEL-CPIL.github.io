import numpy as np
import time
import scipy.optimize as so
import scipy.sparse as sp
import scipy.sparse.linalg as spla

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution peak
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin

    # The objective is to maximize R(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    # Due to scale invariance, we can fix ||h||_1 = 1.
    # Then R(h) = ||h||_2^2 / ||h*h||_inf.
    # Let M = ||h*h||_inf. We want to maximize ||h||_2^2 / M.
    # This is equivalent to: For a fixed M, maximize ||h||_2^2 subject to ||h||_1=1 and (h*h)_k <= M for all k.
    # The constraint (h*h)_k <= M is quadratic. The matrix of quadratic form for (h*h)_k is A_k.
    # (h*h)_k = h^T A_k h.
    # A_k is a matrix where (A_k)_{ij} = 1 if i+j = k, else 0.
    # Note: np.convolve(h, h, mode='full')[k] corresponds to sum_{i+j=k} h_i h_j.
    # Indices i, j in [0, N-1]. k in [0, 2N-2].
    
    # We can solve this using an SDP relaxation or directly as a QCQP if we handle the non-convexity?
    # Wait, maximizing a convex function (||h||_2^2 is convex) subject to convex constraints (||h||_1=1 is convex, h^T A_k h <= M is NOT convex generally? 
    # A_k has 0/1 entries. Eigenvalues of A_k? 
    # A_k is a Hankel-like matrix. It is symmetric. 
    # If A_k is positive semidefinite, then h^T A_k h <= M is a convex constraint.
    # Let's check PSD of A_k.
    # A_k maps h to (h*h)_k.
    # For k=0, A_0 has only A_00=1. PSD.
    # For k=1, A_1 has A_01=A_10=1. Eigenvalues 1, -1. Not PSD.
    # So the constraints are not convex in h.
    # However, we can use the SDP relaxation:
    # Maximize Tr(X P) where P = diag(h_i^2)? No.
    # Let's look at the structure again.
    # R(h) = sum h_i^2 / max_k sum_{i+j=k} h_i h_j.
    
    # Alternative approach: 
    # The maximum of the ratio is related to the independence number or clique number of a graph?
    # Actually, the "Sidon set" intuition was: if support is Sidon, max_k (h*h)_k is small relative to sum h_i^2?
    # If h is indicator of Sidon set S, sum h_i^2 = |S|.
    # (h*h)_k is 1 if k is sum of two elements in S (with multiplicity), 0 otherwise.
    # Since S is Sidon, all sums i+j (i<=j) are distinct.
    # So there are |S|(|S|+1)/2 distinct sums.
    # The max of (h*h)_k is 1.
    # So R = |S| / (|S| * 1) = 1?
    # Wait, the current code scores 1.0 for uniform h.
    # Let's re-evaluate the Sidon argument in the current code's comments.
    # "R = (k*c^2) / (k*c * c^2) = 1/c". If L1=1, c=1/k. R = k.
    # Let's check:
    # h_i = c on S, 0 else.
    # L1 = k*c.
    # L2_sq = k*c^2.
    # (h*h)_k = c^2 for distinct sums. Max is c^2.
    # Ratio = (k*c^2) / (k*c * c^2) = 1/c.
    # If L1=1, c=1/k. Ratio = k.
    # So yes, larger Sidon set -> higher score.
    # Max Sidon set in [0, 49] is size ~7. Score ~7.
    # Current best is 1.19. Why?
    # Because the optimizer might not find the pure Sidon set, or the Sidon set size is small.
    # Or maybe the "peak" constraint is not the only thing.
    
    # Let's try a different structural approach: 
    # Use a Genetic Algorithm or Differential Evolution to optimize the vector h directly,
    # but with a smarter representation: 
    # Instead of optimizing 50 floats, optimize the support and weights.
    # Or, use the fact that the optimal solution likely has sparse support.
    
    # Let's try a specific combinatorial search for the best "near-Sidon" sets and then refine weights.
    # But the directive says "Structural change... different optimization or search method".
    # Let's try **Differential Evolution** (global optimizer) on the full 50-dimensional space, 
    # but with a penalty for non-sparse solutions to encourage sparsity, 
    # or just let it find the global max.
    
    # However, DE on 50D with non-convex objective might be slow.
    
    # Let's look at the SDP idea again.
    # Maximize ||h||_2^2 / M.
    # Equivalent to: Maximize t such that ||h||_2^2 >= t * M, ||h||_1=1, h>=0.
    # This is still non-convex.
    
    # What if we fix the support?
    # If we know the support S, the problem becomes:
    # Maximize sum_{i in S} h_i^2 / max_k (sum_{i,j in S, i+j=k} h_i h_j)
    # subject to sum h_i = 1, h_i >= 0.
    # This is a ratio of quadratic forms.
    # We can solve this by bisection on the ratio R.
    # Check if exists h on S such that sum h_i^2 >= R * max_k (sum_{i,j in S, i+j=k} h_i h_j) and sum h_i = 1.
    # This is still hard.
    
    # Let's try a **Greedy Construction with Local Search** on the support, 
    # but using a more robust local search (Simulated Annealing) on the weights.
    
    # Actually, the most promising "structural" change is to use **Differential Evolution**
    # with a constraint that forces sparsity, or simply run it for a fixed number of iterations.
    # DE is robust to local minima.
    
    def _de_objective(h):
        # Minimize negative score
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        # Normalize to L1=1 to handle scale invariance in the search space?
        # No, DE works better in a bounded box.
        # Let's work with h in [0, 1].
        # The ratio is scale invariant.
        return -_score_raw(h)

    # Strategy:
    # 1. Use Differential Evolution to find a good sparse-like vector.
    # 2. To encourage sparsity, we can add a penalty term to the objective: -score + lambda * ||h||_0?
    #    L0 is not smooth.
    #    Instead, just run DE and hope it finds a sparse solution.
    #    Or, initialize DE with many sparse random vectors (Sidon-like).
    
    # Let's try a hybrid:
    # 1. Generate a pool of candidate supports (Sidon sets, random sparse sets).
    # 2. For each support, solve the weight optimization problem using a local method (SLSQP or Nelder-Mead).
    # 3. Keep the best.
    
    # This is similar to the current program but with more diverse supports and better local optimization.
    # The current program uses SLSQP on the full space or restricted space.
    # SLSQP can get stuck.
    
    # Let's try **Nelder-Mead** on the weights for a fixed support, as it doesn't require gradients
    # and can handle non-smooth peaks (max function) better than SLSQP sometimes.
    
    # And for support selection, let's use a **Randomized Greedy** with more restarts and larger sets.
    
    # Actually, let's look at the "Inspiration 1/2" score 1.1944.
    # 1.1944 is close to 7/6? No.
    # Maybe the optimal score is indeed around 7?
    # If so, we need to find a Sidon set of size 7.
    
    # Let's implement a robust Sidon set finder.
    # The greedy method in the current code might be too weak.
    # Let's use a backtracking search for Sidon sets, or a more powerful greedy.
    
    def _find_sidon_backtrack(N, max_size, time_limit=10.0):
        """Find a Sidon set of size max_size in [0, N) using backtracking."""
        start = time.time()
        S = []
        
        def backtrack(start_idx, current_S):
            if time.time() - start > time_limit:
                return False
            if len(current_S) == max_size:
                return True
            
            # Pruning: if remaining numbers not enough to reach max_size
            if N - start_idx < max_size - len(current_S):
                return False
            
            for i in range(start_idx, N):
                # Check if i can be added
                valid = True
                # New sums: i + s for s in current_S, and 2*i
                # Must not collide with existing sums
                existing_sums = set()
                for a in range(len(current_S)):
                    for b in range(a, len(current_S)):
                        existing_sums.add(current_S[a] + current_S[b])
                
                new_sums = [i + s for s in current_S] + [2*i]
                
                # Check collision with existing
                if any(ns in existing_sums for ns in new_sums):
                    continue
                
                # Check collision within new_sums
                # i+s1 == i+s2 => s1==s2 (ok)
                # i+s1 == 2i => s1==i (i not in current_S, so ok)
                
                current_S.append(i)
                if backtrack(i+1, current_S):
                    return True
                current_S.pop()
                
            return False
        
        if backtrack(0, S):
            return S
        return None

    # Try to find Sidon sets of size 7, 8, etc.
    best_h = None
    best_score = 0.0
    
    # 1. Try to find large Sidon sets
    for size in range(7, 4, -1):
        if time.time() > deadline - 30:
            break
        S = _find_sidon_backtrack(N, size, time_limit=10.0)
        if S:
            # Optimize weights for this support
            h_init = np.zeros(N)
            h_init[S] = 1.0 / size
            
            # Use Nelder-Mead on the weights
            support = np.array(S)
            
            def obj_weights(w):
                h = np.zeros(N)
                h[support] = w
                # Normalize L1 to 1? 
                # Nelder-Mead works on the variables directly.
                # The ratio is scale invariant.
                # We can fix sum(w)=1 by parameterizing w_i = x_i / sum(x) ?
                # Or just let NM search in the simplex.
                # NM doesn't handle constraints well.
                # Let's use a penalty method or just search in [0, 1]^k and normalize.
                # If we normalize, the objective is constant on rays.
                # Let's fix sum(w)=1 by setting w_k = 1 - sum_{i<k} w_i.
                # This reduces dimension to k-1.
                
                if len(w) == 0: return 0
                w = np.asarray(w)
                if np.any(w < 0):
                    return 1e6 # Penalty
                w_sum = np.sum(w)
                if w_sum == 0: return 1e6
                w_norm = w / w_sum
                h = np.zeros(N)
                h[support] = w_norm
                return -_score_raw(h)
            
            # Initial guess: uniform
            w0 = np.ones(size-1) / size # Last one determined
            # Actually, let's just use SLSQP with equality constraint sum(w)=1, it's more reliable for small k.
            # SLSQP worked in the seed. Maybe it just needs a better start or more iterations.
            
            def neg_obj_slsqp(w):
                h = np.zeros(N)
                h[support] = w
                return -_score_raw(h)
            
            cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            bounds = [(0, 1)] * size
            
            try:
                res = so.minimize(neg_obj_slsqp, np.ones(size)/size, method='SLSQP', 
                                  bounds=bounds, constraints=cons, options={'maxiter': 2000, 'ftol': 1e-14})
                if res.success:
                    h_candidate = np.zeros(N)
                    h_candidate[support] = res.x
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate
            except Exception:
                pass
            
            if best_score >= size * 0.9: # If we got close to the theoretical max for this size
                break

    # 2. If Sidon search didn't yield great results, try Differential Evolution on full space
    if best_score < 5.0 and time.time() < deadline - 20:
        # DE on 50 variables
        # Bounds [0, 1]
        bounds_de = [(0, 1)] * N
        # Objective
        def de_obj(h):
            return -_score_raw(h)
        
        # Use a small population to fit in time
        try:
            res_de = so.differential_evolution(de_obj, bounds_de, maxiter=50, popsize=10, tol=1e-6, seed=42, polish=True)
            if res_de.success:
                h_de = res_de.x
                s_de = _score_raw(h_de)
                if s_de > best_score:
                    best_score = s_de
                    best_h = h_de
        except Exception:
            pass

    # Fallback
    if best_h is None or best_score <= 0:
        best_h = np.ones(N) / N
        best_score = _score_raw(best_h)
        
    return list(np.maximum(best_h, 0))
