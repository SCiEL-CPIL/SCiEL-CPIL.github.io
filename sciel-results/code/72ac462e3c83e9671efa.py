import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 1e-15:
        return 0.0
    return l2_sq / (l1 * DELTA * max_conv)

def _objective_sparse(h_full):
    """
    Objective for sparse optimization.
    h_full is the full vector, but we only care about the support indices.
    Actually, it's better to define the objective on the reduced vector of weights.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # 1. Generate candidate supports
    supports = []
    
    # Strategy A: Greedy Sidon Set
    # A set S is Sidon if all sums s1+s2 (s1<=s2) are unique.
    def find_greedy_sidon(n, target_size):
        S = [0]
        sums = set([0])
        while len(S) < target_size:
            found = False
            for x in range(1, n):
                if x in S:
                    continue
                # Check if adding x creates duplicate sums
                new_sums = set()
                valid = True
                for s in S:
                    sm = s + x
                    if sm in sums or sm in new_sums:
                        valid = False
                        break
                    new_sums.add(sm)
                # Also check sums within new_sums? No, s1+s2 with s1,s2 in S U {x}.
                # The condition is that the multiset of sums from S U {x} has unique values.
                # Since sums in S are unique, we just need to ensure new sums (x+s) 
                # don't collide with existing sums or each other.
                if valid:
                    # Check collisions among new sums themselves?
                    # s1+x and s2+x are distinct if s1 != s2.
                    # So we just need new_sums to be disjoint from sums.
                    S.append(x)
                    sums.update(new_sums)
                    found = True
                    break
            if not found:
                break
        return S

    # Try to find Sidon sets of size 7 and 8
    for size in [8, 7, 6]:
        S = find_greedy_sidon(N, size)
        if len(S) >= size:
            supports.append(np.array(S))
            
    # Strategy B: Quadratic Residues
    # For prime p, the set of quadratic residues is a Sidon set (B_2 sequence) 
    # with some exceptions or modifications, but generally has low collision counts.
    # We need to map residues to [0, 49].
    # Let's try small primes p <= 50.
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    for p in range(7, 53):
        if not is_prime(p):
            continue
        # Quadratic residues mod p
        residues = set()
        for x in range(1, p):
            residues.add((x * x) % p)
        # Remove 0 if present (it is, 0*0=0), but 0 is a valid index.
        # Actually, 0 is a residue.
        # We want a subset of [0, 49].
        # If p <= 50, residues are in [0, p-1] which is subset of [0, 49].
        # Size is roughly (p-1)/2.
        S = np.array(sorted(list(residues)))
        if len(S) >= 5:
            supports.append(S)
            
    # Strategy C: Random Sparse
    np.random.seed(42)
    for _ in range(20):
        k = np.random.randint(6, 10)
        idx = np.random.choice(N, k, replace=False)
        supports.append(idx)
        
    # Strategy D: Structured "B_2" like sequences
    # e.g., powers of 2 are Sidon. 1, 2, 4, 8, 16, 32. Size 6.
    supports.append(np.array([1, 2, 4, 8, 16, 32]))
    # Shifted or scaled?
    supports.append(np.array([0, 1, 3, 7, 15, 31]))

    # Deduplicate supports
    unique_supports = []
    for s in supports:
        s_sorted = np.sort(s)
        is_dup = False
        for us in unique_supports:
            if np.array_equal(s_sorted, us):
                is_dup = True
                break
        if not is_dup:
            unique_supports.append(s_sorted)
            
    supports = unique_supports

    # 2. Refine weights for each support
    def refine_weights(support, initial_weights=None):
        k = len(support)
        if k == 0:
            return None, 0.0
            
        if initial_weights is None:
            # Equal weights
            w = np.ones(k) / k
        else:
            w = initial_weights.copy()
            
        # Objective: Maximize score
        # Score = sum(w^2) / (sum(w) * DELTA * max_conv(w))
        # Since we fix support, conv is computed on full vector with zeros elsewhere.
        
        def obj(w_vec):
            h = np.zeros(N)
            h[support] = w_vec
            return -_score_raw(h)
            
        # Bounds: w_i >= 0
        bounds = [(0.0, 1.0)] * k
        
        # Initial point
        w0 = w.copy()
        if np.sum(w0) == 0:
            w0 = np.ones(k) / k
            
        try:
            # Use L-BFGS-B
            res = minimize(
                obj, 
                w0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if res.success:
                w_opt = np.clip(res.x, 0, None)
                h_opt = np.zeros(N)
                h_opt[support] = w_opt
                if np.sum(h_opt) > 1e-15:
                    h_opt /= np.sum(h_opt) # Normalize to L1=1 for consistency, though score is scale invariant
                    s_opt = _score_raw(h_opt)
                    return h_opt, s_opt
        except Exception:
            pass
            
        # Fallback: return equal weights score
        h_eq = np.zeros(N)
        h_eq[support] = 1.0 / k
        s_eq = _score_raw(h_eq)
        return h_eq, s_eq

    # Process supports
    for s in supports:
        if time.time() > deadline - 5:
            break
            
        # Try refining with equal weights
        h_cand, s_cand = refine_weights(s)
        
        if h_cand is not None and s_cand > best_score:
            best_score = s_cand
            best_h = h_cand.copy()
            
        # Also try a few random initializations for the weights on this support
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            w_rand = np.random.rand(len(s))
            w_rand /= np.sum(w_rand)
            h_cand, s_cand = refine_weights(s, w_rand)
            if h_cand is not None and s_cand > best_score:
                best_score = s_cand
                best_h = h_cand.copy()

    # 3. Global Refinement on the best found
    # It's possible that the optimal support is not exactly one of the generated ones,
    # but close. We can try a "mutation" of the best support.
    # However, given the time, the discrete Sidon sets are strong candidates.
    # Let's do one more pass: try to optimize the full vector starting from best_h
    # but with a penalty for non-zero elements not in the support?
    # No, L-BFGS-B on 50 vars is fast enough if we have a good start.
    
    def obj_full(h_vec):
        return -_score_raw(h_vec)
        
    bounds_full = [(0.0, 1.0)] * N
    
    try:
        res = minimize(
            obj_full, 
            best_h, 
            method='L-BFGS-B', 
            bounds=bounds_full,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        if res.success:
            h_full = np.clip(res.x, 0, None)
            if np.sum(h_full) > 1e-15:
                h_full /= np.sum(h_full)
                s_full = _score_raw(h_full)
                if s_full > best_score:
                    best_score = s_full
                    best_h = h_full.copy()
    except Exception:
        pass

    return list(best_h)
