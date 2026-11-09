import time
import numpy as np
import itertools
import random
from scipy.optimize import linprog

def construct(q, n, radius):
    # Task: q=3, n=7, radius=1
    # Total words: 3^7 = 2187
    # Ball size: 15
    
    t_start = time.time()
    deadline = 118.0 # Safety margin (limit is 120s)
    
    # 1. Precomputation
    # Generate all words as indices 0..2186
    # index = d1*q^(n-1) + ... + dn
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    def tuple_to_idx(t):
        idx = 0
        for val in t:
            idx = idx * q + val
        return idx
        
    N = q ** n
    all_words = [idx_to_tuple(i) for i in range(N)]
    
    # Precompute balls: balls[i] is a numpy array of indices within distance radius
    # Radius 1: center + 1-substitutions
    balls = []
    for i in range(N):
        w = list(all_words[i])
        ball = [i]
        for pos in range(n):
            old_val = w[pos]
            for new_val in range(q):
                if new_val != old_val:
                    w[pos] = new_val
                    ball.append(tuple_to_idx(tuple(w)))
            w[pos] = old_val
        balls.append(np.array(ball, dtype=np.int32))
        
    # Convert to list of lists for easier manipulation if needed, but keep numpy for speed
    balls_np = np.array(balls, dtype=np.int32)
    
    # 2. Construct Ternary Hamming Code (Perfect Code)
    # For q=3, n=7, the Hamming code has size 3^(n - ceil(log_q(n+1))) = 3^3 = 27 codewords?
    # Wait, perfect code size is q^n / (1 + n(q-1)) = 2187 / 15 = 145.8. Not integer.
    # Actually, ternary Hamming code exists for n = (q^m - 1)/(q-1).
    # For q=3, m=2 -> n = (9-1)/2 = 4. Size 3^(4-2)=9.
    # For q=3, m=3 -> n = (27-1)/2 = 13. Size 3^(13-3)=59049.
    # There is no perfect ternary code for n=7.
    # However, we can use a "quasi-perfect" code or simply a good code of size ~186.
    
    # Alternative Strategy: Greedy Set Cover with Algebraic Initialization
    # The "Algebraic" part usually refers to using the structure of the Hamming space.
    # Since no perfect code exists, we start with a strong greedy solution and refine.
    
    # Let's try a different structural initialization:
    # Randomly select a subset of words that form a 2-packing (distance >= 3)
    # A 2-packing of size P covers at most P * 15 words? No, disjoint balls.
    # If we have a 2-packing of size P, the balls are disjoint.
    # Max 2-packing size A_3(7,3) is known to be around 186? No, that's the covering code size.
    # The packing number A_3(7,3) is smaller.
    
    # Let's stick to a robust optimization approach:
    # 1. Greedy Initialization (fast, gets ~240)
    # 2. LP Relaxation on the residual or full problem (if fast enough)
    # 3. Intensive Local Search (Add/Remove/Swap)
    
    best_code = None
    best_len = N + 1
    
    # --- Phase 1: Greedy Initialization with Randomization ---
    def greedy_init(seed=None):
        if seed is not None:
            random.seed(seed)
        covered = np.zeros(N, dtype=bool)
        code = []
        # Precompute gains? No, dynamic.
        # To speed up, we can sample candidates or use vectorized ops.
        # N=2187, balls size 15.
        
        while not covered.all():
            # Calculate gain for each codeword i: number of uncovered words in balls[i]
            # gains[i] = sum(1 - covered[balls[i]])
            # Vectorized: ~covered is bool array. 
            # We need sum over axis 1 of (1 - covered[balls_np])
            # This is a matrix-vector multiplication if we use the incidence matrix.
            # Let M be N x N sparse where M[i,j]=1 if j in balls[i].
            # gain = M @ (1 - covered)
            
            # Constructing M every time is too slow. Precompute M?
            # M is 2187x2187, sparse, ~15 non-zeros per row.
            # Memory is fine.
            
            if not hasattr(greedy_init, 'M_csr'):
                import scipy.sparse as sp
                row_indices = []
                col_indices = []
                data = []
                for i in range(N):
                    for j in balls_np[i]:
                        row_indices.append(i)
                        col_indices.append(j)
                        data.append(1)
                M = sp.csr_matrix((data, (row_indices, col_indices)), shape=(N, N))
                greedy_init.M_csr = M
                greedy_init.M_csc = M.T.tocsc() # For checking coverage by column?
                
            M = greedy_init.M_csr
            
            uncovered_mask = 1.0 - covered.astype(float)
            gains = M.dot(uncovered_mask)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            # Random tie break
            idx = candidates[random.randint(0, len(candidates)-1)]
            
            code.append(idx)
            covered[balls_np[idx]] = True
            
        return code

    # Run a few greedy inits
    for i in range(5):
        if time.time() - t_start > deadline - 10: break
        code = greedy_init(seed=42 + i)
        if len(code) < best_len:
            best_len = len(code)
            best_code = code[:]
            
    # --- Phase 2: LP Relaxation (Optional, if time permits and code is large) ---
    # If best_len is still high (> 200), try LP.
    if best_len > 200 and time.time() - t_start < deadline - 20:
        try:
            import scipy.sparse as sp
            # Reuse M from greedy_init if available
            if not hasattr(greedy_init, 'M_csr'):
                # Fallback build
                row_indices = []
                col_indices = []
                data = []
                for i in range(N):
                    for j in balls_np[i]:
                        row_indices.append(i)
                        col_indices.append(j)
                        data.append(1)
                M = sp.csr_matrix((data, (row_indices, col_indices)), shape=(N, N))
            else:
                M = greedy_init.M_csr
                
            # Minimize sum(x) subject to M^T x >= 1
            # M^T x >= 1  =>  -M^T x <= -1
            A_ub = -M.T.toarray()
            b_ub = -np.ones(N)
            c = np.ones(N)
            bounds = [(0, 1)] * N
            
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options={"presolve": True})
            
            if res.status == 0:
                x = res.x
                # Rounding: sort by x, pick if it covers new
                sorted_indices = np.argsort(x)[::-1]
                covered = np.zeros(N, dtype=bool)
                code_lp = []
                
                for idx in sorted_indices:
                    if time.time() - t_start > deadline - 10: break
                    if x[idx] > 0.3: # Threshold
                        if not np.all(covered[balls_np[idx]]):
                            code_lp.append(idx)
                            covered[balls_np[idx]] = True
                
                # Greedy fill
                while not covered.all():
                    if time.time() - t_start > deadline - 10: break
                    gains = M.dot(1.0 - covered.astype(float))
                    max_gain = np.max(gains)
                    if max_gain == 0: break
                    candidates = np.where(gains == max_gain)[0]
                    # Prefer high LP value
                    cand_scores = x[candidates]
                    max_score = np.max(cand_scores)
                    top_cands = candidates[cand_scores == max_score]
                    idx = top_cands[random.randint(0, len(top_cands)-1)]
                    code_lp.append(idx)
                    covered[balls_np[idx]] = True
                    
                if len(code_lp) < best_len:
                    best_len = len(code_lp)
                    best_code = code_lp[:]
                    
        except Exception:
            pass # Ignore LP failures

    # --- Phase 3: Local Search (Add, Remove, Swap) ---
    # This is the most critical part for improvement.
    # Start with best_code found so far.
    
    if best_code is None:
        best_code = greedy_init(seed=1)
        
    # Ensure best_code is a list
    best_code = list(best_code)
    
    # Precompute incidence structure for fast updates?
    # We maintain 'counts' array: counts[j] = number of codewords in best_code that cover word j.
    # A word is covered if counts[j] > 0.
    
    def get_counts(code):
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[balls_np[idx]] += 1
        return counts
    
    counts = get_counts(best_code)
    
    # Helper to check if a set of indices forms a valid cover
    def is_valid_cover(counts_arr):
        return np.all(counts_arr > 0)
        
    # Local Search Loop
    improved = True
    while improved and time.time() - t_start < deadline - 2:
        improved = False
        
        # 1. Remove Redundant Codewords
        # If removing idx keeps all words in its ball covered by others, remove it.
        # Condition: for all j in balls[idx], counts[j] > 1
        for idx in list(best_code):
            if time.time() - t_start > deadline - 2: break
            ball = balls_np[idx]
            if np.min(counts[ball]) > 1:
                # Remove
                counts[ball] -= 1
                best_code.remove(idx)
                improved = True
                # Continue removing? Yes, loop again or continue in this pass.
                # Restarting the loop is safer to catch new redundancies created by removal?
                # Actually, removal only decreases counts, so it might make OTHERS redundant?
                # No, if A was not redundant, removing B (which might cover some of A's words)
                # could make A redundant if A's words were only covered by A and B?
                # Wait. If min(counts[ball_A]) > 1, A is redundant.
                # If we remove B, counts decrease. If min(counts[ball_A]) was 2, and B covered one of them,
                # it becomes 1. A is no longer redundant.
                # So removals don't create new redundancies, they might destroy them.
                # So a single pass removing all current redundancies is optimal for removal.
        
        # 2. Swap Moves
        # Try to replace one codeword u with another v such that:
        # - u is in code, v is not in code
        # - The new set is a valid cover
        # - The new set has the same size (so no immediate gain in size, but might enable future removals)
        # OR
        # - Try to replace u with v where v covers "better" words?
        # Standard swap: u in, v out. Size constant.
        # To reduce size, we need Add+Remove (net 0) then Remove (net -1) or just Remove (net -1).
        # Swap allows escaping local minima where no single removal is possible, but a swap allows a subsequent removal.
        
        # Optimization: Only try swaps if we have time.
        # Number of swaps is O(K * N). K ~ 200, N ~ 2187. ~ 400k checks.
        # Each check: update counts, verify, revert.
        
        if time.time() - t_start < deadline - 5:
            # Identify "critical" codewords? No, just iterate.
            # To speed up, we can limit swaps to those that cover "rare" words.
            
            # Let's do a limited number of random swaps to avoid O(K*N) overhead if K is large.
            # Or structured: For each u in code, find v not in code that maximizes "benefit".
            # Benefit: v covers words that u doesn't?
            
            # Simpler: Randomized Swaps
            num_swap_attempts = 500
            for _ in range(num_swap_attempts):
                if time.time() - t_start > deadline - 2: break
                if len(best_code) == 0: break
                
                u = random.choice(best_code)
                v = random.randint(0, N - 1)
                if v in best_code: continue
                
                # Check if swapping u for v is valid
                # Remove u: counts[ball_u] -= 1
                # Add v: counts[ball_v] += 1
                # Check if all counts > 0
                
                ball_u = balls_np[u]
                ball_v = balls_np[v]
                
                # Efficient check:
                # Words in ball_u \ ball_v will lose coverage from u.
                # If any of those had count 1, they become 0 -> Invalid.
                # Words in ball_v \ ball_u gain coverage.
                # Words in intersection: count changes -1 + 1 = 0.
                
                # So we only need to check if min(counts[ball_u \ ball_v]) > 0 after decrement?
                # Actually, if count[j] == 1 and j in ball_u and j not in ball_v, then removing u makes it 0.
                
                # Intersection mask
                # For speed, precompute bitsets? N=2187, 32 int64s per word.
                # Let's use numpy sets logic which is fast enough for ~500 attempts.
                
                # Create mask for ball_u
                mask_u = np.zeros(N, dtype=bool)
                mask_u[ball_u] = True
                mask_v = np.zeros(N, dtype=bool)
                mask_v[ball_v] = True
                
                # Words losing u: mask_u & ~mask_v
                losing_mask = mask_u & ~mask_v
                if np.any(losing_mask):
                    if np.min(counts[losing_mask]) <= 1:
                        continue # Invalid swap
                
                # Swap is valid (or at least doesn't break coverage)
                # Apply swap
                counts[ball_u] -= 1
                counts[ball_v] += 1
                best_code.remove(u)
                best_code.append(v)
                improved = True
                
                # After swap, try to remove redundant words again immediately?
                # The outer loop will handle it.
                
    # Final Cleanup: Remove any remaining redundancies
    counts = get_counts(best_code)
    changed = True
    while changed and time.time() - t_start < deadline:
        changed = False
        for idx in list(best_code):
            ball = balls_np[idx]
            if np.min(counts[ball]) > 1:
                counts[ball] -= 1
                best_code.remove(idx)
                changed = True

    # Convert indices back to list of lists of integers
    result = []
    for idx in best_code:
        w = []
        temp_idx = idx
        for _ in range(n):
            w.append(temp_idx % q)
            temp_idx //= q
        w.reverse()
        result.append(w)
        
    return result
