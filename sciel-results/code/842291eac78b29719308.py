import time
import numpy as np
import random
from scipy.optimize import linprog

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 113.0  # Safety margin for 120s limit
    
    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute place values for index conversion
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q
        
    # Precompute all words as indices
    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q
            
    # Precompute ball_indices: for each word, list of indices within Hamming distance 1
    ball_indices = [[] for _ in range(total)]
    for i in range(total):
        w = words[i]
        ball_indices[i].append(i)
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball_indices[i].append(new_idx)
                
    # Convert to numpy arrays for faster vectorized operations
    ball_indices_np = [np.array(b, dtype=np.int32) for b in ball_indices]
    
    # Precompute ball membership matrix: ball_mem[i, j] = True if j is in ball(i)
    # This allows O(1) membership checks and fast vectorized operations
    ball_mem = np.zeros((total, total), dtype=bool)
    for i in range(total):
        ball_mem[i, ball_indices_np[i]] = True
    
    # LP-based column generation for set cover
    def lp_column_generation(max_iters=50, time_limit=15.0):
        """Solve LP relaxation via column generation, return solution."""
        if time.time() > deadline - time_limit:
            return None
        
        # Start with a small set of columns
        rng = random.Random(42)
        initial_cols = set(rng.sample(range(total), min(50, total)))
        
        # Build constraint matrix A: A[i,j] = 1 if word i is in ball(column j)
        # We'll use sparse representation
        col_list = list(initial_cols)
        
        # Dual variables for constraints (each word must be covered)
        # Solve: min sum x_j subject to A x >= 1, x >= 0
        # Dual: max sum y_i subject to A^T y <= 1, y >= 0
        
        for iteration in range(max_iters):
            if time.time() > deadline - time_limit:
                break
            
            # Build A matrix for current columns (sparse)
            A = np.zeros((total, len(col_list)), dtype=np.float64)
            for j, col in enumerate(col_list):
                A[ball_indices_np[col], j] = 1.0
            
            # Solve LP: min sum x_j s.t. A x >= 1, x >= 0
            # Convert to standard form for linprog: min c^T x s.t. A_ub x <= b_ub
            # A x >= 1  =>  -A x <= -1
            c = np.ones(len(col_list))
            A_ub = -A
            b_ub = -np.ones(total)
            
            try:
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * len(col_list), 
                             method='highs', options={'presolve': True, 'time_limit': 2.0})
            except Exception:
                break
            
            if not res.success:
                break
            
            # Get dual variables (shadow prices) from the LP
            # In linprog, we need to extract from the result
            # The dual variables correspond to the constraints A_ub x <= b_ub
            # For our problem, dual y_i = -res.ineqlin[0] (approximately)
            
            # Actually, let's compute the dual directly:
            # The dual is: max sum y_i s.t. sum_i A[i,j] y_i <= 1 for all j, y >= 0
            # We can solve this dual LP instead
            c_dual = np.ones(total)
            A_dual = A.T  # (num_cols, total)
            b_dual = np.ones(len(col_list))
            
            try:
                res_dual = linprog(-c_dual, A_ub=A_dual, b_ub=b_dual, 
                                  bounds=[(0, None)] * total, method='highs',
                                  options={'presolve': True, 'time_limit': 2.0})
            except Exception:
                break
            
            if not res_dual.success:
                break
            
            y = -res_dual.x  # Dual variables
            
            # Pricing subproblem: find column j that minimizes (reduced cost)
            # Reduced cost for column j = 1 - sum_i A[i,j] y_i
            # We want columns with negative reduced cost: 1 - sum_i A[i,j] y_i < 0
            # i.e., sum_i A[i,j] y_i > 1
            
            # For each potential new column c, compute gain = sum_i ball_mem[c,i] * y[i]
            # Vectorized: gain[c] = ball_mem[c, :] @ y
            gains = ball_mem @ y
            
            # Find the column with maximum gain (most negative reduced cost)
            max_gain_idx = np.argmax(gains)
            max_gain = gains[max_gain_idx]
            
            if max_gain <= 1.0 + 1e-6:
                # No improving column found, LP is optimal
                break
            
            # Add the new column
            if max_gain_idx not in col_list:
                col_list.append(int(max_gain_idx))
        
        # Round the LP solution
        if len(col_list) == 0:
            return None
        
        # Solve final LP with all columns
        A = np.zeros((total, len(col_list)), dtype=np.float64)
        for j, col in enumerate(col_list):
            A[ball_indices_np[col], j] = 1.0
        
        c = np.ones(len(col_list))
        A_ub = -A
        b_ub = -np.ones(total)
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * len(col_list),
                         method='highs', options={'presolve': True, 'time_limit': 3.0})
        except Exception:
            return None
        
        if not res.success:
            return None
        
        # Round: take columns with x_j > 0.5
        rounded = [col_list[j] for j in range(len(col_list)) if res.x[j] > 0.5]
        
        # If rounding doesn't cover everything, add more columns greedily
        if len(rounded) == 0:
            rounded = [col_list[0]]
        
        # Check coverage and add missing
        covered = np.zeros(total, dtype=bool)
        for c in rounded:
            covered[ball_indices_np[c]] = True
        
        while not np.all(covered):
            # Find uncovered word
            uncovered = np.where(~covered)[0]
            if len(uncovered) == 0:
                break
            
            # Find column that covers most uncovered words
            best_col = -1
            best_cov = -1
            for col in col_list:
                cov = np.sum(~covered[ball_indices_np[col]])
                if cov > best_cov:
                    best_cov = cov
                    best_col = col
            
            if best_col == -1 or best_cov <= 0:
                # Try random column
                best_col = random.choice(col_list)
            
            if best_col not in rounded:
                rounded.append(best_col)
                covered[ball_indices_np[best_col]] = True
        
        return rounded
    
    # Greedy construction with randomization
    def greedy_build(seed=None, dual_guidance=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
            
        covered = np.zeros(total, dtype=bool)
        code = []
        
        while not np.all(covered) and time.time() < deadline - 20.0:
            best_c = -1
            max_new = -1
            candidates = list(range(total))
            rng.shuffle(candidates)
            
            # If dual guidance available, weight candidates by dual values
            if dual_guidance is not None:
                # Prioritize candidates that cover words with high dual values
                weighted_candidates = []
                for c in candidates:
                    ball = ball_indices_np[c]
                    new_cov = np.sum(~covered[ball])
                    if new_cov > 0:
                        # Weight by sum of dual values of newly covered words
                        new_words = ball[~covered[ball]]
                        weight = np.sum(dual_guidance[new_words])
                        weighted_candidates.append((c, new_cov, weight))
                
                if weighted_candidates:
                    # Sort by combined score: new_cov + weight
                    weighted_candidates.sort(key=lambda x: x[1] + 0.1 * x[2], reverse=True)
                    # Take top candidates
                    top_candidates = [c for c, _, _ in weighted_candidates[:min(100, len(weighted_candidates))]]
                    # Add some random candidates for diversity
                    for c in candidates[:min(50, len(candidates))]:
                        if c not in top_candidates:
                            top_candidates.append(c)
                    candidates = top_candidates
            
            for c in candidates:
                ball = ball_indices_np[c]
                new_cov = np.sum(~covered[ball])
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == ball_size:
                        break
            if best_c == -1:
                break
            code.append(best_c)
            covered[ball_indices_np[best_c]] = True
        return code
    
    # Multiple greedy restarts to find good starting points
    best_initial = None
    best_initial_size = float('inf')
    dual_guidance = None
    
    # First try LP column generation
    if time.time() < deadline - 30.0:
        lp_result = lp_column_generation(max_iters=30, time_limit=20.0)
        if lp_result is not None:
            best_initial = set(lp_result)
            best_initial_size = len(lp_result)
            # Extract dual values from LP for guidance (approximate)
            # For now, use uniform guidance
            dual_guidance = np.ones(total, dtype=np.float64)
    
    # If LP didn't give good result or failed, use greedy
    if best_initial is None or best_initial_size > 200:
        init_code = greedy_build(seed=0)
        if best_initial is None or len(init_code) < best_initial_size:
            best_initial = set(init_code)
            best_initial_size = len(init_code)
    
    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 40.0:
            break
        init_code = greedy_build(seed=seed, dual_guidance=dual_guidance)
        if len(init_code) < best_initial_size:
            best_initial_size = len(init_code)
            best_initial = set(init_code)
    
    current_code = set(best_initial)
    best_code = list(current_code)
    best_size = len(current_code)
    
    # Initialize counts
    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        cnt[ball_indices_np[c]] += 1
        
    def run_delete_phase(code_set, cnt_arr):
        """Removes redundant codewords. Returns number removed."""
        removed = 0
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                ball = ball_indices_np[c]
                if np.min(cnt_arr[ball]) >= 2:
                    code_set.remove(c)
                    cnt_arr[ball] -= 1
                    removed += 1
                    changed = True
        return removed
    
    # Compute valid swaps using vectorized operations with ball_mem
    def get_valid_swaps(c, code_set, cnt_arr):
        """Return list of valid c_add for removing c using vectorized ball_mem."""
        ball_rem = ball_indices_np[c]
        
        # Words in ball_rem that would become uncovered if c is removed
        # These are words where cnt[x] == 1 (only covered by c)
        unique_mask = cnt_arr[ball_rem] == 1
        unique_words = ball_rem[unique_mask]
        
        if len(unique_words) == 0:
            # Any word not in code_set is valid
            return None
        
        # For a candidate c_add to be valid, it must cover all unique_words
        # i.e., for all x in unique_words, x must be in ball(c_add)
        # Using ball_mem: ball_mem[c_add, x] must be True for all x in unique_words
        
        # Vectorized check: for each candidate c_add, check if it covers all unique_words
        covers_count = np.sum(ball_mem[:, unique_words], axis=1)
        valid_mask = covers_count == len(unique_words)
        
        # Filter out words already in code_set
        code_mask = np.zeros(total, dtype=bool)
        for c_idx in code_set:
            code_mask[c_idx] = True
        
        valid_mask &= ~code_mask
        
        valid_list = np.where(valid_mask)[0].tolist()
        return valid_list
    
    # Local Search Loop with simulated annealing-inspired acceptance
    no_improve_counter = 0
    max_no_improve = 20
    temperature = 5.0  # Initial "temperature" for occasional worse moves
    min_temperature = 0.1
    
    while time.time() < deadline - 3.0:
        # 1. Delete Phase
        removed = run_delete_phase(current_code, cnt)
        if removed > 0:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                no_improve_counter = 0
            else:
                no_improve_counter += 1
        else:
            no_improve_counter += 1
        
        # 2. Swap Phase with smarter candidate selection
        swaps_attempted = 0
        max_swaps = 800
        while swaps_attempted < max_swaps and time.time() < deadline - 3.0:
            swaps_attempted += 1
            if len(current_code) == 0:
                break
                
            # Pick random codeword to remove
            c = random.choice(list(current_code))
            
            # Get valid swaps
            valid_swaps = get_valid_swaps(c, current_code, cnt)
            
            if valid_swaps is None:
                # Any non-code word works; pick one that maximizes new coverage
                ball_rem = ball_indices_np[c]
                unique_mask = cnt[ball_rem] == 1
                unique_words = ball_rem[unique_mask]
                
                if len(unique_words) == 0:
                    # No unique words, pick random non-code word
                    c_prime = random.randint(0, total - 1)
                    attempts = 0
                    while c_prime in current_code and attempts < 100:
                        c_prime = random.randint(0, total - 1)
                        attempts += 1
                else:
                    # Pick from words that cover the most unique words
                    covers_count = np.sum(ball_mem[:, unique_words], axis=1)
                    code_mask = np.zeros(total, dtype=bool)
                    for c_idx in current_code:
                        code_mask[c_idx] = True
                    covers_count[code_mask] = -1
                    c_prime = int(np.argmax(covers_count))
                    
            elif len(valid_swaps) == 0:
                no_improve_counter += 1
                continue
            else:
                # Pick from valid swaps, preferring those that cover many unique words
                ball_rem = ball_indices_np[c]
                unique_mask = cnt[ball_rem] == 1
                unique_words = ball_rem[unique_mask]
                
                if len(unique_words) > 0:
                    covers_count = np.array([np.sum(ball_mem[v, unique_words]) for v in valid_swaps])
                    # Weighted random selection: prefer higher coverage
                    max_cov = np.max(covers_count)
                    weights = np.exp((covers_count - max_cov) / max(temperature, 0.01))
                    weights /= np.sum(weights)
                    c_prime = valid_swaps[random.choices(range(len(valid_swaps)), weights=weights)[0]]
                else:
                    c_prime = random.choice(valid_swaps)
            
            # Perform swap
            current_code.remove(c)
            current_code.add(c_prime)
            cnt[ball_indices_np[c]] -= 1
            cnt[ball_indices_np[c_prime]] += 1
            
            # After swap, run delete phase
            removed = run_delete_phase(current_code, cnt)
            if removed > 0:
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
            else:
                no_improve_counter += 1
            
            # Cool down temperature
            temperature = max(min_temperature, temperature * 0.995)
        
        # If no improvement for many iterations, try a restart
        if no_improve_counter > max_no_improve and time.time() < deadline - 15.0:
            # Try a new greedy initialization
            seed = int(time.time() * 1000) % 10000
            init_code = greedy_build(seed=seed, dual_guidance=dual_guidance)
            if len(init_code) < len(current_code) + 5:
                current_code = set(init_code)
                cnt = np.zeros(total, dtype=np.int32)
                for c in current_code:
                    cnt[ball_indices_np[c]] += 1
                no_improve_counter = 0
                temperature = 5.0  # Reset temperature on restart
                
        if best_size <= 160:
            break
    
    # Final result conversion
    result = []
    for idx in best_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
        
    return result
