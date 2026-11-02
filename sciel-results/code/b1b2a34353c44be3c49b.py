import numpy as np
import itertools
import time
import random

# Precomputation for q=3, n=6
Q = 3
N = 6
NUM_WORDS = Q ** N

# Generate all words
all_words = np.array(list(itertools.product(range(Q), repeat=N)), dtype=np.int32)

# Precompute a mapping from word tuple to index
word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

# Precompute balls: for each codeword c, ball_indices[c] = array of indices within distance 1
ball_indices = []
ball_matrix = np.zeros((NUM_WORDS, NUM_WORDS), dtype=np.int8)
for i in range(NUM_WORDS):
    c = all_words[i]
    ball_indices.append([i])
    ball_matrix[i, i] = 1
    for pos in range(N):
        for val in range(Q):
            if val == c[pos]:
                continue
            neighbor = c.copy()
            neighbor[pos] = val
            idx = word_to_idx[tuple(neighbor)]
            ball_indices[-1].append(idx)
            ball_matrix[i, idx] = 1

ball_indices = [np.array(bi, dtype=np.int32) for bi in ball_indices]

def get_tetracode_code():
    """Fallback construction yielding 81 codewords."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=N - 4)]
    return code

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        if q is None or n is None or radius is None:
            return []
        all_w = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
        nw = q ** n
        wti = {tuple(w): i for i, w in enumerate(all_w)}
        bm = np.zeros((nw, nw), dtype=np.int8)
        for i in range(nw):
            c = all_w[i]
            bm[i, i] = 1
            for pos in range(n):
                for val in range(q):
                    if val == c[pos]:
                        continue
                    nb = c.copy()
                    nb[pos] = val
                    bm[i, wti[tuple(nb)]] = 1
        covered = np.zeros(nw, dtype=bool)
        code = []
        while not np.all(covered):
            scores = bm.dot(~covered)
            c = int(np.argmax(scores))
            if scores[c] == 0:
                break
            code.append(c)
            covered[bm[c] == 1] = True
        return [list(all_w[c]) for c in code]

    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    best_code = get_tetracode_code()
    best_size = len(best_code)

    def greedy_with_reduction():
        """Run vectorized greedy set cover with randomization, then reduction."""
        covered = np.zeros(NUM_WORDS, dtype=bool)
        uncovered_count = NUM_WORDS
        code = []
        
        while uncovered_count > 0:
            # Compute scores: number of uncovered words in each ball
            # Use int8 dot product for speed
            scores = ball_matrix.dot((~covered).astype(np.int8))
            
            max_score = scores.max()
            if max_score == 0:
                break
                
            # Randomized selection from top candidates
            threshold = max_score * 0.9
            candidates = np.where(scores >= threshold)[0]
            c = candidates[np.random.randint(len(candidates))]
            
            code.append(c)
            # Update covered in-place
            bi = ball_indices[c]
            new_uncovered = (~covered[bi]).sum()
            uncovered_count -= new_uncovered
            covered[bi] = True
            
        # Reduction Phase
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code:
            cov_counts[ball_indices[c]] += 1
            
        changed = True
        while changed:
            changed = False
            indices = list(range(len(code)))
            np.random.shuffle(indices)
            
            for idx in indices:
                c = code[idx]
                # Check if all words in this codeword's ball are covered by >1 codeword
                if np.all(cov_counts[ball_indices[c]] > 1):
                    cov_counts[ball_indices[c]] -= 1
                    code.pop(idx)
                    changed = True
                    break
                    
        return code

    def simulated_annealing(initial_code, max_iterations=50000):
        """Simulated annealing on the binary codeword selection problem."""
        code = list(initial_code)
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code:
            cov_counts[ball_indices[c]] += 1
        
        uncovered = np.sum(cov_counts == 0)
        current_energy = 1000.0 * uncovered + 1.0 * len(code)
        
        best_code = list(code)
        best_energy = current_energy
        
        T_start = 50.0
        T_end = 0.1
        cooling_rate = (T_end / T_start) ** (1.0 / max_iterations)
        T = T_start
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
                
            # Choose a random move
            move_type = np.random.randint(3)
            
            if move_type == 0 and len(code) > 0:
                # Remove a codeword
                idx = np.random.randint(len(code))
                c_remove = code[idx]
                
                # Check if removal keeps coverage valid
                # Words that would become uncovered
                would_lose = cov_counts[ball_indices[c_remove]] == 1
                new_uncovered = np.sum(would_lose)
                delta_energy = 1000.0 * new_uncovered - 1.0
                
                if np.random.rand() < np.exp(-delta_energy / T) or delta_energy < 0:
                    code.pop(idx)
                    cov_counts[ball_indices[c_remove]] -= 1
                    current_energy += delta_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = list(code)
                        
            elif move_type == 1:
                # Add a codeword
                c_add = np.random.randint(NUM_WORDS)
                if c_add in code:
                    continue
                    
                # Check how many new words it covers
                new_covered = np.sum(cov_counts[ball_indices[c_add]] == 0)
                delta_energy = -1000.0 * new_covered + 1.0
                
                if np.random.rand() < np.exp(-delta_energy / T) or delta_energy < 0:
                    code.append(c_add)
                    cov_counts[ball_indices[c_add]] += 1
                    current_energy += delta_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = list(code)
                        
            elif move_type == 2 and len(code) > 0:
                # Swap: remove one, add one
                idx = np.random.randint(len(code))
                c_remove = code[idx]
                c_add = np.random.randint(NUM_WORDS)
                if c_add == c_remove or c_add in code:
                    continue
                
                br = ball_indices[c_remove]
                ba = ball_indices[c_add]
                
                # Words that would become uncovered if we remove c_remove
                # (only those with cov_count == 1 in ball[c_remove])
                would_lose = (cov_counts[br] == 1)
                
                # After removal, cov_counts[br] decreases by 1
                # New uncovered from addition: words in ball[c_add] with cov_count == 0
                # BUT some words in ball[c_add] ∩ ball[c_remove] had their count decreased
                # A word w in ball[c_add] is newly covered if:
                #   - w not in ball[c_remove]: cov_counts[w] == 0
                #   - w in ball[c_remove]: cov_counts[w] - 1 == 0, i.e., cov_counts[w] == 1
                # Net new covered = (words in ba\br with cov==0) + (words in ba∩br with cov==1)
                # Net lose = words in br\ba with cov==1 (words only covered by c_remove)
                #           + words in br∩ba with cov==1 that are NOT newly covered by c_add
                #           Actually: lose = words in br with cov==1 AND NOT in ba
                #                     (words in br∩ba with cov==1 are saved by c_add)
                
                # Simpler exact computation:
                # lose_count = sum over w in br: [cov_counts[w]==1 AND w not in ba]
                # gain_count = sum over w in ba: [w not in br ? cov_counts[w]==0 : cov_counts[w]==1]
                
                in_br = np.zeros(NUM_WORDS, dtype=bool)
                in_br[br] = True
                in_ba = np.zeros(NUM_WORDS, dtype=bool)
                in_ba[ba] = True
                
                # lose: in br, not in ba, cov==1
                lose_mask = in_br & ~in_ba
                lose_count = np.sum(cov_counts[lose_mask] == 1)
                
                # gain: in ba, not in br, cov==0  OR  in ba, in br, cov==1
                gain_mask1 = in_ba & ~in_br
                gain_mask2 = in_ba & in_br
                gain_count = np.sum(cov_counts[gain_mask1] == 0) + np.sum(cov_counts[gain_mask2] == 1)
                
                delta_energy = 1000.0 * (lose_count - gain_count)
                # Codeword count unchanged in swap
                
                if np.random.rand() < np.exp(-delta_energy / T) or delta_energy < 0:
                    code[idx] = c_add
                    cov_counts[br] -= 1
                    cov_counts[ba] += 1
                    current_energy += delta_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = list(code)
            
            T *= cooling_rate
        
        return best_code

    # Phase 1: Find good initial solutions with greedy + reduction
    # Since greedy is now much faster, we can run many more restarts
    initial_solutions = []
    restart_count = 0
    while time.time() < deadline - 20: # Reserve 20s for SA and safety
        sol = greedy_with_reduction()
        if len(sol) < best_size:
            best_size = len(sol)
            best_code = [list(all_words[c]) for c in sol]
        initial_solutions.append(sol)
        restart_count += 1
        
    # Phase 2: Simulated annealing from best initial solutions
    # Sort initial solutions by size, take the best few
    initial_solutions.sort(key=len)
    
    # Run SA on the best few solutions found
    num_sa_trials = min(10, len(initial_solutions))
    for i in range(num_sa_trials):
        if time.time() > deadline - 5:
            break
        sol = initial_solutions[i]
        
        # Run SA with different temperature schedules
        for trial in range(3):
            if time.time() > deadline - 2:
                break
            
            sa_result = simulated_annealing(sol, max_iterations=20000)
            
            if len(sa_result) < best_size:
                best_size = len(sa_result)
                best_code = [list(all_words[c]) for c in sa_result]
                
                if best_size <= 73:
                    # Found the best known, try to improve further
                    pass

    # Phase 3: Final polish - aggressive local search on the best solution
    if time.time() < deadline - 3:
        best_sol = [c for c in range(len(best_code)) if True]
        # Convert best_code back to indices
        best_idx = []
        for cw in best_code:
            best_idx.append(word_to_idx[tuple(cw)])
        
        # Try removals in random order, multiple passes
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in best_idx:
            cov_counts[ball_indices[c]] += 1
        
        polish_changed = True
        while polish_changed and time.time() < deadline - 3:
            polish_changed = False
            idx_list = list(range(len(best_idx)))
            np.random.shuffle(idx_list)
            for idx in idx_list:
                if time.time() > deadline - 3:
                    break
                c = best_idx[idx]
                if np.all(cov_counts[ball_indices[c]] > 1):
                    cov_counts[ball_indices[c]] -= 1
                    best_idx.pop(idx)
                    polish_changed = True
                    break
        
        # Try swaps to reduce size
        if time.time() < deadline - 3:
            for _ in range(200):
                if time.time() > deadline - 3:
                    break
                if len(best_idx) == 0:
                    break
                i = np.random.randint(len(best_idx))
                j = np.random.randint(NUM_WORDS)
                if j == best_idx[i] or j in best_idx:
                    continue
                bi = ball_indices[best_idx[i]]
                bj = ball_indices[j]
                # Check if swap helps: remove i, add j
                # Lose: words in bi with cov==1 and not in bj
                # Gain: words in bj with (cov==0 if not in bi) or (cov==1 if in bi)
                in_bi = np.zeros(NUM_WORDS, dtype=bool)
                in_bi[bi] = True
                lose = np.sum((cov_counts[bi] == 1) & ~in_bi[bj])
                # Actually need to be more careful
                # Words in bi not in bj: lose if cov==1
                # Words in bj not in bi: gain if cov==0
                # Words in both: no net change (cov decreases by 1 from bi removal, increases by 1 from bj addition)
                bi_only = bi[np.isin(bi, bj, invert=True)]
                bj_only = bj[np.isin(bj, bi, invert=True)]
                lose_count = np.sum(cov_counts[bi_only] == 1)
                gain_count = np.sum(cov_counts[bj_only] == 0)
                if gain_count > lose_count:
                    cov_counts[bi] -= 1
                    cov_counts[bj] += 1
                    best_idx[i] = j
                    polish_changed = True
        
        if len(best_idx) < best_size:
            best_size = len(best_idx)
            best_code = [list(all_words[c]) for c in best_idx]

    return best_code
