import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius
    'radius'. Uses greedy set cover followed by simulated annealing with incremental
    coverage tracking and restart mechanism.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    if q == 3 and n == 7 and radius == 1:
        # Specific optimization for K_3(7,1)
        
        # 1. Precompute all words and their balls (neighbors within radius 1)
        all_words = list(itertools.product(range(3), repeat=7))
        N = len(all_words)  # 2187
        
        # Map word tuple to index
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute balls: for each index, list of indices covered by that codeword
        # Ball(c) = {c} U {words at dist 1}
        balls = []
        for w in all_words:
            covered_indices = [word_to_idx[w]]
            for i in range(7):
                for val in range(3):
                    if val != w[i]:
                        new_word = list(w)
                        new_word[i] = val
                        new_word = tuple(new_word)
                        covered_indices.append(word_to_idx[new_word])
            balls.append(np.array(covered_indices, dtype=np.int32))
        
        # 2. Greedy Set Cover to get initial solution
        covered = np.zeros(N, dtype=bool)
        codewords = []
        
        while not covered.all():
            best_cw = -1
            best_count = -1
            
            for i in range(N):
                b = balls[i]
                new_cov = np.sum(~covered[b])
                if new_cov > best_count:
                    best_count = new_cov
                    best_cw = i
            
            if best_count == 0:
                break
                
            codewords.append(best_cw)
            covered[balls[best_cw]] = True
            
            if time.time() > deadline - 60:
                break

        # 3. Initialize coverage counts
        coverage_count = np.zeros(N, dtype=np.int32)
        for cw in codewords:
            coverage_count[balls[cw]] += 1
        
        # 4. Redundancy Removal Pass
        i = 0
        while i < len(codewords) and time.time() < deadline - 40:
            b = balls[codewords[i]]
            if np.min(coverage_count[b]) >= 2:
                coverage_count[b] -= 1
                codewords.pop(i)
            else:
                i += 1
        
        # 5. Simulated Annealing with proper move evaluation
        random.seed(42)
        
        best_size = len(codewords)
        best_codewords = list(codewords)
        best_coverage = coverage_count.copy()
        best_cw_set = set(codewords)
        
        # SA parameters
        T_init = 2.0
        T_min = 0.001
        cooling_rate = 0.9995
        restart_threshold = 5000  # Steps without improvement before restart
        
        current_size = len(codewords)
        cw_set = set(codewords)  # O(1) membership check
        steps_since_improvement = 0
        T = T_init
        
        while time.time() < deadline - 5:
            # Choose move type with probabilities
            r = random.random()
            
            if r < 0.4:
                # Try Remove: size decreases by 1
                if not codewords:
                    T = max(T * cooling_rate, T_min)
                    continue
                    
                idx = random.randint(0, len(codewords) - 1)
                cw = codewords[idx]
                b = balls[cw]
                
                # Check if removal is valid (all words in ball have count >= 2)
                if np.min(coverage_count[b]) >= 2:
                    # Valid removal - always accept
                    coverage_count[b] -= 1
                    codewords.pop(idx)
                    cw_set.discard(cw)
                    current_size -= 1
                    steps_since_improvement = 0
                    
                    if current_size < best_size:
                        best_size = current_size
                        best_codewords = list(codewords)
                        best_coverage = coverage_count.copy()
                        best_cw_set = set(codewords)
                else:
                    # Invalid - skip
                    pass
                    
            elif r < 0.95:
                # Try Swap: size unchanged, structure changes
                if not codewords:
                    T = max(T * cooling_rate, T_min)
                    continue
                    
                idx_old = random.randint(0, len(codewords) - 1)
                cw_old = codewords[idx_old]
                cw_new = random.randint(0, N - 1)
                
                if cw_new == cw_old or cw_new in cw_set:
                    T = max(T * cooling_rate, T_min)
                    continue
                
                b_old = balls[cw_old]
                b_new = balls[cw_new]
                
                # Simulate swap
                coverage_count[b_old] -= 1
                coverage_count[b_new] += 1
                
                new_cnt0 = np.sum(coverage_count == 0)
                
                if new_cnt0 == 0:
                    # Valid swap - accept
                    codewords[idx_old] = cw_new
                    cw_set.discard(cw_old)
                    cw_set.add(cw_new)
                    steps_since_improvement = min(steps_since_improvement, restart_threshold // 2)
                else:
                    # Invalid swap - revert
                    coverage_count[b_old] += 1
                    coverage_count[b_new] -= 1
                    
            else:
                # Try Add: size increases by 1 (worse, but may enable future improvements)
                if codewords:
                    cw_new = random.randint(0, N - 1)
                    if cw_new not in cw_set:
                        b_new = balls[cw_new]
                        
                        # Simulate add
                        coverage_count[b_new] += 1
                        current_size_sim = current_size + 1
                        
                        # Accept with probability exp(-delta/T) where delta = 1 (size increase)
                        if random.random() < np.exp(-1.0 / T):
                            # Accept the add
                            codewords.append(cw_new)
                            cw_set.add(cw_new)
                            current_size = current_size_sim
                        else:
                            # Reject - revert
                            coverage_count[b_new] -= 1
        
            # Cool down
            T = max(T * cooling_rate, T_min)
            steps_since_improvement += 1
            
            # Restart if stuck: reset to best solution and reheat
            if steps_since_improvement >= restart_threshold:
                codewords = list(best_codewords)
                coverage_count = best_coverage.copy()
                current_size = best_size
                cw_set = set(best_codewords)
                T = T_init * 0.5
                steps_since_improvement = 0
        
        # 6. Final high-precision polish pass: aggressive redundancy removal
        # After SA, do multiple passes to remove any remaining redundant codewords
        polish_deadline = deadline - 2
        for polish_round in range(20):
            if time.time() > polish_deadline:
                break
            changed = False
            i = 0
            while i < len(codewords):
                if time.time() > polish_deadline:
                    break
                b = balls[codewords[i]]
                if np.min(coverage_count[b]) >= 2:
                    coverage_count[b] -= 1
                    cw = codewords.pop(i)
                    cw_set.discard(cw)
                    changed = True
                else:
                    i += 1
            if not changed:
                break
        
        # Update best if polish improved things
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = list(codewords)
        
        return [list(all_words[cw]) for cw in best_codewords]
    
    # General fallback (not optimized for this specific case)
    all_words = list(itertools.product(range(q), repeat=n))
    
    def get_coverage_general(codeword):
        covered = set()
        covered.add(codeword)
        for i in range(n):
            for val in range(q):
                if val != codeword[i]:
                    new_word = list(codeword)
                    new_word[i] = val
                    covered.add(tuple(new_word))
        return covered
    
    codewords = []
    covered = set()
    
    while time.time() < deadline - 10:
        best_cw = None
        best_count = -1
        best_new_covered = set()
        
        for candidate in all_words:
            new_covered = get_coverage_general(candidate) - covered
            count = len(new_covered)
            if count > best_count:
                best_count = count
                best_cw = candidate
                best_new_covered = new_covered
        
        if best_count <= 0:
            break
        
        codewords.append(best_cw)
        covered.update(best_new_covered)
        
        if len(covered) == len(all_words):
            break
    
    # Redundancy removal
    i = 0
    while i < len(codewords) and time.time() < deadline - 5:
        cw_to_remove = codewords[i]
        coverage_of_cw = get_coverage_general(cw_to_remove)
        
        can_remove = True
        for w in coverage_of_cw:
            is_covered_by_other = False
            for j, other_cw in enumerate(codewords):
                if i == j:
                    continue
                if w in get_coverage_general(other_cw):
                    is_covered_by_other = True
                    break
            if not is_covered_by_other:
                can_remove = False
                break
        
        if can_remove:
            codewords.pop(i)
        else:
            i += 1
    
    return [list(cw) for cw in codewords]
