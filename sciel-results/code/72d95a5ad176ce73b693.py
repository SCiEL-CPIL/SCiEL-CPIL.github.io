import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius
    'radius'. Uses a greedy set-cover approach followed by local search (deletions and swaps).
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    if q != 3 or n != 7 or radius != 1:
        # Fallback for other cases, though the task is specific
        # Simple greedy for general case
        all_words = list(itertools.product(range(q), repeat=n))
        def get_coverage(cw):
            covered = set([cw])
            for i in range(n):
                for val in range(q):
                    if val != cw[i]:
                        new_cw = list(cw)
                        new_cw[i] = val
                        covered.add(tuple(new_cw))
            return covered
        
        covered = set()
        codewords = []
        uncovered = set(all_words)
        
        while uncovered:
            best_cw = None
            best_count = -1
            for cw in uncovered: # Heuristic: pick from uncovered to maximize new coverage? 
                # Actually standard greedy picks from all candidates
                pass
            
            # Proper greedy
            best_cw = None
            best_count = -1
            for candidate in all_words:
                new_covered = get_coverage(candidate) & uncovered
                if len(new_covered) > best_count:
                    best_count = len(new_covered)
                    best_cw = candidate
                    if best_count == len(uncovered):
                        break
            
            if best_cw is None:
                break
                
            codewords.append(best_cw)
            covered.update(get_coverage(best_cw))
            uncovered -= get_coverage(best_cw)
            
        return [list(cw) for cw in codewords]

    # Specific optimization for q=3, n=7, radius=1
    # Total words: 3^7 = 2187
    
    # 1. Generate all words and map to indices
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = len(all_words)
    
    # 2. Precompute coverage masks
    # M[i] is a boolean array of length N, True if word j is covered by word i
    # We use int8 or bool for memory efficiency, but numpy handles bool well
    # Actually, for fast counting, we might use int8 and sum, but boolean logic is fine for min check if we maintain counts.
    
    # Let's use a 2D array of uint8? Or just list of sets?
    # Numpy vectorization is key.
    # M will be shape (N, N). 2187 * 2187 bytes ~ 4.7 MB. Very small.
    M = np.zeros((N, N), dtype=np.uint8)
    
    # Optimization: Generate coverage vectors
    # For each word i, it covers itself and words at dist 1.
    # We can compute this efficiently.
    # But 2187 * 15 operations is trivial.
    for i, w in enumerate(all_words):
        M[i, i] = 1
        for pos in range(7):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val != orig_val:
                    new_word = list(w)
                    new_word[pos] = new_val
                    j = word_to_idx[tuple(new_word)]
                    M[i, j] = 1
                    
    # 3. Greedy Initialization
    def greedy_init(seed=0):
        rng = random.Random(seed)
        covered_count = np.zeros(N, dtype=np.uint8)
        S = set()
        
        # Randomize order of candidates to break ties?
        # Standard greedy: pick best uncovered coverage.
        # To allow different initializations, we can add small random noise to coverage counts.
        
        while True:
            if np.min(covered_count) > 0:
                break
            
            # Calculate marginal gain for each candidate not in S
            # gain[i] = sum(M[i] * (covered_count == 0))
            # We only care about uncovered words (where covered_count == 0)
            uncovered_mask = (covered_count == 0)
            
            # Compute gains
            # M is (N, N). uncovered_mask is (N,).
            # gains = M @ uncovered_mask ? No, M[i, j] is 1 if i covers j.
            # We want number of j such that M[i, j]=1 and uncovered_mask[j]=1.
            # This is M.dot(uncovered_mask)
            gains = M.dot(uncovered_mask)
            
            # Add noise to break ties and explore different paths
            noise = rng.random(N) * 0.5
            scores = gains + noise
            
            # Pick best
            best_idx = np.argmax(scores)
            
            # Add to set
            S.add(best_idx)
            covered_count += M[best_idx]
            
        return S, covered_count

    best_solution_size = N
    best_S = set()
    
    # Multiple restarts
    num_restarts = 20
    time_per_restart = (deadline - start_time) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        S, cover_count = greedy_init(seed=r)
        
        # Local Search
        # We maintain S as a list for easy iteration, and a set for membership check
        S_list = list(S)
        S_set = set(S_list)
        
        # cover_count is already computed
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Try Deletions
            # Shuffle S_list to randomize deletion order
            rng.shuffle(S_list)
            
            for i in S_list:
                if i not in S_set:
                    continue
                    
                # Tentatively remove i
                cover_count -= M[i]
                
                if np.min(cover_count) > 0:
                    # Valid removal
                    S_set.remove(i)
                    improved = True
                else:
                    # Revert
                    cover_count += M[i]
            
            # 2. Try Swaps
            # For each i in S, try swapping with a random j not in S
            # If valid, accept.
            # We do this for a limited number of attempts per restart to avoid getting stuck
            # or we just loop until no swap improves? 
            # Swaps don't reduce size, they change configuration.
            # We accept a swap if it maintains validity. This allows us to reach a state
            # where a deletion becomes possible.
            
            # Let's do a fixed number of swap attempts based on time
            swap_attempts = 1000
            attempts_done = 0
            while attempts_done < swap_attempts and time.time() < deadline - 5:
                if not S_set:
                    break
                    
                i = rng.choice(list(S_set))
                # Pick random j
                j = rng.randint(0, N - 1)
                
                if j in S_set:
                    continue
                
                # Tentative swap: remove i, add j
                cover_count -= M[i]
                cover_count += M[j]
                
                if np.min(cover_count) > 0:
                    # Valid swap
                    S_set.remove(i)
                    S_set.add(j)
                    improved = True
                else:
                    # Revert
                    cover_count -= M[j]
                    cover_count += M[i]
                
                attempts_done += 1
                
                # If we made a swap, we should probably re-check deletions immediately?
                # The outer loop handles that.
        
        if len(S_set) < best_solution_size:
            best_solution_size = len(S_set)
            best_S = set(S_set)
            
    # Final cleanup: ensure we return the best found
    # Convert indices back to words
    result = [list(all_words[i]) for i in best_S]
    
    # Sort for determinism? Not required, but good practice.
    # The problem doesn't specify order.
    
    return result

