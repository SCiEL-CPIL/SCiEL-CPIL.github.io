"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
import random


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius
    'radius'. Uses greedy set-cover then fast local search with swap moves.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    if q == 3 and n == 7 and radius == 1:
        # Precompute all words
        all_words = list(itertools.product(range(3), repeat=7))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        total_words = len(all_words)

        # Build coverage mask: cov[cw_idx, word_idx] = 1 if cw covers word
        # Each codeword covers itself + 2*n neighbors = 15 words
        cov = np.zeros((total_words, total_words), dtype=np.int8)
        for i, w in enumerate(all_words):
            cov[i, i] = 1
            for pos in range(n):
                for val in range(q):
                    if val != w[pos]:
                        nw = list(w)
                        nw[pos] = val
                        cov[i, word_to_idx[tuple(nw)]] = 1

        # Precompute for each word, the list of codewords that cover it (inverse)
        # This is just cov.T but we'll use it as a matrix
        covered_by = cov.T  # covered_by[word_idx, cw_idx] = 1

        # Phase 1: Pure greedy set cover
        cov_counts = np.zeros(total_words, dtype=np.int32)
        codeword_list = []
        
        while True:
            if time.time() > deadline - 60:
                break
            uncovered = (cov_counts == 0)
            if not uncovered.any():
                break
            # For each candidate, count newly covered words
            gains = (cov & uncovered[None, :]).sum(axis=1)
            best = int(np.argmax(gains))
            if gains[best] == 0:
                break
            codeword_list.append(best)
            cov_counts += cov[best]
        
        # Phase 2: Local search with swap and delete moves
        # Represent current solution as a set of codeword indices
        current = list(codeword_list)
        cur_cov = np.zeros(total_words, dtype=np.int32)
        for cw in current:
            cur_cov += cov[cw]
        
        best_solution = list(current)
        best_size = len(current)
        
        # Helper: try to remove redundant codewords
        def try_remove_redundant(sol, cov_counts):
            """Remove any codeword whose removal doesn't uncover words. Return new sol, cov_counts, removed_count."""
            removed = 0
            i = 0
            while i < len(sol):
                if time.time() > deadline - 30:
                    break
                cw = sol[i]
                new_cov = cov_counts - cov[cw]
                if (new_cov == 0).sum() == 0:
                    sol.pop(i)
                    cov_counts = new_cov
                    removed += 1
                else:
                    i += 1
            return sol, cov_counts, removed

        # Apply initial redundancy removal
        current, cur_cov, _ = try_remove_redundant(current, cur_cov)
        if len(current) < best_size:
            best_size = len(current)
            best_solution = list(current)
        
        # Local search: try swaps (replace one codeword with another) and deletions
        # We'll iterate: for each position, try replacing with a better codeword
        no_improvement_count = 0
        max_no_improvement = 50
        
        while time.time() < deadline - 20:
            improved = False
            cur_size = len(current)
            
            # Try deletions first
            for i in range(cur_size):
                if time.time() > deadline - 20:
                    break
                cw = current[i]
                new_cov = cur_cov - cov[cw]
                if (new_cov == 0).sum() == 0:
                    # Safe deletion
                    current.pop(i)
                    cur_cov = new_cov
                    improved = True
                    no_improvement_count = 0
                    if len(current) < best_size:
                        best_size = len(current)
                        best_solution = list(current)
                    break  # restart loop after modification
            
            if improved:
                continue
            
            # Try swaps: replace current[i] with some other codeword j
            # A swap is beneficial if the new codeword covers words that were only
            # covered by the old one, OR if it allows further reductions
            # Simpler: try replacing and see if we can then remove more
            for i in range(len(current)):
                if time.time() > deadline - 20:
                    break
                old_cw = current[i]
                old_cov = cov[old_cw]
                
                # Find words that are ONLY covered by old_cw (critical words)
                # These are words where cur_cov[w] == 1 and old_cov[w] == 1
                critical = (cur_cov == 1) & (old_cov == 1)
                if not critical.any():
                    # old_cw is redundant! (should have been caught by deletion)
                    continue
                
                # We need a replacement that covers all critical words
                # Find candidates that cover all critical words
                # For speed, just try a few random candidates
                for attempt in range(20):
                    if time.time() > deadline - 20:
                        break
                    new_cw = random.randint(0, total_words - 1)
                    if new_cw == old_cw:
                        continue
                    new_cov_mask = cov[new_cw]
                    # Check if new_cw covers all critical words
                    if (new_cov_mask[critical]).all():
                        # Check if new_cw is already in the list
                        if new_cw in current:
                            continue
                        # Do the swap
                        current[i] = new_cw
                        cur_cov = cur_cov - old_cov + new_cov_mask
                        improved = True
                        no_improvement_count = 0
                        # Try to remove redundancies after swap
                        current, cur_cov, rem = try_remove_redundant(current, cur_cov)
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)
                        break
                if improved:
                    break
            
            if not improved:
                no_improvement_count += 1
                if no_improvement_count >= max_no_improvement:
                    # Perturb: randomly add a codeword to escape local minimum
                    if len(current) < total_words:
                        rand_cw = random.randint(0, total_words - 1)
                        if rand_cw not in current:
                            current.append(rand_cw)
                            cur_cov += cov[rand_cw]
                            no_improvement_count = 0
        
        # Final cleanup
        final = list(best_solution)
        final_cov = np.zeros(total_words, dtype=np.int32)
        for cw in final:
            final_cov += cov[cw]
        final, final_cov, _ = try_remove_redundant(final, final_cov)
        
        # Verify coverage
        if final_cov.min() >= 1:
            return [list(all_words[idx]) for idx in final]
        
        # Fallback: use greedy solution
        return [list(all_words[idx]) for idx in codeword_list]
    
    # General fallback: greedy set cover
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
# EVOLVE-BLOCK-END
