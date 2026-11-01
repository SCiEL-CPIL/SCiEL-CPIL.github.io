"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
import random


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius
    'radius'. Uses a greedy set-cover approach, then simulated annealing for optimization.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    if q == 3 and n == 7 and radius == 1:
        # Precompute all words and their indices
        all_words = list(itertools.product(range(3), repeat=7))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        total_words = len(all_words)
        
        # Precompute coverage masks for each possible codeword position
        # For each word index, compute which words it covers (itself + distance 1)
        coverage_masks = np.zeros((total_words, total_words), dtype=np.int8)
        for i, w in enumerate(all_words):
            coverage_masks[i, i] = 1
            for pos in range(n):
                for val in range(q):
                    if val != w[pos]:
                        new_list = list(w)
                        new_list[pos] = val
                        new_word = tuple(new_list)
                        coverage_masks[i, word_to_idx[new_word]] = 1
        
        # Convert to column-wise view for fast operations
        # cov_col[j] = set of codeword indices that cover word j
        cov_col = coverage_masks.T  # shape (total_words, total_words), cov_col[word_idx, cw_idx] = 1 if cw covers word
        
        # Start with tetracode product construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_codewords = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=3):
                initial_codewords.append(tuple(h + list(tail)))
        
        # Convert to indices
        codeword_indices = [word_to_idx[cw] for cw in initial_codewords]
        
        # Compute coverage counts: how many codewords cover each word
        cov_counts = np.zeros(total_words, dtype=np.int32)
        for cw_idx in codeword_indices:
            cov_counts += coverage_masks[cw_idx]
        
        # Greedily add codewords to cover remaining words
        while cov_counts.min() == 0 and time.time() < deadline - 40:
            uncovered_mask = (cov_counts == 0)
            # For each candidate, count how many uncovered words it covers
            new_covered_counts = (coverage_masks & uncovered_mask[None, :]).sum(axis=1)
            best_cw = np.argmax(new_covered_counts)
            if new_covered_counts[best_cw] <= 0:
                break
            codeword_indices.append(best_cw)
            cov_counts += coverage_masks[best_cw]
        
        # Simulated annealing to reduce code size
        best_indices = list(codeword_indices)
        best_size = len(best_indices)
        current_indices = list(codeword_indices)
        current_size = len(current_indices)
        
        # Build coverage counts for current solution
        cur_cov_counts = np.zeros(total_words, dtype=np.int32)
        for cw_idx in current_indices:
            cur_cov_counts += coverage_masks[cw_idx]
        
        # Track which codeword is at which position
        cw_position = {}
        for pos, cw_idx in enumerate(current_indices):
            if cw_idx not in cw_position:
                cw_position[cw_idx] = []
            cw_position[cw_idx].append(pos)
        
        sa_deadline = deadline - 5
        temperature = 1.0
        cooling_rate = 0.9995
        
        while time.time() < sa_deadline:
            if current_size <= 1:
                break
            
            # Try to remove a random codeword
            pos = random.randint(0, current_size - 1)
            cw_to_remove = current_indices[pos]
            
            # Check if removing this codeword keeps all words covered
            # A word becomes uncovered if its coverage count drops to 0
            new_cov = cur_cov_counts - coverage_masks[cw_to_remove]
            would_uncover = (new_cov == 0).sum()
            
            if would_uncover == 0:
                # Safe removal - always accept
                cur_cov_counts = new_cov
                current_indices.pop(pos)
                current_size -= 1
                # Update position tracking
                for i, p in enumerate(cw_position[cw_to_remove]):
                    if p == pos:
                        cw_position[cw_to_remove].remove(p)
                        break
                # Shift positions for codewords after pos
                for cw_idx in cw_position:
                    for i, p in enumerate(cw_position[cw_idx]):
                        if p > pos:
                            cw_position[cw_idx][i] = p - 1
                if not cw_position[cw_to_remove]:
                    del cw_position[cw_to_remove]
                
                if current_size < best_size:
                    best_size = current_size
                    best_indices = list(current_indices)
            else:
                # Probabilistic acceptance (simulated annealing)
                if random.random() < np.exp(-would_uncover / temperature):
                    # Accept worse state temporarily
                    cur_cov_counts = new_cov
                    current_indices.pop(pos)
                    current_size -= 1
                    for i, p in enumerate(cw_position[cw_to_remove]):
                        if p == pos:
                            cw_position[cw_to_remove].remove(p)
                            break
                    for cw_idx in cw_position:
                        for i, p in enumerate(cw_position[cw_idx]):
                            if p > pos:
                                cw_position[cw_idx][i] = p - 1
                    if not cw_position[cw_to_remove]:
                        del cw_position[cw_to_remove]
                
                # Try to add a codeword that covers the most uncovered words
                uncovered_mask = (cur_cov_counts == 0)
                if uncovered_mask.any():
                    new_covered_counts = (coverage_masks & uncovered_mask[None, :]).sum(axis=1)
                    # Weight by how much it helps
                    best_add = np.argmax(new_covered_counts)
                    if new_covered_counts[best_add] > 0:
                        cur_cov_counts += coverage_masks[best_add]
                        current_indices.append(best_add)
                        current_size += 1
                        if best_add not in cw_position:
                            cw_position[best_add] = []
                        cw_position[best_add].append(current_size - 1)
            
            temperature *= cooling_rate
            if temperature < 0.01:
                temperature = 0.01
        
        # Use the best solution found
        final_indices = best_indices
        
        # Final redundancy removal pass (deterministic)
        final_cov_counts = np.zeros(total_words, dtype=np.int32)
        for cw_idx in final_indices:
            final_cov_counts += coverage_masks[cw_idx]
        
        i = 0
        while i < len(final_indices) and time.time() < deadline - 2:
            cw_to_remove = final_indices[i]
            new_cov = final_cov_counts - coverage_masks[cw_to_remove]
            if (new_cov == 0).sum() == 0:
                final_cov_counts = new_cov
                final_indices.pop(i)
            else:
                i += 1
        
        return [list(all_words[idx]) for idx in final_indices]
    
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
