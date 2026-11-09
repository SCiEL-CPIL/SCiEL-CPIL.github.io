"""Covering code search for K_q(n, radius) using Tetracode initialization + intensive local search."""
import time
import numpy as np
import random
import itertools

_DEADLINE = 118.0  # seconds; evaluator kills at 150s, keep safety margin


def _idx_to_word(idx, n, q):
    """Convert a linear index to a word over {0,1,...} with n digits (base q)."""
    digits = [0] * n
    for i in range(n):
        digits[i] = idx % q
        idx //= q
    return digits


def _word_to_idx(word, q, n):
    """Convert a word to a linear index."""
    idx = 0
    for i in range(n):
        idx = idx * q + word[i]
    return idx


def _build_coverage_matrix(q, n, radius):
    """
    Build a coverage matrix M of shape (total, total) where
    M[i, j] = 1 if word i covers word j (i.e., hamming(word_i, word_j) <= radius).
    """
    total = q ** n
    all_words = np.array([_idx_to_word(i, n, q) for i in range(total)], dtype=np.int8)
    
    # Compute pairwise Hamming distances
    diff = all_words[:, np.newaxis, :] != all_words[np.newaxis, :, :]
    dists = diff.sum(axis=2)
    
    M = (dists <= radius).astype(np.int8)
    return M, all_words


def _direct_sum_tetracode(q, n, radius):
    """
    Direct sum of ternary tetracode (9 words of length 4) with all tails.
    This provides a valid covering code of size 9 * q^(n-4).
    For q=3, n=6, size is 9 * 3^2 = 81.
    """
    if q != 3 or n < 4:
        return None
    
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    if n == 4:
        return [list(h) for h in tetracode]
    
    tails = list(itertools.product(range(q), repeat=n - 4))
    return [h + list(tail) for h in tetracode for tail in tails]


def _verify_code(code_indices, M, total):
    """Verify that every word is covered."""
    if len(code_indices) == 0:
        return total == 0
    selected = np.zeros(total, dtype=bool)
    selected[code_indices] = True
    # Check if M @ selected >= 1 for all rows
    covered = (M @ selected) >= 1
    return covered.all()


def _local_search_polish(code_indices, M, all_words, total, time_limit=60.0, seed=42):
    """
    Intensive local search: greedy deletion followed by single and pair swaps with smart candidates.
    """
    code_indices = list(code_indices)
    start_time = time.time()
    
    # Precompute covers as boolean arrays for speed
    covers = [M[idx] == 1 for idx in code_indices]
    
    # Maintain counts: how many codewords cover each word
    counts = np.zeros(total, dtype=np.int32)
    for cov in covers:
        counts[cov] += 1
        
    code_set = set(code_indices)
    rng = random.Random(seed)
    
    def try_deletion():
        """Attempt to remove redundant codewords. Returns True if any removed."""
        nonlocal code_indices, covers, counts, code_set
        changed = False
        # Iterate backwards to allow popping
        for i in range(len(code_indices) - 1, -1, -1):
            cov = covers[i]
            # If any word in this codeword's ball is covered ONLY by this codeword, we can't remove it
            if (counts[cov] == 1).any():
                continue
            # Remove it
            counts[cov] -= 1
            removed_idx = code_indices.pop(i)
            covers.pop(i)
            code_set.remove(removed_idx)
            changed = True
        return changed

    # Initial greedy deletion
    while try_deletion():
        pass
        
    best_size = len(code_indices)
    best_snapshot = list(code_indices)
    no_improvement_count = 0
    
    def _smart_candidates(cov_target, n_candidates=50):
        """Find candidate words that cover many of the target's covered positions."""
        # Words that are within distance 1 of the target word itself are good candidates
        target_idx = np.where(cov_target)[0]
        if len(target_idx) == 0:
            return []
        
        # Get the center word index
        # Actually, cov_target is a boolean array of what this codeword covers
        # We want other codewords whose balls overlap significantly with cov_target
        # Pick random words and score by overlap
        candidates = []
        for _ in range(n_candidates):
            c = rng.randint(0, total - 1)
            if c in code_set:
                continue
            overlap = np.sum(M[c] & cov_target)
            if overlap > 0:
                candidates.append((overlap, c))
        
        candidates.sort(reverse=True)
        return [c for _, c in candidates[:10]]
    
    def _try_single_swap():
        """Try a single swap. Returns True if it led to a deletion."""
        nonlocal code_indices, covers, counts, code_set, best_size, best_snapshot, no_improvement_count
        if len(code_indices) == 0:
            return False
        
        i = rng.randint(0, len(code_indices) - 1)
        idx_out = code_indices[i]
        cov_out = covers[i]
        
        # Find smart candidates
        cands = _smart_candidates(cov_out, 50)
        if not cands:
            cands = [rng.randint(0, total - 1)]
        
        for idx_in in cands:
            if idx_in in code_set:
                continue
            cov_in = M[idx_in] == 1
            
            # Check if swap is valid
            if not np.all(counts[cov_out] + cov_in[cov_out] >= 2):
                continue
            
            # Perform swap
            counts[cov_out] -= 1
            counts[cov_in] += 1
            code_indices[i] = idx_in
            covers[i] = cov_in
            code_set.remove(idx_out)
            code_set.add(idx_in)
            
            deleted_any = try_deletion()
            
            if deleted_any:
                best_size = len(code_indices)
                best_snapshot = list(code_indices)
                no_improvement_count = 0
                return True
            else:
                # Revert
                counts[cov_in] -= 1
                counts[cov_out] += 1
                code_indices[i] = idx_out
                covers[i] = cov_out
                code_set.remove(idx_in)
                code_set.add(idx_out)
        
        return False
    
    def _try_pair_swap():
        """Try removing 2 and adding 2. Returns True if it led to a net deletion."""
        nonlocal code_indices, covers, counts, code_set, best_size, best_snapshot, no_improvement_count
        if len(code_indices) < 2:
            return False
        
        # Pick two random codewords to remove
        i, j = rng.sample(range(len(code_indices)), 2)
        idx_out1 = code_indices[i]
        idx_out2 = code_indices[j]
        cov_out1 = covers[i]
        cov_out2 = covers[j]
        
        # Combined uncovered risk
        combined = cov_out1 | cov_out2
        
        # Find two candidates that together cover the combined risk
        cands1 = _smart_candidates(cov_out1, 30)
        cands2 = _smart_candidates(cov_out2, 30)
        
        for idx_in1 in cands1[:5]:
            if idx_in1 in code_set or idx_in1 == idx_out2:
                continue
            cov_in1 = M[idx_in1] == 1
            remaining_risk = combined & ~cov_in1
            if not remaining_risk.any():
                continue
            
            for idx_in2 in cands2[:5]:
                if idx_in2 in code_set or idx_in2 == idx_out1 or idx_in2 == idx_out1:
                    continue
                cov_in2 = M[idx_in2] == 1
                if not np.all(counts[combined] + cov_in1[combined] + cov_in2[combined] >= 2):
                    continue
                
                # Perform pair swap
                counts[cov_out1] -= 1
                counts[cov_out2] -= 1
                counts[cov_in1] += 1
                counts[cov_in2] += 1
                code_indices[i] = idx_in1
                code_indices[j] = idx_in2
                covers[i] = cov_in1
                covers[j] = cov_in2
                code_set.remove(idx_out1)
                code_set.remove(idx_out2)
                code_set.add(idx_in1)
                code_set.add(idx_in2)
                
                deleted_any = try_deletion()
                
                if deleted_any:
                    best_size = len(code_indices)
                    best_snapshot = list(code_indices)
                    no_improvement_count = 0
                    return True
                else:
                    # Revert
                    counts[cov_in1] -= 1
                    counts[cov_in2] -= 1
                    counts[cov_out1] += 1
                    counts[cov_out2] += 1
                    code_indices[i] = idx_out1
                    code_indices[j] = idx_out2
                    covers[i] = cov_out1
                    covers[j] = cov_out2
                    code_set.remove(idx_in1)
                    code_set.remove(idx_in2)
                    code_set.add(idx_out1)
                    code_set.add(idx_out2)
        
        return False
    
    while time.time() - start_time < time_limit:
        if len(code_indices) == 0:
            break
        
        # Alternate between single and pair swaps
        if rng.random() < 0.7:
            improved = _try_single_swap()
        else:
            improved = _try_pair_swap()
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > 500:
                # Restore best snapshot and try a different seed
                break
    
    # Restore best found
    if len(best_snapshot) < len(code_indices):
        code_indices = list(best_snapshot)
    
    return code_indices

def _greedy_construct(M, all_words, total):
    """
    Greedy set cover: at each step, pick the word that covers the most uncovered words.
    """
    uncovered = np.ones(total, dtype=bool)
    selected = []
    
    while uncovered.any():
        # Count how many uncovered words each candidate covers
        counts = M[:, uncovered].sum(axis=1)
        # Pick the one with max count
        best_idx = np.argmax(counts)
        if counts[best_idx] == 0:
            break
        selected.append(best_idx)
        uncovered[M[best_idx] == 1] = False
    
    return selected

def construct(q, n, radius):
    """Construct a covering code: choose as few words as possible so every word
    is within Hamming distance radius of some chosen word."""
    start_time = time.time()
    
    total = q ** n
    
    # Build coverage matrix
    M, all_words = _build_coverage_matrix(q, n, radius)
    
    # 1. Try Tetracode initialization if applicable
    tetracode_code = _direct_sum_tetracode(q, n, radius)
    if tetracode_code is not None:
        # Convert words to indices
        tetracode_indices = []
        for w in tetracode_code:
            idx = _word_to_idx(w, q, n)
            tetracode_indices.append(idx)
        best_code = tetracode_indices
        best_size = len(best_code)
    else:
        # Fallback to greedy if tetracode not applicable
        greedy_code = _greedy_construct(M, all_words, total)
        best_code = greedy_code
        best_size = len(best_code)
    
    # 2. Intensive local search polish with multiple restarts
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 8  # Leave 8s margin
    
    if remaining > 10:
        # Split time among multiple independent restarts
        n_restarts = min(4, max(1, int(remaining // 15)))
        per_restart = remaining / n_restarts
        
        for r in range(n_restarts):
            seed = 42 + r * 1000
            t_start = time.time()
            polished_code = _local_search_polish(
                best_code, M, all_words, total, 
                time_limit=per_restart, seed=seed
            )
            if len(polished_code) < best_size:
                best_code = polished_code
                best_size = len(best_code)
            
            # Check time
            if time.time() - t_start > _DEADLINE - 5:
                break
    
    # Convert indices back to words
    best_code_words = [all_words[i].tolist() for i in best_code]
    
    # Final verification
    if not _verify_code(best_code, M, total):
        # Fallback: use a valid greedy solution
        greedy_code = _greedy_construct(M, all_words, total)
        return [all_words[i].tolist() for i in greedy_code]
    
    return best_code_words
