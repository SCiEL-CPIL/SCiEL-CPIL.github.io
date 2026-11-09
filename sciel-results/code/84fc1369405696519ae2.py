"""
Simulated Annealing based covering code construction.
Optimizes the number of codewords for small spaces (q^n <= 10000).
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
import random

_DEADLINE = 120.0  # seconds


def _idx_to_word(idx, q, n):
    """Convert a linear index to a word over {0,...,q-1} with n digits."""
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


def _hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def _build_coverage_matrix(q, n, radius):
    """
    Build a boolean matrix M where M[i, j] = True iff word i is within radius of word j.
    Returns M as a numpy boolean array of shape (total, total).
    """
    total = q ** n
    # Generate all words
    words = np.array([_idx_to_word(i, q, n) for i in range(total)], dtype=np.int8)
    # Compute pairwise Hamming distances
    # diff[i, j, k] = 1 if words[i][k] != words[j][k]
    diff = words[:, np.newaxis, :] != words[np.newaxis, :, :]
    dists = diff.sum(axis=2)
    M = (dists <= radius)
    return M


def _direct_sum_tetracode_indices(q, n, radius):
    """
    Generate indices for the direct sum of ternary tetracode.
    For q=3, n>=4.
    Tetracode: [a, b, a+b, a+2b] for a,b in {0,1,2}. Size 9.
    Direct sum with tail of length n-4.
    """
    if q != 3:
        return [0] # Fallback: just all-zeros
    tetracode_words = []
    for a in range(3):
        for b in range(3):
            w = [a, b, (a + b) % 3, (a + 2 * b) % 3]
            tetracode_words.append(w)
    
    indices = []
    if n == 4:
        for w in tetracode_words:
            indices.append(_word_to_idx(w, q, n))
    else:
        tail_len = n - 4
        for h in tetracode_words:
            for tail in itertools.product(range(3), repeat=tail_len):
                full_word = h + list(tail)
                indices.append(_word_to_idx(full_word, q, n))
    return indices


def _greedy_init_indices(q, n, radius, M):
    """
    Greedy set cover to get an initial valid code.
    Returns list of indices.
    """
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    code = []
    
    # Precompute coverage sets: for each word j, which words i does it cover?
    # M[i, j] is True if i is covered by j.
    # So column j of M is the set of words covered by j.
    
    remaining = np.ones(total, dtype=bool)
    while remaining.any():
        # Find j that covers the most remaining words
        # new_cover_count[j] = sum(remaining[i] for i where M[i, j] is True)
        # M is (total, total). M[:, j] is coverage of j.
        # We want argmax_j (M[:, j] & remaining).sum()
        
        # Efficient calculation:
        # cover_counts = (M & remaining[:, np.newaxis]).sum(axis=0)
        # But M is boolean, remaining is boolean.
        # M.T is (total, total) where M.T[j, i] = M[i, j].
        # So (M.T * remaining).sum(axis=1) gives count of covered remaining words for each j.
        
        # Note: M is symmetric? No, M[i,j] is "i covered by j".
        # Distance is symmetric, so M[i,j] == M[j,i].
        # So M is symmetric.
        
        cover_counts = (M * remaining[:, np.newaxis]).sum(axis=0)
        best_j = int(np.argmax(cover_counts))
        
        if cover_counts[best_j] == 0:
            break
            
        code.append(best_j)
        covered |= M[:, best_j]
        remaining = ~covered
        
        # Optional: check deadline
        if time.time() > _DEADLINE - 20:
            break
            
    return code


def _sa_optimize(q, n, radius, M, initial_code, deadline):
    """
    Simulated Annealing to minimize the size of the code.
    """
    total = q ** n
    code = set(initial_code)
    
    # Data structures for fast updates
    # cover_count[i] = number of codewords covering word i
    # is_in_code[j] = True if j is in code
    cover_count = np.zeros(total, dtype=np.int32)
    is_in_code = np.zeros(total, dtype=bool)
    
    for c in code:
        is_in_code[c] = True
        cover_count += M[:, c]
        
    # Helper to check if a codeword c is redundant
    # c is redundant if no word i has cover_count[i] == 1 and M[i, c] == 1
    # i.e., the set of words covered ONLY by c is empty.
    
    # Precompute for each c, the list of words covered ONLY by c?
    # This changes as code changes. So we must check dynamically.
    # However, we can maintain a list of "critical" words for each c.
    # critical[c] = [i for i in range(total) if cover_count[i] == 1 and M[i, c]]
    # Updating this list on add/remove is expensive if done naively.
    
    # Alternative: Just check on the fly.
    # For removal of c:
    #   Check if any i has cover_count[i] == 1 and M[i, c].
    #   This is a dot product of (cover_count == 1) and M[:, c].
    
    # For swap c -> w:
    #   Let S = { i | cover_count[i] == 1 and M[i, c] }
    #   Swap is valid if for all i in S, M[i, w] is True.
    #   This is equivalent to: (cover_count == 1) & M[:, c] & ~M[:, w] is all False.
    #   Or: sum((cover_count == 1) & M[:, c] & ~M[:, w]) == 0
    
    current_size = len(code)
    best_code = set(code)
    best_size = current_size
    
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    start_time = time.time()
    
    # Precompute M.T for faster access? M is symmetric, so M.T == M.
    
    while T > T_min and time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.7:
            # Try Remove
            if not code:
                continue
            c = random.choice(list(code))
            
            # Check redundancy
            # Words covered only by c:
            # mask_only = (cover_count == 1) & M[:, c]
            # if not mask_only.any(): redundant
            mask_only = (cover_count == 1) & M[:, c]
            if not mask_only.any():
                # Remove c
                code.remove(c)
                is_in_code[c] = False
                cover_count -= M[:, c]
                current_size -= 1
                if current_size < best_size:
                    best_size = current_size
                    best_code = set(code)
                # After removal, T might need to be reset or kept?
                # Keep T.
            else:
                # Not redundant. Try to accept with prob exp(0/T) = 1? No, removal failed.
                # Do nothing.
                pass
                
        elif move_type < 0.95:
            # Try Swap
            if not code:
                continue
            c = random.choice(list(code))
            # Pick w not in code
            # Randomly pick from 0..total-1 until not in code?
            # Better: pick from list of not-in-code?
            # For efficiency, just pick random index and check.
            w = random.randint(0, total - 1)
            if is_in_code[w]:
                continue
                
            # Check validity of swap c -> w
            # Words covered only by c must be covered by w.
            mask_only_c = (cover_count == 1) & M[:, c]
            if mask_only_c.any():
                # Check if w covers all these words
                # mask_only_c & ~M[:, w] must be empty
                if (mask_only_c & ~M[:, w]).any():
                    continue # Invalid swap
            
            # Valid swap
            # Update structures
            code.remove(c)
            code.add(w)
            is_in_code[c] = False
            is_in_code[w] = True
            cover_count -= M[:, c]
            cover_count += M[:, w]
            # Size unchanged
            
        else:
            # Try Add (Escape move)
            # Accept with prob exp(-1/T)
            if random.random() < np.exp(-1.0 / T):
                w = random.randint(0, total - 1)
                if not is_in_code[w]:
                    code.add(w)
                    is_in_code[w] = True
                    cover_count += M[:, w]
                    current_size += 1
                    
        T *= cooling
        
        # Optional: If we reached a new best, restart T?
        # No, standard SA.
        
    return list(best_code)


def construct(q, n, radius):
    """
    Construct a covering code: choose as few words as possible so every word
    is within Hamming distance radius of some chosen word.
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE - 10.0  # 10s safety margin
    
    total = q ** n
    
    # For small spaces, use SA
    if total <= 10000:
        try:
            # Build coverage matrix
            M = _build_coverage_matrix(q, n, radius)
            
            # Get initial code
            # Try direct sum first as it's a good structured start
            init_indices = _direct_sum_tetracode_indices(q, n, radius)
            
            # Verify init is valid? It should be.
            # But let's ensure we have a valid code to start SA.
            # If direct sum is not valid (e.g. wrong params), use greedy.
            # Check validity of init_indices
            covered = np.zeros(total, dtype=bool)
            for c in init_indices:
                covered |= M[:, c]
            if not covered.all():
                # Fallback to greedy
                init_indices = _greedy_init_indices(q, n, radius, M)
            
            # Run SA
            elapsed = time.time() - start_time
            time_for_sa = max(5.0, deadline - elapsed)
            
            optimized_indices = _sa_optimize(q, n, radius, M, init_indices, start_time + time_for_sa)
            
            if optimized_indices:
                # Convert indices to words
                code_words = [_idx_to_word(i, q, n) for i in optimized_indices]
                # Final verification
                # Re-verify just in case
                covered = np.zeros(total, dtype=bool)
                for c in optimized_indices:
                    covered |= M[:, c]
                if covered.all():
                    return code_words
                else:
                    # SA bug? Return initial
                    return [_idx_to_word(i, q, n) for i in init_indices]
            
        except Exception:
            pass
        
        # Fallback to greedy if SA failed
        try:
            M = _build_coverage_matrix(q, n, radius)
            greedy_indices = _greedy_init_indices(q, n, radius, M)
            if greedy_indices:
                return [_idx_to_word(i, q, n) for i in greedy_indices]
        except Exception:
            pass
    
    # Final fallback: Direct sum
    fallback_indices = _direct_sum_tetracode_indices(q, n, radius)
    if not fallback_indices:
        fallback_indices = [0]
        
    return [_idx_to_word(i, q, n) for i in fallback_indices]

# EVOLVE-BLOCK-END
