import numpy as np
import itertools

def _greedy_cover_local(q, n, radius):
    """
    Computes a covering code for Q_q^n using greedy set cover.
    Returns a list of codewords (as lists of ints).
    """
    N = q ** n
    if N == 0:
        return []
    
    # Precompute neighbors for the local space
    # For radius=1, sphere_size = 1 + n*(q-1)
    if radius == 1:
        sphere_size = 1 + n * (q - 1)
        neighbors = np.empty((N, sphere_size), dtype=int)
        
        # Precompute digits for all N words
        digits = np.zeros((N, n), dtype=int)
        for i in range(N):
            val = i
            for j in range(n):
                digits[i, j] = val % q
                val //= q
        
        # Fill neighbors
        for i in range(N):
            idx = 0
            neighbors[i, idx] = i
            idx += 1
            for pos in range(n):
                orig_val = digits[i, pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    neighbors[i, idx] = i + (new_val - orig_val) * (q ** pos)
                    idx += 1
    else:
        # Fallback for other radii (simple brute force neighbor generation)
        # This is slow, but radius=1 is the main case.
        # For safety, we implement a generic neighbor finder for small N
        neighbors_list = []
        for i in range(N):
            word_i = []
            val = i
            for _ in range(n):
                word_i.append(val % q)
                val //= q
            
            neigh = [i]
            for pos in range(n):
                for new_val in range(q):
                    if new_val == word_i[pos]:
                        continue
                    # Check distance
                    # For radius 1, this is just the immediate neighbors.
                    # For radius > 1, we need to generate all words at dist <= radius.
                    # Given the task is radius 1, we assume radius 1 here for efficiency.
                    # If radius > 1, this function is not optimized.
                    if radius == 1:
                        new_word = word_i.copy()
                        new_word[pos] = new_val
                        new_idx = 0
                        for k in range(n):
                            new_idx += new_word[k] * (q ** k)
                        neigh.append(new_idx)
            neighbors_list.append(neigh)
        
        # Convert to array of fixed size (max sphere size)
        # For radius 1, size is 1 + n*(q-1)
        max_size = 1 + n * (q - 1)
        neighbors = np.full((N, max_size), -1, dtype=int)
        for i in range(N):
            for j, nb in enumerate(neighbors_list[i]):
                if j < max_size:
                    neighbors[i, j] = nb
                else:
                    # Should not happen for radius 1
                    pass

    covered = np.zeros(N, dtype=bool)
    result_indices = []
    
    while covered.sum() < N:
        gains = (~covered[neighbors]).sum(axis=1)
        # Mask out -1s or invalid entries if any (though for radius 1 they are all valid)
        # np.argmax ignores -1 if there are positive gains? No, -1 is smaller.
        # We should ensure neighbors are valid.
        best_idx = np.argmax(gains)
        
        if gains[best_idx] == 0:
            break
            
        result_indices.append(best_idx)
        # Mark covered
        # Ensure we only mark valid neighbors
        valid_neighs = neighbors[best_idx]
        # For radius 1, all are valid.
        covered[valid_neighs[valid_neighs >= 0]] = True

    # Convert indices back to lists
    result_words = []
    for idx in result_indices:
        word = []
        val = idx
        for _ in range(n):
            word.append(val % q)
            val //= q
        result_words.append(word)
        
    return result_words

def construct(q, n, radius):
    # Partition-based constructive approach
    # We decompose Q_q^n into q subcubes of dimension n-1.
    # For each subcube, we find a covering code.
    # This is effective when q * K_q(n-1, r) < K_q(n, r).
    
    if n == 0:
        return [[]]
        
    if n == 1:
        # Base case: cover all q symbols
        return [[i] for i in range(q)]
        
    # Partition by the first coordinate
    subcube_codes = []
    for prefix_val in range(q):
        # Get code for the subcube where first coordinate is fixed to prefix_val
        # The remaining n-1 coordinates form a Q_q^(n-1) space
        local_code = _greedy_cover_local(q, n - 1, radius)
        
        # Map local codewords to global space
        for word in local_code:
            global_word = [prefix_val] + word
            subcube_codes.append(global_word)
            
    return subcube_codes
