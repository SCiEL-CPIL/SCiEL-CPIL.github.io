import time
import itertools

def _build_word_from_order(n, order):
    """
    Build a superpermutation word from a given ordering of permutations.
    'order' is a list of permutations (tuples of length n).
    The word is built by concatenating permutations with maximum overlap.
    """
    if not order:
        return ""
    
    word = list(order[0])
    for i in range(1, len(order)):
        prev = order[i-1]
        curr = order[i]
        # Find max overlap k such that prev[-k:] == curr[:k]
        # k can be at most n-1
        max_k = n - 1
        # Optimization: only check k where the last symbol of prev matches first of curr
        # Actually, just loop down. n is small (7).
        for k in range(max_k, -1, -1):
            if k == 0 or prev[-k:] == curr[:k]:
                word.extend(curr[k:])
                break
    return "".join(map(str, word))

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # 1. Generate all permutations
    all_perms = list(itertools.permutations(range(1, n + 1)))
    N = len(all_perms)
    perm_index = {p: i for i, p in enumerate(all_perms)}
    
    # 2. Compute overlap matrix
    # overlap[i][j] = max k such that all_perms[i][-k:] == all_perms[j][:k]
    # This determines the cost of transitioning from i to j, which is n - k.
    # We want to minimize total length = n + sum(n - overlap[i][j]).
    # Equivalent to maximizing sum(overlap[i][j]).
    
    # For n=7, N=5040. N^2 = 25M operations.
    # Python loop for this might be slow (25M * ~100ns = 2.5s, maybe more).
    # Let's try to speed it up.
    
    # Precompute suffixes/prefixes?
    # Or just compute on the fly in the search?
    # No, we need it for the search.
    
    # Let's compute the overlap matrix efficiently.
    # We can use a dictionary of suffixes to indices?
    # For each permutation p, and each k from n-1 down to 0:
    #   suffix = p[-k:]
    #   prefix = p[:k]
    #   If suffix == prefix of q, then overlap is at least k.
    
    # Since n is small, direct comparison is probably fine if optimized.
    # Let's use NumPy if available? Or just pure Python with some tricks.
    
    # Actually, let's just build the adjacency list or matrix.
    # overlap_matrix[i][j] = max overlap
    # We can store this as a list of arrays or just a 2D list.
    
    # Optimization:
    # For each permutation i, we can precompute its suffixes.
    # But comparing tuples is fast in Python.
    
    # Let's try a hybrid approach:
    # For each i, we want to find j such that overlap is high.
    # High overlap means long common suffix/prefix.
    
    # Let's just compute the full matrix.
    # To speed up, we can note that overlap[i][j] >= k implies prefix of j of length k matches suffix of i of length k.
    
    # Let's use a map: prefix_tuple -> list of indices
    # For k = n-1 down to 1:
    #   For each perm, get prefix of length k.
    #   Group perms by prefix.
    #   For each perm i, get suffix of length k.
    #   Look up suffix in map.
    #   If found, then for all j in that group, overlap[i][j] >= k.
    #   Since we iterate k downwards, the first match gives the max overlap.
    
    # This is O(N * n * (N/k)?) No, the map lookup is O(1) average.
    # So for each i and each k, we do a lookup.
    # Total operations: N * n lookups.
    # N=5040, n=7. 35k lookups. Very fast.
    
    # Let's implement this.
    
    # Initialize overlap matrix with 0
    # overlap[i][j] is max k
    # We can store it as a list of bytearrays or lists of ints.
    # N=5040, so 5040x5040 matrix is 25M entries.
    # If we use int8, that's 25MB. Acceptable.
    # But Python lists of lists are heavy.
    # Let's use a flat list or array.
    # Or just a list of lists of small ints.
    
    # Actually, we don't need the full matrix if we use a heuristic that only looks at neighbors.
    # But 2-opt needs to evaluate arbitrary pairs.
    
    # Let's build the matrix using the prefix map method.
    
    # prefix_map[k] = { prefix_tuple: [indices] }
    # Note: we only need to find the max overlap.
    # So for each i, we iterate k from n-1 down to 0.
    # We check if suffix_i_k is in prefix_map[k].
    # If yes, we set overlap[i][j] = k for all j in the list?
    # No, we want the max. So if we find a match at k, we can stop for this i?
    # No, there might be multiple j's with that overlap.
    # But for the search, we just need the cost between any pair.
    
    # Let's store overlap as a 2D array.
    # import numpy as np
    # overlap = np.zeros((N, N), dtype=np.int8)
    
    # But let's stick to standard library to be safe, or use numpy if available.
    # The prompt says NumPy is available.
    
    import numpy as np
    
    overlap = np.zeros((N, N), dtype=np.int8)
    
    # Build prefix maps for each k
    # prefix_maps[k] = dict: prefix -> list of indices
    prefix_maps = [None] * (n)  # indices 1 to n-1
    
    for k in range(1, n):
        pm = {}
        for i, p in enumerate(all_perms):
            prefix = p[:k]
            if prefix not in pm:
                pm[prefix] = []
            pm[prefix].append(i)
        prefix_maps[k] = pm
        
    # Compute overlaps
    for i, p in enumerate(all_perms):
        for k in range(n - 1, 0, -1):
            suffix = p[-k:]
            if suffix in prefix_maps[k]:
                js = prefix_maps[k][suffix]
                for j in js:
                    if overlap[i][j] == 0: # Only set if not already set (higher k processed earlier? No, we go down)
                        # Wait, we are going k from n-1 down to 1.
                        # So the first time we find a match for a specific j, it will be the max k?
                        # No. For a fixed i and j, the overlap is unique.
                        # But different j's might match different k's.
                        # Actually, for a fixed i, we iterate k.
                        # If suffix of length k matches prefix of j, then overlap[i][j] >= k.
                        # Since we iterate k downwards, the first k we find for a specific j is the max?
                        # No, because a specific j might match a smaller k later, but we already found a larger k?
                        # No, if suffix of length k matches, it implies suffix of length k-1 matches.
                        # So if we find a match at k, we know overlap >= k.
                        # We want the MAX overlap.
                        # So if we iterate k from n-1 down to 1, the first time we see a match for j, it is the max overlap.
                        # So we should only set it if it's 0.
                        overlap[i][j] = k
                        # We cannot break the inner loop for i, because other j's might have different max overlaps.
                        # But we can skip setting if already set.
    
    # Wait, the logic above is slightly flawed.
    # If I find a match at k=5 for j=10, I set overlap[i][10]=5.
    # Later I check k=4. suffix of length 4 matches prefix of j=10 (implied by k=5 match).
    # But I skip because overlap[i][10] is not 0.
    # This is correct.
    
    # 3. Initial Order: Classic Superpermutation
    # We need an initial valid superpermutation.
    # We can generate one using the recursive construction from the seed.
    
    def classic_superpermutation(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Overlap logic
                max_overlap = min(len(new), len(block))
                overlap_len = 0
                for o in range(max_overlap, -1, -1):
                    if o == 0 or new[len(new)-o:] == block[:o]:
                        overlap_len = o
                        break
                new += block[overlap_len:]
            word = new
        return word
    
    word = classic_superpermutation(n)
    
    # Extract the order of permutations from the classic word
    # We need to map the windows in the word to the permutation indices
    current_order = []
    seen_perms = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i+n])
        if window not in seen_perms:
            seen_perms.add(window)
            current_order.append(perm_index[window])
            
    # Verify that we have all permutations
    # If not, something went wrong with the classic construction or extraction.
    # The classic construction should contain all permutations.
    # But the order of appearance might not be a simple list if overlaps are tricky?
    # No, the classic construction is a valid superpermutation.
    # So we should find all N permutations.
    
    if len(current_order) < N:
        # Fallback: just use a random permutation of indices?
        # Or use the order from itertools?
        # Let's just use a random shuffle of indices as a fallback, though it will be long.
        import random
        current_order = list(range(N))
        random.shuffle(current_order)
    
    # 4. Local Search: 2-Opt
    # We want to minimize the total length.
    # Length = n + sum(n - overlap[current_order[i]][current_order[i+1]]) for i in 0..N-2
    # = n + (N-1)*n - sum(overlap[...])
    # Minimizing length is equivalent to maximizing sum(overlap[...]).
    
    def calc_score(order):
        total_overlap = 0
        for i in range(len(order) - 1):
            total_overlap += overlap[order[i]][order[i+1]]
        return total_overlap
    
    current_score = calc_score(current_order)
    best_order = current_order[:]
    best_score = current_score
    
    # 2-Opt moves: reverse a segment of the order.
    # If we reverse order[i:j], the edges affected are:
    # (i-1, i), (j-1, j) become (i-1, j), (i, j-1) roughly?
    # Let's trace:
    # ... A, B, C, D ...
    # Reverse B, C -> C, B
    # ... A, C, B, D ...
    # Edges removed: (A,B), (B,C), (C,D)
    # Edges added: (A,C), (C,B), (B,D)
    # Wait, if we reverse segment [i, j] inclusive:
    # Old: ... X, P_i, P_{i+1}, ..., P_j, Y ...
    # New: ... X, P_j, P_{j-1}, ..., P_i, Y ...
    # Edges removed: (X, P_i), (P_i, P_{i+1}), ..., (P_{j-1}, P_j), (P_j, Y)
    # Edges added: (X, P_j), (P_j, P_{j-1}), ..., (P_{i+1}, P_i), (P_i, Y)
    
    # Note: overlap is not symmetric. overlap[A][B] != overlap[B][A].
    # So we must be careful.
    
    # Change in score:
    # Delta = (overlap[X][P_j] + overlap[P_i][Y] + sum_{k=i+1}^j overlap[P_k][P_{k-1}]) - (overlap[X][P_i] + overlap[P_j][Y] + sum_{k=i}^{j-1} overlap[P_k][P_{k+1}])
    
    # We can precompute the sum of overlaps in the current order to speed up?
    # Or just calculate the delta for the specific segment.
    # The sum inside the segment is reversed.
    # sum_{k=i+1}^j overlap[P_k][P_{k-1}] is the sum of overlaps in the reversed segment.
    
    # To make this fast, we can maintain the total score and update it.
    # But for 2-opt, we need to evaluate many segments.
    # O(N^2) segments. For each, O(1) or O(N) cost?
    # If we calculate the sum over the segment each time, it's O(N). Total O(N^3).
    # N=5040. 5040^3 is too big.
    
    # We need O(1) delta calculation.
    # This requires prefix sums of overlaps.
    # Let S[i] = sum of overlaps from 0 to i-1.
    # But the overlaps change when we do a move.
    
    # Alternative: Simulated Annealing or Tabu Search?
    # Given the time limit, maybe a simple greedy local search or a limited 2-opt with random restarts?
    
    # Let's try a simple 2-opt with O(N) delta calculation, but limit the number of iterations.
    # Or use a better heuristic.
    
    # Actually, for N=5040, N^2 is 25M.
    # If each delta calc is O(1), we can do 25M checks in ~1-2 seconds in C, but in Python?
    # 25M Python iterations might take 5-10 seconds.
    # We have 115 seconds.
    # So we can afford a few full passes of 2-opt.
    
    # To get O(1) delta, we can precompute:
    # 1. The current total score.
    # 2. For each i, the "internal" sum of overlaps in the segment? No, that changes.
    
    # Let's just do O(N) delta calculation and see how fast it is.
    # If it's too slow, we reduce the number of attempts.
    
    # Better idea: Use the fact that we only care about the boundary edges and the internal sum.
    # Internal sum of reversed segment:
    # Old internal: sum_{k=i}^{j-1} overlap[P_k][P_{k+1}]
    # New internal: sum_{k=i}^{j-1} overlap[P_{j-k}][P_{j-k+1}] ? No.
    # Reversed segment is P_j, P_{j-1}, ..., P_i.
    # Edges are (P_j, P_{j-1}), (P_{j-1}, P_{j-2}), ..., (P_{i+1}, P_i).
    # So new internal sum = sum_{k=i+1}^{j} overlap[P_k][P_{k-1}].
    
    # We can precompute a matrix of "reverse overlaps"?
    # Or just sum them up.
    
    # Let's implement a fast 2-opt.
    
    # We'll maintain the order as a list.
    # We'll iterate i from 0 to N-2, and j from i+1 to N-1.
    
    # To speed up, we can break early if the potential gain is small?
    
    # Let's try it.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N - 1):
            if time.time() > deadline:
                break
            # Calculate prefix sums of overlaps from i onwards?
            # Or just calculate the delta for each j.
            
            # To make it faster, we can precompute the cumulative sum of overlaps
            # for the current order.
            # Let cum[k] = sum_{m=0}^{k-1} overlap[order[m]][order[m+1]]
            # Then sum_{k=i}^{j-1} overlap[order[k]][order[k+1]] = cum[j] - cum[i]
            
            # But we also need the reverse sum.
            # We can't easily get that from cum.
            
            # Let's just do the O(N) sum for now.
            # If N=5040, inner loop is 5040. Outer is 5040. Total 25M ops.
            # Each op is a few array lookups and adds.
            # Should be feasible in ~10-20s.
            
            for j in range(i + 1, N):
                if time.time() > deadline:
                    break
                
                # Calculate delta
                # Edges removed:
                # Left boundary: if i > 0, overlap[order[i-1]][order[i]]
                # Right boundary: if j < N-1, overlap[order[j]][order[j+1]]
                # Internal: sum_{k=i}^{j-1} overlap[order[k]][order[k+1]]
                
                # Edges added:
                # Left boundary: if i > 0, overlap[order[i-1]][order[j]]
                # Right boundary: if j < N-1, overlap[order[i]][order[j+1]]
                # Internal: sum_{k=i+1}^{j} overlap[order[k]][order[k-1]]
                
                removed = 0
                added = 0
                
                if i > 0:
                    removed += overlap[order[i-1]][order[i]]
                    added += overlap[order[i-1]][order[j]]
                else:
                    # No left boundary edge
                    pass
                
                if j < N - 1:
                    removed += overlap[order[j]][order[j+1]]
                    added += overlap[order[i]][order[j+1]]
                
                # Internal sums
                # Old internal: i to j-1
                old_internal = 0
                for k in range(i, j):
                    old_internal += overlap[order[k]][order[k+1]]
                removed += old_internal
                
                # New internal: i+1 to j (reversed)
                new_internal = 0
                for k in range(i + 1, j + 1):
                    new_internal += overlap[order[k]][order[k-1]]
                added += new_internal
                
                delta = added - removed
                
                if delta > 0:
                    # Apply move
                    # Reverse segment i to j
                    order[i:j+1] = reversed(order[i:j+1])
                    current_score += delta
                    improved = True
                    # Restart search? Or continue?
                    # Usually restart is better for 2-opt to avoid local minima.
                    # But restarting is expensive.
                    # Let's just break and restart the outer loop.
                    break
            
            if improved:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_order = order[:]
    
    # Reconstruct word from best_order
    final_word = _build_word_from_order(n, best_order)
    
    return final_word

# Test for small n
if __name__ == "__main__":
    # n=1
    # print(construct(1))
    # n=2
    # print(construct(2))
    # n=3
    # print(construct(3))
    # n=4
    # print(construct(4))
    pass
