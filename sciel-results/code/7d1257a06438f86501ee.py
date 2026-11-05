import time
import random
import sys

def construct(n):
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # Helper to build word from a list of permutations (indices into a master list or actual tuples)
    # For the standard construction, we build it recursively.
    # For local search, we work with indices into a list of permutations.
    
    # 1. Generate the standard superpermutation word for n
    # This gives us the base "naive" solution.
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract unique (m-1)-permutations in order of appearance
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Overlap with current new
            max_k = min(len(new), len(block))
            k = 0
            for k_try in range(max_k, 0, -1):
                if new[-k_try:] == block[:k_try]:
                    k = k_try
                    break
            new += block[k:]
        word = new
    
    # 2. Extract the sequence of permutations of 1..n in order of first appearance
    # This sequence defines the "order" we can optimize.
    # The word is reconstructed by overlapping consecutive permutations in this order.
    # Note: The standard word is exactly the result of overlapping these permutations in this order.
    
    seen_perms = set()
    perm_seq = []
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        if p not in seen_perms:
            seen_perms.add(p)
            perm_seq.append(p)
            
    if len(perm_seq) != len(list(permutations(range(1, n+1)))):
        # Fallback if something went wrong (unlikely)
        return "".join(map(str, word))
        
    N = len(perm_seq)
    
    # For n > 4, N is large. N=5040 for n=7.
    # Precomputing the full N x N overlap matrix is O(N^2 * n).
    # 5040^2 * 7 ~ 177M operations. In Python, this is too slow (seconds to minutes).
    # We need a faster approach or a smarter heuristic.
    
    # Strategy:
    # Instead of a full TSP solve, we perform local search on the sequence.
    # We calculate overlaps on the fly but cache them.
    # Or, we can precompute overlaps only for "likely" neighbors?
    # Actually, for n=7, 5040 is manageable if we use NumPy or just careful Python.
    # Let's try to precompute the overlap matrix using NumPy if available, or just efficient Python.
    # But 25M entries is ~200MB if using Python ints. Too much.
    # We can store overlaps as bytes (0-6). 25MB. That's fine.
    
    # Let's use a 2D array of bytes for overlaps.
    # overlap[i][j] = max k such that perm_seq[i][n-k:] == perm_seq[j][:k]
    
    # To speed up calculation, we can use the fact that permutations are unique.
    # But checking suffix/prefix equality is still O(n).
    
    # Let's try a different approach:
    # Use the structure of the standard construction.
    # The standard construction has a specific recursive structure.
    # Maybe we can do local search on the "blocks" of the recursive construction?
    # The word for n is built from blocks of word for n-1.
    # Specifically, if W_{n-1} is the word, W_n is formed by inserting n into each (n-1)-perm block.
    # The sequence of (n-1)-perms is fixed by the order in W_{n-1}.
    # The "choice" is how we order the (n-1)-perms? No, they are fixed by W_{n-1}.
    # Wait, the standard construction *defines* the order.
    # Is there a better order?
    # The problem is equivalent to finding a Hamiltonian path in the graph of permutations where edge weight is overlap.
    
    # For n=7, N=5040.
    # A full 2-opt search is O(N^2) per pass. 25M checks.
    # If each check is fast (lookup in precomputed matrix), 25M lookups in Python is ~1-2 seconds.
    # So precomputing the matrix is the bottleneck.
    
    # Can we compute the matrix faster?
    # We can iterate over all pairs (i, j).
    # There is no obvious sub-quadratic way to compute all overlaps.
    # However, we can parallelize or use C-optimized libraries.
    # But we are restricted to standard library + NumPy/SciPy.
    
    # Let's try to compute the matrix using NumPy vectorization?
    # It's hard to vectorize "longest suffix of A == prefix of B" easily without broadcasting huge arrays.
    
    # Alternative:
    # Don't precompute the full matrix.
    # Use a cache.
    # Start with the standard order.
    # Perform 2-opt moves.
    # Only calculate overlaps for the pairs involved in the move.
    # In a 2-opt move (i, j), we check:
    # old: (i-1, i), (j, j+1)
    # new: (i-1, j), (i, j+1)
    # (Assuming internal edges cancel or we approximate? No, internal edges change.)
    # Actually, in 2-opt, the internal edges of the segment are reversed.
    # The cost of the segment changes.
    # This makes the delta calculation complex unless we maintain prefix sums of costs.
    # But the cost of an edge (u, v) is not symmetric.
    # So we need to sum the costs of the reversed segment.
    # This is O(L) per move.
    # Total O(N^3) worst case. Too slow.
    
    # Better local search:
    # 1. Nearest Neighbor construction (Greedy)
    # 2. 2-Opt with "first improvement" or "best improvement" but limited to short segments?
    # Or just adjacent swaps (2-opt with length 1 segment)?
    # Adjacent swap:
    # Old: ... A, B, C ...
    # New: ... B, A, C ...
    # Edges removed: (Prev, A), (A, B), (B, C)
    # Edges added: (Prev, B), (B, A), (A, C)
    # Delta = ov(Prev, B) + ov(B, A) + ov(A, C) - (ov(Prev, A) + ov(A, B) + ov(B, C))
    # This is O(1) per move if we have the overlap matrix.
    # Without the matrix, it's O(n) per move.
    # N^2 moves * O(n) = 25M * 7 = 175M ops. ~10-20s in Python.
    # This is feasible!
    
    # So, plan:
    # 1. Do NOT precompute full matrix.
    # 2. Use a cache for overlaps.
    # 3. Start with standard order.
    # 4. Perform hill climbing with adjacent swaps (2-opt L=1) and maybe L=2, 3.
    # 5. Use time limit to stop.
    
    # To speed up overlap calculation:
    # We can precompute for each permutation its suffixes and prefixes?
    # Or just use string comparison?
    # Tuples are fast.
    
    overlap_cache = {}
    def get_overlap(a, b):
        # a, b are tuples of length n
        key = (a, b)
        if key in overlap_cache:
            return overlap_cache[key]
        # Check k from n-1 down to 0
        for k in range(n - 1, -1, -1):
            if a[n-k:] == b[:k]:
                overlap_cache[key] = k
                return k
        overlap_cache[key] = 0
        return 0
    
    # Initial score
    current_seq = perm_seq
    current_score = sum(get_overlap(current_seq[i], current_seq[i+1]) for i in range(N-1))
    
    # Hill Climbing
    improved = True
    while improved and (time.time() - start_time) < time_limit - 5:
        improved = False
        # Try all adjacent swaps
        # Randomize order to avoid getting stuck in a specific local minimum?
        # Or just iterate 0 to N-2.
        
        # To make it faster, we can skip some checks or use a random subset if time is tight.
        # But N=5040, so N-1 = 5039 swaps.
        # Each swap check takes ~4 overlap lookups.
        # 20k lookups per pass.
        # If each lookup is 1us (optimistic for dict+slice), that's 20ms per pass.
        # We can do ~5000 passes in 100s.
        # This should converge quickly.
        
        for i in range(N - 1):
            if (time.time() - start_time) > time_limit - 5:
                break
            
            A = current_seq[i]
            B = current_seq[i+1]
            Prev = current_seq[i-1] if i > 0 else None
            Next = current_seq[i+2] if i+2 < N else None
            
            # Calculate old cost
            old_cost = 0
            if Prev is not None:
                old_cost += get_overlap(Prev, A)
            old_cost += get_overlap(A, B)
            if Next is not None:
                old_cost += get_overlap(B, Next)
            
            # Calculate new cost
            # Sequence: ... Prev, B, A, Next ...
            new_cost = 0
            if Prev is not None:
                new_cost += get_overlap(Prev, B)
            new_cost += get_overlap(B, A)
            if Next is not None:
                new_cost += get_overlap(A, Next)
            
            if new_cost > old_cost:
                # Swap
                current_seq[i], current_seq[i+1] = current_seq[i+1], current_seq[i]
                current_score += new_cost - old_cost
                improved = True
                # Continue scanning? Or restart?
                # Restarting ensures we find the best local optimum.
                # But restarting is O(N).
                # Let's just continue.
                
    # Reconstruct word
    res = list(current_seq[0])
    for i in range(1, N):
        prev = current_seq[i-1]
        curr = current_seq[i]
        k = get_overlap(prev, curr)
        res.extend(curr[k:])
        
    return "".join(map(str, res))

# Helper for checking if we have all permutations (used in fallback logic above, not strictly needed if standard construction is correct)
def permutations(iterable, r=None):
    # Simple implementation or use itertools
    from itertools import permutations as it_perm
    return it_perm(iterable, r)
