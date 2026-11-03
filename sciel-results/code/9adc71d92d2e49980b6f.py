import numpy as np
from itertools import permutations
import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Classic construction is a strong baseline
    classic_word = _classic_construct(n)
    best_word = classic_word
    best_len = len(classic_word)
    
    if n == 7:
        # Attempt local search to improve the classic construction
        optimized_word = _optimize_superpermutation(n, deadline)
        if optimized_word is not None:
            if len(optimized_word) < best_len:
                best_word = optimized_word
                best_len = len(optimized_word)
                
    return best_word

def _classic_construct(n):
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
            # Find overlap
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def _optimize_superpermutation(n, deadline):
    # Generate all permutations
    perms = list(permutations(range(1, n + 1)))
    N = len(perms)
    
    # Map permutation to index for fast lookup
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Precompute overlap matrix
    # overlap[i][j] = length of longest suffix of perms[i] that is prefix of perms[j]
    # We can compute this efficiently by checking suffixes of length n-1 down to 1.
    # But N=5040, N^2 = 25M. Checking up to n-1=6 chars for each pair is 150M ops.
    # This might be too slow in pure Python.
    # Let's use a smarter way or accept the time if it's vectorized or optimized.
    
    # Alternative: Use the fact that we only need to evaluate the current path and neighbors.
    # We can compute overlap on the fly for the specific pairs involved in 2-opt.
    # But 2-opt checks many pairs.
    
    # Let's try to build the overlap matrix using NumPy if possible, or just accept the cost
    # if we limit the search.
    
    # Actually, for n=7, 25M operations in a tight loop might take ~10-20s in Python.
    # Let's try to build it.
    
    perm_strs = ["".join(map(str, p)) for p in perms]
    
    # We will store overlap in a 2D array of uint8
    overlap = np.zeros((N, N), dtype=np.uint8)
    
    # Efficient overlap calculation:
    # For each l from n-1 down to 1:
    #   Find pairs (i, j) where suffix_l(i) == prefix_l(j)
    #   If overlap[i][j] is 0, set it to l.
    
    # We can use hashing for suffixes and prefixes.
    # Since n is small, we can just use strings.
    
    for l in range(n - 1, 0, -1):
        # Build map from suffix string to list of indices
        suffix_map = {}
        for i in range(N):
            s = perm_strs[i][n-l:]
            if s not in suffix_map:
                suffix_map[s] = []
            suffix_map[s].append(i)
            
        # Build map from prefix string to list of indices
        prefix_map = {}
        for j in range(N):
            p = perm_strs[j][:l]
            if p not in prefix_map:
                prefix_map[p] = []
            prefix_map[p].append(j)
            
        # Iterate over common keys
        for key in suffix_map:
            if key in prefix_map:
                i_list = suffix_map[key]
                j_list = prefix_map[key]
                # For each i in i_list and j in j_list, if overlap[i][j] == 0, set to l
                for i in i_list:
                    for j in j_list:
                        if overlap[i][j] == 0:
                            overlap[i][j] = l
        
        if time.time() > deadline - 30:
            return None

    # Now we have the overlap matrix.
    # Cost of edge i->j is n - overlap[i][j].
    # We want to find a Hamiltonian path minimizing total cost.
    # Start with the path from the classic construction.
    
    # Extract the permutation sequence from the classic word
    classic_word = _classic_construct(n)
    seq = []
    for i in range(len(classic_word) - n + 1):
        w = tuple(map(int, classic_word[i:i+n]))
        if w in perm_to_idx:
            # We need to ensure we get the sequence of DISTINCT permutations in order of appearance
            # The classic construction visits each permutation exactly once?
            # Yes, by definition of superpermutation, every permutation appears.
            # But does it appear exactly once?
            # The classic construction is a path in the permutation graph where nodes are permutations.
            # It visits each node exactly once.
            pass
            
    # Let's just reconstruct the sequence from the word by sliding window and deduplicating?
    # No, the classic construction is built by appending blocks.
    # It is known to be a Hamiltonian path in the directed graph of permutations.
    
    # Let's extract the sequence of permutations from the classic word.
    # Since it's a superpermutation, every window of length n is a permutation.
    # In the classic construction, consecutive windows share n-1 symbols.
    # So the sequence of windows is a path where step k is determined by the new symbol.
    # We can just take the windows in order.
    
    seq_indices = []
    seen = set()
    for i in range(len(classic_word) - n + 1):
        w = tuple(map(int, classic_word[i:i+n]))
        if w not in seen:
            seen.add(w)
            seq_indices.append(perm_to_idx[w])
            
    # Check if we found all N permutations
    if len(seq_indices) != N:
        # Fallback: Just use a random permutation of indices?
        # Or return None
        return None
        
    # Current path
    path = seq_indices
    
    # Calculate current cost
    def get_cost(path):
        cost = 0
        for i in range(len(path) - 1):
            cost += n - overlap[path[i]][path[i+1]]
        return cost
        
    current_cost = get_cost(path)
    
    # 2-Opt Local Search
    # We try to reverse segments of the path to reduce cost.
    # Change in cost for reversing segment [i, j]:
    # Edges removed: (i-1, i) and (j, j+1)
    # Edges added: (i-1, j) and (i, j+1)
    # Note: boundaries if i=0 or j=N-1
    
    improved = True
    max_iters = 20
    iter_count = 0
    while improved and iter_count < max_iters and time.time() < deadline - 5:
        improved = False
        iter_count += 1
        # To speed up, we can sample random i, j pairs or iterate.
        # Iterating all pairs is O(N^2) = 25M checks per iteration.
        # Each check is O(1) with precomputed overlap.
        # 25M * 20 = 500M ops. This is too slow in Python (approx 50s+).
        
        # We need to be smarter.
        # Maybe just do a few passes with random sampling?
        # Or limit the number of checks per iteration.
        
        # Let's try random sampling of (i, j) pairs.
        num_checks = 50000
        for _ in range(num_checks):
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            
            # Calculate cost change
            old_cost = 0
            new_cost = 0
            
            # Edge (i-1, i)
            if i > 0:
                old_cost += n - overlap[path[i-1]][path[i]]
            # Edge (j, j+1)
            if j < N - 1:
                old_cost += n - overlap[path[j]][path[j+1]]
                
            # New Edge (i-1, j)
            if i > 0:
                new_cost += n - overlap[path[i-1]][path[j]]
            # New Edge (i, j+1)
            if j < N - 1:
                new_cost += n - overlap[path[i]][path[j+1]]
                
            if new_cost < old_cost:
                path[i:j+1] = path[i:j+1][::-1]
                current_cost -= (old_cost - new_cost)
                improved = True
                
    # Build the word from the optimized path
    s = perm_strs[path[0]]
    for k in range(1, N):
        p_prev = perm_strs[path[k-1]]
        p_curr = perm_strs[path[k]]
        ov = overlap[path[k-1]][path[k]]
        s += p_curr[ov:]
        
    # Verify
    windows = set()
    for i in range(len(s) - n + 1):
        w = tuple(map(int, s[i:i+n]))
        windows.add(w)
        
    if len(windows) == N:
        return s
    else:
        return None
