"""
Constructs a superpermutation for n symbols.
For n=7, uses a TSP-based approach to find a near-optimal ordering of permutations
to minimize the total length of the superpermutation.
"""
import time
import sys
import math
from itertools import permutations

def construct(n):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill
    
    # Base cases
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    if n >= 7:
        return _construct_tsp(n, deadline)
    else:
        return _classic_construct(n)

def _classic_construct(n):
    """Classic recursive construction: length = sum(i! for i=1..n)"""
    word = [1]
    for m in range(2, n + 1):
        # Extract unique permutations of length m-1 from current word
        perms = []
        seen = set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap between new_word and block
            max_overlap = 0
            limit = min(len(new_word), len(block))
            for o in range(limit, -1, -1):
                if o == 0:
                    new_word.extend(block)
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    new_word.extend(block[o:])
                    break
        word = new_word
    
    return "".join(map(str, word))

def _construct_tsp(n, deadline):
    """
    TSP-based construction for n>=7.
    Models the problem as finding a path through all n! permutations
    where the cost between consecutive permutations is the number of
    new symbols needed (n - overlap).
    """
    all_perms = list(permutations(range(1, n + 1)))
    num_perms = len(all_perms)
    
    # Create index mapping
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    
    # Precompute overlap/cost matrix
    # cost[i][j] = number of new symbols needed to go from perm i to perm j
    # This is n - (length of longest suffix of perm i that is a prefix of perm j)
    # But actually, in a superpermutation, when we move from one n-window to the next,
    # we shift by 1 position. So the overlap is always n-1 if they share n-1 consecutive symbols.
    
    # Actually, the key insight: a superpermutation is a sequence of n-windows where
    # consecutive windows overlap by n-1 symbols. So we need to find an ordering of all
    # n! permutations such that consecutive permutations share n-1 consecutive symbols
    # (i.e., perm_j[:n-1] == perm_i[1:]).
    
    # This is equivalent to finding a Hamiltonian path in the directed graph where
    # edge i->j exists if perm_i[1:] == perm_j[:-1].
    
    # Let's build this graph
    # For each permutation, find which permutation it can transition to
    # perm_i[1:] should equal perm_j[:-1]
    
    # Actually, let me think about this differently.
    # In a superpermutation of length L, there are L-n+1 windows of length n.
    # Each window is a permutation. Consecutive windows overlap by n-1 symbols.
    # So window_k[1:] == window_{k+1}[:-1].
    
    # This means we need a sequence of all n! permutations where each consecutive pair
    # satisfies: perm_{k+1} = (perm_k[1], perm_k[2], ..., perm_k[n-1], x) for some x.
    # i.e., perm_{k+1}[:-1] == perm_k[1:]
    
    # So for each permutation p, the next permutation must have prefix p[1:].
    # There is exactly one such permutation for each choice of the last symbol.
    # Wait, no. perm_{k+1}[:-1] == perm_k[1:] means the first n-1 symbols of perm_{k+1}
    # are the last n-1 symbols of perm_k. The last symbol of perm_{k+1} can be any symbol
    # not in perm_k[1:], which is exactly one symbol (the one that was in perm_k[0]).
    
    # So actually, for each permutation p, there is exactly ONE valid next permutation:
    # next = (p[1], p[2], ..., p[n-1], p[0])  -- this is a cyclic shift!
    
    # Wait, that's not right either. Let me reconsider.
    
    # If window_k = (a1, a2, ..., an) and window_{k+1} = (a2, a3, ..., an, b),
    # then window_{k+1} must also be a permutation. So b must be a1.
    # Therefore, window_{k+1} = (a2, a3, ..., an, a1), which is a cyclic left shift.
    
    # This means the only valid transitions are cyclic shifts! So the superpermutation
    # would just be a cycle of cyclic shifts, which only visits n permutations, not n!.
    
    # I'm making an error. Let me reconsider the problem.
    
    # A superpermutation contains every permutation as a contiguous subsequence, but
    # consecutive windows in the word don't have to be different permutations.
    # The word is a sequence of symbols, and we look at all length-n windows.
    # Consecutive windows in the word overlap by n-1 symbols by definition.
    # But the windows don't have to be permutations - only some of them are.
    
    # So the correct model is: we want to find a word such that every permutation appears
    # as a length-n substring. The word can have arbitrary symbols, not just permutations.
    
    # The TSP approach: think of each permutation as a node. We want to visit all nodes.
    # The cost of going from permutation A to permutation B is the minimum number of new
    # symbols needed to transition from A to B, which is n - overlap(A, B), where overlap
    # is the length of the longest suffix of A that matches a prefix of B.
    
    # Actually, in the classic construction, when we go from one permutation block to the next,
    # we overlap as much as possible. The total length is sum of (n - overlap) for each transition.
    
    # Let me implement a proper TSP approach.
    
    # For n=7, 5040 nodes. Computing all-pairs overlap is 5040^2 = 25M pairs, each O(n)=7.
    # That's about 175M operations, which might be too slow in pure Python.
    
    # Let's use numpy to speed up the overlap computation.
    try:
        import numpy as np
        use_numpy = True
    except ImportError:
        use_numpy = False
    
    if use_numpy:
        # Convert permutations to numpy array
        perm_array = np.array(all_perms, dtype=np.int8)  # shape (5040, 7)
        
        # Compute overlap matrix using numpy
        # overlap[i][j] = length of longest suffix of perm_i that is prefix of perm_j
        # For each possible overlap length k from n down to 1:
        #   perm_i[n-k:] == perm_j[:k]
        
        overlap = np.zeros((num_perms, num_perms), dtype=np.int8)
        
        for k in range(n, 0, -1):
            if time.time() > deadline - 20:
                break
            # Check if perm_i[n-k:] == perm_j[:k] for all i,j
            # This is a matrix comparison
            suffixes = perm_array[:, n-k:]  # shape (5040, k)
            prefixes = perm_array[:, :k]    # shape (5040, k)
            # Compare: suffixes[i] == prefixes[j] for all i,j
            # Use broadcasting
            match = (suffixes[:, np.newaxis, :] == prefixes[np.newaxis, :, :]).all(axis=2)
            overlap[match] = k
        
        # Cost matrix: cost[i][j] = n - overlap[i][j]
        cost = n - overlap
        
        # Now solve TSP on this cost matrix
        # For 5040 nodes, exact TSP is infeasible. Use greedy approximation.
        
        # Greedy TSP: start at node 0, repeatedly go to nearest unvisited node
        visited = np.zeros(num_perms, dtype=bool)
        path = [0]
        visited[0] = True
        
        for _ in range(num_perms - 1):
            if time.time() > deadline - 5:
                break
            current = path[-1]
            # Find nearest unvisited node
            unvisited_costs = cost[current].copy()
            unvisited_costs[visited] = np.inf
            next_node = np.argmin(unvisited_costs)
            path.append(next_node)
            visited[next_node] = True
        
        # Build the word from the path
        word = list(all_perms[path[0]])
        for i in range(1, len(path)):
            prev_perm = all_perms[path[i-1]]
            curr_perm = all_perms[path[i]]
            # Find overlap
            o = overlap[path[i-1]][path[i]]
            word.extend(list(curr_perm[o:]))
        
        return "".join(map(str, word))
    else:
        # Fallback to classic construction
        return _classic_construct(n)

if __name__ == "__main__":
    n = 7
    res = construct(n)
    print(len(res))
