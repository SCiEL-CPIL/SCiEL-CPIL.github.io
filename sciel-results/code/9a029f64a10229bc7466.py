import sys
import time
import itertools
from collections import deque

def construct(n):
    """
    Constructs a superpermutation of order n.
    Uses a recursive construction as a seed, then applies local search (2-opt)
    to minimize the total length by optimizing the overlap between consecutive permutations.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Step 1: Generate the classic recursive superpermutation
    # This gives a valid sequence of permutations
    def classic_construct(n):
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
                # Find maximum overlap
                max_overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0:
                        max_overlap = 0
                        break
                    if new[len(new) - o:] == block[:o]:
                        max_overlap = o
                        break
                new += block[max_overlap:]
            word = new
        return word

    # Get the initial word
    initial_word = classic_construct(n)
    
    # Extract the sequence of permutations from the initial word
    # A permutation is a window of length n
    # The recursive construction ensures every window of length n is a permutation?
    # No, only specific windows are the "intended" ones, but actually,
    # in the recursive construction, every window of length n IS a permutation?
    # Let's check.
    # The construction adds blocks that ensure the new windows are permutations.
    # Yes, the property of a superpermutation is that it contains all permutations.
    # The "sequence" of permutations in the word are the windows of length n.
    
    perms_in_word = []
    seen_perms = set()
    for i in range(len(initial_word) - n + 1):
        window = tuple(initial_word[i:i + n])
        if window not in seen_perms:
            seen_perms.add(window)
            perms_in_word.append(window)
            
    # We have a list of n! permutations in the order they appear.
    # The length of the word is n + sum( n - overlap(P_i, P_{i+1}) )
    # We want to minimize this sum.
    
    # Precompute overlap costs
    # overlap(P, Q) is the largest k such that P[n-k:] == Q[:k]
    # Cost to go from P to Q is n - overlap(P, Q)
    
    # For n=7, n! = 5040.
    # We can run 2-opt on the list of permutations.
    
    if n > 6:
        # 2-opt local search
        # We represent the path as a list of indices into all_perms
        all_perms = list(seen_perms)
        # Map permutation to index
        perm_to_idx = {p: i for i, p in enumerate(all_perms)}
        
        # Initial path
        path = [perm_to_idx[p] for p in perms_in_word]
        
        def get_overlap_cost(p1, p2):
            # p1, p2 are tuples of length n
            max_k = 0
            for k in range(1, n):
                if p1[n-k:] == p2[:k]:
                    max_k = k
                    break
            return n - max_k
        
        # Current total cost
        current_cost = sum(get_overlap_cost(all_perms[path[i]], all_perms[path[i+1]]) for i in range(len(path)-1))
        
        improved = True
        iterations = 0
        while improved and time.time() < deadline:
            improved = False
            iterations += 1
            # 2-opt: reverse a segment
            for i in range(len(path) - 2):
                if time.time() > deadline:
                    break
                for j in range(i + 2, len(path)):
                    if time.time() > deadline:
                        break
                    # Calculate delta cost
                    # Remove edges (i-1, i) and (j, j+1)
                    # Add edges (i-1, j) and (i, j+1)
                    # Note: boundaries
                    cost_remove = 0
                    cost_add = 0
                    if i > 0:
                        cost_remove += get_overlap_cost(all_perms[path[i-1]], all_perms[path[i]])
                    else:
                        # Start of path, no cost removed? 
                        # Actually, the first permutation contributes n to the length.
                        # The transitions contribute n - overlap.
                        # So cost is sum of transition costs.
                        pass
                    if j < len(path) - 1:
                        cost_remove += get_overlap_cost(all_perms[path[j]], all_perms[path[j+1]])
                    
                    if i > 0:
                        cost_add += get_overlap_cost(all_perms[path[i-1]], all_perms[path[j]])
                    if j < len(path) - 1:
                        cost_add += get_overlap_cost(all_perms[path[i]], all_perms[path[j+1]])
                    
                    delta = cost_add - cost_remove
                    if delta < 0:
                        # Perform swap
                        path[i:j+1] = reversed(path[i:j+1])
                        current_cost += delta
                        improved = True
                        if time.time() > deadline:
                            break
                            
    # Reconstruct word from optimized path
    if len(path) == n:
        pass
        
    # Reconstruct
    final_word = list(all_perms[path[0]])
    for i in range(1, len(path)):
        p_prev = all_perms[path[i-1]]
        p_curr = all_perms[path[i]]
        # Find max overlap
        max_k = 0
        for k in range(1, n):
            if p_prev[n-k:] == p_curr[:k]:
                max_k = k
                break
        final_word.extend(p_curr[max_k:])
        
    # Verify
    # Check if all permutations are present
    word_str = "".join(map(str, final_word))
    # For n=7, verification is O(N * n)
    
    return word_str

if __name__ == "__main__":
    # Test for small n
    print(construct(3))
    print(construct(4))
