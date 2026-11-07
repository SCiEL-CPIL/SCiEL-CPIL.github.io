import numpy as np
import time
import random
from itertools import combinations

def construct():
    n = 12
    m = 40
    total_subsets = 1 << n  # 4096
    deadline = time.time() + 80.0  # 80s safety margin
    
    # Precompute subset sizes
    sizes = np.array([bin(i).count('1') for i in range(total_subsets)], dtype=np.int32)
    
    # Precompute strict subset relation matrix
    # A[i,j] = 1 if subset i is a strict subset of subset j
    # Using bit manipulation: i is subset of j iff (i & j) == i
    # Strict: i != j and (i & j) == i
    # We can compute this efficiently
    
    # For each pair (i, j), i is strict subset of j iff:
    # 1. i & j == i  (i is subset of j)
    # 2. i != j (strict)
    
    # Vectorized approach: for each i, compute which j's contain i
    # i is subset of j iff (i & ~j) == 0, i.e., i & j == i
    # Equivalently, j must have all bits that i has
    
    # Let's build the comparability count for each pair
    # comp[i, j] = 1 if i is strict subset of j, or j is strict subset of i
    
    # More efficient: for each i, the sets that are strict supersets of i are those j where
    # (j & i) == i and j != i
    # The sets that are strict subsets of i are those j where (j & i) == j and j != i
    
    # Let's precompute for each mask, the list of masks it's a strict subset of
    # This is O(4096 * 4096) which is ~16M operations, feasible
    
    # Actually, let's think about this differently.
    # For a given family of 40 sets, the score is the number of unordered pairs (i,j) 
    # where one is a strict subset of the other.
    
    # Greedy approach: start with empty family, at each step add the set that maximizes
    # the number of comparability relations with already-selected sets.
    
    # Precompute: for each pair of masks (a, b), whether a is strict subset of b
    # strict_subset[a, b] = 1 if a is strict subset of b
    
    # Efficient computation: a is subset of b iff (a & b) == a
    # We can vectorize this
    
    # For each a, compute which b's are supersets of a:
    # b is superset of a iff (a & ~b) == 0, i.e., all bits in a are in b
    
    # Let's precompute the strict subset relation as a boolean matrix
    # This is 4096 x 4096 = 16.7M booleans, ~16.7MB, feasible
    
    # Actually, let's be smarter. For greedy selection, at each step we need to compute
    # for each candidate set c, the number of selected sets s such that c is comparable to s.
    
    # Let's precompute a matrix where M[i, j] = 1 if i is strict subset of j
    # Then comparability(i, j) = M[i,j] + M[j,i]
    
    # Build M efficiently:
    # For each i, j is a strict superset of i iff (j & i) == i and j != i
    # Equivalently, j must contain all bits of i, and have at least one additional bit
    
    # Let's use a different approach: precompute for each mask, its "supersets" and "subsets"
    # but that's still O(4096^2)
    
    # 4096^2 = 16,777,216 operations. In numpy, we can do this in chunks.
    
    # Let's build the strict subset matrix
    # M[i, j] = 1 iff i is strict subset of j
    
    # For this, i is subset of j iff (i & j) == i
    # Let's compute this in blocks to avoid memory issues
    
    # Actually 4096x4096 int8 is 16MB, fine.
    
    # Compute M
    # For each i, we want to find all j where (i & j) == i and i != j
    # (i & j) == i iff (i & ~j) == 0
    
    # Let's do it row by row in numpy
    M = np.zeros((total_subsets, total_subsets), dtype=np.int8)
    
    # For each i, compute which j's are strict supersets
    # j is strict superset of i iff (j & i) == i and j != i
    # i.e., j must have all bits of i set, and at least one more bit
    
    # For a given i, the supersets of i are all j where (j & i) == i
    # This is equivalent to: j = i | k where k is any subset of (~i) & ((1<<n)-1)
    # and k != 0 (for strict)
    
    # Let's compute this using numpy broadcasting in chunks
    # For chunk of i's, compute (i & j) == i for all j
    
    chunk_size = 512
    for start in range(0, total_subsets, chunk_size):
        end = min(start + chunk_size, total_subsets)
        i_chunk = np.array(range(start, end), dtype=np.int32)
        j_all = np.array(range(total_subsets), dtype=np.int32)
        
        # i_chunk[:, None] & j_all[None, :]
        # Check if (i & j) == i
        i_expanded = i_chunk[:, None]
        j_expanded = j_all[None, :]
        
        # (i & j) == i
        is_subset = (i_expanded & j_expanded) == i_expanded
        # Strict: i != j
        not_equal = i_expanded != j_expanded
        M[start:end, :] = (is_subset & not_equal).astype(np.int8)
    
    # Now M[i, j] = 1 if i is strict subset of j
    # Comparability matrix: C[i, j] = M[i, j] + M[j, i]
    # C is symmetric
    
    # Greedy with random restarts
    best_score = -1
    best_family = None
    
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        random.seed(restart)
        
        # Start with a random initial set of a few elements
        family = []
        remaining = set(range(total_subsets))
        
        # Optionally start with empty set and full set for structure
        if restart % 2 == 0:
            # Start with empty set and full set
            family = [0, (1 << n) - 1]
            remaining.discard(0)
            remaining.discard((1 << n) - 1)
        else:
            # Start with a random chain
            # Pick a random chain of subsets
            chain = []
            current = 0
            while current != (1 << n) - 1 and len(chain) < 5:
                chain.append(current)
                remaining.discard(current)
                # Add a random bit
                available_bits = [b for b in range(n) if not (current & (1 << b))]
                if available_bits:
                    current |= (1 << random.choice(available_bits))
        
        # Greedily add sets
        while len(family) < m:
            if time.time() > deadline - 5.0:
                break
            
            if not remaining:
                break
            
            # For each candidate, compute comparability count with current family
            family_arr = np.array(family, dtype=np.int32)
            remaining_arr = np.array(list(remaining), dtype=np.int32)
            
            # Compute comparability: for each r in remaining, count how many f in family
            # are comparable with r
            # Comparable if M[r, f] == 1 or M[f, r] == 1
            
            # M[r, f] for r in remaining_arr, f in family_arr
            # M is indexed by mask value, so M[remaining_arr[:, None], family_arr[None, :]]
            
            comp_counts = np.zeros(len(remaining_arr), dtype=np.int32)
            
            # M[remaining_arr[:, None], family_arr[None, :]] gives M[r, f] for all r, f
            # Similarly M[family_arr[None, :], remaining_arr[:, None]] gives M[f, r]
            
            # But we need to be careful with indexing
            # M[i, j] where i and j are mask values
            
            mr_f = M[remaining_arr[:, None], family_arr[None, :]]  # M[r, f]
            mf_r = M[family_arr[None, :], remaining_arr[:, None]]  # M[f, r]
            
            # Comparability: at least one direction
            comparable = (mr_f | mf_r).sum(axis=1)
            
            # Pick the one with highest comparability
            best_idx = np.argmax(comparable)
            best_mask = remaining_arr[best_idx]
            
            family.append(best_mask)
            remaining.discard(best_mask)
        
        if len(family) < m:
            continue
        
        # Compute final score
        family_arr = np.array(family, dtype=np.int32)
        score = 0
        for i in range(len(family_arr)):
            for j in range(i+1, len(family_arr)):
                if M[family_arr[i], family_arr[j]] == 1 or M[family_arr[j], family_arr[i]] == 1:
                    score += 1
        
        if score > best_score:
            best_score = score
            best_family = list(family)
    
    # Convert to lists of integers
    if best_family is None:
        # Fallback
        best_family = list(range(m))
    
    result = []
    for mask in best_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    return result
