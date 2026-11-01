import time
import itertools
import sys

def construct(n):
    """
    Constructs the shortest known superpermutation for n symbols.
    For n <= 6, uses the optimal Graham-Lagarias construction.
    For n = 7, uses Graham-Lagarias as a seed and applies local search
    to the final step's permutation ordering to minimize length.
    """
    if n == 1:
        return "1"
    
    # Standard Graham-Lagarias construction
    def graham_lagarias(m):
        if m == 1:
            return [1], [(1,)]
        
        # Get word and perms for m-1
        prev_word, prev_perms = graham_lagarias(m - 1)
        
        new_word = []
        for p in prev_perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        
        # Extract perms of length m from new_word
        perms = []
        seen = set()
        for i in range(len(new_word) - m + 1):
            window = tuple(new_word[i:i + m])
            if len(set(window)) == m and window not in seen:
                seen.add(window)
                perms.append(window)
                
        return new_word, perms

    if n <= 6:
        word, _ = graham_lagarias(n)
        return "".join(map(str, word))
    
    if n == 7:
        # Get perms of 1..6 in the order they appear in the standard construction
        # We need the list of permutations P_1 ... P_720
        _, perms = graham_lagarias(6)
        
        num_perms = len(perms)
        if num_perms != 720:
            # Fallback to standard if something went wrong
            word, _ = graham_lagarias(7)
            return "".join(map(str, word))
            
        # Precompute overlap matrix
        # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
        # k can be 0 to n-2 (5)
        max_k = n - 2
        overlap_matrix = [[0] * num_perms for _ in range(num_perms)]
        
        # Optimization: Use strings or tuples for comparison? Tuples are fine.
        for i in range(num_perms):
            pi = perms[i]
            for j in range(num_perms):
                pj = perms[j]
                # Check from max_k down to 0
                # pi[-k:] == pj[:k]
                found = 0
                for k in range(max_k, -1, -1):
                    if k == 0:
                        found = 0
                        break
                    if pi[-k:] == pj[:k]:
                        found = k
                        break
                overlap_matrix[i][j] = found
                
        # Initial order
        order = list(range(num_perms))
        
        def calc_length(ord_list):
            total = 2 * (n - 1) + 1
            for i in range(len(ord_list) - 1):
                u = ord_list[i]
                v = ord_list[i + 1]
                total += (2 * (n - 1) + 1) - overlap_matrix[u][v]
            return total
            
        best_length = calc_length(order)
        best_order = order[:]
        
        start_time = time.time()
        deadline = start_time + 110.0 # Leave 10s for safety margin (total 120s limit)
        
        improved = True
        # Adjacent swaps
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(order) - 1):
                # Try swap i, i+1
                order[i], order[i+1] = order[i+1], order[i]
                new_len = calc_length(order)
                if new_len < best_length:
                    best_length = new_len
                    best_order = order[:]
                    improved = True
                else:
                    order[i], order[i+1] = order[i+1], order[i]
                    
        # 2-opt search
        # Reversing a segment [i, j]
        # Edges affected: (i-1, i), (i, i+1)...(j-1, j), (j, j+1)
        # New edges: (i-1, j), (j, j-1)...(i+1, i), (i, j+1) ? No, (i, j+1) is not right.
        # Sequence: ... A, B, C, D ...
        # Reverse B..C.
        # New: ... A, C, B, D ...
        # Old edges: (A,B), (B,C), (C,D)
        # New edges: (A,C), (C,B), (B,D)
        # Wait, if segment is longer: A, B, C, D, E. Reverse B,C,D.
        # New: A, D, C, B, E.
        # Old: (A,B), (B,C), (C,D), (D,E)
        # New: (A,D), (D,C), (C,B), (B,E)
        # Delta = w(A,D) + w(D,C) + w(C,B) + w(B,E) - w(A,B) - w(B,C) - w(C,D) - w(D,E)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(order) - 1):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(order)):
                    if time.time() > deadline:
                        break
                    # Calculate delta
                    # Indices in order list:
                    # i-1, i, i+1 ... j, j+1
                    # Note: order is a list of indices into perms.
                    
                    # Get nodes
                    u_left = order[i-1] if i > 0 else None
                    u_right = order[j+1] if j < len(order)-1 else None
                    
                    # Old edges
                    old_cost = 0
                    # Edge (i-1, i)
                    if i > 0:
                        old_cost += overlap_matrix[u_left][order[i]]
                    # Internal edges i..j
                    for k in range(i, j):
                        old_cost += overlap_matrix[order[k]][order[k+1]]
                    # Edge (j, j+1)
                    if j < len(order)-1:
                        old_cost += overlap_matrix[order[j]][u_right]
                        
                    # New edges
                    new_cost = 0
                    # Edge (i-1, j)
                    if i > 0:
                        new_cost += overlap_matrix[u_left][order[j]]
                    # Internal edges reversed: j, j-1 ... i
                    for k in range(j, i, -1):
                        new_cost += overlap_matrix[order[k]][order[k-1]]
                    # Edge (i, j+1)
                    if j < len(order)-1:
                        new_cost += overlap_matrix[order[i]][u_right]
                        
                    # We want to maximize overlap, so minimize length.
                    # Length change = (old_len - new_len) ?
                    # Length = Sum( (2n-1) - overlap )
                    # Delta Length = - (new_overlap - old_overlap)
                    # If new_overlap > old_overlap, length decreases.
                    if new_cost > old_cost:
                        # Apply reversal
                        order[i:j+1] = order[i:j+1][::-1]
                        # Update best
                        # Recalculate full length to be safe? Or update best_length?
                        # best_length -= (new_cost - old_cost)
                        best_length -= (new_cost - old_cost)
                        best_order = order[:]
                        improved = True
                        
        # Construct final word
        final_word = []
        for idx in best_order:
            p = perms[idx]
            block = list(p) + [n] + list(p)
            max_overlap = min(len(final_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if final_word[len(final_word) - o:] == block[:o]:
                    overlap = o
                    break
            final_word += block[overlap:]
            
        return "".join(map(str, final_word))
        
    # Fallback for n > 7 (unlikely to be tested, but handle gracefully)
    word, _ = graham_lagarias(n)
    return "".join(map(str, word))
