import numpy as np
import itertools
import random
import math
import time

def construct(q, n, radius):
    # Parameters for this task are fixed: q=3, n=8, radius=1
    # But we write it generally for the interface.
    
    total_words = q ** n
    
    # Precompute neighbors (ball of radius 1) for all words
    def to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val
        
    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word
            
    neighbors = []
    for i in range(total_words):
        word = to_word(i)
        ball = [i]
        for pos in range(n):
            for delta in range(1, q):
                new_word = word[:]
                new_word[pos] = (word[pos] + delta) % q
                ball.append(to_int(new_word))
        neighbors.append(ball)
        
    # 1. Algebraic Initialization: Linear Code of Size 3^(n-k)
    # For q=3, n=8, we use a [8, 6, 3] code (parity check 2x8 with all non-zero cols).
    # This code has covering radius 1.
    # We generate the 729 codewords.
    
    # Generate parity check matrix H (2 x n)
    # Columns are all non-zero vectors in Z_q^2
    # For general q, n, this only works if n = q^2 - 1.
    # Here q=3, n=8, so q^2-1 = 8. It fits perfectly.
    
    if q == 3 and n == 8:
        H_cols = []
        for a in range(q):
            for b in range(q):
                if a == 0 and b == 0:
                    continue
                H_cols.append((a, b))
        
        # H is 2 x 8
        # Codewords are vectors x such that H * x^T = 0
        # We can generate them by iterating over the first 6 coordinates and solving for the last 2.
        # Or simply iterate over all 3^8 and check? 6561 checks is fast.
        # But we can be smarter.
        
        # Let's just use the greedy/local search structure but initialize with this code.
        # We need to find the 729 codewords.
        
        # System of equations:
        # x0*a0 + x1*a1 + ... + x7*a7 = 0
        # x0*b0 + x1*b1 + ... + x7*b7 = 0
        # where (ai, bi) are the columns.
        
        # Let's solve for x6, x7 given x0..x5.
        # The last two columns are H[0], H[1].
        # Let h6 = H_cols[6], h7 = H_cols[7].
        # We need h6*x6 + h7*x7 = -sum_{i=0}^5 h_i*x_i.
        # This is a 2x2 linear system over Z_3.
        
        code_init = set()
        for x0 in range(q):
            for x1 in range(q):
                for x2 in range(q):
                    for x3 in range(q):
                        for x4 in range(q):
                            for x5 in range(q):
                                # Calculate syndrome of first 6 parts
                                s0 = 0
                                s1 = 0
                                for i in range(6):
                                    xi = [x0, x1, x2, x3, x4, x5][i]
                                    s0 = (s0 + xi * H_cols[i][0]) % q
                                    s1 = (s1 + xi * H_cols[i][1]) % q
                                
                                # We need x6, x7 such that:
                                # H_cols[6][0]*x6 + H_cols[7][0]*x7 = -s0
                                # H_cols[6][1]*x6 + H_cols[7][1]*x7 = -s1
                                
                                a = H_cols[6][0]
                                b = H_cols[7][0]
                                c = H_cols[6][1]
                                d = H_cols[7][1]
                                e = (-s0) % q
                                f = (-s1) % q
                                
                                # Solve:
                                # a*x6 + b*x7 = e
                                # c*x6 + d*x7 = f
                                det = (a*d - b*c) % q
                                if det == 0:
                                    # This shouldn't happen if H_cols are independent
                                    continue
                                det_inv = pow(det, q-2, q) # Fermat's little theorem for inverse in Z_p (q=3 is prime)
                                x6 = (e*d - b*f) * det_inv % q
                                x7 = (a*f - e*c) * det_inv % q
                                
                                word = [x0, x1, x2, x3, x4, x5, x6, x7]
                                code_init.add(to_int(word))
        
        # code_init should have 729 elements
        code = code_init
    else:
        # Fallback to Greedy for other parameters
        uncovered_mask = np.ones(total_words, dtype=np.int8)
        code = set()
        
        neighbor_arrays = [np.array(neighbors[i]) for i in range(total_words)]
        
        while np.any(uncovered_mask):
            best_c = -1
            best_count = -1
            for c in range(total_words):
                if c in code:
                    continue
                count = np.sum(uncovered_mask[neighbor_arrays[c]])
                if count > best_count:
                    best_count = count
                    best_c = c
            if best_c == -1:
                break
            code.add(best_c)
            uncovered_mask[neighbor_arrays[best_c]] = 0

    # 2. Local Search: Minimize Code Size
    # State: code (set), cover_count (array)
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in code:
        cover_count[neighbors[c]] += 1
        
    # Helper to update cover count
    def add_to_cover(c):
        cover_count[neighbors[c]] += 1
        
    def remove_from_cover(c):
        cover_count[neighbors[c]] -= 1
        
    # Iterative Improvement
    # We try to remove redundant words and swap to enable removals.
    max_iterations = 5000
    for _ in range(max_iterations):
        improved = False
        
        # Phase 1: Remove any word that doesn't uniquely cover any point
        candidates = list(code)
        random.shuffle(candidates)
        
        for c in candidates:
            if c not in code:
                continue
            is_redundant = True
            for w in neighbors[c]:
                if cover_count[w] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                code.remove(c)
                remove_from_cover(c)
                improved = True
                break # Break to restart the outer loop
        
        if improved:
            continue
            
        # Phase 2: Try Swaps
        code_list = list(code)
        
        unique_counts = {}
        for c in code_list:
            count = 0
            for w in neighbors[c]:
                if cover_count[w] == 1:
                    count += 1
            unique_counts[c] = count
            
        sorted_code = sorted(code_list, key=lambda x: unique_counts[x])
        
        swapped = False
        for c_out in sorted_code:
            if swapped:
                break
            unique_points = [w for w in neighbors[c_out] if cover_count[w] == 1]
            
            if not unique_points:
                continue
                
            best_c_in = -1
            best_overlap = 0
            
            for c_in in range(total_words):
                if c_in in code:
                    continue
                    
                overlap = 0
                for w in unique_points:
                    if w in neighbors[c_in]:
                        overlap += 1
                            
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_c_in = c_in
                    
                if best_overlap == len(unique_points):
                    break
            
            if best_c_in != -1 and best_overlap > 0:
                code.remove(c_out)
                remove_from_cover(c_out)
                code.add(best_c_in)
                add_to_cover(best_c_in)
                swapped = True
                improved = True
                break
        
        if not improved:
            break
            
    # Convert back to list of lists
    result = [to_word(c) for c in code]
    return result
