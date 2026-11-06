import time
import sys
from math import gcd
from fractions import Fraction

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin

    N = 256
    vertices = list(range(2, N + 1))
    n = len(vertices)

    # Precompute lcm-conflict for all pairs
    # conflict[i] is a set of indices j > i such that lcm(vertices[i], vertices[j]) <= N
    # We use the split: small = 2..S, large = S+1..256
    
    # Try a few split points and pick the best
    best_result = None
    best_val = Fraction(0, 1)

    split_points = [16, 20, 24, 28, 32]
    
    for S in split_points:
        if time.time() > deadline - 5:
            break
        
        small_vertices = list(range(2, S + 1))
        large_vertices = list(range(S + 1, N + 1))
        n_small = len(small_vertices)
        n_large = len(large_vertices)
        
        if n_small > 20:
            # Too many to enumerate all subsets
            continue
        
        # Precompute conflicts within small
        conflict_small = [0] * n_small
        for i in range(n_small):
            a = small_vertices[i]
            for j in range(i + 1, n_small):
                b = small_vertices[j]
                g = gcd(a, b)
                if a * b // g <= N:
                    conflict_small[i] |= (1 << j)
                    conflict_small[j] |= (1 << i)
        
        # Precompute conflicts between small and large
        conflict_small_large = [0] * n_small
        for i in range(n_small):
            a = small_vertices[i]
            mask = 0
            for j in range(n_large):
                b = large_vertices[j]
                g = gcd(a, b)
                if a * b // g <= N:
                    mask |= (1 << j)
            conflict_small_large[i] = mask
        
        # Precompute conflicts within large
        # For large numbers, most pairs have lcm > 256
        # Let's precompute conflict masks
        conflict_large = [0] * n_large
        for i in range(n_large):
            a = large_vertices[i]
            for j in range(i + 1, n_large):
                b = large_vertices[j]
                g = gcd(a, b)
                if a * b // g <= N:
                    conflict_large[i] |= (1 << j)
                    conflict_large[j] |= (1 << i)
        
        weights_small = [Fraction(1, v) for v in small_vertices]
        weights_large = [Fraction(1, v) for v in large_vertices]
        
        # Enumerate valid small subsets
        # For each valid small subset, greedily + local search on large
        
        # Precompute which large vertices are "easy" (few conflicts)
        large_conflict_count = [bin(conflict_large[j]).count('1') for j in range(n_large)]
        
        # Sort large by weight descending (i.e., value ascending)
        large_order = sorted(range(n_large), key=lambda j: large_vertices[j])
        
        # Also try ordering by fewest conflicts
        large_order_conf = sorted(range(n_large), key=lambda j: (large_conflict_count[j], large_vertices[j]))
        
        for sm in range(1 << n_small):
            if time.time() > deadline - 3:
                break
            
            # Check if sm is a valid independent set in small graph
            valid = True
            m = sm
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                if conflict_small[i] & sm:
                    valid = False
                    break
                m ^= lsb
            
            if not valid:
                continue
            
            # Calculate value of small part
            val_s = Fraction(0, 1)
            m = sm
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                val_s += weights_small[i]
                m ^= lsb
            
            # Determine which large vertices are forbidden
            forbidden_large = 0
            m = sm
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                forbidden_large |= conflict_small_large[i]
                m ^= lsb
            
            # Greedy select large vertices, then local search
            for order in [large_order, large_order_conf]:
                if time.time() > deadline - 3:
                    break
                
                selected_large = 0
                val_l = Fraction(0, 1)
                
                for j in order:
                    if selected_large & (1 << j):
                        continue
                    if forbidden_large & (1 << j):
                        continue
                    if conflict_large[j] & selected_large:
                        continue
                    selected_large |= (1 << j)
                    val_l += weights_large[j]
                
                # Local search: try to add, remove, or swap
                improved = True
                while improved and time.time() < deadline - 3:
                    improved = False
                    
                    # Try adding a vertex
                    for j in range(n_large):
                        if selected_large & (1 << j):
                            continue
                        if forbidden_large & (1 << j):
                            continue
                        if conflict_large[j] & selected_large:
                            continue
                        # Can add
                        selected_large |= (1 << j)
                        val_l += weights_large[j]
                        improved = True
                        break
                    
                    # Try swapping: remove one, add another
                    if not improved:
                        # Try removing a low-value vertex and adding a high-value one
                        for j_out in range(n_large):
                            if not (selected_large & (1 << j_out)):
                                continue
                            for j_in in range(n_large):
                                if selected_large & (1 << j_in):
                                    continue
                                if forbidden_large & (1 << j_in):
                                    continue
                                # Check if j_in conflicts with selected_large minus j_out
                                new_selected = selected_large & ~(1 << j_out)
                                if conflict_large[j_in] & new_selected:
                                    continue
                                # Check improvement
                                delta = weights_large[j_in] - weights_large[j_out]
                                if delta > 0:
                                    selected_large = new_selected | (1 << j_in)
                                    val_l += delta
                                    improved = True
                                    break
                            if improved:
                                break
                    
                    # Try removing a vertex that allows adding two others
                    if not improved:
                        for j_out in range(n_large):
                            if not (selected_large & (1 << j_out)):
                                continue
                            new_selected = selected_large & ~(1 << j_out)
                            # Try adding vertices that were blocked by j_out
                            for j_in in range(n_large):
                                if new_selected & (1 << j_in):
                                    continue
                                if forbidden_large & (1 << j_in):
                                    continue
                                if conflict_large[j_in] & new_selected:
                                    continue
                                # j_in was blocked by j_out
                                if not (conflict_large[j_in] & (1 << j_out)):
                                    continue
                                new_selected |= (1 << j_in)
                                delta = weights_large[j_in] - weights_large[j_out]
                                if delta > 0:
                                    selected_large = new_selected
                                    val_l += delta
                                    improved = True
                                    break
                            if improved:
                                break
                
                total_val = val_s + val_l
                
                if total_val > best_val:
                    best_val = total_val
                    # Build result
                    result = []
                    m = sm
                    while m:
                        lsb = m & -m
                        i = lsb.bit_length() - 1
                        result.append(small_vertices[i])
                        m ^= lsb
                    m = selected_large
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        result.append(large_vertices[j])
                        m ^= lsb
                    best_result = result
        
        if time.time() > deadline - 5:
            break
    
    if best_result is None:
        return [2]
    
    return best_result
