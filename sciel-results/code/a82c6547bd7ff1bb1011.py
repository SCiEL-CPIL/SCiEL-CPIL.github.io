import time
import itertools
import sys
import random

def construct(n):
    """
    Constructs the shortest known superpermutation for n symbols.
    For n <= 6, uses the optimal standard construction.
    For n = 7, uses the standard construction as a seed and applies
    intensive local search (fast 2-Opt and Or-Opt) on the permutation ordering.
    """
    if n == 1:
        return "1"
    
    if n == 2:
        return "121"
    
    # Standard Graham-Lagarias construction
    def graham_lagarias(m):
        if m == 1:
            return [1], [(1,)]
        
        prev_word, prev_perms = graham_lagarias(m - 1)
        
        new_word = []
        for p in prev_perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        
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
        start_time = time.time()
        
        # Get perms of 1..6 in the order they appear in the standard construction
        _, perms = graham_lagarias(6)
        
        num_perms = len(perms)
        if num_perms != 720:
            word, _ = graham_lagarias(7)
            return "".join(map(str, word))
            
        # Precompute overlap matrix
        max_k = n - 2  # = 5
        overlap_matrix = [[0] * num_perms for _ in range(num_perms)]
        
        for i in range(num_perms):
            pi = perms[i]
            for j in range(num_perms):
                pj = perms[j]
                found = 0
                for k in range(max_k, 0, -1):
                    if pi[-k:] == pj[:k]:
                        found = k
                        break
                overlap_matrix[i][j] = found
        
        # Edge cost: cost of transitioning from u to v
        # block_length = 2*(n-1)+1 = 13
        block_len = 2 * (n - 1) + 1
        edge_cost = [[0] * num_perms for _ in range(num_perms)]
        for i in range(num_perms):
            for j in range(num_perms):
                edge_cost[i][j] = block_len - overlap_matrix[i][j]
        
        def calc_length(ord_list):
            total = block_len
            for i in range(len(ord_list) - 1):
                total += edge_cost[ord_list[i]][ord_list[i + 1]]
            return total
        
        def build_word(order):
            final_word = []
            for idx in order:
                p = perms[idx]
                block = list(p) + [n] + list(p)
                max_overlap = min(len(final_word), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if final_word[len(final_word) - o:] == block[:o]:
                        overlap = o
                        break
                final_word += block[overlap:]
            return final_word
        
        deadline = start_time + 110.0  # Leave 40s safety margin
        
        # === Fast 2-Opt with prefix sums ===
        # For a given order, precompute prefix sums of edge costs
        # prefix[i] = sum of edge_cost[order[k]][order[k+1]] for k in 0..i-1
        # So prefix has length N, prefix[0]=0, prefix[i] = sum of first i edges
        
        def compute_prefix(order):
            N = len(order)
            prefix = [0] * (N)  # prefix[i] = sum of edges 0..i-1
            for i in range(N - 1):
                prefix[i + 1] = prefix[i] + edge_cost[order[i]][order[i + 1]]
            return prefix
        
        def two_opt_pass(order, deadline):
            """One full pass of 2-Opt. Returns new order and improvement count."""
            N = len(order)
            prefix = compute_prefix(order)
            total = prefix[N - 1] + block_len
            improved = 0
            
            for i in range(N - 1):
                if time.time() > deadline:
                    break
                for j in range(i + 1, N):
                    if time.time() > deadline:
                        break
                    
                    # Delta for reversing order[i..j]
                    # Old edges affected: 
                    #   (i-1, i) if i>0, (i,i+1)..(j-1,j), (j, j+1) if j<N-1
                    # New edges:
                    #   (i-1, j) if i>0, (j,j-1)..(i+1,i), (i, j+1) if j<N-1
                    
                    old_cost = 0
                    if i > 0:
                        old_cost += edge_cost[order[i-1]][order[i]]
                    for k in range(i, j):
                        old_cost += edge_cost[order[k]][order[k+1]]
                    if j < N - 1:
                        old_cost += edge_cost[order[j]][order[j+1]]
                    
                    new_cost = 0
                    if i > 0:
                        new_cost += edge_cost[order[i-1]][order[j]]
                    for k in range(j, i, -1):
                        new_cost += edge_cost[order[k]][order[k-1]]
                    if j < N - 1:
                        new_cost += edge_cost[order[i]][order[j+1]]
                    
                    delta = old_cost - new_cost
                    if delta > 0:
                        order[i:j+1] = order[i:j+1][::-1]
                        total -= delta
                        improved += 1
                        # Recompute prefix (expensive but correct)
                        prefix = compute_prefix(order)
            
            return order, total, improved
        
        # === Or-Opt: move segment of length L from position i to position j ===
        def or_opt_pass(order, deadline):
            """One full pass of Or-Opt for L=1,2,3."""
            N = len(order)
            total = calc_length(order)
            improved = 0
            
            for L in [1, 2, 3]:
                if time.time() > deadline:
                    break
                for i in range(N - L + 1):
                    if time.time() > deadline:
                        break
                    for j in range(N - L + 1):
                        if time.time() > deadline:
                            break
                        if j == i or j == i - 1:
                            continue
                        
                        # Compute delta for moving segment [i, i+L-1] to position j
                        # After removal, if j > i, the insertion point shifts left by L
                        
                        # Save segment
                        seg = order[i:i+L]
                        
                        # Edges affected by removal:
                        #   (i-1, i) if i>0
                        #   (i, i+1), ..., (i+L-2, i+L-1) internal
                        #   (i+L-1, i+L) if i+L < N
                        
                        # Edges affected by insertion at j (in the list after removal):
                        #   (j-1, seg[0]) if j > 0
                        #   (seg[L-1], j+L) if j+L < N-L (i.e., after insertion, next element)
                        
                        # This is complex. Let's just do the move, calc, revert.
                        # For N=720 and L<=3, this is feasible if we're careful.
                        
                        # Actually, let's compute it properly.
                        # Original list: ... A, seg, B ...
                        # Where A ends at index i-1, B starts at index i+L
                        
                        # After removal: ... A, B ...
                        # Insert seg at position j in the new list.
                        
                        # If j < i: insert before the removal point
                        #   New: ... A', seg, A'', B ...
                        #   Edges changed: (A'[-1], seg[0]), (seg[-1], A''[0])
                        #   where A' is order[0..j-1], A'' is order[j..i-1]
                        #   Old edges: (A'[-1], A''[0]) if j>0 and i>j
                        #             plus the removal/insertion edges
                        
                        # This is getting complicated. Let's use a simpler approach:
                        # Just do the move and recalc locally.
                        
                        old_local = 0
                        # Edges that will be broken
                        if i > 0:
                            old_local += edge_cost[order[i-1]][order[i]]
                        for k in range(i, i+L-1):
                            old_local += edge_cost[order[k]][order[k+1]]
                        if i+L < N:
                            old_local += edge_cost[order[i+L-1]][order[i+L]]
                        
                        # Perform the move
                        del order[i:i+L]
                        if j > i:
                            j -= L
                        order[j:j] = seg
                        
                        new_local = 0
                        # Edges that are created
                        if j > 0:
                            new_local += edge_cost[order[j-1]][order[j]]
                        for k in range(j, j+L-1):
                            new_local += edge_cost[order[k]][order[k+1]]
                        if j+L < len(order):
                            new_local += edge_cost[order[j+L-1]][order[j+L]]
                        
                        delta = old_local - new_local
                        if delta > 0:
                            total -= delta
                            improved += 1
                        else:
                            # Revert
                            del order[j:j+L]
                            if j > i:
                                j += L
                            order[i:i] = seg
            
            return order, total, improved
        
        # === Main search loop ===
        # Start with standard order
        order = list(range(num_perms))
        best_length = calc_length(order)
        best_order = order[:]
        
        # Phase 1: 2-Opt
        pass_count = 0
        while time.time() < deadline:
            order, current_length, improved = two_opt_pass(order, deadline)
            pass_count += 1
            if current_length < best_length:
                best_length = current_length
                best_order = order[:]
            if improved == 0:
                break
        
        # Phase 2: Or-Opt
        while time.time() < deadline:
            order, current_length, improved = or_opt_pass(order, deadline)
            if current_length < best_length:
                best_length = current_length
                best_order = order[:]
            if improved == 0:
                break
        
        # Phase 3: Alternate 2-Opt and Or-Opt
        while time.time() < deadline:
            order, current_length, imp1 = two_opt_pass(order, deadline)
            if current_length < best_length:
                best_length = current_length
                best_order = order[:]
            if imp1 == 0 and time.time() > deadline:
                break
            order, current_length, imp2 = or_opt_pass(order, deadline)
            if current_length < best_length:
                best_length = current_length
                best_order = order[:]
            if imp1 == 0 and imp2 == 0:
                break
        
        # Phase 4: Random restarts with perturbation
        remaining = deadline - time.time()
        if remaining > 5:
            for restart in range(20):
                if time.time() > deadline:
                    break
                
                # Perturb best order
                perturbed = best_order[:]
                num_moves = random.randint(10, 30)
                for _ in range(num_moves):
                    L = random.choice([1, 2, 3])
                    i = random.randint(0, len(perturbed) - L)
                    j = random.randint(0, len(perturbed) - L)
                    if j == i or j == i - 1:
                        continue
                    seg = perturbed[i:i+L]
                    del perturbed[i:i+L]
                    if j > i:
                        j -= L
                    perturbed[j:j] = seg
                
                # Quick 2-Opt on perturbed
                order_r = perturbed[:]
                len_r = calc_length(order_r)
                while time.time() < deadline:
                    order_r, len_r, imp = two_opt_pass(order_r, deadline)
                    if imp == 0:
                        break
                
                # Quick Or-Opt
                while time.time() < deadline:
                    order_r, len_r, imp = or_opt_pass(order_r, deadline)
                    if imp == 0:
                        break
                
                if len_r < best_length:
                    best_length = len_r
                    best_order = order_r[:]
        
        # Construct final word
        final_word = build_word(best_order)
        return "".join(map(str, final_word))
        
    # Fallback for n > 7
    word, _ = graham_lagarias(n)
    return "".join(map(str, word))
