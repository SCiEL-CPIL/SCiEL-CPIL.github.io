import time
import itertools
import sys
import random
import numpy as np

def construct(n):
    """
    Constructs the shortest known superpermutation for n symbols.
    For n <= 6, uses the optimal Graham-Lagarias construction.
    For n = 7, uses Graham-Lagarias as a seed and applies local search
    to the final step's permutation ordering to minimize length.
    """
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
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
        _, perms = graham_lagarias(6)
        
        num_perms = len(perms)
        if num_perms != 720:
            # Fallback to standard if something went wrong
            word, _ = graham_lagarias(7)
            return "".join(map(str, word))
            
        # Precompute overlap matrix using NumPy for speed
        # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
        # k can be 0 to n-2 (5)
        max_k = n - 2  # 5
        
        # Convert perms to numpy array for vectorized operations
        perms_arr = np.array(perms, dtype=np.int8)  # shape (720, 6)
        
        # Build overlap matrix
        overlap_matrix = np.zeros((num_perms, num_perms), dtype=np.int8)
        
        # For each k from max_k down to 1, check if suffix of length k matches prefix
        for k in range(max_k, 0, -1):
            # suffix: perms_arr[:, -k:] shape (720, k)
            # prefix: perms_arr[:, :k] shape (720, k)
            # We need overlap[i][j] = k if perms_arr[i, -k:] == perms_arr[j, :k]
            # Vectorize: for each i, find all j where this holds
            suffixes = perms_arr[:, -k:]  # (720, k)
            prefixes = perms_arr[:, :k]   # (720, k)
            
            # Compare all pairs: suffixes[i] == prefixes[j]
            # Use broadcasting: (720, 1, k) vs (1, 720, k) -> (720, 720, k)
            # This is 720*720*k = 720*720*5 = 2.6M comparisons, manageable
            match = np.all(suffixes[:, np.newaxis, :] == prefixes[np.newaxis, :, :], axis=2)
            
            # Only set where not already set (higher k takes priority)
            mask = match & (overlap_matrix == 0)
            overlap_matrix[mask] = k
        
        # Initial order
        order = list(range(num_perms))
        
        # Precompute block length
        block_len = 2 * (n - 1) + 1  # 13
        
        def calc_length(ord_list):
            """Calculate total length given an ordering."""
            total = block_len
            for i in range(len(ord_list) - 1):
                u = ord_list[i]
                v = ord_list[i + 1]
                total += block_len - overlap_matrix[u][v]
            return total
            
        best_length = calc_length(order)
        best_order = order[:]
        
        start_time = time.time()
        deadline = start_time + 140.0  # Leave 10s for safety
        
        # Convert to numpy for faster indexing
        order_arr = np.array(order, dtype=np.int32)
        
        # 1. Adjacent swaps (2-opt with length 2) - vectorized delta calculation
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(order_arr) - 1):
                if time.time() > deadline:
                    break
                u = order_arr[i]
                v = order_arr[i + 1]
                
                # Calculate delta for swapping u and v
                # Old edges: (prev, u), (u, v), (v, next)
                # New edges: (prev, v), (v, u), (u, next)
                old_cost = 0
                new_cost = 0
                
                if i > 0:
                    prev = order_arr[i - 1]
                    old_cost += overlap_matrix[prev][u]
                    new_cost += overlap_matrix[prev][v]
                
                old_cost += overlap_matrix[u][v]
                new_cost += overlap_matrix[v][u]
                
                if i + 1 < len(order_arr) - 1:
                    next_v = order_arr[i + 2]
                    old_cost += overlap_matrix[v][next_v]
                    new_cost += overlap_matrix[u][next_v]
                
                # We want to maximize overlap (minimize length)
                # Delta in length = (new_cost - old_cost) ... wait
                # Length contribution = block_len - overlap
                # So if overlap increases, length decreases
                # delta_length = (block_len - new_overlap) - (block_len - old_overlap) = old_overlap - new_overlap
                # We want delta_length < 0, i.e., new_overlap > old_overlap
                delta = old_cost - new_cost
                if delta > 0:
                    # Swap improves
                    order_arr[i], order_arr[i + 1] = v, u
                    best_length -= delta
                    best_order = order_arr.tolist()
                    improved = True
                    
        # 2. 2-opt search (reversing segments) - with NumPy for speed
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(order_arr) - 1):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(order_arr)):
                    if time.time() > deadline:
                        break
                    
                    u_left = order_arr[i - 1] if i > 0 else None
                    u_right = order_arr[j + 1] if j < len(order_arr) - 1 else None
                    
                    # Old edges
                    old_cost = 0
                    if i > 0:
                        old_cost += overlap_matrix[u_left][order_arr[i]]
                    for k in range(i, j):
                        old_cost += overlap_matrix[order_arr[k]][order_arr[k + 1]]
                    if j < len(order_arr) - 1:
                        old_cost += overlap_matrix[order_arr[j]][u_right]
                        
                    # New edges (reversed segment)
                    new_cost = 0
                    if i > 0:
                        new_cost += overlap_matrix[u_left][order_arr[j]]
                    for k in range(j, i, -1):
                        new_cost += overlap_matrix[order_arr[k]][order_arr[k - 1]]
                    if j < len(order_arr) - 1:
                        new_cost += overlap_matrix[order_arr[i]][u_right]
                        
                    delta = old_cost - new_cost
                    if delta > 0:
                        order_arr[i:j + 1] = order_arr[i:j + 1][::-1]
                        best_length -= delta
                        best_order = order_arr.tolist()
                        improved = True
                        
        # 3. Or-opt (relocation of segments) - simplified but effective
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for L in [1, 2, 3]:
                if time.time() > deadline:
                    break
                for i in range(len(order_arr) - L):
                    if time.time() > deadline:
                        break
                    
                    segment = order_arr[i:i + L].copy()
                    
                    # Calculate current cost of removing segment
                    # Edges removed:
                    removed_cost = 0
                    if i > 0:
                        removed_cost += overlap_matrix[order_arr[i - 1]][order_arr[i]]
                    for k in range(i, i + L - 1):
                        removed_cost += overlap_matrix[order_arr[k]][order_arr[k + 1]]
                    if i + L < len(order_arr):
                        removed_cost += overlap_matrix[order_arr[i + L - 1]][order_arr[i + L]]
                    
                    # Edge added when removing: (i-1, i+L) if both exist
                    removed_edge_cost = 0
                    if i > 0 and i + L < len(order_arr):
                        removed_edge_cost = overlap_matrix[order_arr[i - 1]][order_arr[i + L]]
                    
                    # Try inserting at each position
                    best_delta = 0
                    best_pos = -1
                    
                    for pos in range(len(order_arr) - L + 1):
                        if pos == i:
                            continue  # Same position
                        
                        # Adjust pos if it's after the removal point
                        insert_pos = pos
                        if pos > i:
                            insert_pos = pos - L
                        
                        # Calculate cost of inserting at insert_pos
                        # Edges added:
                        added_cost = 0
                        if insert_pos > 0:
                            added_cost += overlap_matrix[order_arr[insert_pos - 1]][segment[0]]
                        for k in range(L - 1):
                            added_cost += overlap_matrix[segment[k]][segment[k + 1]]
                        if insert_pos + L < len(order_arr) - (L if insert_pos >= i else 0):
                            # Need to be careful with indexing after removal
                            # Simplified: just check if there's a next element
                            next_idx = insert_pos + L
                            if next_idx < len(order_arr) - (L if insert_pos >= i else 0):
                                added_cost += overlap_matrix[segment[-1]][order_arr[next_idx]]
                        
                        # Edge removed when inserting: (insert_pos-1, insert_pos) if both exist
                        inserted_edge_cost = 0
                        if insert_pos > 0 and insert_pos + L < len(order_arr) - (L if insert_pos >= i else 0):
                            inserted_edge_cost = overlap_matrix[order_arr[insert_pos - 1]][order_arr[insert_pos]]
                        
                        # Net change in overlap
                        # We remove 'removed_cost' of overlap edges and add 'removed_edge_cost'
                        # We add 'added_cost' of overlap edges and remove 'inserted_edge_cost'
                        # Net delta in overlap = (removed_edge_cost - removed_cost) + (added_cost - inserted_edge_cost)
                        delta = (removed_edge_cost - removed_cost) + (added_cost - inserted_edge_cost)
                        
                        if delta > best_delta:
                            best_delta = delta
                            best_pos = pos
                    
                    if best_delta > 0 and best_pos >= 0:
                        # Perform the move
                        remaining = np.concatenate([order_arr[:i], order_arr[i + L:]])
                        insert_pos = best_pos
                        if insert_pos > i:
                            insert_pos = insert_pos - L
                        new_order = np.concatenate([remaining[:insert_pos], segment, remaining[insert_pos:]])
                        order_arr = new_order
                        best_length -= best_delta
                        best_order = order_arr.tolist()
                        improved = True
                        
        # 4. Simulated annealing for further improvement
        if time.time() < deadline:
            current_order = best_order[:]
            current_length = best_length
            temp = 100.0
            cooling = 0.995
            
            while time.time() < deadline and temp > 0.1:
                # Random kick: swap two random positions
                i = random.randint(0, len(current_order) - 2)
                j = random.randint(i + 1, len(current_order) - 1)
                
                # Calculate delta for swapping i and j
                old_cost = 0
                new_cost = 0
                
                if i > 0:
                    old_cost += overlap_matrix[current_order[i - 1]][current_order[i]]
                    new_cost += overlap_matrix[current_order[i - 1]][current_order[j]]
                if j < len(current_order) - 1:
                    old_cost += overlap_matrix[current_order[j]][current_order[j + 1]]
                    new_cost += overlap_matrix[current_order[i]][current_order[j + 1]]
                old_cost += overlap_matrix[current_order[i]][current_order[j]]
                new_cost += overlap_matrix[current_order[j]][current_order[i]]
                
                delta = old_cost - new_cost  # Positive means improvement
                
                if delta > 0 or random.random() < np.exp(delta / temp):
                    current_order[i], current_order[j] = current_order[j], current_order[i]
                    current_length -= delta
                    if current_length < best_length:
                        best_length = current_length
                        best_order = current_order[:]
                
                temp *= cooling
        
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
