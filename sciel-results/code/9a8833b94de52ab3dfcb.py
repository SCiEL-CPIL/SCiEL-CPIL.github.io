import time
import sys
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "12341231423124312132413214321"

    # Generate standard S_{n-1}
    word = [1]
    for m in range(2, n):
        perms, seen = [], set()
        w_size = m - 1
        for i in range(len(word) - w_size + 1):
            window = tuple(word[i:i + w_size])
            if len(set(window)) == w_size:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    
    # Extract perms of size n-1
    perms, seen = [], set()
    w_size = n - 1
    for i in range(len(word) - w_size + 1):
        window = tuple(word[i:i + w_size])
        if len(set(window)) == w_size:
            if window not in seen:
                seen.add(window)
                perms.append(window)
                
    N = len(perms)
    
    if n >= 5:
        # Precompute overlap matrix
        # W[i][j] = overlap(perms[i], perms[j])
        W = [[0]*N for _ in range(N)]
        for i in range(N):
            pi = perms[i]
            for j in range(N):
                if i == j: continue
                pj = perms[j]
                max_len = min(len(pi), len(pj))
                ov = 0
                for k in range(max_len, -1, -1):
                    if k == 0:
                        ov = 0
                        break
                    if pi[-k:] == pj[:k]:
                        ov = k
                        break
                W[i][j] = ov

        # Initial order: use natural order from recursive construction
        order = list(range(N))
        
        # Calculate initial total overlap
        total_ov = 0
        for i in range(1, N):
            total_ov += W[order[i-1]][order[i]]
            
        best_order = order[:]
        best_total_ov = total_ov
        
        def compute_total_ov(ord):
            """Compute total overlap for a given ordering."""
            ov = 0
            for i in range(1, len(ord)):
                ov += W[ord[i-1]][ord[i]]
            return ov
        
        # Local search with multiple move types
        def local_search(ord, total_ov, time_limit):
            improved = True
            while improved and time.time() < time_limit:
                improved = False
                
                # Try adjacent swaps
                for i in range(N - 1):
                    if time.time() > time_limit:
                        break
                    
                    a = ord[i]
                    b = ord[i+1]
                    
                    # Old edges: (i-1, a), (a, b), (b, i+2)
                    old_edges = 0
                    if i > 0:
                        old_edges += W[ord[i-1]][a]
                    old_edges += W[a][b]
                    if i+1 < N-1:
                        old_edges += W[b][ord[i+2]]
                    
                    # New edges: (i-1, b), (b, a), (a, i+2)
                    new_edges = 0
                    if i > 0:
                        new_edges += W[ord[i-1]][b]
                    new_edges += W[b][a]
                    if i+1 < N-1:
                        new_edges += W[a][ord[i+2]]
                    
                    if new_edges > old_edges:
                        ord[i], ord[i+1] = b, a
                        total_ov += (new_edges - old_edges)
                        improved = True
                        break  # Restart scan after a move
                
                # Try or-opt: move single element
                for i in range(N):
                    if time.time() > time_limit:
                        break
                    for j in range(N):
                        if i == j or i == j + 1:
                            continue
                        
                        elem = ord[i]
                        
                        # Old edges affected:
                        # (i-1, i) if i > 0
                        # (i, i+1) if i < N-1
                        
                        old_edges = 0
                        if i > 0:
                            old_edges += W[ord[i-1]][elem]
                        if i < N-1:
                            old_edges += W[elem][ord[i+1]]
                        
                        # New edges:
                        # If i < j: inserting at position j (which is now j-1 after removal)
                        #   Left: ord[j-1] (if j-1 > i, else ord[i-1])
                        #   Right: ord[j] (if j < N-1, else none)
                        # If i > j: inserting at position j
                        #   Left: ord[j-1] (if j > 0)
                        #   Right: ord[j] (if j < N-1)
                        
                        new_edges = 0
                        if i < j:
                            # After removing i, the element at position j is now at j-1
                            left_idx = j - 1
                            right_idx = j
                            if left_idx >= 0 and left_idx != i:
                                left_elem = ord[left_idx]
                                new_edges += W[left_elem][elem]
                            if right_idx < N and right_idx != i:
                                right_elem = ord[right_idx]
                                new_edges += W[elem][right_elem]
                        else:
                            # i > j, inserting at position j
                            left_idx = j - 1
                            right_idx = j
                            if left_idx >= 0:
                                left_elem = ord[left_idx]
                                new_edges += W[left_elem][elem]
                            if right_idx < N and right_idx != i:
                                right_elem = ord[right_idx]
                                new_edges += W[elem][right_elem]
                        
                        # Also, the edge that connected the neighbors of i changes
                        if i > 0 and i < N-1:
                            # Old: (i-1, i+1) was not an edge, now it is
                            new_edges += W[ord[i-1]][ord[i+1]]
                        
                        if new_edges > old_edges:
                            # Perform the move
                            # Remove from i
                            ord.pop(i)
                            # Insert at j (adjust if i < j)
                            if i < j:
                                j -= 1
                            ord.insert(j, elem)
                            total_ov += (new_edges - old_edges)
                            improved = True
                            break  # Restart scan after a move
                    if improved:
                        break
                
                # Try 2-opt: reverse segment from i to j (limited range for speed)
                max_2opt_range = min(50, N)  # Limit reversal length for performance
                for i in range(N - 1):
                    if time.time() > time_limit:
                        break
                    for j in range(i+1, min(i+max_2opt_range, N)):
                        if time.time() > time_limit:
                            break
                        
                        # Reverse ord[i..j]
                        # Old edges: (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
                        # New edges: (i-1, j), (j, j-1), ..., (i+1, i), (i, j+1)
                        
                        old_edges = 0
                        new_edges = 0
                        
                        # Edge before segment
                        if i > 0:
                            old_edges += W[ord[i-1]][ord[i]]
                            new_edges += W[ord[i-1]][ord[j]]
                        
                        # Edges within segment (reversed)
                        for k in range(i, j):
                            old_edges += W[ord[k]][ord[k+1]]
                            new_edges += W[ord[k+1]][ord[k]]
                        
                        # Edge after segment
                        if j < N-1:
                            old_edges += W[ord[j]][ord[j+1]]
                            new_edges += W[ord[i]][ord[j+1]]
                        
                        if new_edges > old_edges:
                            # Reverse the segment
                            ord[i:j+1] = ord[i:j+1][::-1]
                            total_ov += (new_edges - old_edges)
                            improved = True
                            break  # Restart scan after a move
                    if improved:
                        break
                
                if not improved:
                    break
            
            return ord, total_ov
        
        # Run local search with restarts
        random.seed(42)
        num_restarts = 100
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
            
            # For first few restarts, use natural order; then perturb
            if restart == 0:
                order = list(range(N))
                total_ov = compute_total_ov(order)
            else:
                # Perturb the best order found so far
                order = best_order[:]
                # Randomly swap a few pairs
                num_swaps = min(N // 10, 30)
                for _ in range(num_swaps):
                    i = random.randint(0, N-2)
                    j = random.randint(i+1, N-1)
                    order[i], order[j] = order[j], order[i]
                total_ov = compute_total_ov(order)
            
            # Run local search
            time_limit = min(deadline - 15, time.time() + 1.5)
            order, total_ov = local_search(order, total_ov, time_limit)
            
            if total_ov > best_total_ov:
                best_total_ov = total_ov
                best_order = order[:]
        
        order = best_order

    # Construct final word using precomputed overlaps for speed
    final_word = []
    for idx, perm_idx in enumerate(order):
        p = perms[perm_idx]
        block = list(p) + [n] + list(p)
        if idx == 0:
            final_word = block[:]
        else:
            prev_perm = perms[order[idx-1]]
            # Get overlap from matrix
            ov = W[order[idx-1]][order[idx]]
            # The overlap is between the end of prev block and start of current block
            # Since block = p + [n] + p, and prev block ended with prev_p,
            # the overlap ov means we can skip ov characters from the start of block
            # But we need to be careful: the overlap is between prev_p and p
            # block starts with p, so we skip ov characters
            final_word += block[ov:]
        
    return "".join(map(str, final_word))
