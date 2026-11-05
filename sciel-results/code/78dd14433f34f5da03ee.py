"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234123142312431243121342132413214321"
    
    if n != 7:
        # For other n, use standard recursive construction
        return _standard_superperm_str(n)

    # Optimization for n=7
    # 1. Get the list of permutations of 1..6 in the order they appear in SP(6)
    word_6 = [1]
    for m in range(2, 7): # up to 6
        perms, seen = [], set()
        for i in range(len(word_6) - m + 2):
            window = tuple(word_6[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Fast overlap check
            max_ov = min(len(new), len(block))
            ov = 0
            for k in range(max_ov, -1, -1):
                if new[len(new) - k:] == block[:k]:
                    ov = k
                    break
            new += block[ov:]
        word_6 = new
        
    perms_6, seen = [], set()
    for i in range(len(word_6) - 6 + 2):
        window = tuple(word_6[i:i + 6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            perms_6.append(window)
            
    num_perms = len(perms_6)
    
    # 2. Precompute overlap matrix
    # M[i][j] = longest k such that perms_6[i][-k:] == perms_6[j][:k]
    M = [[0]*num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        p1 = perms_6[i]
        for j in range(num_perms):
            p2 = perms_6[j]
            # Max overlap 6, but since i!=j, max 5.
            # Check k from 5 down to 0
            for k in range(5, -1, -1):
                if k == 0 or p1[-k:] == p2[:k]:
                    M[i][j] = k
                    break
                    
    # 3. Greedy Nearest Neighbor for initial order
    def greedy_nn():
        remaining = set(range(num_perms))
        order = [random.choice(list(remaining))]
        remaining.remove(order[0])
        while remaining:
            last = order[-1]
            best_next = -1
            best_score = -1
            # Optimization: pick random subset if too slow? 720^2 is fine.
            for j in remaining:
                score = M[last][j]
                if score > best_score:
                    best_score = score
                    best_next = j
            order.append(best_next)
            remaining.remove(best_next)
        return order

    def calc_score(ord):
        score = 0
        for i in range(len(ord) - 1):
            score += M[ord[i]][ord[i+1]]
        return score

    # Generate a few initial orders
    random.seed(42)
    initial_orders = []
    for _ in range(5):
        initial_orders.append(greedy_nn())
    
    best_order = max(initial_orders, key=calc_score)
    best_score = calc_score(best_order)
    current_order = best_order[:]
    current_score = best_score

    # 4. Simulated Annealing with O(1) delta moves
    T = 1.0
    T_min = 0.001
    cooling = 0.9995 # Slower cooling to allow more exploration
    
    # Helper for O(1) Swap Delta
    def delta_swap(ord, i):
        # Swap i and i+1
        if i < 0 or i >= num_perms - 1:
            return 0
        A = ord[i-1] if i > 0 else None
        B = ord[i]
        C = ord[i+1]
        D = ord[i+2] if i+2 < num_perms else None
        
        old = 0
        new = 0
        if A is not None:
            old += M[A][B]
            new += M[A][C]
        old += M[B][C]
        new += M[C][B]
        if D is not None:
            old += M[C][D]
            new += M[B][D]
            
        return new - old

    # Helper for O(1) Move Delta
    # Move element at i to j (in the list after removal)
    def delta_move(ord, i, j):
        if i == j:
            return 0
        
        x = ord[i]
        
        # Calculate Delta Remove
        delta_rem = 0
        left_i = ord[i-1] if i > 0 else None
        right_i = ord[i+1] if i < num_perms - 1 else None
        
        if left_i is not None and right_i is not None:
            delta_rem -= M[left_i][x]
            delta_rem -= M[x][right_i]
            delta_rem += M[left_i][right_i]
        elif left_i is not None:
            delta_rem -= M[left_i][x]
        elif right_i is not None:
            delta_rem -= M[x][right_i]
        
        # Calculate Delta Insert
        # Determine neighbors in the list AFTER removal
        # If j > i, the indices in the original list for the neighbors are shifted
        # Let's find the values of the neighbors at position j in the reduced list
        
        # The reduced list is ord with x removed.
        # We want to insert x at index j in this reduced list.
        # The elements at indices < j in reduced list are:
        #   if k < i: ord[k]
        #   if k >= i: ord[k+1]
        
        # Left neighbor at j-1 in reduced list (if j > 0)
        left_j_val = None
        if j > 0:
            if j - 1 < i:
                left_j_val = ord[j-1]
            else:
                left_j_val = ord[j] # ord[j] is at index j-1 in reduced list? 
                # Wait. 
                # Original: ... ord[j-1], ord[j], ...
                # If j-1 >= i, then ord[j-1] was removed? No, x was at i.
                # If j > i, then indices >= i in original shift left.
                # So index k in reduced list corresponds to:
                #   ord[k] if k < i
                #   ord[k+1] if k >= i
                # So index j-1 in reduced list:
                #   if j-1 < i: ord[j-1]
                #   if j-1 >= i: ord[j]
                # Yes.
        
        # Right neighbor at j in reduced list (if j < num_perms - 1)
        right_j_val = None
        if j < num_perms - 1:
            if j < i:
                right_j_val = ord[j]
            else:
                right_j_val = ord[j+1]
                # If j >= i, index j in reduced list corresponds to ord[j+1].
                # Example: i=2, j=2. Reduced list has ord[0], ord[1], ord[3]...
                # Index 0: ord[0], 1: ord[1], 2: ord[3].
                # So index j=2 is ord[3]. Correct.
        
        delta_ins = 0
        if left_j_val is not None and right_j_val is not None:
            delta_ins -= M[left_j_val][right_j_val]
            delta_ins += M[left_j_val][x]
            delta_ins += M[x][right_j_val]
        elif left_j_val is not None:
            delta_ins += M[left_j_val][x]
        elif right_j_val is not None:
            delta_ins += M[x][right_j_val]
            
        return delta_rem + delta_ins

    steps = 0
    no_improve_count = 0
    max_no_improve = 10000
    
    while time.time() < deadline and T > T_min:
        move_type = random.random()
        if move_type < 0.5:
            # Swap
            i = random.randint(0, num_perms - 2)
            delta = delta_swap(current_order, i)
            
            if delta >= 0 or random.random() < pow(2.71828, delta / T):
                current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_order = current_order[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            # Move
            i = random.randint(0, num_perms - 1)
            j = random.randint(0, num_perms - 1)
            if i == j:
                continue
            delta = delta_move(current_order, i, j)
            
            if delta >= 0 or random.random() < pow(2.71828, delta / T):
                # Perform move
                x = current_order.pop(i)
                # Adjust j if i < j
                if i < j:
                    j -= 1
                current_order.insert(j, x)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_order = current_order[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        
        T *= cooling
        steps += 1
        
        # Optional: If stuck for too long, restart with higher T?
        # Or just continue.
        
    # 5. Local Polish (Hill Climbing with Swaps)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(num_perms - 1):
            delta = delta_swap(best_order, i)
            if delta > 0:
                best_order[i], best_order[i+1] = best_order[i+1], best_order[i]
                best_score += delta
                improved = True
                # Restart loop from beginning to catch cascading improvements
                break
        # Also try Moves? Hill climbing with Moves is harder to implement efficiently 
        # without O(N) checks. Given time, Swap polish is good.
        
    # 6. Construct word from best_order
    word = []
    for idx, p_idx in enumerate(best_order):
        p = perms_6[p_idx]
        block = list(p) + [7] + list(p)
        if idx == 0:
            word = block
        else:
            # Find overlap with current word
            # Suffix of word vs Prefix of block
            # We know the overlap is M[best_order[idx-1]][p_idx]
            # But we must verify because the word might have accumulated more?
            # No, the overlap is strictly determined by the blocks.
            ov = M[best_order[idx-1]][p_idx]
            word += block[ov:]
            
    return "".join(map(str, word))

def _standard_superperm_str(n):
    """Generate the standard recursive superpermutation as a string."""
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

# EVOLVE-BLOCK-END
