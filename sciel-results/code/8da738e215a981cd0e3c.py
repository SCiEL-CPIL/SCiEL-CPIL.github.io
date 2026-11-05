import time
import random
import sys
import math

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Classic construction for n != 7
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
                # Calculate overlap with new
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    if n != 7:
        return classic_construct(n)

    # Optimization for n=7
    # 1. Generate permutations of 1..6 in the order of SP(6)
    word_6 = [1]
    for m in range(2, 7):
        perms, seen = [], set()
        for i in range(len(word_6) - m + 2):
            window = tuple(word_6[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word_6 = new
        
    perms_6, seen = [], set()
    for i in range(len(word_6) - 6 + 2):
        window = tuple(word_6[i:i + 6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            perms_6.append(window)
            
    num_perms = len(perms_6) # 720
    
    # 2. Precompute overlap matrix
    # M[i][j] = longest suffix of perms_6[i] matching prefix of perms_6[j]
    # We store the cost as (6 - overlap) to minimize sum of costs.
    # Or just maximize overlap. Let's stick to maximizing overlap for clarity in SA.
    M = [[0]*num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        p1 = perms_6[i]
        for j in range(num_perms):
            p2 = perms_6[j]
            # Max overlap 6
            for k in range(6, -1, -1):
                if k == 0 or p1[-k:] == p2[:k]:
                    M[i][j] = k
                    break
                    
    # 3. Initial Order: Greedy Construction
    # This is a strong starting point.
    order = [0]
    unvisited = set(range(1, num_perms))
    while unvisited:
        last = order[-1]
        best_next = -1
        best_ov = -1
        for j in unvisited:
            ov = M[last][j]
            if ov > best_ov:
                best_ov = ov
                best_next = j
        order.append(best_next)
        unvisited.remove(best_next)
        
    # Score function: Sum of overlaps. Higher is better.
    def calc_score(ord):
        score = 0
        for i in range(len(ord) - 1):
            score += M[ord[i]][ord[i+1]]
        return score

    current_order = order[:]
    current_score = calc_score(current_order)
    best_order = current_order[:]
    best_score = current_score
    
    # 4. Simulated Annealing with Optimized Moves
    # We use a hybrid approach:
    # - Fast delta calculation for local moves (swap, 2-opt, move).
    # - Multiple restarts if time permits.
    
    random.seed(42)
    
    # Precompute neighbors for quick access? No, M is dense.
    # But we can precompute the list of best outgoing edges for each node to guide "smart" random moves?
    # Or just rely on SA.
    
    # To speed up SA, we implement delta calculation.
    # Score = sum M[ord[i]][ord[i+1]]
    
    def get_delta_swap(ord, i):
        # Swap ord[i] and ord[i+1]
        # Old edges: (ord[i-1], ord[i]), (ord[i], ord[i+1]), (ord[i+1], ord[i+2])
        # New edges: (ord[i-1], ord[i+1]), (ord[i+1], ord[i]), (ord[i], ord[i+2])
        a = ord[i-1] if i > 0 else None
        b = ord[i]
        c = ord[i+1]
        d = ord[i+2] if i+2 < len(ord) else None
        
        delta = 0
        if a is not None:
            delta -= M[a][b]
            delta += M[a][c]
        delta -= M[b][c]
        delta += M[c][b]
        if d is not None:
            delta -= M[c][d]
            delta += M[b][d]
        return delta

    def get_delta_move(ord, i, j):
        # Move ord[i] to position j
        # This is more complex. For simplicity and safety, we might just recalc
        # or use a simpler delta if i and j are close.
        # Given 720 items, full recalc is ~720 ops. 
        # SA does many iterations. Let's try to optimize.
        # Actually, let's just use full recalc for Move and 2-Opt to avoid bugs, 
        # but swap is very fast.
        # If we do 100k iterations, 720*100k = 72M ops. Python might be slow.
        # Let's try to do delta for move too.
        
        # Remove item at i, insert at j.
        # Case 1: i < j
        # Case 2: i > j
        
        # It's error prone. Let's stick to Swap and 2-Opt with delta, and Move with full recalc 
        # but limit Move frequency.
        # Or just use full recalc for everything but optimize the inner loop?
        # NumPy?
        # Let's try a different tactic: use a list of arrays and vectorized score?
        # No, standard python list is fine if we optimize the loop.
        
        # Let's just do full recalc for Move and 2-Opt, and delta for Swap.
        # Swap is 40% of moves.
        
        pass # Placeholder

    # Better: Implement delta for 2-Opt (reverse segment)
    def get_delta_2opt(ord, i, j):
        # Reverse segment ord[i..j]
        # Old edges: 
        # (ord[i-1], ord[i]) if i>0
        # (ord[k], ord[k+1]) for k in i..j-1
        # (ord[j], ord[j+1]) if j < n-1
        
        # New edges:
        # (ord[i-1], ord[j]) if i>0
        # (ord[k+1], ord[k]) for k in i..j-1  (i.e. M[ord[k+1]][ord[k]])
        # (ord[i], ord[j+1]) if j < n-1
        
        delta = 0
        if i > 0:
            delta -= M[ord[i-1]][ord[i]]
            delta += M[ord[i-1]][ord[j]]
        
        for k in range(i, j):
            delta -= M[ord[k]][ord[k+1]]
            delta += M[ord[k+1]][ord[k]]
            
        if j < len(ord) - 1:
            delta -= M[ord[j]][ord[j+1]]
            delta += M[ord[i]][ord[j+1]]
            
        return delta

    # For Move, let's just do full recalc if we don't have a fast delta.
    # Actually, Move is powerful. Let's implement a fast delta for Move.
    def get_delta_move(ord, i, j):
        # Move item at i to j.
        # If i == j, return 0.
        # If i < j:
        #   Remove at i. Insert at j (which shifts i..j-1 left by 1).
        #   Old sequence: ... A, B, C, D ... (B is at i, C is at i+1, D is at j)
        #   New sequence: ... A, C, D, B ... 
        #   Wait, if we move B from i to j:
        #   Original: ... X, B, C, D, ... Y ... (B at i, C at i+1, ..., Y at j-1, Z at j)
        #   Remove B: ... X, C, D, ... Y, Z ...
        #   Insert B at j (before Z): ... X, C, D, ... Y, B, Z ...
        #   
        #   Affected edges:
        #   Old: (X, B), (B, C), (Y, Z)
        #   New: (X, C), (Y, B), (B, Z)
        #   
        #   Delta = M[X][C] + M[Y][B] + M[B][Z] - M[X][B] - M[B][C] - M[Y][Z]
        
        # If i > j:
        #   Original: ... X, Y, ... C, B, Z ... (B at i, C at i-1, ..., Y at j, X at j-1)
        #   Remove B: ... X, Y, ... C, Z ...
        #   Insert B at j (before Y): ... X, B, Y, ... C, Z ...
        #   
        #   Affected edges:
        #   Old: (X, Y), (C, B), (B, Z)
        #   New: (X, B), (B, Y), (C, Z)
        
        if i == j:
            return 0
        
        n_ord = len(ord)
        
        if i < j:
            # X is ord[i-1] (if i>0), B is ord[i], C is ord[i+1]
            # Y is ord[j-1], Z is ord[j]
            # Note: The segment i+1..j-1 shifts left. Their internal edges don't change.
            
            X = ord[i-1] if i > 0 else None
            B = ord[i]
            C = ord[i+1]
            Y = ord[j-1]
            Z = ord[j]
            
            delta = 0
            if X is not None:
                delta -= M[X][B]
                delta += M[X][C]
            
            delta -= M[B][C]
            
            delta -= M[Y][Z]
            delta += M[Y][B]
            delta += M[B][Z]
            
            return delta
        else: # i > j
            # X is ord[j-1] (if j>0), Y is ord[j]
            # C is ord[i-1], B is ord[i], Z is ord[i+1] (if i < n-1)
            
            X = ord[j-1] if j > 0 else None
            Y = ord[j]
            C = ord[i-1]
            B = ord[i]
            Z = ord[i+1] if i < n_ord - 1 else None
            
            delta = 0
            if X is not None:
                delta -= M[X][Y]
                delta += M[X][B]
            
            delta += M[B][Y]
            
            delta -= M[C][B]
            if Z is not None:
                delta -= M[B][Z]
                delta += M[C][Z]
            else:
                delta -= M[B][Z] if Z is not None else 0 # Z is None, so M[B][Z] is not valid. 
                # Wait, if Z is None, the edge (B, Z) doesn't exist.
                # So we just subtract M[C][B] and add nothing for the Z part?
                # Let's re-verify.
                # Old: ... C, B, [END]
                # New: ... C, [END]  (B moved away)
                # So we lose edge (C, B). We don't gain anything at the end.
                # So delta -= M[C][B]. Correct.
                # In the code above, if Z is None, we skip the Z part.
                pass
            
            # Correction for Z handling in i > j case:
            # If Z is not None:
            #   Old edges: (C, B), (B, Z)
            #   New edges: (C, Z)
            #   Delta part: M[C][Z] - M[C][B] - M[B][Z]
            # If Z is None:
            #   Old edges: (C, B)
            #   New edges: (none)
            #   Delta part: - M[C][B]
            
            # The code above:
            # delta -= M[C][B]
            # if Z is not None:
            #    delta -= M[B][Z]
            #    delta += M[C][Z]
            # This matches.
            
            return delta

    T = 1.0
    cooling = 0.9999
    min_T = 0.01
    
    # We can also add a "restart" mechanism if we get stuck.
    # But let's first try to make the main loop efficient.
    
    while time.time() < deadline:
        move_type = random.random()
        new_order = current_order[:]
        
        if move_type < 0.4:
            # Swap
            i = random.randint(0, num_perms - 2)
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            delta = get_delta_swap(current_order, i)
        elif move_type < 0.7:
            # 2-Opt
            i = random.randint(0, num_perms - 2)
            j = random.randint(i + 1, num_perms - 1)
            new_order[i:j+1] = new_order[i:j+1][::-1]
            delta = get_delta_2opt(current_order, i, j)
        else:
            # Move
            i = random.randint(0, num_perms - 1)
            j = random.randint(0, num_perms - 1)
            if i == j: continue
            new_order.pop(i)
            new_order.insert(j, current_order[i]) # Insert the item that was at i
            # Note: The item to insert is current_order[i].
            # In get_delta_move, we assumed B = ord[i].
            # So delta is calculated based on current_order.
            delta = get_delta_move(current_order, i, j)
            
        new_score = current_score + delta
        
        if new_score > current_score:
            current_order = new_order
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_order = new_order[:]
        else:
            # Accept with probability exp(delta / T)
            if random.random() < math.exp(delta / T):
                current_order = new_order
                current_score = new_score
        
        T *= cooling
        if T < min_T:
            T = min_T
            
    # 5. Construct word from best_order
    word = []
    for idx, p_idx in enumerate(best_order):
        p = perms_6[p_idx]
        block = list(p) + [7] + list(p)
        if idx == 0:
            word = block
        else:
            # Find overlap with current word
            max_ov = min(len(word), len(block))
            ov = 0
            for k in range(max_ov, -1, -1):
                if word[-k:] == block[:k]:
                    ov = k
                    break
            word += block[ov:]
            
    return "".join(map(str, word))
