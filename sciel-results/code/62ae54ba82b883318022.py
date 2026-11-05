import time
import random
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # Classic construction
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
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    if n != 7:
        return classic_construct(n)
    
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word_6 = new
        
    perms_6, seen = [], set()
    for i in range(len(word_6) - 6 + 2):
        window = tuple(word_6[i:i + 6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            perms_6.append(window)
            
    # perms_6 is a list of 720 tuples of length 6
    num_perms = len(perms_6)
    
    # 2. Precompute overlap matrix
    # M[i][j] = overlap between perms_6[i] and perms_6[j]
    # Overlap is longest suffix of i matching prefix of j
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
                    
    # 3. Local Search with Delta Calculations
    order = list(range(num_perms))
    random.seed(42)
    
    # Calculate initial score
    def calc_score(ord):
        score = 0
        for i in range(len(ord) - 1):
            score += M[ord[i]][ord[i+1]]
        return score
        
    best_order = order[:]
    best_score = calc_score(order)
    current_score = best_score
    
    # Helper to get neighbor indices safely
    def get_prev(i):
        return order[i-1] if i > 0 else None
    def get_next(i):
        return order[i+1] if i < num_perms - 1 else None

    # Simulated Annealing parameters
    temp = 1.0
    min_temp = 0.01
    cool_rate = 0.9999

    while time.time() < deadline and temp > min_temp:
        # Pick a random move
        move_type = random.random()
        if move_type < 0.5:
            # Swap adjacent
            i = random.randint(0, num_perms - 2)
            a = order[i]
            b = order[i+1]
            
            # Calculate Delta
            # Old: prev(a)->a->b->next(b)
            # New: prev(a)->b->a->next(b)
            delta = 0
            prev_a = order[i-1] if i > 0 else None
            next_b = order[i+2] if i+2 < num_perms else None
            
            if prev_a is not None:
                delta -= M[prev_a][a]
                delta += M[prev_a][b]
            # Middle edge
            delta -= M[a][b]
            delta += M[b][a]
            if next_b is not None:
                delta -= M[b][next_b]
                delta += M[a][next_b]
                
            new_score = current_score + delta
            if new_score > best_score:
                # Apply swap
                order[i], order[i+1] = b, a
                current_score = new_score
                best_score = current_score
                best_order = order[:]
            else:
                # Accept with probability exp(-delta/temp) if delta < 0
                if delta < 0 and random.random() < pow(2.718281828, -delta / temp):
                    order[i], order[i+1] = b, a
                    current_score = new_score
        else:
            # Move element i to position j
            i = random.randint(0, num_perms - 1)
            j = random.randint(0, num_perms - 1)
            if i == j: continue
            
            # We need to handle i < j and i > j carefully
            # Let's just swap for now if move is complex? No, move is better.
            # Let's implement Move with Delta.
            
            if i < j:
                # Remove i, insert at j
                # Sequence: ... A, x, B, y, C ...
                # x at i, y at j.
                # B is i+1 to j-1.
                # If j == i+1, B is empty.
                
                x = order[i]
                y = order[j]
                
                prev_x = order[i-1] if i > 0 else None
                next_y = order[j+1] if j+1 < num_perms else None
                
                delta = 0
                
                # Edges involving x and y and their neighbors change
                # Old:
                # (prev_x, x) if prev_x
                # (x, order[i+1]) if i+1 <= j (i.e., B not empty or y is next)
                # ...
                # (order[j-1], y) if j-1 >= i (i.e., B not empty or x is prev)
                # (y, next_y) if next_y
                
                # New:
                # (prev_x, order[i+1]) if i+1 <= j
                # ...
                # (order[j-1], x) if j-1 >= i
                # (x, y)
                # (y, next_y)
                
                # Let's identify the specific edges
                # Edge 1: (prev_x, x) -> (prev_x, order[i+1])  [if i+1 <= j]
                # Edge 2: (x, order[i+1]) -> REMOVED
                # Edge 3: (order[j-1], y) -> (order[j-1], x) [if j-1 >= i]
                # Edge 4: (y, next_y) -> (y, next_y) [UNCHANGED]
                # New Edge: (x, y)
                
                # Wait, if B is empty (j == i+1):
                # Old: (prev_x, x), (x, y), (y, next_y)
                # New: (prev_x, y), (y, x), (x, next_y)
                
                # Case j == i+1:
                if j == i + 1:
                    if prev_x is not None:
                        delta -= M[prev_x][x]
                        delta += M[prev_x][y]
                    delta -= M[x][y]
                    delta += M[y][x]
                    if next_y is not None:
                        delta -= M[y][next_y]
                        delta += M[x][next_y]
                else:
                    # B is not empty
                    b_first = order[i+1]
                    b_last = order[j-1]
                    
                    if prev_x is not None:
                        delta -= M[prev_x][x]
                        delta += M[prev_x][b_first]
                    else:
                        # x was first, now b_first is first
                        # No edge before x, so no change there?
                        # Actually, if i=0, prev_x is None.
                        # Old: x->b_first
                        # New: b_first is first? No, b_first is at i.
                        # The edge (x, b_first) is removed.
                        # Is there a new edge before b_first? No.
                        pass 
                    
                    # Remove (x, b_first)
                    delta -= M[x][b_first]
                    
                    # Change (b_last, y) to (b_last, x)
                    delta -= M[b_last][y]
                    delta += M[b_last][x]
                    
                    # Add (x, y)
                    delta += M[x][y]
                    
                    # (y, next_y) is unchanged.
                    
                    # Perform the move
                    item = order.pop(i)
                    order.insert(j, item)
                    current_score += delta
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_order = order[:]
                    elif delta < 0 and random.random() < pow(2.718281828, -delta / temp):
                        # Accept worse move
                        pass
                    else:
                        # Revert
                        order.pop(j)
                        order.insert(i, item)
                        current_score -= delta
                        
            else:
                # i > j
                # Move i to j.
                # Sequence: ... y, B, x, C ...
                # y at j, x at i.
                # B is j+1 to i-1.
                
                x = order[i]
                y = order[j]
                
                prev_y = order[j-1] if j > 0 else None
                next_x = order[i+1] if i+1 < num_perms else None
                
                delta = 0
                
                if j == i - 1:
                    # Adjacent swap backwards
                    # Same as swap
                    if prev_y is not None:
                        delta -= M[prev_y][y]
                        delta += M[prev_y][x]
                    delta -= M[y][x]
                    delta += M[x][y]
                    if next_x is not None:
                        delta -= M[x][next_x]
                        delta += M[y][next_x]
                else:
                    # B not empty
                    b_first = order[j+1]
                    b_last = order[i-1]
                    
                    # Old:
                    # (prev_y, y)
                    # (y, b_first)
                    # ...
                    # (b_last, x)
                    # (x, next_x)
                    
                    # New:
                    # (prev_y, b_first)
                    # ...
                    # (b_last, y)
                    # (y, x)
                    # (x, next_x)
                    
                    if prev_y is not None:
                        delta -= M[prev_y][y]
                        delta += M[prev_y][b_first]
                    
                    delta -= M[y][b_first]
                    
                    delta -= M[b_last][x]
                    delta += M[b_last][y]
                    
                    delta += M[y][x]
                    
                    # (x, next_x) unchanged.
                    
                    item = order.pop(i)
                    order.insert(j, item)
                    current_score += delta
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_order = order[:]
                    elif delta < 0 and random.random() < pow(2.718281828, -delta / temp):
                        # Accept worse move
                        pass
                    else:
                        # Revert
                        order.pop(j)
                        order.insert(i, item)
                        current_score -= delta

        temp *= cool_rate

    # 4. Construct word from best_order
    word = []
    for idx, p_idx in enumerate(best_order):
        p = perms_6[p_idx]
        block = list(p) + [7] + list(p)
        if idx == 0:
            word = block
        else:
            # Find overlap with current word
            # Suffix of word vs Prefix of block
            max_ov = min(len(word), len(block))
            ov = 0
            for k in range(max_ov, -1, -1):
                if word[-k:] == block[:k]:
                    ov = k
                    break
            word += block[ov:]
            
    return "".join(map(str, word))
