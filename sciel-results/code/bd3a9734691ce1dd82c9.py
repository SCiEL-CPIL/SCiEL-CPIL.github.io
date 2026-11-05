"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit (kill at 150s, but be safe)

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
    # We need to construct SP(6) first to get the specific order of perms
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
                    
    # 3. Local Search
    # Initial order: the order in which they appear in SP(6)
    order = list(range(num_perms))
    
    # Calculate initial score (sum of overlaps)
    # Total length = sum(len(block)) - sum(overlaps)
    # len(block) is constant (13) for all. So minimizing length == maximizing sum(overlaps)
    
    def calc_score(ord):
        score = 0
        for i in range(len(ord) - 1):
            score += M[ord[i]][ord[i+1]]
        return score
        
    best_order = order[:]
    best_score = calc_score(order)
    current_order = order[:]
    current_score = best_score
    
    random.seed(42)
    
    # Simple Hill Climbing / Simulated Annealing lite
    # We can try swaps and moves
    # 720 elements.
    
    while time.time() < deadline:
        # Pick a random move
        move_type = random.random()
        if move_type < 0.5:
            # Swap adjacent
            i = random.randint(0, num_perms - 2)
            # Calculate delta for swap i, i+1
            # Old: ... A, B, C ...
            # New: ... B, A, C ...
            # Delta = (M[B][A] + M[A][C]) - (M[A][B] + M[B][C])
            # Need to handle boundaries
            old_score_part = 0
            new_score_part = 0
            
            if i > 0:
                old_score_part += M[current_order[i-1]][current_order[i]]
                new_score_part += M[current_order[i-1]][current_order[i+1]]
            
            old_score_part += M[current_order[i]][current_order[i+1]]
            new_score_part += M[current_order[i+1]][current_order[i]]
            
            if i+2 < num_perms:
                old_score_part += M[current_order[i+1]][current_order[i+2]]
                new_score_part += M[current_order[i]][current_order[i+2]]
                
            delta = new_score_part - old_score_part
            
            if delta >= 0:
                # Perform swap
                current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_order = current_order[:]
            else:
                # Accept with some probability? Or just reject?
                # For simplicity, reject negative delta in hill climbing.
                # But maybe we should allow some exploration.
                # Let's stick to strict hill climbing first to see if it helps.
                pass
        else:
            # Move element i to position j
            i = random.randint(0, num_perms - 1)
            j = random.randint(0, num_perms - 1)
            if i == j: continue
            
            # Calculate delta for move
            # This is more complex.
            # Element x = current_order[i]
            # Remove x from i, insert at j.
            
            # For efficiency, we can just recalc full score if the move is potentially good?
            # No, 720 iterations is fast.
            # Let's just do a full recalc for moves to be safe, since swaps are frequent.
            # Actually, let's just do swaps for now. Moves are harder to delta-calc.
            # Or we can just do a full recalc for moves.
            
            new_order = current_order[:]
            x = new_order.pop(i)
            new_order.insert(j, x)
            
            new_score = calc_score(new_order)
            
            if new_score > current_score:
                current_order = new_order
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_order = current_order[:]
            # Else reject (or accept with prob for SA, but let's keep it simple)

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
