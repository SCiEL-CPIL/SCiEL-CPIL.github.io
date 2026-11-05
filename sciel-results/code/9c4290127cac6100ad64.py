"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import math

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
    
    # 2. Precompute overlap matrix efficiently
    # M[i][j] = longest k such that suffix of perms_6[i] of length k == prefix of perms_6[j] of length k
    # Use hashing for speed: precompute prefix hashes and suffix hashes
    # For each permutation, compute rolling hash of prefixes and suffixes
    
    # Actually, let's just do it directly but more carefully.
    # For each pair (i,j), the max overlap is at most 5 (since if overlap=6, i=j)
    # We check k=5,4,3,2,1,0
    
    M = [[0]*num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        p1 = perms_6[i]
        # Precompute suffixes of p1 as tuples for quick comparison
        for j in range(num_perms):
            p2 = perms_6[j]
            # Check from k=5 down to 0
            # k=5: p1[1:6] == p2[0:5]
            if p1[1:6] == p2[0:5]:
                M[i][j] = 5
            elif p1[2:6] == p2[0:4]:
                M[i][j] = 4
            elif p1[3:6] == p2[0:3]:
                M[i][j] = 3
            elif p1[4:6] == p2[0:2]:
                M[i][j] = 2
            elif p1[5:6] == p2[0:1]:
                M[i][j] = 1
            else:
                M[i][j] = 0

    # 3. Simulated Annealing with multiple move types
    random.seed(42)
    
    def calc_score(ord):
        score = 0
        for i in range(len(ord) - 1):
            score += M[ord[i]][ord[i+1]]
        return score
    
    # Initial order
    current_order = list(range(num_perms))
    current_score = calc_score(current_order)
    best_order = current_order[:]
    best_score = current_score
    
    # SA parameters
    T = 5.0  # Initial temperature
    T_min = 0.01
    cooling = 0.9999  # Geometric cooling
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Pick move type
        r = random.random()
        if r < 0.4:
            # Adjacent swap
            i = random.randint(0, num_perms - 2)
            a, b = current_order[i], current_order[i+1]
            
            old_part = M[a][b]
            new_part = M[b][a]
            
            if i > 0:
                old_part += M[current_order[i-1]][a]
                new_part += M[current_order[i-1]][b]
            if i+2 < num_perms:
                old_part += M[b][current_order[i+2]]
                new_part += M[a][current_order[i+2]]
            
            delta = new_part - old_part
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_order[i], current_order[i+1] = b, a
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_order = current_order[:]
        
        elif r < 0.7:
            # 2-opt: reverse a segment [i, j]
            i = random.randint(0, num_perms - 2)
            j = random.randint(i+1, num_perms - 1)
            seg_len = j - i + 1
            if seg_len > 30:  # Limit segment length for efficiency
                continue
            
            # Compute delta for reversing segment [i, j]
            # Before: ... X, A, B, ..., C, Y ...
            # After:  ... X, C, ..., B, A, Y ...
            # Edges affected: X-A, A-B, B-C, ..., C-Y
            # New edges: X-C, C-..., ..., B-A, A-Y
            
            x_node = current_order[i-1] if i > 0 else None
            y_node = current_order[j+1] if j+1 < num_perms else None
            a = current_order[i]
            c = current_order[j]
            
            old_edges = 0
            new_edges = 0
            
            # Edge into segment
            if x_node is not None:
                old_edges += M[x_node][a]
                new_edges += M[x_node][c]
            
            # Edges within segment (reversed)
            for k in range(i, j):
                old_edges += M[current_order[k]][current_order[k+1]]
                new_edges += M[current_order[k+1]][current_order[k]]
            
            # Edge out of segment
            if y_node is not None:
                old_edges += M[c][y_node]
                new_edges += M[a][y_node]
            
            delta = new_edges - old_edges
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                # Perform reversal
                current_order[i:j+1] = current_order[i:j+1][::-1]
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_order = current_order[:]
        
        else:
            # Relocate: move element i to position j
            i = random.randint(0, num_perms - 1)
            j = random.randint(0, num_perms - 1)
            if i == j:
                continue
            
            x = current_order[i]
            
            # Remove x from position i
            # Edges affected: (i-1, i), (i, i+1)
            old_edges = 0
            new_edges = 0
            
            if i > 0:
                old_edges += M[current_order[i-1]][x]
            if i < num_perms - 1:
                old_edges += M[x][current_order[i+1]]
            if i > 0 and i < num_perms - 1:
                new_edges += M[current_order[i-1]][current_order[i+1]]
            
            # Determine insertion position in the modified list
            # After removal, the list has num_perms-1 elements
            # If j > i, insert at j-1 in the new list (since one element was removed before j)
            # If j <= i, insert at j in the new list
            
            # For simplicity, just do full recalc if the move looks potentially good
            # But that's expensive. Let's do it properly.
            
            # Simpler approach: just do the move and recalc the affected edges
            # This is complex, so let's skip relocate for now and rely on swap + 2-opt
            
            pass
        
        # Cool down
        T = max(T_min, T * cooling)
        
        # Periodic restart: if stuck, perturb
        if step % 50000 == 0 and T > T_min:
            # Perturb current solution
            for _ in range(10):
                i = random.randint(0, num_perms - 2)
                current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
            current_score = calc_score(current_order)
    
    # 4. Construct word from best_order
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
