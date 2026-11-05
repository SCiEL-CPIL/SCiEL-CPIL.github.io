import time
import random
import math
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if n != 7:
        return _standard_construct(n)
        
    # 1. Generate permutations of 1..6
    perms = list(itertools.permutations(range(1, 7)))
    num_perms = len(perms) # 720
    
    # 2. Compute Overlap Matrix
    # M[i][j] = overlap between perms[i] and perms[j]
    # We store this as a list of lists of small ints
    M = [[0]*num_perms for _ in range(num_perms)]
    
    # Precompute suffixes/prefixes? 
    # Overlap k means perms[i][-k:] == perms[j][:k]
    # Since perms are permutations, we can check k=5, 4, 3, 2, 1, 0
    # k=5: last 5 of i == first 5 of j
    # This implies the set of elements in i[-5:] is {1..6} \ {i[0]}
    # and j[:5] is {1..6} \ {j[5]}
    # So i[0] must equal j[5] for k=5 to be possible?
    # No, i[-5:] is i[1:]. j[:5] is j[0:5].
    # If i[1:] == j[0:5], then i[1]=j[0], i[2]=j[1], etc.
    # This is a strong condition.
    
    for i in range(num_perms):
        p1 = perms[i]
        for j in range(num_perms):
            p2 = perms[j]
            # Check k from 5 down to 0
            # Max overlap is 5 (since 7 is in the middle, and blocks are P 7 P)
            # Wait, block is P 7 P. Length 13.
            # Overlap between Block A and Block B.
            # Suffix of A: ... P1[-k:]
            # Prefix of B: P2[:k] ...
            # If k <= 6, it's just suffix of P1 and prefix of P2.
            # If k > 6, it involves the 7.
            # Max overlap between two blocks P1 7 P1 and P2 7 P2.
            # Can overlap be 6?
            # P1[-6:] = P1. P2[:6] = P2.
            # P1 == P2? If P1==P2, overlap is 6?
            # Block: 1234567123456
            # Next: 1234567123456
            # Suffix 123456, Prefix 123456. Match.
            # But we cannot repeat permutations in a superpermutation?
            # The problem asks for a word containing *every* permutation.
            # It does not say we cannot repeat them, but to be *shortest*, we shouldn't repeat.
            # The recursive construction uses each permutation of 1..6 exactly once as a "block center".
            # So we are ordering the 720 unique permutations.
            # Thus P1 != P2 for i != j.
            # If P1 != P2, max overlap is 5.
            
            ov = 0
            if p1[1:] == p2[:5]: ov = 5
            elif p1[2:] == p2[:4]: ov = 4
            elif p1[3:] == p2[:3]: ov = 3
            elif p1[4:] == p2[:2]: ov = 2
            elif p1[5] == p2[0]: ov = 1
            M[i][j] = ov

    # 3. Greedy Initialization
    order = [0]
    used = [False]*num_perms
    used[0] = True
    for _ in range(num_perms - 1):
        last = order[-1]
        best_next = -1
        best_ov = -1
        for j in range(num_perms):
            if not used[j]:
                ov = M[last][j]
                if ov > best_ov:
                    best_ov = ov
                    best_next = j
        order.append(best_next)
        used[best_next] = True
        
    # Calculate initial score
    def calc_score(ord_list):
        s = 0
        for i in range(len(ord_list) - 1):
            s += M[ord_list[i]][ord_list[i+1]]
        return s
        
    current_score = calc_score(order)
    best_order = order[:]
    best_score = current_score
    
    # 4. Simulated Annealing
    T = 1.0
    cooling_rate = 0.99
    min_T = 1e-4
    random.seed(42)
    
    while T > min_T and time.time() < deadline:
        # Choose move: 2-opt or Or-opt
        if random.random() < 0.5:
            # 2-opt: reverse segment
            i = random.randint(0, num_perms - 2)
            j = random.randint(i + 1, num_perms - 1)
            
            # Calculate delta
            old_score = 0
            new_score = 0
            
            if i > 0:
                old_score += M[order[i-1]][order[i]]
                new_score += M[order[i-1]][order[j]]
            else:
                # No left neighbor
                pass
                
            if j < num_perms - 1:
                old_score += M[order[j]][order[j+1]]
                new_score += M[order[i]][order[j+1]]
            else:
                pass
                
            # Internal links
            # Old: (i, i+1), ..., (j-1, j)
            # New: (j, j-1), ..., (i+1, i)
            for k in range(i, j):
                old_score += M[order[k]][order[k+1]]
            for k in range(j, i, -1):
                new_score += M[order[k]][order[k-1]]
                
            delta = new_score - old_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                order[i:j+1] = reversed(order[i:j+1])
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_order = order[:]
        else:
            # Or-opt: move a segment of length 1, 2, or 3
            seg_len = random.randint(1, 3)
            i = random.randint(0, num_perms - seg_len)
            j = random.randint(0, num_perms - seg_len)
            
            if i == j:
                continue
            
            # Extract segment
            seg = order[i:i+seg_len]
            
            # Remove segment
            # This is expensive in Python lists. 
            # For N=720, it's okay.
            
            # Calculate delta is complex with removal/insertion.
            # Let's stick to 2-opt for simplicity and speed, or implement Or-opt carefully.
            # Given time constraints, 2-opt with SA is robust.
            # Let's just use 2-opt but with a better cooling schedule.
            pass

        T *= cooling_rate
        
    # 5. Construct Word
    word = []
    for idx, p_idx in enumerate(best_order):
        p = perms[p_idx]
        block = list(p) + [7] + list(p)
        if idx == 0:
            word = block
        else:
            ov = M[best_order[idx-1]][best_order[idx]]
            word += block[ov:]
            
    return "".join(map(str, word))

def _standard_construct(n):
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
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
