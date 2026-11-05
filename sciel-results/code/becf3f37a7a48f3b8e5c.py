import time
import random
import sys

def construct(n):
    # Deadline guard: 120s limit, use 110s for safety
    start_time = time.time()
    deadline = start_time + 110.0

    # For n < 7, the standard construction is optimal or close enough,
    # and local search overhead might not be worth it or might be complex.
    # The task specifically mentions n=7 in the context of "shortest known".
    # We will apply the optimization only for n=7 to be safe and fast for others.
    if n != 7:
        return _standard_construct(n)

    # 1. Construct SP(6) and extract permutations in order
    word_6 = _standard_construct(6)
    w6_ints = [int(c) for c in word_6]
    
    perms_6, seen = [], set()
    for i in range(len(w6_ints) - 6 + 1):
        window = tuple(w6_ints[i:i + 6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            perms_6.append(window)
            
    num_perms = len(perms_6) # Should be 720
    
    # 2. Precompute overlap matrix M
    # M[i][j] = overlap between perms_6[i] and perms_6[j]
    M = [[0] * num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        p1 = perms_6[i]
        for j in range(num_perms):
            p2 = perms_6[j]
            # Max overlap is 6
            # Check k from 6 down to 0
            for k in range(6, -1, -1):
                if k == 0:
                    M[i][j] = 0
                    break
                if p1[-k:] == p2[:k]:
                    M[i][j] = k
                    break

    # 3. Local Search to maximize sum of overlaps
    # Initial order: natural order of appearance
    order = list(range(num_perms))
    
    def calc_score(ord):
        score = 0
        for i in range(len(ord) - 1):
            score += M[ord[i]][ord[i+1]]
        return score

    best_order = order[:]
    best_score = calc_score(order)
    
    random.seed(42)
    
    current_order = order[:]
    current_score = best_score
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Perturb: random restart or kick
            # Let's try a random 2-opt with large segment as a kick
            i = random.randint(0, num_perms - 100)
            j = random.randint(i + 100, num_perms - 1)
            current_order[i:j+1] = reversed(current_order[i:j+1])
            current_score = calc_score(current_order)
            no_improve_count = 0
            max_no_improve = int(max_no_improve * 1.2)
            continue

        # Pick a random move type
        # 70% 2-opt, 30% Or-Opt
        if random.random() < 0.7:
            # 2-opt
            i = random.randint(0, num_perms - 2)
            j = random.randint(i + 1, num_perms - 1)
            
            # Calculate delta
            old_links_score = 0
            if i > 0:
                old_links_score += M[current_order[i-1]][current_order[i]]
            if j < num_perms - 1:
                old_links_score += M[current_order[j]][current_order[j+1]]
            for k in range(i, j):
                old_links_score += M[current_order[k]][current_order[k+1]]
                
            new_links_score = 0
            if i > 0:
                new_links_score += M[current_order[i-1]][current_order[j]]
            if j < num_perms - 1:
                new_links_score += M[current_order[i]][current_order[j+1]]
            for k in range(j, i, -1):
                new_links_score += M[current_order[k]][current_order[k-1]]
                
            delta = new_links_score - old_links_score
            
            if delta > 0:
                current_order[i:j+1] = reversed(current_order[i:j+1])
                current_score += delta
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_order = current_order[:]
            else:
                no_improve_count += 1
        else:
            # Or-Opt
            k = random.choice([1, 2, 3])
            i = random.randint(0, num_perms - k)
            j = random.randint(0, num_perms - k)
            
            if i == j:
                continue
                
            # Segment S is current_order[i : i+k]
            # If j < i: move S to j
            # If j > i+k: move S to j
            
            if j < i:
                # Remove from i, insert at j
                # Edges removed: (i-1, i), (i+k-1, i+k)
                # Edges added: (j-1, i), (i+k-1, j)
                old_left = M[current_order[i-1]][current_order[i]] if i > 0 else 0
                old_right = M[current_order[i+k-1]][current_order[i+k]] if i+k < num_perms else 0
                
                new_left = M[current_order[j-1]][current_order[i]] if j > 0 else 0
                new_right = M[current_order[i+k-1]][current_order[j]] if j < num_perms else 0
                
                delta = new_left + new_right - old_left - old_right
                
                if delta > 0:
                    # Perform move
                    seg = current_order[i:i+k]
                    del current_order[i:i+k]
                    current_order[j:j] = seg
                    current_score += delta
                    no_improve_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_order = current_order[:]
                else:
                    no_improve_count += 1
                    
            elif j > i + k:
                # Remove from i, insert at j
                # Edges removed: (i-1, i), (i+k-1, i+k), (j-1, j)
                # Edges added: (i-1, i+k), (j-1, i), (i+k-1, j)
                old_left = M[current_order[i-1]][current_order[i]] if i > 0 else 0
                old_mid = M[current_order[i+k-1]][current_order[i+k]] if i+k < num_perms else 0
                old_right = M[current_order[j-1]][current_order[j]] if j > 0 else 0
                
                new_left = M[current_order[i-1]][current_order[i+k]] if i > 0 else 0
                new_mid_left = M[current_order[j-1]][current_order[i]] if j > 0 else 0
                new_mid_right = M[current_order[i+k-1]][current_order[j]] if j < num_perms else 0
                
                delta = new_left + new_mid_left + new_mid_right - old_left - old_mid - old_right
                
                if delta > 0:
                    # Perform move
                    seg = current_order[i:i+k]
                    del current_order[i:i+k]
                    # Adjust j if j > i+k? 
                    # If we delete i..i+k-1, the index j (which was > i+k) becomes j-k.
                    # But we are inserting at the position that was j.
                    # In the new list, the element that was at j is now at j-k.
                    # We want to insert before the element that was at j.
                    # So we insert at index j-k.
                    insert_idx = j - k
                    current_order[insert_idx:insert_idx] = seg
                    current_score += delta
                    no_improve_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_order = current_order[:]
                else:
                    no_improve_count += 1
            else:
                # j is between i and i+k? Overlapping move, skip or handle carefully.
                # For simplicity, skip if j is in [i, i+k)
                continue

    # 4. Construct word from best_order
    word = []
    for idx, p_idx in enumerate(best_order):
        p = perms_6[p_idx]
        block = list(p) + [7] + list(p)
        if idx == 0:
            word = block
        else:
            ov = M[best_order[idx-1]][best_order[idx]]
            word += block[ov:]
            
    return "".join(map(str, word))

def _standard_construct(n):
    """Classic recursive construction: 1! + 2! + ... + n!"""
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
