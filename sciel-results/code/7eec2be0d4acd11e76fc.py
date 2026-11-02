import time
import sys
from itertools import permutations

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412431324142314321234124313241423143212341243132414231432"
    if n == 5:
        # Known optimal length is 153
        return _classic_construct(5)
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    def _classic_construct(m):
        """Classic recursive construction of length 1! + 2! + ... + m!"""
        word = [1]
        for k in range(2, m + 1):
            perms, seen = [], set()
            for i in range(len(word) - k + 2):
                window = tuple(word[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [k] + list(p)
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if o == 0:
                        overlap = 0
                        break
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    if n <= 6:
        return _classic_construct(n)
    
    # For n >= 7, use the ordering-based approach
    # Get the 720 permutations of 1..6 in their natural order from classic construction
    def get_perms_order(m):
        """Get permutations of 1..m in order of appearance in classic construction"""
        word = [1]
        for k in range(2, m + 1):
            perms, seen = [], set()
            for i in range(len(word) - k + 2):
                window = tuple(word[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [k] + list(p)
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if o == 0:
                        overlap = 0
                        break
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        # Extract all m-length windows
        result = []
        seen = set()
        for i in range(len(word) - m + 1):
            w = tuple(word[i:i + m])
            if len(set(w)) == m and w not in seen:
                seen.add(w)
                result.append(w)
        return result
    
    perms = get_perms_order(n - 1)  # Permutations of 1..6
    num_perms = len(perms)
    
    if num_perms != 720:
        # Fallback to classic
        return _classic_construct(n)
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    max_k = n - 2  # 5 for n=7
    overlap = [[0] * num_perms for _ in range(num_perms)]
    
    for i in range(num_perms):
        pi = perms[i]
        for j in range(num_perms):
            pj = perms[j]
            found = 0
            for k in range(max_k, 0, -1):
                if pi[-k:] == pj[:k]:
                    found = k
                    break
            overlap[i][j] = found
    
    # Block length: each block is P, n, P where P is a perm of 1..(n-1)
    # Block length = 2*(n-1) + 1 = 13 for n=7
    block_len = 2 * (n - 1) + 1
    
    # Initial order: natural order
    order = list(range(num_perms))
    
    def calc_length(ord_list):
        """Calculate total word length given an ordering of permutation indices"""
        total = block_len
        for i in range(len(ord_list) - 1):
            u = ord_list[i]
            v = ord_list[i + 1]
            total += block_len - overlap[u][v]
        return total
    
    best_length = calc_length(order)
    best_order = order[:]
    
    # Local search: 2-opt (reverse segments)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(num_perms - 1):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, num_perms):
                if time.time() > deadline - 5:
                    break
                
                # Calculate delta for reversing order[i..j]
                u_left = order[i - 1] if i > 0 else None
                u_right = order[j + 1] if j < num_perms - 1 else None
                
                # Old cost
                old_cost = 0
                if i > 0:
                    old_cost += overlap[u_left][order[i]]
                for k in range(i, j):
                    old_cost += overlap[order[k]][order[k + 1]]
                if j < num_perms - 1:
                    old_cost += overlap[order[j]][u_right]
                
                # New cost
                new_cost = 0
                if i > 0:
                    new_cost += overlap[u_left][order[j]]
                for k in range(j, i, -1):
                    new_cost += overlap[order[k]][order[k - 1]]
                if j < num_perms - 1:
                    new_cost += overlap[order[i]][u_right]
                
                delta = new_cost - old_cost
                if delta < 0:
                    # Accept and apply
                    order[i:j + 1] = reversed(order[i:j + 1])
                    best_length += delta
                    best_order = order[:]
                    improved = True
    
    # Try some random restarts with perturbation if time allows
    if time.time() < deadline - 10:
        import random
        rng = random.Random(42)
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            
            # Perturb: reverse a random segment
            i = rng.randint(0, num_perms - 2)
            j = rng.randint(i + 1, num_perms - 1)
            order = best_order[:]
            order[i:j + 1] = reversed(order[i:j + 1])
            
            # Local search from this perturbed state
            cur_len = calc_length(order)
            cur_improved = True
            while cur_improved and time.time() < deadline - 5:
                cur_improved = False
                for i2 in range(num_perms - 1):
                    if time.time() > deadline - 5:
                        break
                    for j2 in range(i2 + 1, num_perms):
                        if time.time() > deadline - 5:
                            break
                        
                        u_left = order[i2 - 1] if i2 > 0 else None
                        u_right = order[j2 + 1] if j2 < num_perms - 1 else None
                        
                        old_cost = 0
                        if i2 > 0:
                            old_cost += overlap[u_left][order[i2]]
                        for k in range(i2, j2):
                            old_cost += overlap[order[k]][order[k + 1]]
                        if j2 < num_perms - 1:
                            old_cost += overlap[order[j2]][u_right]
                        
                        new_cost = 0
                        if i2 > 0:
                            new_cost += overlap[u_left][order[j2]]
                        for k in range(j2, i2, -1):
                            new_cost += overlap[order[k]][order[k - 1]]
                        if j2 < num_perms - 1:
                            new_cost += overlap[order[i2]][u_right]
                        
                        delta = new_cost - old_cost
                        if delta < 0:
                            order[i2:j2 + 1] = reversed(order[i2:j2 + 1])
                            cur_len += delta
                            cur_improved = True
            
            if cur_len < best_length:
                best_length = cur_len
                best_order = order[:]
    
    # Build the final word from best_order
    word = []
    # First block
    p0 = perms[best_order[0]]
    word.extend(list(p0) + [n] + list(p0))
    
    for idx in range(1, len(best_order)):
        pi = perms[best_order[idx - 1]]
        pj = perms[best_order[idx]]
        block = list(pj) + [n] + list(pj)
        
        # Find max overlap between end of word and start of block
        max_overlap = min(len(word), len(block))
        overlap_amt = 0
        for o in range(max_overlap, 0, -1):
            if word[len(word) - o:] == block[:o]:
                overlap_amt = o
                break
        word.extend(block[overlap_amt:])
    
    return "".join(map(str, word))
