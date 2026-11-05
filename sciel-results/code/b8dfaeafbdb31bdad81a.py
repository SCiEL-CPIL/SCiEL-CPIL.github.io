import sys
import time
import random
from itertools import permutations

_DEADLINE = 145.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def construct(n):
    global _start_time
    _start_time = time.time()
    
    if n <= 1:
        return "1" if n == 1 else ""
    
    # Base case for n=2
    if n == 2:
        return "121"
    
    # For n < 7, the classic recursive construction is optimal or very close.
    # We can just use the classic method for n < 7 to be safe and fast.
    if n < 7:
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
                # Calculate overlap
                ov = 0
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        ov = o
                        break
                new += block[ov:]
            word = new
        return "".join(map(str, word))

    # Special optimization for n=7
    # The blocks are formed by permutations of 1..6, with 7 in the middle.
    # Block for perm P is P + [7] + P.
    # We need to order the 720 permutations of 1..6 to maximize total overlap.
    
    prev_syms = list(range(1, 7))
    perms = list(permutations(prev_syms))
    num_perms = len(perms) # 720
    
    # Map permutation tuple to index for faster lookup if needed, 
    # but we just need the overlap matrix M[i][j].
    
    # Precompute overlap matrix
    # M[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    M = [[0] * num_perms for _ in range(num_perms)]
    
    # Optimization: Group permutations by their prefix/suffix?
    # Or just brute force since 720 is small.
    # 720 * 720 * 6 checks is ~3 million operations.
    
    for i in range(num_perms):
        pi = perms[i]
        for j in range(num_perms):
            if i == j:
                continue
            pj = perms[j]
            # Max overlap is 6 (if pi == pj, but we skip i==j)
            # Actually, if pi != pj, max overlap is 5?
            # If pi[-5:] == pj[:5], and pi[0] != pj[5], then overlap is 5.
            # If pi[-6:] == pj[:6], then pi == pj.
            # So max overlap for i != j is 5.
            # Wait, is it possible to have overlap 6? No, because then the permutations would be identical.
            # So we check k from 5 down to 0.
            for k in range(5, -1, -1):
                if k == 0 or pi[-k:] == pj[:k]:
                    M[i][j] = k
                    break

    # Initialize order with the "classic" order.
    # We can generate the classic order for n=6 to get the sequence of perms.
    def get_classic_order(n):
        word = [1]
        for m in range(2, n + 1):
            perms_local, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_local.append(window)
            new = []
            for p in perms_local:
                block = list(p) + [m] + list(p)
                ov = 0
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        ov = o
                        break
                new += block[ov:]
            word = new
        # Extract order of perms of size n
        # The perms of size n are the windows of size n in the final word?
        # No, the construction for n uses perms of n-1.
        # To get the order of perms of size n-1 used in step n, we look at the 
        # sequence of blocks added.
        # Actually, simpler: just run the construction for n=6 and extract the 
        # sequence of 6-perms in the order they were used as 'p' in the step m=7?
        # No, the step m=7 uses perms of 1..6.
        # The classic construction for n=7 uses the perms of 1..6 in the order 
        # they appear in the superpermutation of 1..6?
        # Let's trace:
        # S_2 = 121. Perms of 1: [1].
        # S_3: blocks from perms of 1: [1,2,1]. Order: 1.
        #      S_3 = 121 3 121 = 1213121.
        #      Perms of 2 in S_3: 12, 21.
        # S_4: blocks from perms of 2 in order [12, 21].
        #      Block 12: 12412.
        #      Block 21: 21421.
        #      S_4 = 1213121 4 12412 4 21421 ...
        # The order of perms of n-1 is the order in which they appear as 
        # "unique windows" in S_{n-1}?
        # Actually, the standard algorithm iterates through the windows of S_{n-1} 
        # and picks the first occurrence of each permutation.
        
        # Let's just extract the order of 6-perms from the classic S_6.
        # S_6 is built from perms of 5.
        # We want the order of perms of 6 that would be used to build S_7 
        # in the classic way.
        # This is exactly the order in which 6-perms appear in S_6?
        # No. The classic S_7 is built by taking the sequence of 6-perms 
        # from S_6 (in order of appearance) and wrapping them with 7.
        
        # So we need to generate S_6 and extract the 6-perms in order.
        
        word_6 = [1]
        for m in range(2, 7):
            perms_step, seen = [], set()
            for i in range(len(word_6) - m + 2):
                window = tuple(word_6[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_step.append(window)
            new = []
            for p in perms_step:
                block = list(p) + [m] + list(p)
                ov = 0
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        ov = o
                        break
                new += block[ov:]
            word_6 = new
            
        # Extract 6-perms from word_6
        order = []
        seen = set()
        for i in range(len(word_6) - 6 + 1):
            window = tuple(word_6[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                # Map window to index in perms
                # We need to find the index of this perm in the 'perms' list
                # This is O(N) if we do it naively. 720 * 720 is fine.
                # Or use a dict.
                order.append(window)
        
        # Map to indices
        perm_to_idx = {p: i for i, p in enumerate(perms)}
        return [perm_to_idx[p] for p in order]

    initial_order = get_classic_order(6)
    
    # Calculate initial score
    def calc_score(order):
        s = 0
        for i in range(len(order) - 1):
            s += M[order[i]][order[i+1]]
        return s

    current_score = calc_score(initial_order)
    best_order = initial_order[:]
    best_score = current_score
    
    random.seed(42)
    
    # Simulated Annealing
    T = 10.0
    cooling_rate = 0.9995
    
    # We can also try 2-opt moves (reverse segment)
    # Move types:
    # 1. Swap adjacent
    # 2. Reverse segment of length 2 (same as swap)
    # 3. Reverse segment of length 3, 4, 5
    
    while time.time() < _DEADLINE - 5:
        # Choose move type
        # Mostly swaps, some reversals
        if random.random() < 0.8:
            # Adjacent swap
            i = random.randint(0, num_perms - 2)
            j = i + 1
            # Calculate delta
            # Affected edges: (i-1, i), (i, i+1), (i+1, i+2)
            old_edges = 0
            new_edges = 0
            
            if i > 0:
                old_edges += M[order[i-1]][order[i]]
                new_edges += M[order[i-1]][order[i+1]]
            old_edges += M[order[i]][order[i+1]]
            new_edges += M[order[i+1]][order[i]]
            if i + 2 < num_perms:
                old_edges += M[order[i+1]][order[i+2]]
                new_edges += M[order[i]][order[i+2]]
                
            delta = new_edges - old_edges
            
            if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                order[i], order[i+1] = order[i+1], order[i]
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_order = order[:]
        else:
            # Reverse segment
            length = random.randint(2, 5)
            i = random.randint(0, num_perms - length)
            j = i + length - 1
            
            # Calculate delta for reversal
            # Edges affected:
            # (i-1, i) -> (i-1, j)
            # (i, i+1) ... (j-1, j) -> internal edges reversed
            # (j, j+1) -> (i, j+1)
            
            old_edges = 0
            new_edges = 0
            
            if i > 0:
                old_edges += M[order[i-1]][order[i]]
                new_edges += M[order[i-1]][order[j]]
            
            # Internal edges
            for k in range(i, j):
                old_edges += M[order[k]][order[k+1]]
                # New order: order[j], order[j-1], ..., order[i]
                # Edge from order[k+1] to order[k] in new order?
                # The segment is order[i..j]. Reversed: order[j], order[j-1], ..., order[i].
                # Edge between index k and k+1 in old segment becomes edge between 
                # original index k+1 and k in new segment.
                new_edges += M[order[k+1]][order[k]]
            
            if j + 1 < num_perms:
                old_edges += M[order[j]][order[j+1]]
                new_edges += M[order[i]][order[j+1]]
                
            delta = new_edges - old_edges
            
            if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                order[i:j+1] = reversed(order[i:j+1])
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_order = order[:]
        
        T *= cooling_rate
        if T < 0.01:
            break
    
    # Construct final word
    word = []
    for idx, p_idx in enumerate(best_order):
        p = perms[p_idx]
        block = list(p) + [7] + list(p)
        if idx == 0:
            word = block
        else:
            ov = 0
            max_ov = min(len(word), len(block))
            # We know the overlap is M[best_order[idx-1]][best_order[idx]]
            # But let's verify or just use it to slice.
            # Using M is faster.
            ov = M[best_order[idx-1]][best_order[idx]]
            word += block[ov:]
    
    return "".join(map(str, word))
