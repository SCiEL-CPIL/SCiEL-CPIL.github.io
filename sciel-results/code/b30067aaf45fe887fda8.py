import time
import sys
import itertools
import random

def construct(n):
    if n < 7:
        # Classic recursive construction is optimal for n <= 6
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
                # Find overlap
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    if n != 7:
        # Fallback for other n
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
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # n == 7 specific search
    start_time = time.time()
    deadline = start_time + 110 # 10s margin for output and safety

    # 1. Generate all permutations
    perms = list(itertools.permutations(range(1, n+1)))
    perm_to_id = {p: i for i, p in enumerate(perms)}
    num_perms = len(perms)
    
    # 2. Precompute cost matrix
    # cost[i][j] = n - overlap(perms[i], perms[j])
    # This is the number of new symbols added when going from i to j.
    # If overlap is k, we add n-k symbols.
    # Note: The first permutation adds n symbols.
    # Total length = n + sum(cost[i][j] for consecutive i, j)
    
    # Optimization: We only need to calculate overlap.
    # Overlap(i, j) is max k such that perms[i][n-k:] == perms[j][:k].
    
    # To speed up, we can use string matching or just simple loops since n=7 is small.
    # 5040 * 5040 = 25,401,600 pairs.
    # In Python, a tight loop might take ~10-20 seconds.
    
    # We will store costs in a flat list or list of lists.
    # Using a list of lists of ints.
    cost = [[0]*num_perms for _ in range(num_perms)]
    
    # Precompute suffixes and prefixes for faster comparison?
    # Or just compare tuples.
    # perms[i][n-k:] is a slice.
    
    # Let's try to optimize the cost calculation.
    # For a fixed i, perms[i] is fixed.
    # For each j, we check k from n-1 down to 0.
    
    # We can precompute for each perm, the set of its suffixes of length k?
    # No, simple loop is fine for n=7.
    
    # Parallelization is not allowed/available easily.
    # Let's just run it.
    
    # If time is tight, we might skip precomputation and calculate on the fly in the search,
    # but that would be slower per swap. Precomputation is O(N^2), search is O(Swaps).
    # Swaps can be millions. Precomputation is better.
    
    # Check time
    if time.time() > deadline - 20:
        # Not enough time for precomputation, return classic
        return _classic_construction(n)

    for i in range(num_perms):
        pi = perms[i]
        for j in range(num_perms):
            pj = perms[j]
            max_k = n - 1
            # Find max overlap
            # Check k from n-1 down to 0
            # pi[n-k:] == pj[:k]
            overlap = 0
            # Optimization: If pi[-1] != pj[0], overlap is 0? No, overlap can be 0.
            # If pi[n-1] == pj[0], check k=1.
            # Actually, we can just loop.
            for k in range(n - 1, -1, -1):
                if pi[n-k:] == pj[:k]:
                    overlap = k
                    break
            cost[i][j] = n - overlap
            
    if time.time() > deadline - 10:
        return _classic_construction(n)

    # 3. Initial Solution: Classic Construction
    # The classic construction can be generated as a sequence of permutations.
    # We need the order of permutations in the classic superpermutation.
    
    # Recursively generate the list of permutations in the classic order.
    # S(n) is formed by S(n-1), then n, then S(n-1) shifted...
    # It's easier to generate the string and extract permutations.
    
    word_str = _classic_construction(n)
    word_list = [int(c) for c in word_str]
    
    # Extract permutations from the classic word
    # The classic word contains every permutation.
    # We need the specific order they appear.
    # Note: Some permutations might appear multiple times?
    # In a superpermutation, every permutation appears at least once.
    # The "classic" construction is designed such that each permutation appears exactly once?
    # No, for n=3, 123121321.
    # 123, 231, 312, 121(no), 213, 132, 321.
    # Wait, 121 is not a perm.
    # The windows that ARE permutations are 123, 231, 312, 213, 132, 321.
    # Do they appear in that order?
    # Indices:
    # 0: 123
    # 1: 231
    # 2: 312
    # 3: 121 (skip)
    # 4: 213
    # 5: 132
    # 6: 321
    # So the order is 123, 231, 312, 213, 132, 321.
    
    # We can extract this order from the classic word.
    classic_order = []
    seen = set()
    for i in range(len(word_list) - n + 1):
        w = tuple(word_list[i:i+n])
        if w not in seen:
            seen.add(w)
            classic_order.append(perm_to_id[w])
            
    # Verify we have all permutations
    if len(classic_order) != num_perms:
        # Fallback
        return word_str

    # 4. Local Search
    # Current path: classic_order
    # Current length: n + sum(cost[path[i]][path[i+1]])
    
    path = classic_order[:]
    
    def get_length(p):
        l = n
        for i in range(len(p) - 1):
            l += cost[p[i]][p[i+1]]
        return l
        
    best_len = get_length(path)
    best_path = path[:]
    
    # Simulated Annealing / Hill Climbing
    # We will try random swaps.
    # To avoid O(N) update for length, we can maintain the length incrementally.
    # But for simplicity and safety, we might just recalculate if N is small enough?
    # N=5040. Recalc is 5039 additions.
    # If we do 10,000 swaps, that's 50 million ops. Doable in Python? Maybe 5-10s.
    # Let's try incremental update.
    
    current_len = best_len
    current_path = best_path[:]
    
    # Precompute neighbors? No, random swaps.
    
    # We need to handle the cost of swapping i and j.
    # Old: ... A, B, C ... and ... X, Y, Z ... (where B is at i, Y is at j)
    # New: ... A, Y, C ... and ... X, B, Z ...
    # Change in length:
    # Remove cost(A,B) + cost(B,C) + cost(X,Y) + cost(Y,Z)
    # Add cost(A,Y) + cost(Y,C) + cost(X,B) + cost(B,Z)
    # Note: If i and j are adjacent or close, terms overlap.
    
    def try_swap(i, j):
        nonlocal current_len, current_path
        if i == j:
            return
        if i > j:
            i, j = j, i
            
        p = current_path
        # Get neighbors
        # Indices in path:
        # i-1, i, i+1
        # j-1, j, j+1
        
        # Handle boundaries
        # If i==0, no left neighbor.
        # If j==N-1, no right neighbor.
        
        # To simplify, we can just recalculate the length if the swap is promising?
        # Or just recalculate fully every time?
        # Let's try full recalculation first to ensure correctness.
        # 5040 length list. Summing 5039 items.
        # In Python, sum(list) is fast (C loop).
        
        # Swap
        p[i], p[j] = p[j], p[i]
        
        new_len = get_length(p)
        
        if new_len < current_len:
            current_len = new_len
            if current_len < best_len:
                best_len = current_len
                best_path = p[:]
        else:
            # Revert
            p[i], p[j] = p[j], p[i]
            
    # Run search
    # Number of iterations depends on time.
    # Let's run until deadline - 5 seconds.
    
    while time.time() < deadline - 5:
        # Random swap
        i = random.randint(0, num_perms - 1)
        j = random.randint(0, num_perms - 1)
        try_swap(i, j)
        
        # Also try reversing a segment?
        # Reversing a segment is more complex to update incrementally.
        # Stick to swaps for now.
        
        # Occasionally, try a random restart?
        # No, we want to improve on the classic.
        
        # If we found 5905, we can stop.
        if best_len <= 5905:
            break

    # Construct final string from best_path
    if best_len < 5913: # Classic length
        # Reconstruct word from path
        # Start with first perm
        word = list(perms[best_path[0]])
        for i in range(1, len(best_path)):
            prev_id = best_path[i-1]
            curr_id = best_path[i]
            # Add new symbols
            # Overlap k
            # We need to find k such that perms[prev_id][n-k:] == perms[curr_id][:k]
            # We can use the cost matrix?
            # cost[prev_id][curr_id] = n - k.
            # So k = n - cost[prev_id][curr_id]
            k = n - cost[prev_id][curr_id]
            # Append last k symbols of curr_id?
            # No, append symbols from index k to end of curr_id?
            # Wait.
            # Word ends with perms[prev_id].
            # We want Word + X = ... perms[prev_id] X ...
            # And the next window is perms[curr_id].
            # perms[curr_id] = perms[prev_id][n-k:] + X?
            # No.
            # Overlap means:
            # Suffix of prev (len k) == Prefix of curr (len k).
            # So the new symbols to append are curr_id[k:].
            word.extend(perms[curr_id][k:])
        
        return "".join(map(str, word))
    else:
        return _classic_construction(n)

def _classic_construction(n):
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
            # Find overlap
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
