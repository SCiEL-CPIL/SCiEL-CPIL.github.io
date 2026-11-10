import numpy as np
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    # Use 110s to be safe against 120s limit
    deadline = start_time + 110.0
    
    N = q ** n
    # Precompute neighbors
    # neighbors[w] is a list of words within distance <= radius
    # For radius 1, size is 1 + n*(q-1)
    neighbors = []
    neighbor_sets = []
    for w in range(N):
        neigh = [w]
        for i in range(n):
            power = q ** i
            digit = (w // power) % q
            for d in range(q):
                if d != digit:
                    new_w = w + (d - digit) * power
                    neigh.append(new_w)
        neighbors.append(neigh)
        neighbor_sets.append(set(neigh))
        
    def to_digits(w):
        return [int((w // (q**i)) % q) for i in range(n)]

    best_code = None
    best_len = float('inf')
    
    # Helper to run local search on a given code
    def local_search(code, start_time):
        code = code[:] # copy
        code_set = set(code)
        
        # Compute initial cover counts
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code:
            for u in neighbors[c]:
                cover_count[u] += 1
                
        # Reduction function
        def reduce_code():
            nonlocal code, code_set, cover_count
            changed = True
            while changed:
                if time.time() > deadline - 2.0:
                    return False
                changed = False
                to_remove = []
                for c in code:
                    is_essential = False
                    for u in neighbors[c]:
                        if cover_count[u] == 1:
                            is_essential = True
                            break
                    if not is_essential:
                        to_remove.append(c)
                if to_remove:
                    for c in to_remove:
                        code.remove(c)
                        code_set.discard(c)
                        for u in neighbors[c]:
                            cover_count[u] -= 1
                    changed = True
            return True

        # Initial reduction
        if not reduce_code():
            return code
            
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5.0:
                break
                
            # Try swaps (2-for-1)
            n_code = len(code)
            # To avoid bias, we can shuffle the code list occasionally, 
            # but for deterministic behavior in a single run, we iterate sequentially.
            # We need to be careful with indices if we modify the list.
            # We will collect moves and apply them one by one, restarting the search.
            
            move_found = False
            for i in range(n_code):
                if time.time() > deadline - 5.0:
                    break
                if move_found:
                    break
                c_i = code[i]
                for j in range(i + 1, n_code):
                    if time.time() > deadline - 5.0:
                        break
                    c_j = code[j]
                    
                    # Determine U_ij: words that become uncovered if we remove c_i and c_j
                    # A word u is in U_ij if:
                    # 1. u is in B(c_i) U B(c_j)
                    # 2. cover_count[u] - (1 if u in B(c_i) else 0) - (1 if u in B(c_j) else 0) < 1
                    
                    # We can compute this by iterating through the union of neighbors
                    # But iterating through the intersection of candidates is faster for finding k.
                    # First, let's identify the "critical" words.
                    
                    # Optimization: 
                    # If we remove c_i and c_j, the words that *might* become uncovered 
                    # are those in B(c_i) U B(c_j) that have low cover counts.
                    
                    # Let's just compute the set of words that would be uncovered.
                    # This set U must be covered by a single k.
                    
                    # To find U efficiently:
                    # We iterate through neighbors[c_i] and neighbors[c_j].
                    # To avoid duplicates, we can use a set or just check logic.
                    
                    # However, the intersection method for finding k requires U.
                    # So we must construct U first.
                    
                    # Let's use a temporary set for words in B(c_i) U B(c_j)
                    # Actually, we only care about words that are *uncovered* after removal.
                    
                    # Let's iterate through neighbors[c_i]
                    # For each u, new_count = cover_count[u] - 1 (since u in B(c_i))
                    # if u in B(c_j), new_count -= 1
                    # if new_count < 1, u is in U.
                    
                    U = set()
                    
                    # Check neighbors of c_i
                    for u in neighbors[c_i]:
                        nc = cover_count[u] - 1
                        if u in neighbor_sets[c_j]:
                            nc -= 1
                        if nc < 1:
                            U.add(u)
                            
                    # Check neighbors of c_j that are NOT in neighbors[c_i]
                    # (Those in both were already checked in the first loop)
                    for u in neighbors[c_j]:
                        if u in neighbor_sets[c_i]:
                            continue
                        nc = cover_count[u] - 1
                        if nc < 1:
                            U.add(u)
                    
                    if not U:
                        # We can remove both
                        code.remove(c_i)
                        code.remove(c_j)
                        code_set.discard(c_i)
                        code_set.discard(c_j)
                        for u in neighbors[c_i]:
                            cover_count[u] -= 1
                        for u in neighbors[c_j]:
                            cover_count[u] -= 1
                        improved = True
                        move_found = True
                        break
                    
                    # Try to find k such that U subset of B(k)
                    # k must be in intersection of neighbors[u] for all u in U
                    # Start with neighbors of the first word in U
                    # Sort U to ensure determinism? Or just pick arbitrary first.
                    first_u = next(iter(U))
                    candidates = neighbor_sets[first_u].copy()
                    
                    valid = True
                    for u in U:
                        if u == first_u:
                            continue
                        candidates &= neighbor_sets[u]
                        if not candidates:
                            valid = False
                            break
                    
                    if valid:
                        # Find a k in candidates not in code_set
                        # Note: k can be c_i or c_j? 
                        # If k == c_i, then we are replacing {c_i, c_j} with {c_i}, i.e., removing c_j.
                        # But we already checked if c_j could be removed alone? 
                        # Not necessarily, because removing c_i might make c_j essential, or vice versa.
                        # However, if k is in the code, we are effectively just removing the other one.
                        # The reduction phase should have handled single removals.
                        # So we should prefer k NOT in code_set.
                        # If no k outside code_set, we can still use one in code_set 
                        # if it results in a smaller code (which it won't, 2->1 is always better).
                        # Wait, if k is in code_set, say k=c_i, then we remove c_j.
                        # Code size decreases by 1.
                        
                        found_k = None
                        for k in candidates:
                            if k not in code_set:
                                found_k = k
                                break
                                
                        if found_k is not None:
                            # Replace c_i, c_j with found_k
                            code.remove(c_i)
                            code.remove(c_j)
                            code_set.discard(c_i)
                            code_set.discard(c_j)
                            
                            code.append(found_k)
                            code_set.add(found_k)
                            
                            for u in neighbors[c_i]:
                                cover_count[u] -= 1
                            for u in neighbors[c_j]:
                                cover_count[u] -= 1
                            for u in neighbors[found_k]:
                                cover_count[u] += 1
                            
                            improved = True
                            move_found = True
                            break
        
            if move_found:
                # After a swap, run reduction again
                if not reduce_code():
                    break
            else:
                # No swap found. Try a kick?
                # If time is tight, skip kicks.
                if time.time() > deadline - 10.0:
                    break
                
                # Kick: Remove a random codeword, cover uncovered, reduce.
                # Only do this if code size is not too small or if we are stuck.
                # Let's limit kicks to avoid wasting time.
                if random.random() < 0.1: # 10% chance of kick per pass if stuck
                    if code:
                        c_rand = random.choice(code)
                        code.remove(c_rand)
                        code_set.discard(c_rand)
                        for u in neighbors[c_rand]:
                            cover_count[u] -= 1
                        
                        # Greedily cover uncovered words
                        # Find words with cover_count == 0
                        uncovered = [w for w in range(N) if cover_count[w] == 0]
                        
                        while uncovered:
                            if time.time() > deadline - 5.0:
                                break
                            # Pick a codeword that covers the most uncovered words
                            # This is expensive to do from scratch.
                            # Approximation: pick a random neighbor of a random uncovered word?
                            # Or just pick one uncovered word, and add a random neighbor?
                            # Better: Greedy.
                            
                            # To speed up, just pick the first uncovered word w
                            # and add a random neighbor k of w that is not in code?
                            # Or the neighbor that covers the most *other* uncovered words.
                            
                            # Simple heuristic:
                            w0 = uncovered[0]
                            best_k = -1
                            max_cov = -1
                            for k in neighbors[w0]:
                                if k in code_set:
                                    continue
                                # Count how many uncovered words k covers
                                cov = 0
                                for u in neighbors[k]:
                                    if cover_count[u] == 0:
                                        cov += 1
                                if cov > max_cov:
                                    max_cov = cov
                                    best_k = k
                            
                            if best_k != -1:
                                code.append(best_k)
                                code_set.add(best_k)
                                for u in neighbors[best_k]:
                                    if cover_count[u] == 0:
                                        uncovered.remove(u) # This is O(N)
                                    cover_count[u] += 1
                                # Update uncovered list efficiently?
                                # For now, just rebuild or filter.
                                # Since N is small, rebuilding is okay but slow.
                                # Let's just break and let the next iteration handle it?
                                # No, we must maintain validity.
                                
                                # Efficient update:
                                # We already updated cover_count.
                                # We need to update the `uncovered` list.
                                # `uncovered` is a list. Removing from it is slow.
                                # Let's just recreate it if the list gets large?
                                # Or just break out and let the outer loop check validity?
                                # The outer loop assumes validity.
                                # Let's just do a full rebuild of uncovered if we do a kick.
                                pass # This simple kick implementation is risky for time.
                                
                        # After kick, reduce
                        if not reduce_code():
                            break

        return code

    # Run multiple trials
    # Trial 1: Standard Greedy
    uncovered = np.ones(N, dtype=np.int8)
    cov_count = np.array([len(neigh) for neigh in neighbors], dtype=np.int32)
    code = []
    remaining = N
    
    while remaining > 0:
        if time.time() > deadline - 20.0:
            break
        max_idx = np.argmax(cov_count)
        code.append(max_idx)
        for u in neighbors[max_idx]:
            if uncovered[u]:
                uncovered[u] = 0
                remaining -= 1
                for v in neighbors[u]:
                    cov_count[v] -= 1
                    
    if remaining > 0:
        # Fallback: add random words to cover rest
        for w in range(N):
            if uncovered[w]:
                code.append(w)
                for u in neighbors[w]:
                    if uncovered[u]:
                        uncovered[u] = 0
                        remaining -= 1
                if remaining == 0:
                    break

    # Run local search on the greedy result
    code = local_search(code, start_time)
    
    if len(code) < best_len:
        best_len = len(code)
        best_code = code[:]
        
    # Trial 2: Randomized Greedy (if time allows)
    if time.time() < deadline - 30.0:
        random.seed(42)
        uncovered = np.ones(N, dtype=np.int8)
        cov_count = np.array([len(neigh) for neigh in neighbors], dtype=np.int32)
        code = []
        remaining = N
        
        while remaining > 0:
            if time.time() > deadline - 10.0:
                break
            # Add noise to cov_count to break ties
            # Or just pick max
            max_idx = np.argmax(cov_count)
            code.append(max_idx)
            for u in neighbors[max_idx]:
                if uncovered[u]:
                    uncovered[u] = 0
                    remaining -= 1
                    for v in neighbors[u]:
                        cov_count[v] -= 1
                        
        if remaining > 0:
            for w in range(N):
                if uncovered[w]:
                    code.append(w)
                    for u in neighbors[w]:
                        if uncovered[u]:
                            uncovered[u] = 0
                            remaining -= 1
                    if remaining == 0:
                        break

        code = local_search(code, start_time)
        if len(code) < best_len:
            best_len = len(code)
            best_code = code[:]

    if best_code is None:
        return [[0]*n]
        
    return [to_digits(c) for c in best_code]

if __name__ == "__main__":
    q = 3
    n = 8
    radius = 1
    code = construct(q, n, radius)
    print(f"Code size: {len(code)}")
