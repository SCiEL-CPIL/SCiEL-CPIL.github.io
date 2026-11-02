import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code using a greedy set-cover algorithm with multiple restarts 
    and redundancy removal.
    """
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit
    
    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start_time, deadline)
    else:
        return _generic_greedy(q, n, radius, start_time, deadline)

def _solve_ternary_7_1(start_time, deadline):
    q = 3
    n = 7
    radius = 1
    total_words = q ** n
    
    # Precompute digits for each index
    # index -> list of 7 digits
    all_codewords = []
    for i in range(total_words):
        digits = [0] * n
        tmp = i
        for pos in range(n):
            digits[pos] = tmp % q
            tmp //= q
        all_codewords.append(tuple(digits))
    
    # Precompute covers: covers[i] is a set of indices covered by codeword i
    # For radius 1, covers[i] contains i and all words at Hamming distance 1.
    # There are 1 + n*(q-1) = 1 + 7*2 = 15 words.
    covers = [set() for _ in range(total_words)]
    
    # Precompute powers of 3 for fast index calculation
    p3 = [1] * n
    for i in range(1, n):
        p3[i] = p3[i-1] * q
        
    for i, c in enumerate(all_codewords):
        covers[i].add(i)
        c_list = list(c)
        for pos in range(n):
            orig_val = c_list[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                c_list[pos] = new_val
                idx = 0
                for digit in c_list:
                    idx = idx * q + digit
                covers[i].add(idx)
            c_list[pos] = orig_val
            
    # Precompute reverse_cover: reverse_cover[w] is set of codeword indices covering w
    reverse_cover = [set() for _ in range(total_words)]
    for c_idx in range(total_words):
        for w_idx in covers[c_idx]:
            reverse_cover[w_idx].add(c_idx)
            
    # We will run multiple restarts of a greedy algorithm with randomized tie-breaking.
    # To speed up, we maintain cover_count[c] = number of uncovered words covered by c.
    # When we select c*, we decrement cover_count[c] for all c that cover any word in covers[c*].
    
    best_code_size = total_words
    best_codewords = None
    
    # Number of restarts is dynamic based on time
    # Each restart takes roughly O(K * N * avg_degree) where K is code size (~200), N=2187, avg_degree=15
    # But the inner loop for updating cover_count is efficient if we just iterate covers[c*] and then reverse_cover[w].
    # Let's estimate time per restart.
    # We will run as many as possible.
    
    run_id = 0
    while time.time() < deadline - 2.0:
        run_id += 1
        rng = random.Random(run_id)
        
        uncovered = set(range(total_words))
        chosen_indices = []
        
        # cover_count[c] = number of words in covers[c] that are in uncovered
        cover_count = [len(covers[c]) for c in range(total_words)]
        
        while uncovered:
            # Find max cover_count
            max_count = -1
            candidates = []
            # Optimization: Instead of scanning all 2187 every time, 
            # we could maintain a heap, but Python's heapq doesn't support decrease-key easily.
            # Scanning 2187 integers is fast (~microseconds).
            # 200 iterations * 2187 scans = 437k operations per restart.
            # Plus the update step.
            
            for c in range(total_words):
                cnt = cover_count[c]
                if cnt > max_count:
                    max_count = cnt
                    candidates = [c]
                elif cnt == max_count:
                    candidates.append(c)
                    
            if not candidates or max_count == 0:
                # If max_count is 0 but uncovered is not empty, something is wrong or we are stuck?
                # Should not happen if covers are correct.
                break
                
            # Pick randomly from candidates to break ties
            c_star = rng.choice(candidates)
            chosen_indices.append(c_star)
            
            # Update uncovered and cover_count
            # Words in covers[c_star] are now covered
            for w in covers[c_star]:
                if w in uncovered:
                    uncovered.remove(w)
                    # For every codeword c that covers w, decrement its count
                    for c in reverse_cover[w]:
                        cover_count[c] -= 1
        
        # Redundancy Removal
        # The greedy algorithm might include redundant codewords.
        # We try to remove any codeword that is not essential.
        # A codeword c is essential if there is a word w covered ONLY by c.
        # We can check this by maintaining a count of how many chosen codewords cover each word.
        
        # Let's build a count array for the chosen set.
        # count[w] = number of chosen codewords that cover w.
        count = [0] * total_words
        for c in chosen_indices:
            for w in covers[c]:
                count[w] += 1
                
        # Try to remove codewords
        # We iterate through chosen_indices and check if any word has count 1.
        # If we remove c, and all words it covers have count >= 2, it's safe to remove.
        
        # We do this iteratively until no more can be removed.
        changed = True
        while changed:
            changed = False
            new_chosen = []
            for c in chosen_indices:
                # Check if c is essential
                is_essential = False
                for w in covers[c]:
                    if count[w] == 1:
                        is_essential = True
                        break
                if not is_essential:
                    # Remove c
                    for w in covers[c]:
                        count[w] -= 1
                    changed = True
                else:
                    new_chosen.append(c)
            chosen_indices = new_chosen
            
        if len(chosen_indices) < best_code_size:
            best_code_size = len(chosen_indices)
            best_codewords = [all_codewords[i] for i in chosen_indices]
            
            # If we reach the known best (186), we can stop or continue to verify?
            # The task says best known is 186. If we find it, we can't do better.
            # But maybe we can find a smaller one? No, 186 is best known.
            # However, the lower bound is 156. It's unlikely we find < 186 easily.
            # Let's continue searching to see if we can consistently get 186 or better.
            if best_code_size <= 186:
                # We found a very good solution. Let's do a few more quick checks?
                # Or just break to save time?
                # Since 186 is the best known, breaking is reasonable if we want to be safe.
                # But maybe randomization finds 185? Unlikely.
                # Let's keep searching a bit more to be sure, but prioritize speed.
                pass

    if best_codewords is None:
        # Fallback: Return the direct sum construction if greedy failed (shouldn't happen)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Convert tuples to lists
    return [list(c) for c in best_codewords]

def _generic_greedy(q, n, radius, start_time, deadline):
    # Fallback for other parameters
    total_words = q ** n
    all_codewords = list(itertools.product(range(q), repeat=n))
    
    covers = [set() for _ in range(total_words)]
    for i, c in enumerate(all_codewords):
        c_list = list(c)
        # Generate all words within radius
        # This is expensive for large n, but okay for small
        for delta in range(radius + 1):
            for positions in itertools.combinations(range(n), delta):
                for changes in itertools.product(range(1, q), repeat=delta):
                    new_c = list(c_list)
                    for pos, ch in zip(positions, changes):
                        new_c[pos] = (new_c[pos] + ch) % q
                    idx = 0
                    for digit in new_c:
                        idx = idx * q + digit
                    covers[i].add(idx)
                    
    best_code_size = total_words
    best_codewords = None
    
    for run_id in range(100):
        if time.time() > deadline:
            break
        random.seed(run_id)
        uncovered = set(range(total_words))
        chosen_indices = []
        
        while uncovered:
            max_count = -1
            candidates = []
            for c in range(total_words):
                count = len(covers[c] & uncovered)
                if count > max_count:
                    max_count = count
                    candidates = [c]
                elif count == max_count:
                    candidates.append(c)
            if not candidates:
                break
            c_star = random.choice(candidates)
            chosen_indices.append(c_star)
            uncovered -= covers[c_star]
            
        if len(chosen_indices) < best_code_size:
            best_code_size = len(chosen_indices)
            best_codewords = [all_codewords[i] for i in chosen_indices]
            
    if best_codewords is None:
        return [list(all_codewords[i]) for i in range(total_words)]
        
    return [list(c) for c in best_codewords]
