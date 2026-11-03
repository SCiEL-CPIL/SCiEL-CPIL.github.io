"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict, deque
import time
import heapq

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Combines heap-based greedy initialization with 1-swap, 2-swap, and merge local search.
    """
    if n == 0:
        return [[]]
    
    total_words = q ** n
    deadline = time.time() + 115  # Safety margin before 120s kill
    
    # Precompute powers for fast index arithmetic
    powers = [q ** i for i in range(n)]
    
    def int_to_word(val, n, q):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        return word
    
    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val
    
    # Precompute neighbors using fast index arithmetic
    neighbors = [None] * total_words
    
    if radius == 1:
        for w in range(total_words):
            neigh = [w]
            for pos in range(n):
                cur = (w // powers[pos]) % q
                for val in range(q):
                    if val == cur:
                        continue
                    diff = val - cur
                    neigh.append(w + diff * powers[pos])
            neighbors[w] = neigh
    elif radius == 0:
        for w in range_words:
            neighbors[w] = [w]
    else:
        for w in range(total_words):
            neigh = [w]
            if radius >= 1:
                for pos in range(n):
                    cur = (w // powers[pos]) % q
                    for val in range(q):
                        if val == cur:
                            continue
                        diff = val - cur
                        neigh.append(w + diff * powers[pos])
            if radius >= 2:
                for pos1 in range(n):
                    cur1 = (w // powers[pos1]) % q
                    for val1 in range(q):
                        if val1 == cur1:
                            continue
                        diff1 = val1 - cur1
                        base = w + diff1 * powers[pos1]
                        for pos2 in range(pos1 + 1, n):
                            cur2 = (base // powers[pos2]) % q
                            for val2 in range(q):
                                if val2 == cur2:
                                    continue
                                diff2 = val2 - cur2
                                neigh.append(base + diff2 * powers[pos2])
            neighbors[w] = neigh
    
    # Reverse neighbors: for each word u, which codewords cover it
    rev_neighbors = [[] for _ in range(total_words)]
    for w in range(total_words):
        for u in neighbors[w]:
            rev_neighbors[u].append(w)
    
    def greedy_init():
        """Heap-based greedy: pick candidate with max new coverage."""
        covered = np.zeros(total_words, dtype=np.int8)
        codewords = []
        uncovered_count = total_words
        
        # Initialize gains for all codewords
        gains = np.full(total_words, len(neighbors[0]), dtype=np.int32)
        
        # Heap entries: (-gain, codeword)
        heap = [(-gains[c], c) for c in range(total_words)]
        heapq.heapify(heap)
        
        while uncovered_count > 0:
            # Find valid candidate with max gain
            valid_c = None
            while heap:
                neg_g, c = heapq.heappop(heap)
                if covered[c]:
                    continue
                if -neg_g != gains[c]:
                    continue
                if gains[c] <= 0:
                    continue
                valid_c = c
                break
            
            if valid_c is None:
                # Fallback: pick any uncovered word's neighbor
                for u in range(total_words):
                    if not covered[u]:
                        for c in rev_neighbors[u]:
                            if not covered[c]:
                                valid_c = c
                                break
                        if valid_c is not None:
                            break
                if valid_c is None:
                    break
            
            # Add codeword
            covered[valid_c] = 1
            codewords.append(valid_c)
            
            # Update coverage and gains
            for u in neighbors[valid_c]:
                if not covered[u]:
                    covered[u] = 1
                    uncovered_count -= 1
                    # Decrease gain for all codewords covering u
                    for c_prime in rev_neighbors[u]:
                        if not covered[c_prime]:
                            gains[c_prime] -= 1
                            heapq.heappush(heap, (-gains[c_prime], c_prime))
        
        return codewords
    
    def optimize(codewords, time_limit):
        """Local search: remove redundant, 1-swap, 2-swap, and merge moves."""
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            for u in neighbors[cw]:
                coverage_count[u] += 1
        
        cw_set = set(codewords)
        start_time = time.time()
        
        def remove_redundant():
            removed_any = False
            changed = True
            while changed:
                changed = False
                indices = list(range(len(codewords)))
                random.shuffle(indices)
                for i in indices:
                    cw = codewords[i]
                    can_remove = True
                    for u in neighbors[cw]:
                        if coverage_count[u] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for u in neighbors[cw]:
                            coverage_count[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        removed_any = True
                        break
            return removed_any
        
        def try_merge():
            """Try to replace 2 codewords with 1 (net -1)."""
            if len(codewords) < 2:
                return False
            
            # Sample pairs
            max_attempts = 50
            for _ in range(max_attempts):
                if time.time() - start_time > time_limit:
                    return False
                i, j = random.sample(range(len(codewords)), 2)
                c1, c2 = codewords[i], codewords[j]
                
                # Find unique points for c1 and c2
                U1 = [u for u in neighbors[c1] if coverage_count[u] == 1]
                U2 = [u for u in neighbors[c2] if coverage_count[u] == 1]
                
                if not U1 and not U2:
                    # Both redundant, should have been caught
                    continue
                
                all_unique = set(U1) | set(U2)
                if not all_unique:
                    continue
                
                # Find candidates covering all_unique
                first_p = next(iter(all_unique))
                cands = set(rev_neighbors[first_p])
                for p in all_unique:
                    cands.intersection_update(rev_neighbors[p])
                
                cands -= cw_set
                cands.discard(c1)
                cands.discard(c2)
                
                if cands:
                    # Pick best candidate (max overlap with already-covered)
                    best_c = None
                    best_score = -1
                    cand_list = list(cands)
                    random.shuffle(cand_list)
                    for c_new in cand_list[:50]:
                        score = 0
                        for u in neighbors[c_new]:
                            if coverage_count[u] > 0:
                                score += coverage_count[u] - 1
                        if score > best_score:
                            best_score = score
                            best_c = c_new
                    
                    if best_c is not None:
                        # Remove c1, c2 and add best_c
                        for u in neighbors[c1]:
                            coverage_count[u] -= 1
                        for u in neighbors[c2]:
                            coverage_count[u] -= 1
                        for u in neighbors[best_c]:
                            coverage_count[u] += 1
                        codewords[i] = best_c
                        codewords.pop(j)
                        cw_set.discard(c1)
                        cw_set.discard(c2)
                        cw_set.add(best_c)
                        return True
            
            return False
        
        def try_2swap():
            """Try to swap 2 codewords with 2 different codewords (net 0, may enable further removal)."""
            if len(codewords) < 2:
                return False
            
            max_attempts = 30
            for _ in range(max_attempts):
                if time.time() - start_time > time_limit:
                    return False
                i, j = random.sample(range(len(codewords)), 2)
                c1, c2 = codewords[i], codewords[j]
                
                U1 = [u for u in neighbors[c1] if coverage_count[u] == 1]
                U2 = [u for u in neighbors[c2] if coverage_count[u] == 1]
                
                if not U1 and not U2:
                    continue
                
                # Find c1' covering U1 and c2' covering U2
                if U1:
                    cands1 = set(rev_neighbors[U1[0]])
                    for p in U1[1:]:
                        cands1.intersection_update(rev_neighbors[p])
                else:
                    cands1 = set(range(total_words))
                
                if U2:
                    cands2 = set(rev_neighbors[U2[0]])
                    for p in U2[1:]:
                        cands2.intersection_update(rev_neighbors[p])
                else:
                    cands2 = set(range(total_words))
                
                cands1 -= cw_set
                cands1.discard(c1)
                cands1.discard(c2)
                cands2 -= cw_set
                cands2.discard(c1)
                cands2.discard(c2)
                
                if not cands1 or not cands2:
                    continue
                
                # Pick best pair
                cand1_list = list(cands1)
                cand2_list = list(cands2)
                random.shuffle(cand1_list)
                random.shuffle(cand2_list)
                
                best_pair = None
                best_score = -1
                
                for c1_new in cand1_list[:20]:
                    for c2_new in cand2_list[:20]:
                        if c1_new == c2_new:
                            continue
                        # Score: maximize overlap
                        score = 0
                        for u in neighbors[c1_new]:
                            if coverage_count[u] > 0:
                                score += coverage_count[u] - 1
                        for u in neighbors[c2_new]:
                            if coverage_count[u] > 0:
                                score += coverage_count[u] - 1
                        if score > best_score:
                            best_score = score
                            best_pair = (c1_new, c2_new)
                
                if best_pair:
                    c1_new, c2_new = best_pair
                    for u in neighbors[c1]:
                        coverage_count[u] -= 1
                    for u in neighbors[c2]:
                        coverage_count[u] -= 1
                    for u in neighbors[c1_new]:
                        coverage_count[u] += 1
                    for u in neighbors[c2_new]:
                        coverage_count[u] += 1
                    codewords[i] = c1_new
                    codewords[j] = c2_new
                    cw_set.discard(c1)
                    cw_set.discard(c2)
                    cw_set.add(c1_new)
                    cw_set.add(c2_new)
                    return True
            
            return False
        
        remove_redundant()
        
        while time.time() - start_time < time_limit:
            if remove_redundant():
                continue
            
            if not codewords:
                break
            
            # Try 1-swap
            idx = random.randrange(len(codewords))
            old_cw = codewords[idx]
            
            private_words = [u for u in neighbors[old_cw] if coverage_count[u] == 1]
            
            if not private_words:
                for u in neighbors[old_cw]:
                    coverage_count[u] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue
            
            potential = set(rev_neighbors[private_words[0]])
            for pw in private_words[1:]:
                potential.intersection_update(rev_neighbors[pw])
                if not potential:
                    break
            
            if not potential:
                # Try 2-swap or merge
                if try_merge():
                    continue
                if try_2swap():
                    continue
                continue
            
            valid_candidates = [c for c in potential if c not in cw_set and c != old_cw]
            
            if not valid_candidates:
                if try_merge():
                    continue
                if try_2swap():
                    continue
                continue
            
            random.shuffle(valid_candidates)
            valid_candidates = valid_candidates[:200]
            
            best_new_cw = -1
            best_score = -1
            
            for new_cw in valid_candidates:
                score = 0
                for u in neighbors[new_cw]:
                    if coverage_count[u] > 0:
                        score += coverage_count[u] - 1
                if score > best_score:
                    best_score = score
                    best_new_cw = new_cw
            
            if best_new_cw != -1:
                for u in neighbors[old_cw]:
                    coverage_count[u] -= 1
                for u in neighbors[best_new_cw]:
                    coverage_count[u] += 1
                codewords[idx] = best_new_cw
                cw_set.discard(old_cw)
                cw_set.add(best_new_cw)
            else:
                if try_merge():
                    continue
                if try_2swap():
                    continue
        
        remove_redundant()
        return codewords
    
    def polish(codewords, time_limit):
        """Final polish: systematic 1-swap, merge, and 2-swap."""
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            for u in neighbors[cw]:
                coverage_count[u] += 1
        
        cw_set = set(codewords)
        start_time = time.time()
        
        def try_remove_all():
            changed = True
            while changed:
                changed = False
                for i in range(len(codewords)):
                    cw = codewords[i]
                    can_remove = True
                    for u in neighbors[cw]:
                        if coverage_count[u] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for u in neighbors[cw]:
                            coverage_count[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        break
        
        try_remove_all()
        
        while time.time() - start_time < time_limit:
            if try_remove_all():
                continue
            
            if not codewords:
                break
            
            improved = False
            
            # Systematic 1-swap
            for idx in range(len(codewords)):
                if time.time() - start_time > time_limit:
                    break
                old_cw = codewords[idx]
                
                private_words = [u for u in neighbors[old_cw] if coverage_count[u] == 1]
                
                if not private_words:
                    for u in neighbors[old_cw]:
                        coverage_count[u] -= 1
                    codewords.pop(idx)
                    cw_set.discard(old_cw)
                    improved = True
                    break
                
                potential = set(rev_neighbors[private_words[0]])
                for pw in private_words[1:]:
                    potential.intersection_update(rev_neighbors[pw])
                    if not potential:
                        break
                
                if not potential:
                    continue
                
                valid_candidates = [c for c in potential if c not in cw_set and c != old_cw]
                
                if not valid_candidates:
                    continue
                
                best_new_cw = -1
                best_score = -1
                
                for new_cw in valid_candidates:
                    score = 0
                    for u in neighbors[new_cw]:
                        if coverage_count[u] > 0:
                            score += coverage_count[u] - 1
                    if score > best_score:
                        best_score = score
                        best_new_cw = new_cw
                
                if best_new_cw != -1:
                    for u in neighbors[old_cw]:
                        coverage_count[u] -= 1
                    for u in neighbors[best_new_cw]:
                        coverage_count[u] += 1
                    codewords[idx] = best_new_cw
                    cw_set.discard(old_cw)
                    cw_set.add(best_new_cw)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try merge
            if len(codewords) >= 2:
                merged = False
                for _ in range(30):
                    if time.time() - start_time > time_limit:
                        break
                    i, j = random.sample(range(len(codewords)), 2)
                    c1, c2 = codewords[i], codewords[j]
                    
                    U1 = [u for u in neighbors[c1] if coverage_count[u] == 1]
                    U2 = [u for u in neighbors[c2] if coverage_count[u] == 1]
                    
                    all_unique = set(U1) | set(U2)
                    if not all_unique:
                        continue
                    
                    first_p = next(iter(all_unique))
                    cands = set(rev_neighbors[first_p])
                    for p in all_unique:
                        cands.intersection_update(rev_neighbors[p])
                    
                    cands -= cw_set
                    cands.discard(c1)
                    cands.discard(c2)
                    
                    if cands:
                        best_c = None
                        best_score = -1
                        for c_new in list(cands)[:50]:
                            score = 0
                            for u in neighbors[c_new]:
                                if coverage_count[u] > 0:
                                    score += coverage_count[u] - 1
                            if score > best_score:
                                best_score = score
                                best_c = c_new
                        
                        if best_c is not None:
                            for u in neighbors[c1]:
                                coverage_count[u] -= 1
                            for u in neighbors[c2]:
                                coverage_count[u] -= 1
                            for u in neighbors[best_c]:
                                coverage_count[u] += 1
                            codewords[i] = best_c
                            codewords.pop(j)
                            cw_set.discard(c1)
                            cw_set.discard(c2)
                            cw_set.add(best_c)
                            merged = True
                            break
                
                if merged:
                    continue
            
            # Try 2-swap
            if len(codewords) >= 2:
                swapped = False
                for _ in range(20):
                    if time.time() - start_time > time_limit:
                        break
                    i, j = random.sample(range(len(codewords)), 2)
                    c1, c2 = codewords[i], codewords[j]
                    
                    U1 = [u for u in neighbors[c1] if coverage_count[u] == 1]
                    U2 = [u for u in neighbors[c2] if coverage_count[u] == 1]
                    
                    if U1:
                        cands1 = set(rev_neighbors[U1[0]])
                        for p in U1[1:]:
                            cands1.intersection_update(rev_neighbors[p])
                    else:
                        cands1 = set(range(total_words))
                    
                    if U2:
                        cands2 = set(rev_neighbors[U2[0]])
                        for p in U2[1:]:
                            cands2.intersection_update(rev_neighbors[p])
                    else:
                        cands2 = set(range(total_words))
                    
                    cands1 -= cw_set
                    cands1.discard(c1)
                    cands1.discard(c2)
                    cands2 -= cw_set
                    cands2.discard(c1)
                    cands2.discard(c2)
                    
                    if not cands1 or not cands2:
                        continue
                    
                    cand1_list = list(cands1)
                    cand2_list = list(cands2)
                    random.shuffle(cand1_list)
                    random.shuffle(cand2_list)
                    
                    best_pair = None
                    best_score = -1
                    
                    for c1_new in cand1_list[:15]:
                        for c2_new in cand2_list[:15]:
                            if c1_new == c2_new:
                                continue
                            score = 0
                            for u in neighbors[c1_new]:
                                if coverage_count[u] > 0:
                                    score += coverage_count[u] - 1
                            for u in neighbors[c2_new]:
                                if coverage_count[u] > 0:
                                    score += coverage_count[u] - 1
                            if score > best_score:
                                best_score = score
                                best_pair = (c1_new, c2_new)
                    
                    if best_pair:
                        c1_new, c2_new = best_pair
                        for u in neighbors[c1]:
                            coverage_count[u] -= 1
                        for u in neighbors[c2]:
                            coverage_count[u] -= 1
                        for u in neighbors[c1_new]:
                            coverage_count[u] += 1
                        for u in neighbors[c2_new]:
                            coverage_count[u] += 1
                        codewords[i] = c1_new
                        codewords[j] = c2_new
                        cw_set.discard(c1)
                        cw_set.discard(c2)
                        cw_set.add(c1_new)
                        cw_set.add(c2_new)
                        swapped = True
                        break
                
                if swapped:
                    continue
        
        try_remove_all()
        return codewords
    
    # Multiple restarts
    start_time = time.time()
    best_codewords = []
    best_len = float('inf')
    
    # First restart gets more time
    cw = greedy_init()
    if time.time() < deadline - 5:
        opt_time = min(25, deadline - time.time() - 5)
        if opt_time > 0:
            cw = optimize(cw, opt_time)
        if time.time() < deadline - 3:
            pol_time = min(10, deadline - time.time() - 3)
            if pol_time > 0:
                cw = polish(cw, pol_time)
    
    if len(cw) < best_len:
        best_len = len(cw)
        best_codewords = cw
    
    # Subsequent restarts with less time each
    restart_count = 0
    while time.time() < deadline - 10:
        restart_count += 1
        remaining = deadline - time.time() - 5
        if remaining < 15:
            break
        iter_time = min(15, remaining * 0.4)
        if iter_time < 5:
            break
        
        cw = greedy_init()
        opt_time = min(iter_time - 2, 10)
        if opt_time > 0 and time.time() < deadline - 5:
            cw = optimize(cw, opt_time)
        
        if len(cw) < best_len:
            best_len = len(cw)
            best_codewords = cw
            if time.time() < deadline - 3:
                pol_time = min(5, deadline - time.time() - 3)
                if pol_time > 0:
                    polished = polish(best_codewords[:], pol_time)
                    if len(polished) < best_len:
                        best_len = len(polished)
                        best_codewords = polished
    
    # Final polish on best result
    if time.time() < deadline - 2 and best_codewords:
        pol_time = min(5, deadline - time.time() - 2)
        if pol_time > 0:
            polished = polish(best_codewords[:], pol_time)
            if len(polished) < best_len:
                best_len = len(polished)
                best_codewords = polished
    
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
