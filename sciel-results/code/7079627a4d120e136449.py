"""
Optimized ternary covering code construction using Greedy + Local Search.
Targets K_3(6,1).
"""
import time
import random
import numpy as np

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # 5s margin before 120s limit
    
    # We specifically optimize for the case q=3, n=6, radius=1
    # But we must handle the general interface.
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: simple greedy
        return _greedy_construct(q, n, radius, deadline - time.time())
    
    # Specific optimization for q=3, n=6, radius=1
    return _construct_k3_6_1(deadline)

def _greedy_construct(q, n, radius, time_limit):
    """Simple greedy fallback for non-standard parameters."""
    deadline = time.time() + time_limit
    total_words = q ** n
    if total_words > 100000:
        return []
        
    # Precompute neighbors as list of lists
    neighbors = []
    for i in range(total_words):
        w = [(i // (q ** (n - 1 - pos))) % q for pos in range(n)]
        nbrs = [i]
        if radius == 1:
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        nw = w[:]
                        nw[pos] = new_val
                        idx = 0
                        for j in range(n):
                            idx = idx * q + nw[j]
                        nbrs.append(idx)
        neighbors.append(nbrs)
        
    covered = np.zeros(total_words, dtype=bool)
    code = []
    
    while time.time() < deadline:
        gains = np.array([np.sum(~covered[neighbors[i]]) for i in range(total_words)])
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        code.append(best_idx)
        covered[neighbors[best_idx]] = True
        
    result = []
    for idx in code:
        w = [(idx // (q ** (n - 1 - pos))) % q for pos in range(n)]
        result.append(w)
    return result

def _construct_k3_6_1(deadline):
    """
    Specialized constructor for K_3(6,1).
    Uses greedy initialization followed by local search with swap moves.
    """
    n = 6
    q = 3
    total_words = 729
    
    # Convert indices to words
    words = np.array([[(i // (3 ** (5 - pos))) % 3 for pos in range(6)] for i in range(total_words)])
    
    # Build neighbor lists (sphere of radius 1)
    neighbor_lists = []
    for i in range(total_words):
        w = words[i]
        nbrs = [i]
        for pos in range(6):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val != orig_val:
                    new_w = w.copy()
                    new_w[pos] = new_val
                    idx = 0
                    for j in range(6):
                        idx = idx * 3 + new_w[j]
                    nbrs.append(idx)
        neighbor_lists.append(nbrs)
    
    # Boolean sphere matrix
    sphere = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        sphere[i, neighbor_lists[i]] = True
    
    # Greedy Initialization
    covered = np.zeros(total_words, dtype=bool)
    code_indices = []
    
    greedy_deadline = deadline - 60.0  # Reserve 60s for local search
    while time.time() < greedy_deadline:
        gains = (sphere & (~covered)[np.newaxis, :]).sum(axis=1)
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        code_indices.append(best_idx)
        covered[sphere[best_idx]] = True
    
    best_code = code_indices.copy()
    best_size = len(best_code)
    
    # Local Search with multiple restarts
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize from best code with some random perturbation
        current_code = best_code.copy()
        if restart > 0 and random.random() < 0.3:
            # Perturb: remove 2-3 random codewords, add 2-3 random ones
            num_remove = min(random.randint(2, 3), len(current_code) - 5)
            for _ in range(num_remove):
                rm = random.choice(current_code)
                current_code.remove(rm)
            num_add = random.randint(2, 3)
            for _ in range(num_add):
                ra = random.randint(0, total_words - 1)
                if ra not in current_code:
                    current_code.append(ra)
        
        # Build count array
        count = np.zeros(total_words, dtype=np.int32)
        for c in current_code:
            count[sphere[c]] += 1
        
        # Verify coverage - if not fully covered, fix by adding codewords
        uncovered = np.where(count == 0)[0]
        while len(uncovered) > 0 and time.time() < deadline - 5.0:
            # Pick the uncovered word with fewest covering options among remaining
            # Just pick the first uncovered and find a codeword covering it
            u = int(uncovered[0])
            # Find codewords that cover u
            candidates = np.where(sphere[:, u])[0]
            # Pick one not already in code, or if all are, pick random
            not_in_code = [c for c in candidates if c not in current_code]
            if not_in_code:
                choice = random.choice(not_in_code)
            else:
                choice = int(random.choice(candidates))
            if choice not in current_code:
                current_code.append(choice)
                count[sphere[choice]] += 1
            uncovered = np.where(count == 0)[0]
        
        # Local search: try removals and swaps
        local_iters = 0
        max_local_iters = 20000
        while time.time() < deadline - 5.0 and local_iters < max_local_iters:
            local_iters += 1
            random.shuffle(current_code)
            modified = False
            
            # Try removals
            i = 0
            while i < len(current_code):
                c = current_code[i]
                min_cnt = 999
                for j in neighbor_lists[c]:
                    if count[j] < min_cnt:
                        min_cnt = count[j]
                        if min_cnt <= 1:
                            break
                if min_cnt > 1:
                    count[sphere[c]] -= 1
                    current_code.pop(i)
                    modified = True
                else:
                    i += 1
            
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = current_code.copy()
            
            if not modified:
                # Try swap: remove one, add another
                if len(current_code) > 10:
                    swap_done = False
                    for _ in range(20):  # Try up to 20 random swaps
                        if time.time() > deadline - 5.0:
                            break
                        rm_idx = random.randint(0, len(current_code) - 1)
                        rm_c = current_code[rm_idx]
                        # Temporarily remove
                        count[sphere[rm_c]] -= 1
                        # Find uncovered
                        unc = np.where(count == 0)[0]
                        if len(unc) == 0:
                            # Safe removal, just keep it removed
                            current_code.pop(rm_idx)
                            modified = True
                            if len(current_code) < best_size:
                                best_size = len(current_code)
                                best_code = current_code.copy()
                            break
                        else:
                            # Need to add something to cover unc
                            # Find a word that covers all uncovered
                            # Pick the uncovered word with most covering options
                            best_add = None
                            best_cover = 0
                            for u in unc[:10]:  # Limit search
                                cands = np.where(sphere[:, u])[0]
                                for cand in cands:
                                    if cand not in current_code and cand != rm_c:
                                        # Check how many uncovered it covers
                                        cov = np.sum(count[sphere[cand]] == 0)
                                        if cov > best_cover:
                                            best_cover = cov
                                            best_add = cand
                            if best_add is not None and best_cover > 0:
                                current_code.append(best_add)
                                count[sphere[best_add]] += 1
                                # Now check if we can remove rm_c
                                # rm_c was already removed from count, so just check coverage
                                unc2 = np.where(count == 0)[0]
                                if len(unc2) == 0:
                                    modified = True
                                    if len(current_code) < best_size:
                                        best_size = len(current_code)
                                        best_code = current_code.copy()
                                else:
                                    # Revert
                                    count[sphere[best_add]] -= 1
                                    current_code.pop()
                                    count[sphere[rm_c]] += 1
                            else:
                                # Revert
                                count[sphere[rm_c]] += 1
                            if swap_done or modified:
                                break
                    if modified:
                        break
                    # If no swap worked, try adding a random codeword
                    ra = random.randint(0, total_words - 1)
                    if ra not in current_code:
                        current_code.append(ra)
                        count[sphere[ra]] += 1
        
        if time.time() > deadline - 10.0:
            break
    
    # Convert best_code indices back to words
    result = []
    for idx in best_code:
        w = words[idx].tolist()
        result.append(w)
        
    return result


