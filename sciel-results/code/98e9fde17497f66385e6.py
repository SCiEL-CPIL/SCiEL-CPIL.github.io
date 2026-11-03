import time
import sys
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We aim to finish well before.
    deadline = start_time + 140.0
    
    # 1. Improved Construction: Greedy Hamiltonian Path with Lookahead
    # Build a directed graph where nodes are (n-1)-permutations and edges
    # represent valid symbol insertions. Use greedy path finding with
    # lookahead to minimize isolated vertices in remaining subgraph.
    
    if n <= 4:
        # For small n, use classic construction
        word = [1]
        for m in range(2, n + 1):
            perms = []
            seen = set()
            L = len(word)
            for i in range(L - (m - 1) + 1):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            new_word = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_overlap = 0
                limit = min(len(new_word), len(block))
                for o in range(limit, -1, -1):
                    if o == 0:
                        break
                    if new_word[-o:] == block[:o]:
                        max_overlap = o
                        break
                new_word.extend(block[max_overlap:])
            word = new_word
        return "".join(map(str, word))
    
    # For n >= 5, use the improved greedy Hamiltonian path approach
    # Build (n-1)-permutation graph
    base_perms = list(permutations(range(1, n)))
    perm_set = set(base_perms)
    perm_index = {p: i for i, p in enumerate(base_perms)}
    num_base_perms = len(base_perms)
    
    # Build adjacency: from perm p, we can go to perm q if q can be formed
    # by inserting n into p (or removing n from q gives p)
    # Actually, for superpermutation construction, we think of it as:
    # Each (n-1)-perm appears as a window. To transition from perm p to perm q,
    # we need to add symbols such that q appears.
    
    # Simpler approach: Use the classic recursive construction but with
    # optimized ordering based on lookahead heuristics
    
    word = [1]
    for m in range(2, n + 1):
        # Extract unique permutations of length m-1 from word
        perms = []
        seen = set()
        L = len(word)
        for i in range(L - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Sort perms using lookahead heuristic:
        # Prioritize transitions that keep more options open
        # Use a greedy approach with 2-step lookahead
        if len(perms) > 1:
            remaining = set(range(len(perms)))
            ordered_perms = []
            current_pos = 0  # Start with first perm as anchor
            
            while remaining:
                # Pick the next perm that minimizes "cost"
                best_idx = None
                best_cost = float('inf')
                
                for idx in remaining:
                    p = perms[idx]
                    # Cost heuristic: how many other remaining perms can follow p?
                    # Fewer options = higher priority (to avoid isolating)
                    follow_options = 0
                    for other_idx in remaining:
                        if other_idx == idx:
                            continue
                        q = perms[other_idx]
                        # Check if q can follow p with minimal overlap
                        # This is a simplification; full lookahead would be expensive
                        if p[-1] != q[0]:
                            follow_options += 1
                    
                    # We want to pick perms that have FEWER follow options first
                    # (to avoid getting stuck), but also consider overlap with current
                    overlap_with_current = 0
                    if ordered_perms:
                        last_p = ordered_perms[-1]
                        for o in range(min(len(last_p), len(p)), -1, -1):
                            if o == 0:
                                break
                            if last_p[-o:] == p[:o]:
                                overlap_with_current = o
                                break
                    
                    # Combined cost: prioritize high overlap, then fewer options
                    cost = -overlap_with_current * 10 + follow_options
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = idx
                
                if best_idx is not None:
                    ordered_perms.append(perms[best_idx])
                    remaining.remove(best_idx)
            
            perms = ordered_perms
        
        # Construct new word
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            
            max_overlap = 0
            limit = min(len(new_word), len(block))
            for o in range(limit, -1, -1):
                if o == 0:
                    break
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            
            new_word.extend(block[max_overlap:])
        
        word = new_word
        
        if time.time() > deadline - 60.0:
            break
    
    # 2. Optimization Phase: Greedy Deletion with Perturbation
    
    # Map permutations to IDs for fast counting
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts
    def compute_counts(w):
        counts = [0] * num_perms
        L = len(w)
        for i in range(L - n + 1):
            p = tuple(w[i:i+n])
            if p in perm_to_id:
                counts[perm_to_id[p]] += 1
        return counts

    # Initial check
    initial_counts = compute_counts(word)
    if any(c == 0 for c in initial_counts):
        # Should not happen with valid construction, but just in case
        return "".join(map(str, word))

    best_word = word[:]
    best_len = len(word)
    
    # Helper to check if deletion at index i is valid
    def try_delete(i, w, counts):
        L = len(w)
        if L <= n:
            return False, None
        
        old_start_min = max(0, i - n + 1)
        old_start_max = min(i, L - n)
        
        new_start_min = max(0, i - n + 1)
        new_start_max = min(i - 1, L - n - 1)
        
        changes = []
        
        # 1. Decrement old windows
        for k in range(old_start_min, old_start_max + 1):
            p = tuple(w[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                counts[pid] -= 1
                changes.append((pid, -1))
                if counts[pid] < 0:
                    for cp, delta in changes:
                        counts[cp] -= delta
                    return False, None

        # 2. Increment new windows
        for k in range(new_start_min, new_start_max + 1):
            w1 = w[k:i]
            w2 = w[i+1:k+n+1]
            p = tuple(w1 + w2)
            pid = perm_to_id.get(p)
            if pid is not None:
                counts[pid] += 1
                changes.append((pid, 1))

        valid = True
        for pid, delta in changes:
            if delta == -1:
                if counts[pid] == 0:
                    valid = False
                    break
        
        if not valid:
            for pid, delta in changes:
                counts[pid] -= delta
            return False, None
        
        return True, counts

    # Main optimization loop
    current_word = best_word[:]
    current_counts = compute_counts(current_word)
    
    # First, do a clean greedy pass
    i = 0
    while i < len(current_word):
        if time.time() > deadline - 10.0:
            break
        valid, new_counts = try_delete(i, current_word, current_counts)
        if valid:
            current_word.pop(i)
            current_counts = new_counts
            if len(current_word) < best_len:
                best_word = current_word[:]
                best_len = len(current_word)
        else:
            i += 1
            
    # Now, try to perturb and re-optimize if we have time
    iterations = 0
    while time.time() < deadline - 5.0 and iterations < 200:
        iterations += 1
        
        cand_word = best_word[:]
        cand_counts = compute_counts(cand_word)
        
        # Perturb: pick a random index and move the character to a random other index
        if len(cand_word) > 2:
            idx1 = random.randint(0, len(cand_word) - 1)
            idx2 = random.randint(0, len(cand_word) - 1)
            if idx1 != idx2:
                val = cand_word.pop(idx1)
                if idx2 > idx1:
                    idx2 -= 1
                cand_word.insert(idx2, val)
                
                cand_counts = compute_counts(cand_word)
                
                if any(c == 0 for c in cand_counts):
                    continue
                
                # Run greedy deletion on the perturbed word
                i = 0
                while i < len(cand_word):
                    if time.time() > deadline - 5.0:
                        break
                    valid, new_counts = try_delete(i, cand_word, cand_counts)
                    if valid:
                        cand_word.pop(i)
                        cand_counts = new_counts
                        if len(cand_word) < best_len:
                            best_word = cand_word[:]
                            best_len = len(cand_word)
                    else:
                        i += 1
    
    return "".join(map(str, best_word))
