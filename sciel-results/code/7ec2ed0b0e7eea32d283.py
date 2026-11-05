# EVOLVE-BLOCK-START
import time
import sys
import random
import math
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    if n <= 3:
        # Standard construction is optimal for n<=3
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
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    if n == 4:
        # Known shortest superpermutation for n=4 is 33
        # Standard construction gives 33 as well, so just use it
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
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # For n >= 5, use a greedy nearest-neighbor construction with local search improvements
    # The key insight: instead of the recursive construction, we can find a good ordering
    # of all n! permutations that maximizes overlaps.
    
    all_perms = list(permutations(range(1, n+1)))
    num_perms = len(all_perms)
    
    # Map permutation to index for fast lookup
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    
    # Precompute overlap matrix: overlap[i][j] = max k such that all_perms[i][n-k:] == all_perms[j][:k]
    # For n=7, num_perms = 5040, so matrix is 5040x5040 = ~25M entries. Too big for full precompute.
    # Instead, compute overlaps on the fly but cache them.
    
    # Actually, let's use a different approach: greedy construction with lookahead
    # Start with a random permutation, then repeatedly append the unused permutation
    # that maximizes overlap with the current suffix.
    
    def get_overlap(a, b):
        """Max overlap between suffix of a and prefix of b, both length n."""
        for k in range(n-1, -1, -1):
            if k == 0:
                return 0
            if a[n-k:] == b[:k]:
                return k
        return 0
    
    # Greedy construction with random restarts
    def greedy_construct(used_set, start_perm):
        """Build a word starting with start_perm, greedily adding unused perms."""
        if not used_set:
            # This is a fresh start
            used_set = set()
        
        order = [start_perm]
        used_set = set(used_set)
        used_set.add(start_perm)
        current_suffix = start_perm
        
        for _ in range(num_perms - 1):
            best_perm = None
            best_overlap = -1
            # Sample a subset of unused permutations for speed
            unused = [p for p in all_perms if p not in used_set]
            # If too many, sample
            if len(unused) > 200:
                sample = random.sample(unused, 200)
            else:
                sample = unused
            
            for p in sample:
                ov = get_overlap(current_suffix, p)
                if ov > best_overlap:
                    best_overlap = ov
                    best_perm = p
            
            if best_perm is None:
                # Fallback: pick any unused
                best_perm = random.choice(unused)
                best_overlap = 0
            
            order.append(best_perm)
            used_set.add(best_perm)
            current_suffix = best_perm
        
        return order, used_set
    
    # Calculate length of a given order
    def calc_length(order):
        length = n
        for i in range(len(order) - 1):
            length += n - get_overlap(order[i], order[i+1])
        return length
    
    # Local search: 2-opt style moves
    def local_search(order, max_iters=100000):
        """Improve order via 2-opt moves."""
        current_len = calc_length(order)
        best_order = order[:]
        best_len = current_len
        
        # Precompute overlaps for adjacent pairs
        ov = [0] * (len(order) - 1)
        for i in range(len(ov)):
            ov[i] = get_overlap(order[i], order[i+1])
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            i = random.randint(0, len(order) - 2)
            j = random.randint(i+1, len(order) - 1)
            
            # Try reversing the segment from i to j
            if i == j:
                continue
            
            # Calculate delta
            # Old: ... order[i-1] -> order[i] ... order[j] -> order[j+1] ...
            # New: ... order[i-1] -> order[j] ... order[i] -> order[j+1] ...
            
            old_cost = 0
            if i > 0:
                old_cost += n - ov[i-1]
            if j < len(ov):
                old_cost += n - ov[j]
            # Internal overlaps in segment i..j also change, but for simplicity
            # let's just do adjacent swaps first
            
            # For now, just do adjacent swaps (i, i+1)
            j = i + 1
            
            old_cost = 0
            if i > 0:
                old_cost += n - ov[i-1]
            old_cost += n - ov[i]
            if i < len(ov) - 1:
                old_cost += n - ov[i+1]
            
            new_cost = 0
            if i > 0:
                new_cost += n - get_overlap(order[i-1], order[i+1])
            new_cost += n - get_overlap(order[i+1], order[i])
            if i < len(ov) - 1:
                new_cost += n - get_overlap(order[i], order[i+2])
            
            delta = new_cost - old_cost
            
            if delta < 0:
                order[i], order[i+1] = order[i+1], order[i]
                if i > 0:
                    ov[i-1] = get_overlap(order[i-1], order[i])
                ov[i] = get_overlap(order[i], order[i+1])
                if i < len(ov) - 1:
                    ov[i+1] = get_overlap(order[i+1], order[i+2])
                current_len += delta
                if current_len < best_len:
                    best_len = current_len
                    best_order = order[:]
        
        return best_order, best_len
    
    # Strategy: 
    # 1. Build initial solution using greedy with the standard construction's order
    # 2. Run local search to improve
    
    # First, get the standard construction's permutation order
    def standard_construction(n):
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
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word
    
    std_word = standard_construction(n)
    std_len = len(std_word)
    
    # Extract permutation order from standard construction
    perm_list = []
    seen = set()
    for i in range(len(std_word) - n + 1):
        p = tuple(std_word[i:i+n])
        if p not in seen:
            seen.add(p)
            perm_list.append(p)
    
    # Verify we have all permutations
    if len(perm_list) < num_perms:
        # Fallback: just return standard construction
        return "".join(map(str, std_word))
    
    # Start with the standard order and improve via local search
    current_order = perm_list[:]
    current_len = calc_length(current_order)
    
    # Run local search with time budget
    max_search_time = deadline - 15
    while time.time() < max_search_time:
        # Phase 1: Hill climbing with adjacent swaps
        improved = False
        ov = [0] * (len(current_order) - 1)
        for i in range(len(ov)):
            ov[i] = get_overlap(current_order[i], current_order[i+1])
        
        # Multiple passes of hill climbing
        for pass_num in range(50):
            if time.time() > max_search_time:
                break
            
            any_improved = False
            for i in range(len(current_order) - 1):
                if time.time() > max_search_time:
                    break
                
                # Try swapping i and i+1
                old_cost = 0
                if i > 0:
                    old_cost += n - ov[i-1]
                old_cost += n - ov[i]
                if i < len(ov) - 1:
                    old_cost += n - ov[i+1]
                
                new_cost = 0
                if i > 0:
                    new_cost += n - get_overlap(current_order[i-1], current_order[i+1])
                new_cost += n - get_overlap(current_order[i+1], current_order[i])
                if i < len(ov) - 1:
                    new_cost += n - get_overlap(current_order[i], current_order[i+2])
                
                delta = new_cost - old_cost
                
                if delta < 0:
                    current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
                    if i > 0:
                        ov[i-1] = get_overlap(current_order[i-1], current_order[i])
                    ov[i] = get_overlap(current_order[i], current_order[i+1])
                    if i < len(ov) - 1:
                        ov[i+1] = get_overlap(current_order[i+1], current_order[i+2])
                    current_len += delta
                    any_improved = True
                    improved = True
            
            if not any_improved:
                break
        
        # Phase 2: Random restarts with greedy + local search
        if time.time() < max_search_time - 20:
            # Try a few random greedy constructions
            for restart in range(3):
                if time.time() > max_search_time - 10:
                    break
                
                random_start = random.choice(all_perms)
                greedy_order, _ = greedy_construct(set(), random_start)
                greedy_len = calc_length(greedy_order)
                
                if greedy_len < current_len:
                    current_order = greedy_order
                    current_len = greedy_len
                
                # Local search on this
                improved_order, improved_len = local_search(current_order, max_iters=50000)
                if improved_len < current_len:
                    current_order = improved_order
                    current_len = improved_len
    
    # Reconstruct word from best order
    final_word = list(current_order[0])
    for i in range(1, len(current_order)):
        p = current_order[i]
        max_ov = 0
        for k in range(n-1, -1, -1):
            if k == 0:
                break
            if final_word[n-k:] == list(p[:k]):
                max_ov = k
                break
        final_word.extend(p[max_ov:])
    
    # Verify validity
    if len(final_word) >= n:
        seen_perms = set()
        for i in range(len(final_word) - n + 1):
            p = tuple(final_word[i:i+n])
            seen_perms.add(p)
        if len(seen_perms) < num_perms:
            # Invalid, fall back to standard
            return "".join(map(str, std_word))
    
    return "".join(map(str, final_word))
# EVOLVE-BLOCK-END
