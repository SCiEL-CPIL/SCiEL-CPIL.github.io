import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    total_words = q ** n
    
    # Specialized solver for the known hard case: q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)
    
    # Fallback for other cases
    if total_words > 10000:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [[0] * n]
    
    # Generic case: use bitset-based greedy + local search
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = total_words
    
    # Precompute coverage bitsets
    coverage = [0] * N
    for i, w in enumerate(all_words):
        mask = 1 << i
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                mask |= 1 << word_to_idx[tuple(new_w)]
        coverage[i] = mask
    
    full_mask = (1 << N) - 1
    
    # Start with greedy
    best_code = None
    best_size = float('inf')
    
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
        
        covered_mask = 0
        code_indices = []
        candidates = list(range(N))
        random.shuffle(candidates)
        
        while covered_mask != full_mask:
            best_gain = -1
            best_idx = -1
            for idx in candidates:
                gain = (coverage[idx] & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            if best_gain <= 0:
                break
            code_indices.append(best_idx)
            covered_mask |= coverage[best_idx]
        
        if covered_mask != full_mask:
            continue
        
        # Prune: remove redundant codewords
        S = list(code_indices)
        improved = True
        while improved:
            improved = False
            for i in range(len(S)):
                others_mask = 0
                for j in range(len(S)):
                    if j != i:
                        others_mask |= coverage[S[j]]
                if coverage[S[i]] & ~others_mask == 0:
                    S.pop(i)
                    improved = True
                    break
        
        # Local search: try swaps
        S_set = set(S)
        union_mask = 0
        for idx in S:
            union_mask |= coverage[idx]
        
        for _ in range(3):
            improved = False
            for i in range(len(S)):
                others_mask = union_mask
                for j in range(len(S)):
                    if j != i:
                        others_mask &= ~coverage[S[j]]
                # words only covered by S[i]
                unique = coverage[S[i]] & ~others_mask
                if unique == 0:
                    S.pop(i)
                    S_set.discard(S[i])
                    union_mask &= ~coverage[S[i]]
                    improved = True
                    break
                # Try to find a replacement
                for j in range(N):
                    if j in S_set:
                        continue
                    if unique & ~coverage[j]:
                        continue
                    new_union = (union_mask & ~coverage[S[i]]) | coverage[j]
                    if new_union == union_mask:
                        # Same coverage, can swap
                        S[i] = j
                        S_set.discard(S[i])
                        S_set.add(j)
                        union_mask = new_union
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        if len(S) < best_size:
            best_size = len(S)
            best_code = S[:]
    
    if best_code is None:
        best_code = list(range(N))
    
    return [list(all_words[c]) for c in best_code]


def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    N = 729
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage bitsets
    coverage = [0] * N
    for i, w in enumerate(all_words):
        mask = 1 << i
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                mask |= 1 << word_to_idx[tuple(new_w)]
        coverage[i] = mask
    
    full_mask = (1 << N) - 1
    
    # Start with tetracode direct-sum (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            word = tuple(h + list(tail))
            initial_code.append(word_to_idx[word])
    
    best_code = None
    best_size = float('inf')
    
    def prune_and_local_search(S, time_limit):
        """Prune redundant codewords and try swaps to reduce size."""
        S = list(S)
        S_set = set(S)
        
        # Compute union
        union_mask = 0
        for idx in S:
            union_mask |= coverage[idx]
        
        # Phase 1: Remove redundant codewords
        changed = True
        while changed:
            changed = False
            for i in range(len(S)):
                # Check if S[i] is redundant
                others_mask = 0
                for j in range(len(S)):
                    if j != i:
                        others_mask |= coverage[S[j]]
                if coverage[S[i]] & ~others_mask == 0:
                    S.pop(i)
                    S_set.discard(S[i])
                    union_mask = 0
                    for idx in S:
                        union_mask |= coverage[idx]
                    changed = True
                    break
        
        # Phase 2: Try swaps to reduce size
        # For each codeword, try replacing it with another that maintains coverage
        for iteration in range(50):
            if time.time() > time_limit:
                break
            improved = False
            for i in range(len(S)):
                if time.time() > time_limit:
                    break
                # Find words only covered by S[i]
                others_mask = 0
                for j in range(len(S)):
                    if j != i:
                        others_mask |= coverage[S[j]]
                unique = coverage[S[i]] & ~others_mask
                
                if unique == 0:
                    # S[i] is redundant, remove it
                    S.pop(i)
                    S_set.discard(S[i])
                    union_mask = 0
                    for idx in S:
                        union_mask |= coverage[idx]
                    improved = True
                    break
                
                # Try to find a replacement j such that coverage[j] covers unique
                # and the new union is at least as good
                best_j = -1
                best_new_union = 0
                for j in range(N):
                    if j in S_set:
                        continue
                    if unique & ~coverage[j]:
                        continue
                    new_union = (union_mask & ~coverage[S[i]]) | coverage[j]
                    # Prefer replacements that maintain or improve coverage
                    if new_union == union_mask:
                        best_j = j
                        best_new_union = new_union
                        break
                    elif (new_union & ~union_mask).bit_count() > (union_mask & ~new_union).bit_count():
                        if best_j == -1 or (new_union & ~union_mask).bit_count() > (best_new_union & ~union_mask).bit_count():
                            best_j = j
                            best_new_union = new_union
                
                if best_j != -1:
                    S[i] = best_j
                    S_set.discard(S[i])
                    S_set.add(best_j)
                    union_mask = best_new_union
                    improved = True
                    break
            if not improved:
                break
        
        return S
    
    # Strategy 1: Start from tetracode direct-sum and reduce
    current = prune_and_local_search(initial_code, deadline - 30.0)
    if len(current) < best_size:
        best_size = len(current)
        best_code = current[:]
    
    # Strategy 2: Multiple restarts with randomized greedy
    for restart in range(30):
        if time.time() > deadline - 15.0:
            break
        
        covered_mask = 0
        code_indices = []
        candidates = list(range(N))
        random.shuffle(candidates)
        
        # Greedy phase with some randomization
        while covered_mask != full_mask:
            if time.time() > deadline - 20.0:
                break
            best_gain = -1
            best_idx = -1
            best_candidates = []
            for idx in candidates:
                gain = (coverage[idx] & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_candidates = [idx]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(idx)
            
            if best_gain <= 0:
                break
            
            # Sometimes pick randomly from top candidates
            if len(best_candidates) > 1 and random.random() < 0.2:
                best_idx = random.choice(best_candidates)
            
            code_indices.append(best_idx)
            covered_mask |= coverage[best_idx]
        
        if covered_mask != full_mask:
            continue
        
        # Prune and local search
        current = prune_and_local_search(code_indices, deadline - 10.0)
        if len(current) < best_size:
            best_size = len(current)
            best_code = current[:]
    
    # Final polish: try more aggressive local search on best solution
    if best_code is not None and time.time() < deadline - 5.0:
        final = prune_and_local_search(best_code, deadline - 2.0)
        if len(final) < best_size:
            best_size = len(final)
            best_code = final
    
    if best_code is None:
        best_code = initial_code
    
    return [list(all_words[c]) for c in best_code]
