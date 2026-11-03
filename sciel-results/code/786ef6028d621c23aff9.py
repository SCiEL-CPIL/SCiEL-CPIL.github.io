import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    total_words = q ** n
    
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute neighbor indices
    neighbor_indices = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = idx_to_tuple(i)
        idxs = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                idxs.append(j)
        neighbor_indices[i] = idxs
    
    # Build adjacency as bitsets (Python integers)
    # adj[i] is a bitset where bit j is set if j is in neighbor_indices[i]
    adj = [0] * total_words
    for i in range(total_words):
        mask = 0
        for j in neighbor_indices[i]:
            mask |= (1 << j)
        adj[i] = mask
    
    # Precompute: for each word w, which codeword indices cover it
    # cover_mask[w] = bitset of all codewords that cover w
    # But we need the reverse: for each codeword c, which words it covers = neighbor_indices[c]
    # We already have that.
    
    # Also precompute: for each word w, list of codewords that cover it
    # This is the transpose of the adjacency
    covered_by = [[] for _ in range(total_words)]
    for c in range(total_words):
        for w in neighbor_indices[c]:
            covered_by[w].append(c)
    
    def greedy_construct_bitset(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered_mask = (1 << total_words) - 1
        code = []
        
        while uncovered_mask:
            # Compute gain for each candidate: popcount(adj[i] & uncovered_mask)
            # Vectorized approach: for each i, count bits in adj[i] & uncovered_mask
            best_gain = -1
            best_candidates = []
            
            for i in range(total_words):
                gain = bin(adj[i] & uncovered_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if not best_candidates or best_gain <= 0:
                break
            
            chosen = random.choice(best_candidates)
            code.append(chosen)
            uncovered_mask &= ~adj[chosen]
        
        return code
    
    def reduce_code_bitset(code):
        counts = [0] * total_words
        for c in code:
            for w in neighbor_indices[c]:
                counts[w] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                for w in neighbor_indices[c]:
                    counts[w] -= 1
                
                if any(counts[w] == 0 for w in neighbor_indices[c]):
                    for w in neighbor_indices[c]:
                        counts[w] += 1
                    new_code.append(c)
                else:
                    changed = True
            
            code = new_code
        return code
    
    def local_search_improve_bitset(code, deadline):
        code = list(code)
        counts = [0] * total_words
        for c in code:
            for w in neighbor_indices[c]:
                counts[w] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                for w in neighbor_indices[c_remove]:
                    counts[w] -= 1
                
                # Check if any word becomes uncovered
                uncovered = [w for w in neighbor_indices[c_remove] if counts[w] == 0]
                
                if not uncovered:
                    code.pop(i)
                    improved = True
                    break
                else:
                    for w in neighbor_indices[c_remove]:
                        counts[w] += 1
                    
                    # Try to find a replacement that covers all uncovered words
                    # An uncovered word w must be in neighbor_indices[replacement]
                    # So replacement must be in covered_by[w] for all w in uncovered
                    candidates = None
                    for w in uncovered:
                        if candidates is None:
                            candidates = set(covered_by[w])
                        else:
                            candidates &= set(covered_by[w])
                    
                    if candidates:
                        # Pick a candidate that maximizes coverage of currently uncovered words
                        best_cand = None
                        best_gain = -1
                        for cand in candidates:
                            if cand == c_remove:
                                continue
                            # Count how many of the uncovered words this candidate covers
                            gain = sum(1 for w in uncovered if w in neighbor_indices[cand])
                            if gain > best_gain:
                                best_gain = gain
                                best_cand = cand
                        
                        if best_cand is not None and best_gain >= len(uncovered):
                            # Remove c_remove, add best_cand
                            code[i] = best_cand
                            for w in neighbor_indices[c_remove]:
                                counts[w] -= 1
                            for w in neighbor_indices[best_cand]:
                                counts[w] += 1
                            improved = True
                            break
        
        return code
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline:
        code = greedy_construct_bitset(seed)
        code = reduce_code_bitset(code)
        
        if len(code) < best_size or (best_code is None):
            if time.time() < deadline - 5:
                code = local_search_improve_bitset(code, deadline - 2)
                code = reduce_code_bitset(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
        
        seed += 1
    
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
        
    return result
