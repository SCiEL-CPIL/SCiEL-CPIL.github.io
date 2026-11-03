"""Ternary covering code K_3(8,1) via criticality greedy + strict 2-opt."""
import time
import random
import numpy as np
from itertools import product

def _precompute_neighbors(q, n, radius):
    """Precompute neighbor lists for all words in {0..q-1}^n within Hamming distance radius.
    Returns a list of lists: neighbors[i] is the list of indices j such that dist(i,j) <= radius.
    """
    total = q ** n
    # Precompute word representations as tuples for distance computation
    words = []
    for idx in range(total):
        w = []
        t = idx
        for _ in range(n):
            w.append(t % q)
            t //= q
        w.reverse()
        words.append(tuple(w))
    
    # Build neighbor list
    neighbors = [None] * total
    for i in range(total):
        ni = []
        wi = words[i]
        for j in range(total):
            # Compute Hamming distance
            d = 0
            for k in range(n):
                if wi[k] != words[j][k]:
                    d += 1
                    if d > radius:
                        break
            if d <= radius:
                ni.append(j)
        neighbors[i] = ni
    return neighbors, words

def _word_to_idx(word, q, n):
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx

def _idx_to_word(idx, q, n):
    w = [0] * n
    for i in range(n - 1, -1, -1):
        w[i] = idx % q
        idx //= q
    return tuple(w)

def _criticality_greedy(q, n, radius, neighbors, max_time=30.0):
    """
    Greedy set cover that prioritizes words with highest criticality.
    Criticality = number of uncovered words that would be covered by this candidate.
    We pick the candidate that covers the most uncovered words.
    """
    start = time.time()
    total = q ** n
    
    # covered[i] = True if word i is covered
    covered = np.zeros(total, dtype=bool)
    # cover_count[i] = how many codewords cover word i
    cover_count = np.zeros(total, dtype=np.int32)
    
    code = []  # list of indices
    
    # For efficiency, maintain a list of uncovered words
    uncovered = set(range(total))
    
    # Precompute: for each candidate word c, which words it covers = neighbors[c]
    
    while uncovered:
        if time.time() - start > max_time:
            break
        
        # Find the candidate that covers the most uncovered words
        best_c = -1
        best_count = -1
        best_covered_words = None
        
        # We can't check all candidates every iteration (too slow)
        # Instead, pick a random uncovered word u, then find candidates that cover u
        # and pick the one that covers the most uncovered words among those.
        
        # Pick a random uncovered word
        u = random.choice(list(uncovered))
        
        # Find all candidates c such that u in neighbors[c]
        # i.e., c in neighbors[u] (symmetric)
        candidates = neighbors[u]
        
        # Among these candidates, find the one that covers the most uncovered words
        for c in candidates:
            if c in code:
                continue
            # Count how many uncovered words c covers
            cnt = 0
            covered_by_c = []
            for nb in neighbors[c]:
                if not covered[nb]:
                    cnt += 1
                    covered_by_c.append(nb)
            if cnt > best_count:
                best_count = cnt
                best_c = c
                best_covered_words = covered_by_c
        
        if best_c == -1:
            # Shouldn't happen, but fallback: pick random uncovered
            best_c = u
            best_covered_words = [u]
        
        # Add best_c to code
        code.append(best_c)
        for nb in neighbors[best_c]:
            if not covered[nb]:
                uncovered.discard(nb)
            covered[nb] = True
            cover_count[nb] += 1
    
    return code, cover_count, covered

def _strict_2opt(code, q, n, radius, neighbors, cover_count, max_time=100.0):
    """
    Strict 2-opt local search:
    - Try removals: remove a codeword if all its neighbors have cover_count > 1
    - Try swaps: replace codeword c_i with c_new if:
      (a) all neighbors of c_i with cover_count == 1 are covered by c_new, AND
      (b) the swap doesn't reduce the minimum coverage multiplicity among covered words
    """
    start = time.time()
    code = list(code)
    cover_count = cover_count.copy()
    total = q ** n
    
    best_code = code[:]
    best_len = len(code)
    best_min_cover = int(np.min(cover_count))
    
    def current_min_cover():
        # Only consider covered words (cover_count > 0)
        positive = cover_count[cover_count > 0]
        if len(positive) == 0:
            return 0
        return int(np.min(positive))
    
    no_improve = 0
    max_no_improve = 30
    
    while time.time() - start < max_time:
        improved = False
        
        # Phase 1: Try removals
        for i in range(len(code)):
            if time.time() - start > max_time:
                return best_code
            
            c = code[i]
            # Check if all neighbors of c have cover_count > 1
            can_remove = True
            for nb in neighbors[c]:
                if cover_count[nb] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove c
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                code.pop(i)
                improved = True
                
                if len(code) < best_len:
                    best_len = len(code)
                    best_code = code[:]
                    best_min_cover = current_min_cover()
                no_improve = 0
                break
        
        if improved:
            continue
        
        # Phase 2: Try swaps
        # For each codeword, try replacing with a random candidate
        for _ in range(200):  # Try 200 random swaps per pass
            if time.time() - start > max_time:
                return best_code
            
            if len(code) == 0:
                break
                
            i = random.randint(0, len(code) - 1)
            c_i = code[i]
            
            # Generate random candidate
            c_new = random.randint(0, total - 1)
            if c_new in code:
                continue
            
            # Check validity of swap
            # Step 1: All neighbors of c_i with cover_count == 1 must be in neighbors[c_new]
            set_new = set(neighbors[c_new])
            valid = True
            for nb in neighbors[c_i]:
                if cover_count[nb] == 1:
                    if nb not in set_new:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Step 2: Check that the swap doesn't reduce min coverage
            # Compute new cover_count after swap
            # We need to check: for all words, new_count >= 1 (always true if current is valid and step 1 passes)
            # And we want new_min_cover >= current_min_cover (or at least not worse)
            
            # Quick check: compute the minimum coverage after swap
            # Words affected: neighbors of c_i (decrease by 1) and neighbors of c_new (increase by 1)
            # Words in both: net change 0
            
            # Find min coverage among affected words
            min_after = 10**9
            for nb in neighbors[c_i]:
                new_cnt = cover_count[nb] - 1
                if nb in set_new:
                    new_cnt += 1
                if new_cnt < min_after:
                    min_after = new_cnt
            for nb in neighbors[c_new]:
                if nb not in set(neighbors[c_i]):
                    new_cnt = cover_count[nb] + 1
                    if new_cnt < min_after:
                        min_after = new_cnt
            
            # Also check unaffected words have current cover_count
            # The global min is min(min_after, current_min_cover for unaffected)
            # But we only care if min_after < best_min_cover
            
            if min_after < best_min_cover:
                continue
            
            # Perform swap
            for nb in neighbors[c_i]:
                cover_count[nb] -= 1
            for nb in neighbors[c_new]:
                cover_count[nb] += 1
            code[i] = c_new
            
            improved = True
            no_improve = 0
            
            # Update best if improved
            cur_min = current_min_cover()
            if len(code) < best_len or (len(code) == best_len and cur_min > best_min_cover):
                best_len = len(code)
                best_code = code[:]
                best_min_cover = cur_min
            break
        
        if not improved:
            no_improve += 1
            if no_improve > max_no_improve:
                break
    
    return best_code

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Precompute neighbors
    neighbors, words = _precompute_neighbors(q, n, radius)
    
    # Criticality greedy initialization
    code, cover_count, covered = _criticality_greedy(q, n, radius, neighbors, max_time=20.0)
    
    # Strict 2-opt local search
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        code = _strict_2opt(code, q, n, radius, neighbors, cover_count, max_time=remaining_time - 5.0)
    
    # Convert to list of lists
    result = [list(words[c]) for c in code]
    
    return result
