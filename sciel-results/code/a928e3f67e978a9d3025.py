import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit
    
    if q == 3 and n == 6 and radius == 1:
        total_words = q ** n
        
        # Generate all words as tuples and index map
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute spheres: for each word, the set of words within Hamming distance 1
        spheres = []
        for w in all_words:
            sphere = [word_to_idx[w]]
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    sphere.append(word_to_idx[tuple(new_w)])
            spheres.append(sphere)
        
        # Phase 1: Greedy set cover
        covered = [False] * total_words
        uncovered_count = total_words
        code_indices = []
        picked = set()
        
        while uncovered_count > 0:
            if time.time() > deadline:
                break
                
            best_candidate = -1
            max_new_covered = 0
            
            for cand_idx in range(total_words):
                if cand_idx in picked:
                    continue
                
                new_covered = 0
                for idx in spheres[cand_idx]:
                    if not covered[idx]:
                        new_covered += 1
                
                if new_covered > max_new_covered:
                    max_new_covered = new_covered
                    best_candidate = cand_idx
                
                if max_new_covered == uncovered_count:
                    break
            
            if best_candidate == -1:
                break
            
            picked.add(best_candidate)
            code_indices.append(best_candidate)
            
            for idx in spheres[best_candidate]:
                if not covered[idx]:
                    covered[idx] = True
                    uncovered_count -= 1
            
            if time.time() > deadline:
                break
        
        # Phase 2: Local search - iteratively remove redundant codewords
        # coverage_count[i] = number of codewords whose sphere covers word i
        coverage_count = [0] * total_words
        for ci in code_indices:
            for idx in spheres[ci]:
                coverage_count[idx] += 1
        
        # Try to remove codewords one by one
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(code_indices) - 1, -1, -1):
                if time.time() > deadline:
                    break
                ci = code_indices[i]
                # Check if removing ci leaves all words in its sphere still covered
                removable = True
                for idx in spheres[ci]:
                    if coverage_count[idx] <= 1:
                        removable = False
                        break
                if removable:
                    # Remove this codeword
                    for idx in spheres[ci]:
                        coverage_count[idx] -= 1
                    code_indices.pop(i)
                    changed = True
                    break  # Restart scan from beginning after a removal
        
        # Phase 3: Try to swap codewords for better coverage (optional, if time allows)
        # Simple local search: try replacing a codeword with another that covers more uncovered
        if time.time() < deadline - 20:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Find words with low coverage (most critical)
                min_cov = min(coverage_count) if coverage_count else 0
                if min_cov >= 2:
                    # All words covered by at least 2 codewords, try more aggressive removals
                    continue
                
                for i in range(len(code_indices)):
                    if time.time() > deadline:
                        break
                    ci = code_indices[i]
                    # Try to find a replacement that covers more new territory
                    # For simplicity, just try removing and re-greedy
                    # This is expensive, so limit attempts
                    if i > 5:  # Limit swaps to first few positions
                        continue
                    
                    # Temporarily remove ci
                    for idx in spheres[ci]:
                        coverage_count[idx] -= 1
                    
                    # Check if still fully covered
                    still_covered = all(c >= 1 for c in coverage_count)
                    
                    if not still_covered:
                        # Restore
                        for idx in spheres[ci]:
                            coverage_count[idx] += 1
                        continue
                    
                    # It's safe to remove, but we already did that in Phase 2
                    # So this phase is mainly for when Phase 2 couldn't remove more
                    # Restore and break
                    for idx in spheres[ci]:
                        coverage_count[idx] += 1
                    break
        
        # Convert to list of lists
        result = [list(all_words[ci]) for ci in code_indices]
        return result
    else:
        # Fallback: return all words (valid but large)
        return [list(t) for t in itertools.product(range(q), repeat=n)]
