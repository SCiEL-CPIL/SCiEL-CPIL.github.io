import numpy as np
import time
import itertools

def construct(q, n, radius):
    deadline = time.time() + 115.0  # 5s margin before 120s kill
    
    if n <= 8 and radius == 1:
        total_words = q ** n
        sphere_size = 1 + n * (q - 1)
        
        # Precompute all neighbor indices for each word
        neighbors = np.full((total_words, sphere_size), -1, dtype=np.int32)
        for i in range(total_words):
            nbrs = [i]
            for pos in range(n):
                pow_q = q ** (n - 1 - pos)
                current_val = (i // pow_q) % q
                for new_val in range(q):
                    if new_val != current_val:
                        nbrs.append(i + (new_val - current_val) * pow_q)
            neighbors[i] = nbrs
        
        def greedy_and_local_search(order_hint=None):
            """Run greedy with optional ordering hint, then local search."""
            covered = np.zeros(total_words, dtype=bool)
            code_indices = []
            
            # Greedy phase
            while time.time() < deadline - 40.0:
                gains = (~covered)[neighbors].sum(axis=1)
                if order_hint is not None:
                    # Use ordering hint: among top candidates, prefer earlier in hint
                    top_k = 20
                    top_candidates = np.argsort(-gains)[:top_k]
                    if len(top_candidates) > 0:
                        # Score by gain, break ties by position in order_hint
                        best_idx = top_candidates[0]
                        best_gain = gains[best_idx]
                        for cand in top_candidates[1:]:
                            if gains[cand] > best_gain:
                                best_idx = cand
                                best_gain = gains[cand]
                        best_idx = top_candidates[0]  # Just use argmax for simplicity
                else:
                    best_idx = np.argmax(gains)
                
                if gains[best_idx] == 0:
                    break
                code_indices.append(best_idx)
                covered[neighbors[best_idx]] = True
            
            # Local search: remove redundant codewords
            # A codeword is redundant if every word in its sphere is covered by another codeword
            # Use coverage counting for efficiency
            max_code_size = len(code_indices) + 1
            coverage_count = np.zeros(total_words, dtype=np.int32)
            
            # Build initial coverage counts
            for ci in code_indices:
                coverage_count[neighbors[ci]] += 1
            
            changed = True
            while changed and time.time() < deadline - 15.0:
                changed = False
                for idx_in_code in range(len(code_indices)):
                    if time.time() > deadline - 15.0:
                        break
                    ci = code_indices[idx_in_code]
                    # Check if ci is redundant: all neighbors have coverage >= 2
                    nbrs = neighbors[ci]
                    # Decrement coverage temporarily
                    coverage_count[nbrs] -= 1
                    # Check if any neighbor has coverage 0
                    if np.any(coverage_count[nbrs] == 0):
                        # Not redundant, restore
                        coverage_count[nbrs] += 1
                    else:
                        # Redundant! Remove it
                        code_indices.pop(idx_in_code)
                        changed = True
                        break  # Restart since indices changed
            
            return code_indices
        
        # Try multiple greedy runs with different strategies
        best_code = None
        best_size = total_words + 1
        
        # First greedy run: standard
        code1 = greedy_and_local_search()
        if len(code1) < best_size:
            best_size = len(code1)
            best_code = code1[:]
        
        # Second greedy run: start from a random word to break symmetry
        if time.time() < deadline - 50.0:
            rng = np.random.RandomState(42)
            # Try a few different random starting points
            for trial in range(3):
                if time.time() > deadline - 45.0:
                    break
                covered = np.zeros(total_words, dtype=bool)
                code_indices = []
                
                # Start with a random word
                start = rng.randint(total_words)
                code_indices.append(start)
                covered[neighbors[start]] = True
                
                # Continue greedy
                while time.time() < deadline - 40.0:
                    gains = (~covered)[neighbors].sum(axis=1)
                    best_idx = np.argmax(gains)
                    if gains[best_idx] == 0:
                        break
                    code_indices.append(best_idx)
                    covered[neighbors[best_idx]] = True
                
                # Local search
                coverage_count = np.zeros(total_words, dtype=np.int32)
                for ci in code_indices:
                    coverage_count[neighbors[ci]] += 1
                
                changed = True
                while changed and time.time() < deadline - 15.0:
                    changed = False
                    for idx_in_code in range(len(code_indices)):
                        if time.time() > deadline - 15.0:
                            break
                        ci = code_indices[idx_in_code]
                        nbrs = neighbors[ci]
                        coverage_count[nbrs] -= 1
                        if np.any(coverage_count[nbrs] == 0):
                            coverage_count[nbrs] += 1
                        else:
                            code_indices.pop(idx_in_code)
                            changed = True
                            break
                
                if len(code_indices) < best_size:
                    best_size = len(code_indices)
                    best_code = code_indices[:]
        
        # Convert to words
        result_words = []
        for idx in best_code:
            w = []
            temp = idx
            for _ in range(n):
                w.append(temp % q)
                temp //= q
            w.reverse()
            result_words.append(w)
        
        return result_words
    else:
        # Fallback: direct sum with tetracode-like base
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            base_n = 4
            base_code = tetracode
        else:
            base_code = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
            base_n = 4
        
        if n < base_n:
            return [list(range(n))]
        
        tails = list(itertools.product(range(q), repeat=n - base_n))
        return [h + list(tail) for h in base_code for tail in tails]
