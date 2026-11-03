import itertools
import time
import random

def construct(n):
    if n == 1:
        return "1"
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    full_mask = (1 << N) - 1
    
    # Map from tuple prefix (length n-1) to list of permutation indices
    # Note: In the original, prefix_to_indices used p[:k]. 
    # For overlap, we need suffix of current == prefix of next.
    # So we map prefix (length n-1) to indices.
    prefix_map = {}
    for i, p in enumerate(perms):
        prefix = tuple(p[:n-1])
        if prefix not in prefix_map:
            prefix_map[prefix] = []
        prefix_map[prefix].append(i)
    
    # Precompute suffixes for each permutation (for next state)
    perm_suffixes = [tuple(p[1:]) for p in perms]
    
    # Fallback Greedy (Original Logic)
    def run_greedy():
        perm_to_idx = {p: i for i, p in enumerate(perms)}
        visited = set()
        current_word = list(perms[0])
        visited.add(tuple(perms[0]))
        
        last_occ = {i: -1 for i in range(1, n+1)}
        for idx, sym in enumerate(current_word):
            last_occ[sym] = idx
            
        for _ in range(N - 1):
            found = False
            if len(current_word) >= n - 1:
                suffix = tuple(current_word[-(n-1):])
                symbols = list(range(1, n + 1))
                symbols.sort(key=lambda x: last_occ[x])
                
                for s in symbols:
                    candidate_window = suffix + (s,)
                    if candidate_window in perm_to_idx:
                        if candidate_window not in visited:
                            current_word.append(s)
                            visited.add(candidate_window)
                            last_occ[s] = len(current_word) - 1
                            found = True
                            break
            if not found:
                # Fallback to greedy overlap
                for k in range(n - 1, 0, -1):
                    suffix = tuple(current_word[-k:])
                    candidates = prefix_map.get(suffix, [])
                    next_idx = -1
                    for idx in candidates:
                        if tuple(perms[idx]) not in visited:
                            next_idx = idx
                            break
                    if next_idx != -1:
                        next_perm = perms[next_idx]
                        current_word.extend(next_perm[k:])
                        visited.add(tuple(next_perm))
                        appended = next_perm[k:]
                        start_idx = len(current_word) - len(appended)
                        for offset, sym in enumerate(appended):
                            last_occ[sym] = start_idx + offset
                        found = True
                        break
                if not found:
                    for i in range(N):
                        if tuple(perms[i]) not in visited:
                            current_word.extend(perms[i])
                            visited.add(tuple(perms[i]))
                            for offset, sym in enumerate(perms[i]):
                                last_occ[sym] = len(current_word) - n + offset
                            break
        return "".join(map(str, current_word))

    # Try Beam Search
    K = 200
    start_time = time.time()
    time_limit = 110.0
    best_word = None
    best_len = N * N
    
    # Run a few beam searches with different starting points
    for start_idx in range(min(N, 5)):
        if time.time() - start_time > time_limit - 10:
            break
        
        init_perm = perms[start_idx]
        init_mask = 1 << start_idx
        # The state needs the last n-1 symbols to find the next prefix.
        # The last n-1 symbols of init_perm are init_perm[1:]
        init_suffix = tuple(init_perm[1:])
        init_word = list(init_perm)
        
        beam = [(init_mask, init_suffix, init_word)]
        
        for step in range(N - 1):
            if time.time() - start_time > time_limit - 5:
                break
                
            new_beam = []
            
            for mask, suffix, word in beam:
                # Find permutations that start with 'suffix'
                # Wait, the overlap is: last n-1 of current == first n-1 of next.
                # So we look for perms where prefix == suffix.
                candidates = prefix_map.get(suffix, [])
                
                for idx in candidates:
                    if not (mask & (1 << idx)):
                        new_mask = mask | (1 << idx)
                        new_suffix = perm_suffixes[idx]
                        # Append the last symbol of the new permutation
                        # The new permutation is perms[idx].
                        # The overlap is n-1 symbols.
                        # So we add perms[idx][n-1]
                        new_word = list(word)
                        new_word.append(perms[idx][n-1])
                        
                        new_beam.append((new_mask, new_suffix, new_word))
                        
            if not new_beam:
                break
                
            if len(new_beam) > K:
                # Heuristic: Score by number of unvisited neighbors
                def score_state(state):
                    m, s, w = state
                    cands = prefix_map.get(s, [])
                    count = 0
                    for idx in cands:
                        if not (m & (1 << idx)):
                            count += 1
                    return count

                # Sort by score descending
                # To avoid O(K^2) or high constant, we just sort.
                new_beam.sort(key=lambda x: score_state(x), reverse=True)
                new_beam = new_beam[:K]
                
            beam = new_beam
            
            # Check for completion
            for mask, suffix, word in beam:
                if mask == full_mask:
                    if len(word) < best_len:
                        best_len = len(word)
                        best_word = word
                    break
            
            if best_word and best_len <= 5905:
                break

    if best_word:
        return "".join(map(str, best_word))
    
    # Fallback
    return run_greedy()
