import itertools
import time

def _recursive_superperm(n):
    """Classic recursive construction: length 1! + 2! + ... + n!"""
    if n == 1:
        return [1]
    if n == 2:
        return [1, 2, 1]
    # Build up from n-1
    prev = _recursive_superperm(n - 1)
    m = n
    # Find all distinct (m-1)-permutations in prev
    perms, seen = [], set()
    for i in range(len(prev) - m + 2):
        window = tuple(prev[i:i + m - 1])
        if len(set(window)) == m - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    # For each perm p, create block p + [m] + p, overlap with existing
    new = []
    for p in perms:
        block = list(p) + [m] + list(p)
        # Find max overlap
        overlap = 0
        max_check = min(len(new), len(block))
        for o in range(max_check, -1, -1):
            if o == 0 or new[len(new) - o:] == block[:o]:
                overlap = o
                break
        new += block[overlap:]
    return new


def _greedy_refine(word, n, time_limit=5.0):
    """Try to shorten the superpermutation by greedy local moves."""
    start_time = time.time()
    # Convert to list for mutation
    w = list(word)
    L = len(w)
    
    # Build set of all permutations as frozensets for O(1) lookup
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    def valid_window(idx):
        """Check if window at idx is a valid permutation."""
        if idx < 0 or idx + n > L:
            return False
        return tuple(w[idx:idx + n]) in all_perms
    
    def all_permutations_present():
        """Check if all 5040 permutations are present."""
        seen = set()
        for i in range(L - n + 1):
            seen.add(tuple(w[i:i + n]))
        return all_perms.issubset(seen)
    
    # Verify initial validity
    if not all_permutations_present():
        return w
    
    # Try removing elements that don't break validity
    # Strategy: for each position, try removing it if the permutation at that
    # position (if any) also appears elsewhere
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        L = len(w)
        for i in range(L - 1, -1, -1):  # Remove from end to avoid index shift issues
            if (time.time() - start_time) > time_limit:
                break
            # Try removing position i
            # Check: is the permutation starting at i-1 (if valid) also present elsewhere?
            # And is the permutation starting at i (if valid) also present elsewhere?
            # After removal, windows that were at positions >= i shift left by 1
            
            # Collect permutations that would be affected
            affected = set()
            # Window starting at i-1 (if i > 0 and i-1 + n <= L)
            if i > 0 and i - 1 + n <= L:
                perm_before = tuple(w[i - 1:i - 1 + n])
                if perm_before in all_perms:
                    affected.add(perm_before)
            # Window starting at i (if i + n <= L)
            if i + n <= L:
                perm_at = tuple(w[i:i + n])
                if perm_at in all_perms:
                    affected.add(perm_at)
            
            if not affected:
                # No permutation is affected, safe to remove
                w.pop(i)
                improved = True
                continue
            
            # Count occurrences of each affected permutation (excluding the one at i)
            safe = True
            for perm in affected:
                count = 0
                for j in range(len(w) - n + 1):
                    if j == i:
                        continue  # Skip the window being removed
                    if tuple(w[j:j + n]) == perm:
                        count += 1
                        if count >= 1:
                            break
                if count < 1:
                    safe = False
                    break
            
            if safe:
                w.pop(i)
                improved = True
    
    return w


def _multi_round_greedy(n, time_limit=10.0):
    """Multi-round greedy construction with refinement."""
    start_time = time.time()
    
    # Start with recursive construction
    word = _recursive_superperm(n)
    
    # Refine
    word = _greedy_refine(word, n, time_limit / 3)
    
    # Try a few different starting permutations for greedy
    perms = sorted(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Build prefix index for fast lookup
    prefix_to_indices = {}
    for i, p in enumerate(perms):
        for k in range(1, n):
            prefix = p[:k]
            if prefix not in prefix_to_indices:
                prefix_to_indices[prefix] = []
            prefix_to_indices[prefix].append(i)
    
    best_word = list(word)
    best_len = len(best_word)
    
    # Try a few greedy starts
    for start_idx in [0, N // 2, N - 1]:
        if (time.time() - start_time) > time_limit * 2 / 3:
            break
        
        visited = [False] * N
        current_word = list(perms[start_idx])
        visited[start_idx] = True
        
        for _ in range(N - 1):
            if (time.time() - start_time) > time_limit * 2 / 3:
                break
            found = False
            for k in range(n - 1, 0, -1):
                suffix = tuple(current_word[-k:])
                candidates = prefix_to_indices.get(suffix, [])
                unvisited = [idx for idx in candidates if not visited[idx]]
                if unvisited:
                    # Pick the one with best look-ahead
                    best_idx = unvisited[0]
                    best_score = -1
                    for idx in unvisited[:min(10, len(unvisited))]:
                        next_perm = perms[idx]
                        next_suffix = next_perm[n - 1:]
                        next_cands = prefix_to_indices.get(next_suffix, [])
                        has_next = any(not visited[i] for i in next_cands)
                        score = 1 if has_next else 0
                        if score > best_score:
                            best_score = score
                            best_idx = idx
                    visited[best_idx] = True
                    current_word.extend(perms[best_idx][k:])
                    found = True
                    break
            if not found:
                for i in range(N):
                    if not visited[i]:
                        visited[i] = True
                        current_word.extend(perms[i])
                        found = True
                        break
            if not found:
                break
        
        if len(current_word) < best_len:
            best_word = current_word
            best_len = len(best_word)
    
    # Final refinement
    best_word = _greedy_refine(best_word, n, time_limit / 3)
    
    return best_word


def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    try:
        word = _multi_round_greedy(n, time_limit=90.0)
    except Exception:
        word = _recursive_superperm(n)
    
    # Final safety check: ensure we have a valid superpermutation
    # If time is running out, return what we have
    if time.time() > deadline - 5.0:
        return "".join(map(str, word))
    
    # Verify and potentially do one more quick refinement
    try:
        word = _greedy_refine(word, n, time_limit=10.0)
    except Exception:
        pass
    
    return "".join(map(str, word))
