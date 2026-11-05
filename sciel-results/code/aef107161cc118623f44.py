import itertools
import time
import sys

def _standard_superperm(n):
    """Generate standard superpermutation for n using recursive construction."""
    if n == 1:
        return [1]
    if n == 2:
        return [1, 2, 1]
    if n == 3:
        return [1, 2, 3, 1, 2, 1, 3, 2, 1]
    
    prev = _standard_superperm(n - 1)
    
    seen = set()
    order = []
    for i in range(len(prev) - (n - 1) + 1):
        window = tuple(prev[i:i + n - 1])
        if window not in seen:
            seen.add(window)
            if sorted(window) == list(range(1, n)):
                order.append(window)
    
    new = []
    for perm in order:
        block = list(perm) + [n] + list(perm)
        max_ov = min(len(new), len(block))
        overlap = 0
        for ov in range(max_ov, -1, -1):
            if ov == 0 or new[len(new) - ov:] == block[:ov]:
                overlap = ov
                break
        new.extend(block[overlap:])
    
    return new

def _verify_superpermutation(word, n):
    """Check if word contains all permutations of 1..n as consecutive subsequences."""
    all_perms = set(itertools.permutations(range(1, n + 1)))
    found = set()
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i + n])
        if p in all_perms:
            found.add(p)
        if len(found) == len(all_perms):
            return True
    return len(found) == len(all_perms)

def _construct_from_order(order, n):
    """
    Construct superpermutation for n from the given order of (n-1)-permutations.
    Returns the word as a list of ints.
    """
    new = []
    for perm in order:
        block = list(perm) + [n] + list(perm)
        max_ov = min(len(new), len(block))
        overlap = 0
        for ov in range(max_ov, -1, -1):
            if ov == 0 or new[len(new) - ov:] == block[:ov]:
                overlap = ov
                break
        new.extend(block[overlap:])
    return new

def _extract_order_from_word(word, n):
    """Extract the order of (n-1)-permutations from a superpermutation for n."""
    seen = set()
    order = []
    for i in range(len(word) - (n - 1) + 1):
        window = tuple(word[i:i + n - 1])
        if window not in seen:
            seen.add(window)
            if sorted(window) == list(range(1, n)):
                order.append(window)
    return order

def _compute_length_from_order(order, n):
    """
    Compute the exact length of the superpermutation built from the given order.
    Uses incremental string building with overlap detection.
    """
    total = 0
    current = []
    for perm in order:
        block = list(perm) + [n] + list(perm)
        max_ov = min(len(current), len(block))
        overlap = 0
        for ov in range(max_ov, -1, -1):
            if ov == 0 or current[len(current) - ov:] == block[:ov]:
                overlap = ov
                break
        total += len(block) - overlap
        current.extend(block[overlap:])
        # Keep current bounded for speed - we only need the tail
        if len(current) > 2 * n + 10:
            current = current[-(2 * n + 10):]
    return total

def _vns_improve_order(order, n, deadline):
    """
    Variable-neighborhood search on the block ordering.
    Neighborhoods: 2-opt (reverse segment), Or-opt (move segment).
    Accepts moves that strictly reduce total length.
    """
    best_order = list(order)
    best_len = _compute_length_from_order(best_order, n)
    
    # Define neighborhoods
    # N1: 2-opt - reverse a contiguous segment
    # N2: Or-opt - move a contiguous segment to another position
    
    def try_2opt(order, n):
        """Try 2-opt moves: reverse segment [i, j]."""
        best = list(order)
        best_len = _compute_length_from_order(best, n)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best) - 1):
                if time.time() > deadline:
                    break
                # Limit segment length for efficiency
                max_j = min(i + 50, len(best) - 1)
                for j in range(i + 1, max_j + 1):
                    if time.time() > deadline:
                        break
                    new_order = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    new_len = _compute_length_from_order(new_order, n)
                    if new_len < best_len:
                        best = new_order
                        best_len = new_len
                        improved = True
                        break
                if improved:
                    break
        return best, best_len
    
    def try_oropt(order, n):
        """Try Or-opt moves: move segment [i, i+k-1] to position j."""
        best = list(order)
        best_len = _compute_length_from_order(best, n)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                for k in range(1, min(4, len(best) - i + 1)):
                    if time.time() > deadline:
                        break
                    segment = best[i:i+k]
                    rest = best[:i] + best[i+k:]
                    for j in range(len(rest) + 1):
                        if time.time() > deadline:
                            break
                        if j == i or j == i + k:
                            continue
                        new_order = rest[:j] + segment + rest[j:]
                        new_len = _compute_length_from_order(new_order, n)
                        if new_len < best_len:
                            best = new_order
                            best_len = new_len
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return best, best_len
    
    # Apply neighborhoods in sequence
    best_order, best_len = try_2opt(best_order, n)
    if time.time() < deadline:
        best_order, best_len = try_oropt(best_order, n)
    if time.time() < deadline:
        best_order, best_len = try_2opt(best_order, n)
    
    return best_order, best_len

def _delete_pass(word, n, deadline):
    """
    Character-level deletion pass. Try to remove characters that are redundant.
    A character can be removed if no permutation window that is unique is destroyed.
    """
    S = list(word)
    counts = {}
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i+n])
        counts[p] = counts.get(p, 0) + 1
    
    best_len = len(S)
    
    while time.time() < deadline:
        made_removal = False
        # Scan from right to left
        for i in range(len(S) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Old windows that include position i: indices i-n+1 to i
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            
            valid = True
            old_perms_to_dec = []
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j+n])
                if counts.get(p, 0) == 1:
                    valid = False
                    break
                old_perms_to_dec.append(p)
            
            if not valid:
                continue
            
            # Perform removal
            for p in old_perms_to_dec:
                counts[p] -= 1
                if counts[p] == 0:
                    del counts[p]
            
            # New windows formed by straddling the gap
            # New string: S[:i] + S[i+1:]
            # Windows that straddle: indices k where k < i and k+n-1 >= i
            # i.e., k in [i-n+1, i-1] in new string coordinates
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, len(S) - 2 - n + 1)  # new len is len(S)-1
            for k in range(start_new, end_new + 1):
                window_chars = []
                for idx in range(k, k+n):
                    s_idx = idx if idx < i else idx + 1
                    window_chars.append(S[s_idx])
                p_new = tuple(window_chars)
                counts[p_new] = counts.get(p_new, 0) + 1
            
            S.pop(i)
            best_len = len(S)
            made_removal = True
        
        if not made_removal:
            break
    
    return S

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # For n <= 5, return standard (optimal for n<=3)
    if n <= 5:
        word = _standard_superperm(n)
        # Apply deletion pass for potential improvement
        if time.time() < deadline:
            word = _delete_pass(word, n, deadline)
        return "".join(map(str, word))
    
    # For n >= 6: build from standard base, optimize ordering, then delete
    # Step 1: Generate standard superpermutation for n-1
    base = _standard_superperm(n - 1)
    
    # Step 2: Extract the order of (n-1)-permutations
    order = _extract_order_from_word(base, n - 1)
    
    # Step 3: VNS on the order
    if time.time() < deadline - 30:
        deadline_vns = deadline - 30
        order, _ = _vns_improve_order(order, n, deadline_vns)
    
    # Step 4: Construct the final superpermutation
    word = _construct_from_order(order, n)
    
    # Step 5: Character-level deletion pass
    if time.time() < deadline - 5:
        deadline_del = deadline - 5
        word = _delete_pass(word, n, deadline_del)
    
    # Verify
    if not _verify_superpermutation(word, n):
        # Fallback to standard
        word = _standard_superperm(n)
    
    return "".join(map(str, word))

# Quick self-test
if __name__ == "__main__":
    for n in range(1, 6):
        w = construct(n)
        valid = _verify_superpermutation(list(map(int, w)), n)
        print(f"n={n}, len={len(w)}, valid={valid}")
    
    for n in [6, 7]:
        start = time.time()
        w = construct(n)
        elapsed = time.time() - start
        valid = _verify_superpermutation(list(map(int, w)), n)
        print(f"n={n}, len={len(w)}, valid={valid}, time={elapsed:.2f}s")
