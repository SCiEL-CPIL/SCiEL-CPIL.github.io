"""Superpermutation for n=7 with smart local search."""
import time

def construct(n):
    start = time.time()
    deadline = start + 140.0
    
    def build_superperm(n):
        if n == 1:
            return [1]
        word = [1]
        for m in range(2, n + 1):
            if time.time() > deadline:
                break
            seen = set()
            blocks = []
            for i in range(len(word) - m + 2):
                w = tuple(word[i:i+m-1])
                if len(set(w)) == m-1 and w not in seen:
                    seen.add(w)
                    blocks.append(list(w))
            new_word = []
            for b in blocks:
                block = list(b) + [m] + list(b)
                max_overlap = 0
                for o in range(min(len(new_word), len(block)), 0, -1):
                    if new_word[-o:] == block[:o]:
                        max_overlap = o
                        break
                new_word.extend(block[max_overlap:])
            word = new_word
        return word
    
    word = build_superperm(n)
    
    if time.time() > deadline:
        return "".join(map(str, word))
    
    target_count = 1
    for i in range(2, n+1):
        target_count *= i
    
    # Build occurrence map: perm -> list of positions
    def build_occ_map(w, n):
        occ = {}
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                if window not in occ:
                    occ[window] = []
                occ[window].append(i)
        return occ
    
    def is_valid_with_map(w, n, occ):
        return len(occ) == target_count and all(len(v) > 0 for v in occ.values())
    
    occ = build_occ_map(word, n)
    if len(occ) < target_count:
        return "".join(map(str, word))
    
    improved = list(word)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(improved)):
            if time.time() > deadline:
                break
            # Deleting position i affects windows at positions i-n+1 to i
            # Check if any permutation becomes uncovered
            candidate_ok = True
            for j in range(max(0, i-n+1), min(i+1, len(improved)-n+1)):
                window = tuple(improved[j:j+n])
                if len(set(window)) == n:
                    if len(occ[window]) == 1:
                        # This perm only appears at position j
                        # Check if it appears elsewhere in candidate
                        # After deletion, windows shift
                        candidate_ok = False
                        break
            if candidate_ok:
                candidate = improved[:i] + improved[i+1:]
                # Quick check: rebuild occ for affected region
                # For now, do full check
                new_occ = build_occ_map(candidate, n)
                if len(new_occ) == target_count:
                    improved = candidate
                    occ = new_occ
                    changed = True
                    break
    
    return "".join(map(str, improved))
