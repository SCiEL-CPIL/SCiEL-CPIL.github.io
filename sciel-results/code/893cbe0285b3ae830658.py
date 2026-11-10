"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110

    def find_overlap(current, block):
        """Find the maximum overlap between the suffix of current and prefix of block."""
        max_ov = min(len(current), len(block))
        # Try overlaps from largest to smallest
        for o in range(max_ov, 0, -1):
            if current[-o:] == block[:o]:
                return o
        return 0

    word = "1"
    for m in range(2, n + 1):
        if time.time() > deadline:
            break

        # Collect all (m-1)-length distinct permutations from the current word
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = word[i:i + m - 1]
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)

        if m < n:
            # Standard superpermutation construction: insert new symbol m
            new = ""
            for p in perms:
                block = p + str(m) + p
                ov = find_overlap(new, block)
                new += block[ov:]
            word = new
        else:
            # For the final step, try different orderings to find the shortest valid superpermutation
            candidates = [perms, list(reversed(perms)), sorted(perms)]
            best_word, best_len = None, float('inf')
            for ordering in candidates:
                if time.time() > deadline - 5:
                    break
                new = ""
                for p in ordering:
                    block = p + str(m) + p
                    ov = find_overlap(new, block)
                    new += block[ov:]
                if len(new) < m:
                    continue
                all_perms = set(permutations(range(1, m + 1)))
                found = set()
                for i in range(len(new) - m + 1):
                    w = tuple(int(c) for c in new[i:i + m])
                    if len(set(w)) == m:
                        found.add(w)
                if found == all_perms and len(new) < best_len:
                    best_len, best_word = len(new), new
            if best_word is not None:
                word = best_word
            else:
                # Fallback to original ordering if no valid superpermutation found
                new = ""
                for p in perms:
                    block = p + str(m) + p
                    ov = find_overlap(new, block)
                    new += block[ov:]
                word = new

    return word
# EVOLVE-BLOCK-END
