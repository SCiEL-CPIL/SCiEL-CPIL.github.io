"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from math import factorial

def construct(n):
    if n <= 1:
        return "1" if n == 1 else ""
    if n <= 3:
        from itertools import permutations
        perms = list(permutations(range(1, n + 1)))
        # Greedy: build shortest string containing all permutations as substrings
        best = None
        best_len = n * factorial(n) + 1
        # Try all starting permutations, greedy extend
        for start in perms:
            s = list(start)
            remaining = set(perms)
            remaining.discard(start)
            # Build a map from last n-1 elements to available next
            # Greedy: at each step, try to extend by 1 char that covers most new perms
            while remaining:
                last = tuple(s[-(n - 1):])
                best_next = None
                best_count = -1
                for sym in range(1, n + 1):
                    new_win = last[1:] + (sym,)
                    # count how many remaining perms start with new_win
                    cnt = 0
                    for p in remaining:
                        if p[:n - 1] == new_win:
                            cnt += 1
                    if cnt > best_count:
                        best_count = cnt
                        best_next = sym
                if best_next is None:
                    break
                s.append(best_next)
                # Remove all perms that are now fully contained
                # Check if last n elements form a complete permutation
                if len(s) >= n:
                    win = tuple(s[-n:])
                    if len(set(win)) == n:
                        remaining.discard(win)
                # Also remove any permutation that ends at current position
                # More efficiently: remove all perms that are substring of s
                # This is expensive; just check the most recent window
            if len(s) < best_len:
                best_len = len(s)
                best = s
        if best is not None:
            return "".join(map(str, best))
        # Fallback: naive
        result = ""
        for p in perms:
            # Check if p is already in result
            p_str = "".join(map(str, p))
            if p_str not in result:
                # Append with maximum overlap
                for k in range(n - 1, -1, -1):
                    if result[-k:] == p_str[:k]:
                        result += p_str[k:]
                        break
                else:
                    result += p_str
        return result
    # For n >= 4, use the standard recursive superpermutation construction
    # This gives length 1! + 2! + ... + n!
    # For n=7, that's 1+2+6+24+120+720+5040 = 5913
    # The shortest known is 5905, so we're 8 over.
    # We'll use the standard construction and hope it's close enough.
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
