"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START
def construct(n):
    # Deadline guard: return best valid result before the 120s limit (safety margin 15s).
    start = time.time()
    deadline = start + 105.0  # 105s to leave margin for evaluator overhead

    # Trivial cases
    if n <= 1:
        return "1"

    # --- Step 1: Build the classic recursive superpermutation (length sum_{k=1}^n k!) ---
    # This is a valid superpermutation but not shortest.
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
            # Find maximum overlap
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    # Check time
    if time.time() > deadline:
        return "".join(map(str, word))

    # --- Step 2: Greedy deletion to shorten the word ---
    # We want to remove symbols that are "redundant" — i.e., their removal doesn't
    # destroy any permutation window.
    # A symbol at position i is redundant if:
    #   - It's not at the very start or end (so removing it doesn't change window count by 1)
    #   - Every permutation that appears in windows containing position i also appears
    #     in at least one window that does NOT contain position i.
    #
    # More precisely: after removing position i, the new word has all windows of size n.
    # We need to verify that every permutation of 1..n still appears.
    #
    # Efficient approach: maintain a set of all required permutations and track counts.
    # For each position, check if it's safe to remove.

    def all_perms(n):
        return set(permutations(range(1, n + 1)))

    required = all_perms(n)
    total_perms = len(required)

    # Build initial count of each permutation in windows
    perm_counts = {}
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i + n])
        perm_counts[p] = perm_counts.get(p, 0) + 1

    # Verify all perms are present
    for p in required:
        if p not in perm_counts:
            return "".join(map(str, word))  # fallback

    # Greedy deletion: repeatedly try to remove redundant positions
    # We process from left to right, and for each position check if removal is safe.
    # To be efficient, we use a heuristic: try all positions, pick the one whose removal
    # is safe and that gives the most "slack" (i.e., the position that appears in the
    # fewest unique permutation windows, or simply the first safe one).

    # For speed, we'll do multiple passes. In each pass, we scan left-to-right and
    # remove any position that is safe.

    # A position i is safe to remove if:
    #   - i is not 0 and not len(word)-1 (boundary positions define window edges)
    #   - Actually, even boundary positions can be removed if all perms still appear.
    #   - After removal, every permutation in `required` still has count >= 1.
    #
    # Efficient check: removing position i affects windows [i-n+1, i] (those that include i).
    # Specifically, windows starting at positions j where j <= i <= j+n-1, i.e., j in [i-n+1, i].
    # After removal, the word shifts: positions after i shift left by 1.
    # The new windows are:
    #   - All old windows that don't include i (unchanged)
    #   - New windows formed by concatenating the parts around the gap
    #
    # This is complex. Let's use a simpler but slower check: just rebuild the count
    # after removal and verify. But that's O(n * len(word)) per check, which is too slow.
    #
    # Alternative: use a randomized/local search approach.
    #
    # Given time constraints, let's try a simple greedy: for each position (left to right),
    # tentatively remove it, check if all perms still appear (using a fast check), and
    # keep the removal if safe.

    # Fast check: after removing position i, which windows are affected?
    # Old windows containing position i: starts j in [max(0, i-n+1), min(i, len(word)-n)]
    # New windows: the gap at i is closed. New windows that cross the gap:
    #   start j in [max(0, i-n+1), min(i, len(word)-n)] but shifted...
    #
    # Actually, let's just do a direct check for small n. For n=7, len(word) ~ 13699.
    # Number of windows ~ 13693. Checking each removal naively is O(n * 13693 * 13693) which is too much.
    #
    # Let's use a different strategy: local search with simulated annealing or hill climbing.
    #
    # Given the time budget, let's try a simple approach:
    # 1. Start with the recursive construction.
    # 2. Repeatedly try to remove a random position and accept if valid.
    # 3. Also try to swap adjacent positions and accept if valid.
    #
    # But even one validity check is O(n * len(word)) ~ 7 * 13699 ~ 96k ops. With 105s budget,
    # we can do maybe ~10000 checks if each takes ~10ms. Let's be optimistic.

    # Let's implement a fast validity check using a set of seen perms.
    def is_valid(w):
        seen = set()
        for i in range(len(w) - n + 1):
            seen.add(tuple(w[i:i + n]))
        return seen == required

    # This is O(n * len(w)) which for n=7, len~13699 is ~96k tuple creations. Might be ~50ms in Python.
    # Too slow for many iterations. Let's optimize.

    # Alternative: use a rolling hash or just accept that we can do few iterations.
    # Let's try a smarter greedy: identify positions that are "critical" (appear in few windows
    # that are unique to them) and try removing non-critical ones.

    # For now, let's just do a limited number of random removals.
    best_word = list(word)
    best_len = len(best_word)

    # Try greedy left-to-right deletion
    for _pass in range(5):  # limited passes
        if time.time() > deadline:
            break
        changed = False
        i = 0
        while i < len(best_word) - 1:  # don't remove last element
            # Try removing position i
            if i == 0 or i == len(best_word) - 1:
                i += 1
                continue
            # Quick check: only try if time allows
            if time.time() > deadline:
                break
            # Tentatively remove
            old = best_word[i]
            del best_word[i]
            if is_valid(best_word):
                best_len = len(best_word)
                changed = True
                # Don't increment i, check same position again (which is now old i+1)
            else:
                best_word.insert(i, old)
                i += 1
        if not changed:
            break

    return "".join(map(str, best_word))
# EVOLVE-BLOCK-END
