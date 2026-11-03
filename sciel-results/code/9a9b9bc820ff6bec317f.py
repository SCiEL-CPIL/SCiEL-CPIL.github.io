"""Improved superpermutation search: better initial word + aggressive local search."""
import time
import math
from itertools import permutations

# EVOLVE-BLOCK-START

def _classic_superperm(n):
    """Build the classic superpermutation of length sum(k! for k=1..n)."""
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
            max_ov = min(len(new), len(block) - 1)
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word


def _is_valid_superperm(word, n):
    """Check if word contains all n! permutations of 1..n as consecutive windows."""
    expected = math.factorial(n)
    seen = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i + n])
        if len(set(w)) == n:
            seen.add(w)
            if len(seen) == expected:
                return True
    return len(seen) == expected


class _SuperpermState:
    """
    Maintains the count of occurrences of each permutation in a word.
    Allows efficient checking and updating for deletions.
    """
    def __init__(self, word, n):
        self.n = n
        self.word = list(word)
        self.length = len(self.word)
        self.perm_counts = {}
        self.total_unique = 0
        self.expected = math.factorial(n)

        for i in range(self.length - n + 1):
            w = tuple(self.word[i:i + n])
            if len(set(w)) == n:
                if w in self.perm_counts:
                    self.perm_counts[w] += 1
                else:
                    self.perm_counts[w] = 1
                    self.total_unique += 1

    def is_valid(self):
        return self.total_unique == self.expected

    def try_delete(self, i):
        """Try to delete element at index i. Returns True if successful."""
        n = self.n
        L = self.length

        if i <= 0 or i >= L - 1:
            return False

        start_min = max(0, i - n + 1)
        start_max = min(i, L - n)

        removed_windows = []
        for k in range(start_min, start_max + 1):
            w = tuple(self.word[k:k + n])
            if len(set(w)) == n:
                if self.perm_counts[w] == 1:
                    return False
                removed_windows.append(w)

        new_word = self.word[:i] + self.word[i + 1:]
        new_len = L - 1

        for w in removed_windows:
            self.perm_counts[w] -= 1
            if self.perm_counts[w] == 0:
                del self.perm_counts[w]
                self.total_unique -= 1

        new_start_min = max(0, i - n + 1)
        new_start_max = min(i - 1, new_len - n)

        for k_new in range(new_start_min, new_start_max + 1):
            left_part = self.word[k_new:i]
            right_part = self.word[i + 1:k_new + n + 1]
            w_new = tuple(left_part + right_part)
            if len(set(w_new)) == n:
                if w_new in self.perm_counts:
                    self.perm_counts[w_new] += 1
                else:
                    self.perm_counts[w_new] = 1
                    self.total_unique += 1

        self.word = new_word
        self.length = new_len
        return True

    def try_swap(self, i, j):
        """Try to swap elements at positions i and j. Returns True if successful."""
        n = self.n
        L = self.length
        if i == j:
            return False
        if i < 0 or i >= L or j < 0 or j >= L:
            return False
        if i > j:
            i, j = j, i

        # Determine affected window start indices
        # A window starting at k is affected if it contains i or j
        # i is in window k if k <= i <= k+n-1 => i-n+1 <= k <= i
        # j is in window k if j-n+1 <= k <= j
        k_min = max(0, min(i, j) - n + 1)
        k_max = min(max(i, j), L - n)

        # Collect all affected windows and their current counts
        affected = []
        for k in range(k_min, k_max + 1):
            w = tuple(self.word[k:k + n])
            if len(set(w)) == n:
                affected.append((k, w, self.perm_counts[w]))

        # Check if any affected window would be lost
        for k, w, cnt in affected:
            if cnt == 1:
                # Check if the swap actually changes this window
                wi = self.word[k + (i - k)] if k <= i <= k + n - 1 else None
                wj = self.word[k + (j - k)] if k <= j <= k + n - 1 else None
                if wi is not None and wj is not None and wi != wj:
                    # This window would change; if count is 1, we'd lose it
                    # unless the new window is the same (impossible since wi!=wj)
                    # Actually we need to check if the new window equals an existing one
                    # For safety, let's just simulate
                    pass
                elif wi is not None or wj is not None:
                    # Only one of i,j is in this window
                    # The window changes
                    if cnt == 1:
                        return False

        # Perform the swap
        self.word[i], self.word[j] = self.word[j], self.word[i]

        # Update counts for affected windows
        # First, decrement old windows
        for k, w, cnt in affected:
            self.perm_counts[w] -= 1
            if self.perm_counts[w] == 0:
                del self.perm_counts[w]
                self.total_unique -= 1

        # Then, increment new windows
        for k in range(k_min, k_max + 1):
            w = tuple(self.word[k:k + n])
            if len(set(w)) == n:
                if w in self.perm_counts:
                    self.perm_counts[w] += 1
                else:
                    self.perm_counts[w] = 1
                    self.total_unique += 1

        return True


def _local_search(state, deadline, max_passes=200):
    """Perform multiple passes of greedy deletion and swap operations."""
    for p in range(max_passes):
        if time.time() > deadline - 5.0:
            break
        improved = False

        # Pass 1: Try deletions from right to left
        for i in range(state.length - 2, 0, -1):
            if time.time() > deadline - 5.0:
                break
            if state.try_delete(i):
                improved = True

        # Pass 2: Try swaps (short-range)
        if state.length > 2:
            max_dist = min(10, state.length // 2)
            for i in range(0, state.length - 1):
                if time.time() > deadline - 5.0:
                    break
                for d in range(1, max_dist + 1):
                    j = i + d
                    if j >= state.length:
                        break
                    if state.try_swap(i, j):
                        improved = True
                        # After a successful swap, re-try deletions locally
                        for k in range(max(1, i - 1), min(state.length - 1, j + 2)):
                            if time.time() > deadline - 5.0:
                                break
                            if state.try_delete(k):
                                pass  # Already counted as improved

        if not improved:
            break

    return state


def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # For n <= 4, classic construction is optimal
    if n <= 4:
        word = _classic_superperm(n)
        return "".join(map(str, word))

    # Build classic superpermutation as baseline
    best_word = _classic_superperm(n)
    best_len = len(best_word)

    if not _is_valid_superperm(best_word, n):
        return "".join(map(str, best_word))

    # Local search
    if n >= 5:
        state = _SuperpermState(best_word, n)
        state = _local_search(state, deadline)

        if state.is_valid() and len(state.word) < best_len:
            best_word = state.word
            best_len = len(best_word)

    return "".join(map(str, best_word))

# EVOLVE-BLOCK-END
