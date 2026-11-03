import time
import random
import numpy as np

def construct(q, n, radius):
    start = time.time()
    deadline = start + 140  # safety margin under 150s kill

    # ---- Structural construction: lexicographic tetracode product ----
    # Tetracode C4 (length 4 over GF(3)): 9 codewords
    C4 = [
        (0,0,0,0),(0,1,1,2),(0,2,2,1),
        (1,0,2,2),(1,1,0,1),(1,2,1,0),
        (2,0,1,1),(2,1,2,0),(2,2,0,2),
    ]
    # Product: 9*9 = 81 codewords of length 8
    code = []
    for a in C4:
        for b in C4:
            code.append(list(a) + list(b))

    # ---- Coverage helpers (vectorized with numpy) ----
    def all_words():
        # 6561 x 8 array
        base = np.array([list(x) for x in np.ndindex(*([q]*n))], dtype=np.int32)
        return base

    def neighbors_mask(codeword_list):
        """Return boolean array of size 6561: True if word is within radius of any codeword."""
        W = all_words()
        # For radius 1: a word is covered if it equals a codeword or differs in exactly one position
        covered = np.zeros(W.shape[0], dtype=bool)
        cw = np.array(codeword_list, dtype=np.int32)  # k x n
        # Equality
        for c in cw:
            covered |= np.all(W == c, axis=1)
        # One-position differences
        for c in cw:
            for pos in range(n):
                # W differs from c in exactly position pos
                diff_mask = (W != c)
                diff_mask[:, pos] = False  # allow difference at pos
                same_mask = (W == c)
                same_mask[:, pos] = False  # require difference at pos
                # covered if same in all other positions AND different at pos
                covered |= (same_mask.all(axis=1) & (W[:, pos] != c[pos]))
        return covered

    # Faster: precompute all words once
    W = all_words()  # 6561 x 8

    def is_covered(codeword_list):
        """Check if all 6561 words are within radius 1 of some codeword."""
        cw = np.array(codeword_list, dtype=np.int32)
        covered = np.zeros(W.shape[0], dtype=bool)
        for c in cw:
            # exact match
            covered |= np.all(W == c, axis=1)
            # one-position difference
            for pos in range(n):
                same = (W == c).copy()
                same[:, pos] = True  # ignore pos
                diff = (W[:, pos] != c[pos])
                covered |= (same.all(axis=1) & diff)
        return covered.all()

    def coverage_count(codeword_list):
        """Return number of covered words (out of 6561)."""
        cw = np.array(codeword_list, dtype=np.int32)
        covered = np.zeros(W.shape[0], dtype=bool)
        for c in cw:
            covered |= np.all(W == c, axis=1)
            for pos in range(n):
                same = (W == c).copy()
                same[:, pos] = True
                diff = (W[:, pos] != c[pos])
                covered |= (same.all(axis=1) & diff)
        return covered.sum()

    # ---- Verify initial construction covers everything ----
    if not is_covered(code):
        # Fallback: greedy set cover
        covered = np.zeros(W.shape[0], dtype=bool)
        for c in code:
            covered |= np.all(W == c, axis=1)
            for pos in range(n):
                same = (W == c).copy()
                same[:, pos] = True
                diff = (W[:, pos] != c[pos])
                covered |= (same.all(axis=1) & diff)
        # Add greedily
        for _ in range(200):
            if covered.all():
                break
            if time.time() > deadline - 10:
                break
            best_w = None
            best_gain = 0
            for i in range(W.shape[0]):
                if covered[i]:
                    continue
                w = W[i]
                gain = 0
                for pos in range(n):
                    same = (W == w).copy()
                    same[:, pos] = True
                    diff = (W[:, pos] != w[pos])
                    gain += ((same.all(axis=1) & diff) & ~covered).sum()
                gain += (~covered & np.all(W == w, axis=1)).sum()
                if gain > best_gain:
                    best_gain = gain
                    best_w = w
            if best_w is None or best_gain == 0:
                break
            code.append(list(best_w))
            covered |= np.all(W == best_w, axis=1)
            for pos in range(n):
                same = (W == best_w).copy()
                same[:, pos] = True
                diff = (W[:, pos] != best_w[pos])
                covered |= (same.all(axis=1) & diff)

    # ---- Deterministic 2-opt exchange to reduce size ----
    # For each pair (i, j) with i < j, try removing i and j, then add best single replacement
    # Or: try removing one word and see if still covered
    rng = random.Random(123)
    code_set = set(tuple(c) for c in code)

    # First pass: remove any word whose removal keeps full coverage
    changed = True
    while changed and time.time() < deadline - 15:
        changed = False
        for i in range(len(code)):
            if time.time() > deadline - 15:
                break
            candidate = code[:i] + code[i+1:]
            if is_covered(candidate):
                code = candidate
                code_set.discard(tuple(code[i]))
                changed = True
                break  # restart scan

    # Second pass: 2-opt swap — replace two words with one that covers their union
    for _pass in range(50):
        if time.time() > deadline - 10:
            break
        improved = False
        # Try all pairs
        for i in range(len(code)):
            if time.time() > deadline - 10:
                break
            for j in range(i+1, len(code)):
                if time.time() > deadline - 10:
                    break
                # Remove i and j, find best single word to add
                without = code[:i] + code[i+1:j] + code[j+1:]
                # Find uncovered
                covered = np.zeros(W.shape[0], dtype=bool)
                for c in without:
                    covered |= np.all(W == c, axis=1)
                    for pos in range(n):
                        same = (W == c).copy()
                        same[:, pos] = True
                        diff = (W[:, pos] != c[pos])
                        covered |= (same.all(axis=1) & diff)
                if covered.all():
                    # Both removable!
                    code = without
                    code_set.discard(tuple(code[i]))
                    code_set.discard(tuple(code[j]))
                    improved = True
                    break
                uncovered = ~covered
                # Find best single word
                best_w = None
                best_gain = 0
                for k in range(W.shape[0]):
                    if uncovered[k]:
                        w = W[k]
                        gain = 0
                        for pos in range(n):
                            same = (W == w).copy()
                            same[:, pos] = True
                            diff = (W[:, pos] != w[pos])
                            gain += ((same.all(axis=1) & diff) & uncovered).sum()
                        gain += (uncovered & np.all(W == w, axis=1)).sum()
                        if gain > best_gain:
                            best_gain = gain
                            best_w = w
                if best_w is not None:
                    # Check if adding best_w to without covers everything
                    test = without + [list(best_w)]
                    if is_covered(test):
                        code = test
                        code_set.discard(tuple(code[i]))
                        code_set.discard(tuple(code[j]))
                        code_set.add(tuple(best_w))
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break

    # Third pass: single-word removal again
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for i in range(len(code)):
            if time.time() > deadline - 5:
                break
            candidate = code[:i] + code[i+1:]
            if is_covered(candidate):
                code = candidate
                code_set.discard(tuple(code[i]))
                changed = True
                break

    # ---- Final verification and repair ----
    if not is_covered(code):
        covered = np.zeros(W.shape[0], dtype=bool)
        for c in code:
            covered |= np.all(W == c, axis=1)
            for pos in range(n):
                same = (W == c).copy()
                same[:, pos] = True
                diff = (W[:, pos] != c[pos])
                covered |= (same.all(axis=1) & diff)
        while not covered.all() and time.time() < deadline - 2:
            best_w = None
            best_gain = 0
            for k in range(W.shape[0]):
                if not covered[k]:
                    w = W[k]
                    gain = 0
                    for pos in range(n):
                        same = (W == w).copy()
                        same[:, pos] = True
                        diff = (W[:, pos] != w[pos])
                        gain += ((same.all(axis=1) & diff) & ~covered).sum()
                    gain += (~covered & np.all(W == w, axis=1)).sum()
                    if gain > best_gain:
                        best_gain = gain
                        best_w = w
            if best_w is None or best_gain == 0:
                break
            code.append(list(best_w))
            covered |= np.all(W == best_w, axis=1)
            for pos in range(n):
                same = (W == best_w).copy()
                same[:, pos] = True
                diff = (W[:, pos] != best_w[pos])
                covered |= (same.all(axis=1) & diff)

    return code
