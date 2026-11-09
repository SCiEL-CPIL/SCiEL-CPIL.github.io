"""Vectorized greedy packing with forbidden-zone marking for A_2(17,6)."""
import numpy as np
import time

N = 17
MIN_DISTANCE = 6
UNIVERSE_SIZE = 1 << N  # 131072

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Precompute all vectors as uint8 array: shape (UNIVERSE_SIZE, N)
    # This is 131072 * 17 bytes = ~2.2 MB, very fast
    idx = np.arange(UNIVERSE_SIZE, dtype=np.uint32)
    # Extract bits: bit j of value x is (x >> j) & 1
    # We want vector[i][j] = bit j of i
    bits = ((idx[:, None] >> np.arange(N, dtype=np.uint32)[None, :]) & 1).astype(np.uint8)
    # bits shape: (131072, 17), uint8

    # Precompute the "forbidden zone" lookup: for each possible word, which words are at distance < 6?
    # That's too expensive to precompute all 131072 x 131072. Instead, we compute on the fly.
    # But we can precompute the weight distribution: for a given word w, the set of words at distance < 6
    # can be enumerated. The number of words at distance d from w is C(17, d).
    # Sum for d=0..5: C(17,0)+C(17,1)+...+C(17,5) = 1+17+136+680+2380+6188 = 9402
    # So for each codeword we add, we mark ~9402 positions as forbidden.

    # Precompute all words at distance <= 5 from the all-zeros word, as offsets.
    # Then for word w, the forbidden set is {w XOR o : o in offsets}
    # Actually, Hamming distance is symmetric, so dist(w, x) = dist(0, w XOR x).
    # So forbidden(x) means dist(w, x) < 6, i.e., dist(0, w XOR x) < 6, i.e., w XOR x in S where S = {y : wt(y) <= 5}.
    # Equivalently, x in {w XOR y : y in S}.

    # Precompute S: all 17-bit words of weight <= 5
    from itertools import combinations
    S_list = []
    for d in range(6):
        for combo in combinations(range(N), d):
            w = 0
            for j in combo:
                w |= (1 << j)
            S_list.append(w)
    S_arr = np.array(S_list, dtype=np.uint32)
    # S_arr has 9402 entries

    # Forbidden zone array: bool array of size UNIVERSE_SIZE
    forbidden = np.zeros(UNIVERSE_SIZE, dtype=bool)
    selected = []  # list of selected word indices

    def mark_forbidden(word_idx):
        """Mark all words within distance < 6 of word_idx as forbidden."""
        # The forbidden words are word_idx XOR S_arr
        targets = (word_idx ^ S_arr).astype(np.uint32)
        forbidden[targets] = True

    def unmark_forbidden(word_idx):
        """Unmark (not used in greedy, but for potential backtracking)."""
        targets = (word_idx ^ S_arr).astype(np.uint32)
        forbidden[targets] = False

    # Greedy phase: scan the universe in a smart order
    # Strategy: multiple passes with different orderings to find denser packings

    # Initial seed: start with 0 (all-zeros word)
    # Actually, let's try a randomized greedy with multiple restarts

    best_count = 0
    best_selection = None

    def greedy_pass(order, time_limit):
        """Run greedy packing following the given order. Returns list of selected indices."""
        sel = []
        forbidden_local = np.zeros(UNIVERSE_SIZE, dtype=bool)

        for idx_in_order in order:
            if time.time() > time_limit:
                break
            if not forbidden_local[idx_in_order]:
                sel.append(idx_in_order)
                # Mark forbidden zone
                targets = (idx_in_order ^ S_arr).astype(np.uint32)
                forbidden_local[targets] = True

        return sel

    # Pass 1: natural order 0, 1, 2, ..., 2^17-1
    # This is fast because we can vectorize the scan
    # Actually, let's do a vectorized greedy: repeatedly find the first unforbidden position

    # Better approach: iterative vectorized greedy
    # In each "round", find all positions that are not forbidden, pick the first one (or a random one), mark its zone.
    # But that's still sequential. Let's just do a simple scan.

    # Fast vectorized greedy: scan in natural order
    sel = []
    forbidden = np.zeros(UNIVERSE_SIZE, dtype=bool)

    # We'll do multiple passes with different orderings
    # Pass 1: natural order
    # To make it faster, we can use numpy to find unforbidden positions in chunks

    # Actually, the simplest fast approach: iterate through all 131072 positions, if not forbidden, select it.
    # Each selection marks 9402 positions. Total selections might be ~250-350, so total marks ~2.5M, very fast.

    # Let's just do it in a loop but with numpy marking
    for idx in range(UNIVERSE_SIZE):
        if time.time() > DEADLINE:
            break
        if not forbidden[idx]:
            sel.append(idx)
            targets = (idx ^ S_arr).astype(np.uint32)
            forbidden[targets] = True

    current_count = len(sel)
    if current_count > best_count:
        best_count = current_count
        best_selection = sel[:]

    # If we have time, try other orderings
    if time.time() < DEADLINE - 5:
        # Try a random shuffle
        np.random.seed(42)
        order = np.random.permutation(UNIVERSE_SIZE)
        sel2 = greedy_pass(order, DEADLINE - 2)
        if len(sel2) > best_count:
            best_count = len(sel2)
            best_selection = sel2

    # Try a few more random shuffles if time permits
    if time.time() < DEADLINE - 3:
        for seed in [123, 456, 789]:
            if time.time() > DEADLINE - 1:
                break
            np.random.seed(seed)
            order = np.random.permutation(UNIVERSE_SIZE)
            sel3 = greedy_pass(order, DEADLINE - 1)
            if len(sel3) > best_count:
                best_count = len(sel3)
                best_selection = sel3

    # Return the best selection
    if not best_selection:
        # Fallback: return the Reed-Muller code (32 words)
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    return best_selection
