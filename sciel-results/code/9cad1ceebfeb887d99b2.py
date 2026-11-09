"""Combined program: bitset-based backtracking + multi-start local refinement for A_2(18,6)."""
import time
import random
import sys

N = 18
MIN_DISTANCE = 6
FULL_MASK = (1 << N) - 1
TOTAL_WORDS = 1 << N

# Deadline guard: keep a safety margin before the 90s kill.
DEADLINE = 85.0  # seconds from start of construct()


def _reed_muller_base():
    """First-order Reed-Muller code on 16 coords, extended by two zero bits.
    32 words, pairwise distance 8."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words


def _precompute_neighbors():
    """Precompute all masks with weight < MIN_DISTANCE (forbidden offsets)."""
    offsets = []
    for mask in range(TOTAL_WORDS):
        if mask.bit_count() < MIN_DISTANCE:
            offsets.append(mask)
    return offsets


# Precompute once at import time; this is cheap (2^18 * bit_count).
_NEIGHBOR_OFFSETS = _precompute_neighbors()


def _valid_code(words):
    """Check pairwise Hamming distance >= MIN_DISTANCE using XOR + popcount."""
    m = len(words)
    for i in range(m):
        wi = words[i]
        for j in range(i + 1, m):
            if (wi ^ words[j]).bit_count() < MIN_DISTANCE:
                return False
    return True


def _greedy_fill(code, forbidden, rng, deadline, allow_swap=False):
    """
    Greedily add words to `code` using the forbidden bitset.
    `code`: list of selected words.
    `forbidden`: int bitset where bit w is 1 if word w is forbidden.
    `rng`: random.Random instance for candidate ordering.
    `deadline`: absolute time (time.time()) to stop before.
    `allow_swap`: if True, also try simple swap moves to escape local optima.
    Returns (code, forbidden, improved_flag).
    """
    code_set = set(code)
    improved = False

    # Helper: add a word and update forbidden bitset.
    def add_word(w):
        nonlocal forbidden, code, code_set, improved
        if w in code_set:
            return False
        # Check forbidden (should be 0 for w if we picked it correctly)
        if (forbidden >> w) & 1:
            return False
        code.append(w)
        code_set.add(w)
        # Mark w and its neighbors as forbidden.
        # forbidden |= neighbor_bitset(w)
        # We'll build the neighbor bitset on the fly using offsets.
        # For speed, we could precompute neighbor_bitset for all w, but that's huge.
        # Instead, iterate offsets and OR bits.
        # This is O(len(offsets)) per add; offsets ~ 1 + 18 + 153 + 816 = ~988.
        # That's fine.
        for off in _NEIGHBOR_OFFSETS:
            nb = w ^ off
            forbidden |= (1 << nb)
        improved = True
        return True

    # Helper: remove a word and clear its forbidden bits (approximate: we just clear w and its neighbors).
    def remove_word(w):
        nonlocal forbidden, code, code_set, improved
        if w not in code_set:
            return False
        code_set.discard(w)
        # Remove from list (O(n) but n is small)
        code.remove(w)
        # Clear forbidden bits for w and its neighbors.
        for off in _NEIGHBOR_OFFSETS:
            nb = w ^ off
            forbidden &= ~(1 << nb)
        improved = True
        return True

    # Main greedy loop: repeatedly pick a candidate word not forbidden, add it.
    # To avoid O(2^18) scans each time, we maintain a list of "available" words
    # by scanning the forbidden bitset occasionally, or use a random sampling approach.
    # Simpler: maintain a list of all words and filter by forbidden bit.
    # But 2^18 = 262144; scanning each iteration is okay if we don't do too many.
    # We'll use a random sampling + occasional full scan hybrid.

    # Precompute list of all words for random access.
    all_words = list(range(TOTAL_WORDS))

    # For deterministic passes, we can sort by weight or index; for random passes, shuffle.
    # We'll do a few passes with different orderings.

    # Pass 1: try words in increasing index order (deterministic).
    for w in all_words:
        if time.time() > deadline:
            break
        if (forbidden >> w) & 1:
            continue
        if add_word(w):
            # Continue; we'll keep adding as we scan.
            pass

    # Pass 2: try words in random order to break symmetry.
    if time.time() < deadline - 5.0:
        shuffled = all_words[:]
        rng.shuffle(shuffled)
        for w in shuffled:
            if time.time() > deadline:
                break
            if (forbidden >> w) & 1:
                continue
            if add_word(w):
                pass

    # Optional: simple swap moves to escape local optima.
    if allow_swap and time.time() < deadline - 10.0:
        # Try a limited number of swap attempts.
        # For each word in code, try to remove it and add a different word.
        # This is expensive; do a few random attempts.
        n_swaps = 50
        for _ in range(n_swaps):
            if time.time() > deadline:
                break
            if not code:
                break
            # Pick a random word to remove.
            idx = rng.randrange(len(code))
            w_remove = code[idx]
            # Save state.
            old_code = code[:]
            old_forbidden = forbidden
            # Remove it.
            remove_word(w_remove)
            # Now try to add a new word (different from w_remove) that is not forbidden.
            # Sample a few candidates.
            added = False
            for _ in range(20):
                if time.time() > deadline:
                    break
                w_new = rng.randrange(TOTAL_WORDS)
                if w_new == w_remove:
                    continue
                if (forbidden >> w_new) & 1:
                    continue
                if add_word(w_new):
                    added = True
                    break
            # If we didn't add, revert.
            if not added:
                code = old_code
                forbidden = old_forbidden
                # Rebuild code_set from code.
                code_set = set(code)

    return code, forbidden, improved


def construct():
    """
    Entry point. Returns a list of distinct integers in [0, 2**N) representing
    codewords with pairwise Hamming distance >= MIN_DISTANCE.
    """
    t_start = time.time()
    deadline = t_start + DEADLINE

    # Precompute base code.
    base_words = _reed_muller_base()
    best_words = base_words[:]
    best_size = len(best_words)

    # We'll try multiple strategies:
    # 1) Start from base code and greedily fill.
    # 2) Start from scratch with symmetry fixing (force 0 in code).
    # 3) Random restarts with small seeds.

    rng = random.Random(12345)

    # Strategy 1: start from base code.
    forbidden = 0
    for w in best_words:
        for off in _NEIGHBOR_OFFSETS:
            nb = w ^ off
            forbidden |= (1 << nb)
    # Also mark the base words themselves as forbidden (they are in the code).
    for w in best_words:
        forbidden |= (1 << w)

    code1, forbidden1, _ = _greedy_fill(best_words, forbidden, rng, deadline - 5.0, allow_swap=True)
    if len(code1) > best_size:
        best_words = code1
        best_size = len(best_words)

    # Strategy 2: start from scratch, force 0 in code (symmetry breaking).
    if time.time() < deadline - 10.0:
        code2 = [0]
        forbidden2 = 0
        for off in _NEIGHBOR_OFFSETS:
            nb = 0 ^ off
            forbidden2 |= (1 << nb)
        # Mark 0 as forbidden (it's in the code).
        forbidden2 |= (1 << 0)
        code2, forbidden2, _ = _greedy_fill(code2, forbidden2, rng, deadline - 5.0, allow_swap=True)
        if len(code2) > best_size:
            best_words = code2
            best_size = len(best_words)

    # Strategy 3: random restarts with small seeds.
    if time.time() < deadline - 5.0:
        for seed in range(5):
            if time.time() > deadline - 2.0:
                break
            rng2 = random.Random(1000 + seed)
            # Start with a small random seed set (size 4-8) that is valid.
            seed_size = rng2.randint(4, 8)
            seed_words = []
            seed_set = set()
            attempts = 0
            while len(seed_words) < seed_size and attempts < 1000:
                attempts += 1
                w = rng2.randrange(TOTAL_WORDS)
                if w in seed_set:
                    continue
                # Check against current seed_words.
                ok = True
                for sw in seed_words:
                    if (w ^ sw).bit_count() < MIN_DISTANCE:
                        ok = False
                        break
                if ok:
                    seed_words.append(w)
                    seed_set.add(w)
            if len(seed_words) < 2:
                continue
            # Build forbidden bitset for seed_words.
            forbidden3 = 0
            for w in seed_words:
                for off in _NEIGHBOR_OFFSETS:
                    nb = w ^ off
                    forbidden3 |= (1 << nb)
                forbidden3 |= (1 << w)
            code3, forbidden3, _ = _greedy_fill(seed_words, forbidden3, rng2, deadline - 2.0, allow_swap=True)
            if len(code3) > best_size:
                best_words = code3
                best_size = len(best_words)

    # Final validity check.
    if not _valid_code(best_words):
        # Fallback to base code.
        best_words = _reed_muller_base()

    # Ensure distinct and in range.
    result = list(dict.fromkeys(best_words))
    return result


if __name__ == "__main__":
    # Quick self-test.
    words = construct()
    print(f"Code size: {len(words)}")
    # Verify.
    assert _valid_code(words), "Invalid code!"
    print("Valid code.")
