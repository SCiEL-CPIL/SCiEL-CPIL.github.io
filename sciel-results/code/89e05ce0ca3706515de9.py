"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 85.0

    def compute_block_sums(seq):
        """Compute all nonempty consecutive block sums."""
        sums = set()
        n = len(seq)
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += seq[j]
                sums.add(total)
        return sums

    def is_valid(seq):
        """Check if sequence is strictly increasing and all block sums distinct."""
        if not seq:
            return True
        for i in range(1, len(seq)):
            if seq[i] <= seq[i-1]:
                return False
        sums = compute_block_sums(seq)
        return len(sums) == len(seq) * (len(seq) + 1) // 2

    # Greedy baseline
    greedy_seq = []
    greedy_forbidden = set()
    for x in range(1, 257):
        tails = []
        total = x
        for a in reversed(greedy_seq):
            tails.append(total)
            total += a
        if not greedy_forbidden.intersection(tails):
            greedy_seq.append(x)
            greedy_forbidden.update(tails)

    best_seq = list(greedy_seq)
    best_len = len(best_seq)

    if time.time() > DEADLINE:
        return best_seq

    # Backtracking search with time limit
    # We'll try to find a sequence longer than best_len
    
    def backtrack(seq, forbidden, deadline):
        """
        Extend seq to be as long as possible.
        Returns the longest valid extension found.
        """
        nonlocal best_seq, best_len

        if time.time() > deadline:
            return list(seq)

        current_len = len(seq)
        if current_len > best_len:
            best_seq = list(seq)
            best_len = current_len

        if not seq:
            candidates = range(1, 20)
        else:
            last = seq[-1]
            # We need to leave room: if we pick x, we need at least 1 more number > x
            # So x <= 255
            candidates = range(last + 1, min(last + 25, 257))

        found_extension = False
        for x in candidates:
            if time.time() > deadline:
                break

            # Compute new tails (block sums ending at x)
            tails = []
            total = x
            for a in reversed(seq):
                tails.append(total)
                total += a

            # Check collision
            if any(s in forbidden for s in tails):
                continue

            new_forbidden = forbidden | set(tails)
            new_seq = seq + [x]

            # Recurse
            result = backtrack(new_seq, new_forbidden, deadline)
            if len(result) > len(seq):
                found_extension = True
                # Continue searching for even longer sequences
                # But don't return immediately; keep looking

        return list(seq) if not found_extension else best_seq

    # Try backtracking from empty and from greedy prefixes
    # Use a shorter deadline for each attempt to allow multiple tries
    attempt_deadline = min(DEADLINE, start_time + 40)
    
    # Try from empty with a time limit
    result = backtrack([], set(), attempt_deadline)
    if len(result) > best_len:
        best_seq = list(result)
        best_len = len(result)

    if time.time() > DEADLINE:
        return best_seq

    # Try from greedy prefixes of various lengths
    for prefix_len in [3, 4, 5]:
        if time.time() > DEADLINE:
            break
        if prefix_len > len(greedy_seq):
            continue

        prefix = greedy_seq[:prefix_len]
        # Compute forbidden for prefix
        forbidden = set()
        n = len(prefix)
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += prefix[j]
                forbidden.add(total)

        # Try to extend
        attempt_deadline = min(DEADLINE, time.time() + 15)
        result = backtrack(prefix, forbidden, attempt_deadline)
        if len(result) > best_len:
            best_seq = list(result)
            best_len = len(result)

    return best_seq
# EVOLVE-BLOCK-END
