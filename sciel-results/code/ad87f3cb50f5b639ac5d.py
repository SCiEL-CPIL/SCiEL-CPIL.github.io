"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85.0
    random.seed(42)

    def try_construct(start_val=1, end_val=257, skip_set=frozenset()):
        values = []
        sums = set()
        for x in range(start_val, end_val):
            if time.time() > deadline - 1:
                return values
            if x in skip_set:
                continue
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    # First pass: standard greedy from 1 to 256
    result = try_construct(1, 257)
    best_len = len(result)
    best_seq = result

    # Try different starting offsets
    if time.time() < deadline - 15.0:
        for offset in range(1, 15):
            if time.time() > deadline - 5.0:
                break
            candidate = try_construct(1 + offset, 257)
            if len(candidate) > best_len:
                best_len = len(candidate)
                best_seq = candidate

    # Try skipping specific small values to see if it allows longer sequences
    # The idea is that sometimes skipping a "bad" small value allows many more
    # larger values to fit
    if time.time() < deadline - 20.0:
        # Try skipping different combinations of small values
        small_values = list(range(1, 20))
        
        # Try skipping individual small values
        for skip_val in small_values:
            if time.time() > deadline - 5.0:
                break
            skip_set = frozenset([skip_val])
            candidate = try_construct(1, 257, skip_set)
            if len(candidate) > best_len:
                best_len = len(candidate)
                best_seq = candidate

        # Try skipping pairs of small values
        if time.time() < deadline - 10.0:
            for i in range(len(small_values)):
                for j in range(i + 1, len(small_values)):
                    if time.time() > deadline - 5.0:
                        break
                    skip_set = frozenset([small_values[i], small_values[j]])
                    candidate = try_construct(1, 257, skip_set)
                    if len(candidate) > best_len:
                        best_len = len(candidate)
                        best_seq = candidate

    # Try a local search approach: start with greedy, then try to insert
    # missing values into the sequence
    if time.time() < deadline - 10.0:
        current = list(best_seq)
        used_sums = set()
        # Recompute sums for current sequence
        for i in range(len(current)):
            total = 0
            for j in range(i, len(current)):
                total += current[j]
                used_sums.add(total)
        
        # Try to insert each missing value
        for x in range(1, 257):
            if time.time() > deadline - 5.0:
                break
            if x in current:
                continue
            # Try inserting at different positions
            for pos in range(len(current) + 1):
                # Check if insertion maintains strictly increasing order
                if pos > 0 and current[pos - 1] >= x:
                    continue
                if pos < len(current) and current[pos] <= x:
                    continue
                # Check if all new consecutive block sums are distinct
                new_sums = set()
                # Sums ending at position pos (the new element)
                total = 0
                for j in range(pos, max(pos - 1, -1), -1):
                    if j == pos:
                        total = x
                    else:
                        total += current[j]
                    new_sums.add(total)
                # Sums starting at position pos and going right
                total = 0
                for j in range(pos, len(current)):
                    if j == pos:
                        total = x
                    else:
                        total += current[j]
                    new_sums.add(total)
                # Check if any new sum conflicts with existing sums
                if new_sums.intersection(used_sums):
                    continue
                # Check if new sums are all distinct
                if len(new_sums) != len(list(new_sums)):
                    continue
                # This insertion is valid
                new_seq = current[:pos] + [x] + current[pos:]
                # Verify it's strictly increasing
                valid = True
                for k in range(len(new_seq) - 1):
                    if new_seq[k] >= new_seq[k + 1]:
                        valid = False
                        break
                if not valid:
                    continue
                current = new_seq
                # Recompute used sums
                used_sums = set()
                for i in range(len(current)):
                    total = 0
                    for j in range(i, len(current)):
                        total += current[j]
                        used_sums.add(total)
                break

        if len(current) > best_len:
            best_len = len(current)
            best_seq = current

    return best_seq
# EVOLVE-BLOCK-END
