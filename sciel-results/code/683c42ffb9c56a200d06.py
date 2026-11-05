import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 58

    # --- Phase 1: Beam search with efficient incremental product tracking ---
    # Each beam: (seq, product_set, last_val)
    # When appending c to seq, new products are: c, c*seq[-1], c*seq[-1]*seq[-2], ...
    # We only need to check these O(n) new products against the existing set.

    def beam_search(max_time):
        """Beam search over sequences, maintaining product sets incrementally."""
        BEAM_WIDTH = 400
        MAX_CANDIDATES = 30  # how many next values to try per beam

        # Initial beam: empty sequence
        beams = [([], set())]
        best_seq = []
        best_len = 0

        while time.time() < max_time:
            new_beams = []

            for seq, prods in beams:
                if not seq:
                    last_val = 1
                else:
                    last_val = seq[-1]

                # Try candidates greater than last_val
                # We look at a window of candidates to balance exploration
                start_c = last_val + 1
                end_c = min(last_val + MAX_CANDIDATES + 1, 129)

                for c in range(start_c, end_c):
                    # Compute new tail products: c, c*seq[-1], c*seq[-1]*seq[-2], ...
                    new_prods = []
                    prod = c
                    new_prods.append(prod)
                    for a in reversed(seq):
                        prod *= a
                        new_prods.append(prod)

                    # Check for collision
                    has_collision = False
                    for p in new_prods:
                        if p in prods:
                            has_collision = True
                            break

                    if not has_collision:
                        new_seq = seq + [c]
                        new_prods_set = prods.union(new_prods)
                        new_beams.append((new_seq, new_prods_set))

                        if len(new_seq) > best_len:
                            best_len = len(new_seq)
                            best_seq = new_seq[:]

            if not new_beams:
                break

            # Prune: sort by length desc, then last value asc (more room to grow)
            def beam_sort_key(b):
                seq, prods = b
                return (-len(seq), seq[-1] if seq else 0)

            new_beams.sort(key=beam_sort_key)
            beams = new_beams[:BEAM_WIDTH]

        return best_seq, best_len

    # --- Phase 2: Greedy insertion to fill gaps ---
    def is_valid_full(s):
        """Full validation using prefix products."""
        n = len(s)
        if n == 0:
            return True
        pref = [1] * (n + 1)
        for i in range(n):
            pref[i + 1] = pref[i] * s[i]
        prods = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                p = pref[j] // pref[i]
                if p in prods:
                    return False
                prods.add(p)
        return True

    def greedy_insert(seed, deadline):
        """Try to insert all remaining numbers into a valid seed."""
        seed = sorted(seed)
        if not is_valid_full(seed):
            return seed

        current_seq = seed[:]
        current_set = set(current_seq)
        all_numbers = list(range(2, 129))
        remaining = [x for x in all_numbers if x not in current_set]

        for c in remaining:
            if time.time() > deadline - 5:
                break

            pos = 0
            while pos < len(current_seq) and current_seq[pos] < c:
                pos += 1

            if pos < len(current_seq) and current_seq[pos] == c:
                continue

            new_seq = current_seq[:pos] + [c] + current_seq[pos:]

            if is_valid_full(new_seq):
                current_seq = new_seq
                current_set.add(c)

        return current_seq

    def local_search(seq, deadline):
        """Try single removal + addition to improve."""
        all_numbers = list(range(2, 129))
        current_seq = seq[:]
        current_set = set(current_seq)
        improved = True

        while improved and time.time() < deadline - 10:
            improved = False
            missing = [x for x in all_numbers if x not in current_set]

            # Try adding a missing element without removing
            for add_val in missing:
                if time.time() > deadline - 10:
                    break
                pos = 0
                while pos < len(current_seq) and current_seq[pos] < add_val:
                    pos += 1
                candidate = current_seq[:pos] + [add_val] + current_seq[pos:]
                if is_valid_full(candidate) and len(candidate) > len(current_seq):
                    current_seq = candidate
                    current_set = set(candidate)
                    improved = True
                    break

            # Try removing one element and adding a different one
            if not improved:
                for idx in range(len(current_seq)):
                    if time.time() > deadline - 10:
                        break
                    without = current_seq[:idx] + current_seq[idx + 1:]
                    for add_val in missing:
                        if time.time() > deadline - 10:
                            break
                        pos = 0
                        while pos < len(without) and without[pos] < add_val:
                            pos += 1
                        candidate = without[:pos] + [add_val] + without[pos:]
                        if is_valid_full(candidate) and len(candidate) > len(current_seq):
                            current_seq = candidate
                            current_set = set(candidate)
                            improved = True
                            break
                    if improved:
                        break

        return current_seq

    # --- Main logic ---
    # Phase 1: Beam search
    beam_seq, beam_len = beam_search(deadline - 25)

    if beam_len == 0:
        # Fallback: simple greedy
        beam_seq = []
        prods = set()
        for x in range(2, 129):
            new_prods = [x]
            prod = x
            for a in reversed(beam_seq):
                prod *= a
                new_prods.append(prod)
            if not any(p in prods for p in new_prods):
                beam_seq.append(x)
                prods.update(new_prods)
        beam_len = len(beam_seq)

    # Phase 2: Greedy insertion on the beam result
    if time.time() < deadline - 15:
        inserted_seq = greedy_insert(beam_seq, deadline)
        if len(inserted_seq) > beam_len:
            beam_seq = inserted_seq
            beam_len = len(beam_seq)

    # Phase 3: Local search
    if time.time() < deadline - 10:
        refined_seq = local_search(beam_seq, deadline)
        if len(refined_seq) > beam_len:
            beam_seq = refined_seq
            beam_len = len(beam_seq)

    # Final validation
    if not is_valid_full(beam_seq):
        # Fallback to simple greedy if something went wrong
        beam_seq = []
        prods = set()
        for x in range(2, 129):
            new_prods = [x]
            prod = x
            for a in reversed(beam_seq):
                prod *= a
                new_prods.append(prod)
            if not any(p in prods for p in new_prods):
                beam_seq.append(x)
                prods.update(new_prods)

    return beam_seq
