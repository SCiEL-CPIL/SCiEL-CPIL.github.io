import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    def time_left():
        return deadline - time.time()

    def compute_block_products(seq):
        """Return set of all nonempty consecutive block products."""
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods

    def try_extend(seq, x, prods):
        """Try to append x to seq. Return new prods set or None if collision."""
        n = len(seq)
        new_prods = set()
        # Products involving x: x, x*seq[-1], x*seq[-2]*seq[-1], ...
        p = x
        new_prods.add(p)
        for k in range(n - 1, -1, -1):
            p *= seq[k]
            if p in prods:
                return None
            new_prods.add(p)
        # Also need to check: the new products don't collide with each other
        # (they can't since all involve x and different suffix lengths give different values
        #  for strictly increasing positive integers > 1)
        return prods | new_prods

    def greedy_build():
        """Greedy: pick smallest possible next number."""
        seq = []
        prods = set()
        for x in range(2, 129):
            if time_left() < 0.5:
                break
            new_prods = try_extend(seq, x, prods)
            if new_prods is not None:
                seq.append(x)
                prods = new_prods
        return seq, prods

    def beam_search(beam_width=50, max_depth=128):
        """Beam search: maintain top-K partial sequences."""
        # Each beam entry: (seq, prods_set, score)
        # Score = length (primary), we want max length
        beam = [([], set(), 0)]
        best_seq = []
        best_len = 0

        for depth in range(1, max_depth + 1):
            if time_left() < 2.0:
                break

            candidates = []
            for seq, prods, _ in beam:
                last = seq[-1] if seq else 1
                # Try each candidate x > last
                for x in range(last + 1, 129):
                    if time_left() < 0.5:
                        break
                    new_prods = try_extend(seq, x, prods)
                    if new_prods is not None:
                        new_len = len(seq) + 1
                        candidates.append((seq + [x], new_prods, new_len))

            if not candidates:
                break

            # Sort by length descending, keep top beam_width
            candidates.sort(key=lambda t: t[2], reverse=True)
            # Deduplicate: keep only unique sequences
            seen = set()
            new_beam = []
            for c in candidates:
                key = tuple(c[0])
                if key not in seen:
                    seen.add(key)
                    new_beam.append(c)
                    if len(new_beam) >= beam_width:
                        break

            # Update best
            if new_beam:
                top_len = new_beam[0][2]
                if top_len > best_len:
                    best_len = top_len
                    best_seq = new_beam[0][0]

            beam = new_beam

        return best_seq

    def local_improve(seq, max_iter=200):
        """Try to improve seq by inserting missing numbers or reordering."""
        if not seq:
            return seq
        seq = list(seq)
        prods = compute_block_products(seq)
        used = set(seq)
        improved = True
        iteration = 0
        while improved and iteration < max_iter and time_left() > 1.0:
            improved = False
            iteration += 1
            # Try inserting each unused number at each position
            for x in range(2, 129):
                if x in used:
                    continue
                for pos in range(len(seq) + 1):
                    # Check strictly increasing
                    if pos > 0 and seq[pos - 1] >= x:
                        continue
                    if pos < len(seq) and seq[pos] <= x:
                        continue
                    new_seq = seq[:pos] + [x] + seq[pos:]
                    new_prods = compute_block_products(new_seq)
                    if len(new_prods) == len(new_seq) * (len(new_seq) + 1) // 2:
                        # All products distinct
                        seq = new_seq
                        prods = new_prods
                        used.add(x)
                        improved = True
                        break
                if improved:
                    break
        return seq

    def local_remove_improve(seq, max_iter=100):
        """Try removing elements to allow more insertions."""
        if not seq:
            return seq
        best = list(seq)
        for _ in range(max_iter):
            if time_left() < 1.0:
                break
            improved = False
            # Try removing each element
            for i in range(len(best)):
                trial = best[:i] + best[i+1:]
                if not trial:
                    continue
                # Try to greedily extend trial
                trial_seq, _ = greedy_build_from(trial)
                if len(trial_seq) > len(best):
                    best = trial_seq
                    improved = True
                    break
            if not improved:
                break
        return best

    def greedy_build_from(initial_seq):
        """Greedy extend from a given initial sequence."""
        seq = list(initial_seq)
        prods = compute_block_products(seq)
        for x in range(seq[-1] + 1 if seq else 2, 129):
            if time_left() < 0.5:
                break
            new_prods = try_extend(seq, x, prods)
            if new_prods is not None:
                seq.append(x)
                prods = new_prods
        return seq, prods

    # Phase 1: Greedy seed
    greedy_seq, _ = greedy_build()
    best_seq = greedy_seq
    best_len = len(greedy_seq)

    # Phase 2: Beam search
    beam_seq = beam_search(beam_width=80)
    if len(beam_seq) > best_len:
        best_seq = beam_seq
        best_len = len(beam_seq)

    # Phase 3: Local improvement on best sequence
    if time_left() > 2.0:
        improved = local_improve(best_seq)
        if len(improved) > best_len:
            best_seq = improved
            best_len = len(improved)

    # Phase 4: Try multiple random restarts with beam search
    if time_left() > 5.0:
        random.seed(42)
        for restart in range(10):
            if time_left() < 3.0:
                break
            # Randomized greedy: with some probability skip a candidate
            seq = []
            prods = set()
            for x in range(2, 129):
                if time_left() < 0.5:
                    break
                # 80% chance to try, 20% skip
                if random.random() < 0.8:
                    new_prods = try_extend(seq, x, prods)
                    if new_prods is not None:
                        seq.append(x)
                        prods = new_prods
            if len(seq) > best_len:
                best_seq = seq
                best_len = len(seq)
            # Local improve
            if time_left() > 2.0:
                improved = local_improve(seq)
                if len(improved) > best_len:
                    best_seq = improved
                    best_len = len(improved)

    # Phase 5: Final local improvement on best
    if time_left() > 1.0:
        final = local_improve(best_seq, max_iter=100)
        if len(final) > best_len:
            best_seq = final
            best_len = len(final)

    return best_seq
