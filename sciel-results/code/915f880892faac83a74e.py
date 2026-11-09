"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _deadline = _t0 + 140.0  # safety margin before 150s kill

    def _elapsed():
        return time.time() - _t0

    def _is_valid_insertion(chosen, counts, x, chosen_set):
        """Check if x can be added to chosen without violating B2[2]."""
        if x in chosen_set:
            return False
        for y in chosen:
            s = x + y
            if counts.get(s, 0) + 1 > 2:
                return False
        s2 = 2 * x
        if counts.get(s2, 0) + 1 > 2:
            return False
        return True

    def _add_element(chosen, counts, chosen_set, x):
        """Add x to the set and update counts."""
        for y in chosen:
            s = x + y
            counts[s] = counts.get(s, 0) + 1
        s2 = 2 * x
        counts[s2] = counts.get(s2, 0) + 1
        chosen.append(x)
        chosen_set.add(x)

    def _remove_element(chosen, counts, chosen_set, x):
        """Remove x from the set and update counts."""
        # Find index
        idx = chosen.index(x)
        chosen.pop(idx)
        chosen_set.discard(x)
        for y in chosen:
            s = x + y
            if s in counts:
                counts[s] -= 1
                if counts[s] == 0:
                    del counts[s]
        s2 = 2 * x
        if s2 in counts:
            counts[s2] -= 1
            if counts[s2] == 0:
                del counts[s2]

    # Phase 1: Greedy baseline (fast, deterministic)
    chosen, counts = [], {}
    chosen_set = set()
    for x in range(1, 1001):
        ok = True
        for y in chosen:
            s = x + y
            if counts.get(s, 0) + 1 > 2:
                ok = False
                break
        if ok:
            s2 = 2 * x
            if counts.get(s2, 0) + 1 > 2:
                ok = False
        if ok:
            _add_element(chosen, counts, chosen_set, x)

    best_size = len(chosen)
    best_chosen = list(chosen)
    best_counts = dict(counts)
    best_chosen_set = set(chosen_set)

    if _elapsed() > _deadline - 5.0:
        return best_chosen

    # Build candidate list
    candidates = [x for x in range(1, 1001) if x not in chosen_set]
    random.shuffle(candidates)

    # Phase 2: Greedy fill - try adding candidates
    for x in candidates:
        if _elapsed() > _deadline - 10.0:
            break
        if _is_valid_insertion(chosen, counts, x, chosen_set):
            _add_element(chosen, counts, chosen_set, x)
            chosen.sort()
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_chosen = list(chosen)
                best_counts = dict(counts)
                best_chosen_set = set(chosen_set)

    if _elapsed() > _deadline - 5.0:
        return best_chosen

    # Phase 3: Improved local search with targeted swaps
    # Strategy: For each candidate, find which chosen elements block it (cause sum > 2)
    # Then try removing one blocker and adding the candidate

    def _get_blockers(chosen, counts, x, chosen_set):
        """Return list of chosen elements that block insertion of x."""
        blockers = []
        for y in chosen:
            s = x + y
            if counts.get(s, 0) + 1 > 2:
                blockers.append(y)
        s2 = 2 * x
        if counts.get(s2, 0) + 1 > 2:
            blockers.append(x)  # self-block
        return blockers

    # Multi-round local search
    for round_num in range(20):
        if _elapsed() > _deadline - 10.0:
            break

        # Refresh candidates
        remaining = [x for x in range(1, 1001) if x not in chosen_set]
        random.shuffle(remaining)

        improved = False
        for x in remaining[:200]:  # Limit per round
            if _elapsed() > _deadline - 10.0:
                break

            if _is_valid_insertion(chosen, counts, x, chosen_set):
                _add_element(chosen, counts, chosen_set, x)
                chosen.sort()
                improved = True
                if len(chosen) > best_size:
                    best_size = len(chosen)
                    best_chosen = list(chosen)
                    best_counts = dict(counts)
                    best_chosen_set = set(chosen_set)
                continue

            # Try swap: remove a blocker, add x
            blockers = _get_blockers(chosen, counts, x, chosen_set)
            if not blockers:
                continue

            # Try removing each blocker (up to 3 attempts)
            random.shuffle(blockers)
            for blocker in blockers[:3]:
                if _elapsed() > _deadline - 10.0:
                    break
                if blocker == x:
                    continue

                # Remove blocker
                _remove_element(chosen, counts, chosen_set, blocker)

                # Try to add x
                if _is_valid_insertion(chosen, counts, x, chosen_set):
                    _add_element(chosen, counts, chosen_set, x)
                    chosen.sort()
                    improved = True
                    if len(chosen) > best_size:
                        best_size = len(chosen)
                        best_chosen = list(chosen)
                        best_counts = dict(counts)
                        best_chosen_set = set(chosen_set)
                    break
                else:
                    # Restore blocker
                    _add_element(chosen, counts, chosen_set, blocker)
                    chosen.sort()

        if not improved:
            # Try random swaps to escape local minima
            for _trial in range(100):
                if _elapsed() > _deadline - 10.0:
                    break
                if len(chosen) < 2:
                    break

                idx = random.randint(0, len(chosen) - 1)
                removed = chosen[idx]
                _remove_element(chosen, counts, chosen_set, removed)

                # Try to add a random candidate
                if remaining:
                    x = random.choice(remaining)
                    if x not in chosen_set and _is_valid_insertion(chosen, counts, x, chosen_set):
                        _add_element(chosen, counts, chosen_set, x)
                        chosen.sort()
                        if len(chosen) > best_size:
                            best_size = len(chosen)
                            best_chosen = list(chosen)
                            best_counts = dict(counts)
                            best_chosen_set = set(chosen_set)
                    else:
                        _add_element(chosen, counts, chosen_set, removed)
                        chosen.sort()

    if _elapsed() > _deadline - 5.0:
        return best_chosen

    # Phase 4: Final polish - try all remaining candidates
    remaining = [x for x in range(1, 1001) if x not in chosen_set]
    random.shuffle(remaining)
    for x in remaining:
        if _elapsed() > _deadline - 3.0:
            break
        if _is_valid_insertion(chosen, counts, x, chosen_set):
            _add_element(chosen, counts, chosen_set, x)
            chosen.sort()
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_chosen = list(chosen)

    return best_chosen
# EVOLVE-BLOCK-END
