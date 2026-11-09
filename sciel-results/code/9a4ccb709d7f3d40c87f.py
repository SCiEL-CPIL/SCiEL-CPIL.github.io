N = 256

import random
import time
import numpy as np


def construct():
    start_time = time.time()
    deadline = start_time + 28.0  # 2s buffer before 30s limit

    def build_sums(selected):
        """Build the set of all pair sums a+b with a<=b from selected."""
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                sums.add(a + b)
        return sums

    def is_sidon(selected):
        """Check if selected is a valid Sidon set."""
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True

    def check_maximal_fast(selected, sums):
        """Check if set is inclusion-maximal. Returns True if maximal.
        Uses vectorized NumPy for speed."""
        sel_set = set(selected)
        if not selected:
            return False
        sel_arr = np.array(selected, dtype=np.int32)
        # For each missing x, check if all new sums x+a (a in selected, a<=x) and 2*x
        # conflict with existing sums
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            # New sums: x + a for a in selected where a <= x (to avoid double counting with a>x handled when a is processed)
            # Actually for maximality we just need: adding x creates a duplicate sum
            # New sums are {x+a : a in selected} union {2x}
            # We need at least one of these to be in sums (existing)
            # OR one of the new sums must equal another new sum (internal conflict)
            # Internal conflict: x+a1 == x+a2 implies a1==a2, so no internal conflict among x+a
            # x+a == 2x implies a == x, but x not in selected, so no
            # So we just need: any (x+a) in sums, OR 2x in sums
            # Vectorized: check if any (x + sel_arr) is in sums
            new_sums_arr = x + sel_arr
            # Also check 2*x
            two_x = 2 * x
            # Check if two_x is in sums
            if two_x in sums:
                continue  # blocked by 2x
            # Check if any x+a is in sums
            # Convert to set for fast lookup - but sums is already a set
            # Use numpy to find intersection
            # For small selected, just loop
            blocked = False
            for s in new_sums_arr:
                if s in sums:
                    blocked = True
                    break
            if not blocked:
                return False  # x can be added, not maximal
        return True

    def greedy_randomized():
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)

        selected = []
        sums = set()

        for x in candidates:
            added_sums = [x + a for a in selected] + [2 * x]
            if len(set(added_sums)) != len(added_sums):
                continue
            if sums.intersection(added_sums):
                continue
            selected.append(x)
            sums.update(added_sums)

        return selected

    def greedy_deterministic():
        selected = []
        sums = set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected

    best_set = None
    best_size = float('inf')

    # Phase 1: Many fast greedy restarts
    phase1_deadline = deadline - 12.0
    while time.time() < phase1_deadline:
        current_set = greedy_randomized()
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set

    if best_set is None:
        best_set = greedy_deterministic()
        best_size = len(best_set)

    # Phase 2: Local search - iterative removal with efficient checks
    # Key insight: after removing x, the new sums = old_sums - {x+a : a in selected} - {2x}
    # We can maintain sums incrementally

    def try_remove_one(selected, sums):
        """Try to remove one element from selected while maintaining maximality.
        Returns (new_selected, new_sums) or (None, None)."""
        n = len(selected)
        if n <= 1:
            return None, None

        # Try each removal in random order for diversity
        order = list(range(n))
        random.shuffle(order)

        for idx in order:
            x = selected[idx]
            # Compute sums to remove: all sums involving x
            # These are: x + a for all a in selected (a != x, but since a<=x or a>x, we have x+a for all a)
            # Plus 2x
            # Actually sums involving x: {x + a : a in selected} union {2x}
            # But note: x+a where a is in selected includes x+x=2x
            # So sums to remove = {x + a : a in selected}
            sums_to_remove = {x + a for a in selected}

            new_sums = sums - sums_to_remove
            new_selected = selected[:idx] + selected[idx+1:]

            if not new_selected:
                continue

            # Check if new set is still maximal
            if check_maximal_fast(new_selected, new_sums):
                return new_selected, new_sums

        return None, None

    # Phase 2: Iterative removal
    phase2_deadline = deadline - 3.0
    current_sums = build_sums(best_set)
    no_improvement_count = 0

    while time.time() < phase2_deadline:
        if not best_set:
            break

        new_selected, new_sums = try_remove_one(best_set, current_sums)
        if new_selected is not None and len(new_selected) < best_size:
            best_set = new_selected
            best_size = len(best_set)
            current_sums = new_sums
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > 5:
                # Try fresh greedy and restart local search
                current_set = greedy_randomized()
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = current_set
                    current_sums = build_sums(best_set)
                no_improvement_count = 0

    # Phase 3: Final polish - try more aggressive multi-step removals
    phase3_deadline = deadline - 1.0
    if best_set and time.time() < phase3_deadline:
        current_sums = build_sums(best_set)
        # Try removing up to 2 elements at a time
        for _ in range(50):
            if time.time() >= phase3_deadline:
                break
            if not best_set:
                break
            n = len(best_set)
            if n <= 2:
                break
            # Try removing pairs
            found = False
            for i in range(n):
                for j in range(i + 1, n):
                    if time.time() >= phase3_deadline:
                        break
                    x1, x2 = best_set[i], best_set[j]
                    new_selected = best_set[:i] + best_set[i+1:j] + best_set[j+1:]
                    if not new_selected:
                        continue
                    new_sums = build_sums(new_selected)
                    if check_maximal_fast(new_selected, new_sums):
                        best_set = new_selected
                        best_size = len(best_set)
                        current_sums = new_sums
                        found = True
                        break
                if found or time.time() >= phase3_deadline:
                    break
            if not found:
                break

    # Final verification
    def verify_final(selected):
        sel_set = set(selected)
        if len(sel_set) != len(selected):
            return False
        if not selected:
            return False
        if any(x < 1 or x > N for x in selected):
            return False
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        # Check maximality
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            # Check if adding x would create a duplicate sum
            # New sums: {x+a for a in selected} union {2x}
            # Internal: x+a1 == x+a2 => a1==a2, OK
            # x+a == 2x => a==x, but x not in selected, OK
            # So just check if any new sum is in existing sums
            two_x = 2 * x
            if two_x in sums:
                continue
            for a in selected:
                if (x + a) in sums:
                    break
            else:
                return False  # x can be added
        return True

    if not verify_final(best_set):
        # Fallback: deterministic greedy
        best_set = greedy_deterministic()

    return best_set
