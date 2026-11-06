"""
Extended Sidon set construction using systematic backtracking with sum-set pruning
plus heuristic local search.
Goal: Maximize size of set A subset [1, 284] containing {1, 2, 5, 11}
such that all pairwise sums a+b (a<=b) are distinct.

Strategy:
1. Fast greedy initialization to get a baseline.
2. Systematic backtracking with aggressive pruning to find larger valid sets.
3. Local search (swaps) for refinement.
4. Time-bounded: return best found if search doesn't complete.
"""
import time
import sys
import random
from typing import Set, List, Tuple

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    max_val = 284
    n = max_val

    def get_sums(s: Set[int]) -> Set[int]:
        sums = set()
        sl = sorted(s)
        for i in range(len(sl)):
            for j in range(i, len(sl)):
                sums.add(sl[i] + sl[j])
        return sums

    # Precompute conflict table: conflict[x][y] = True if adding x to a set
    # containing y would create a sum collision (i.e., 2*x == x+y or x+y already used).
    # More precisely: for a candidate x, it conflicts with existing element y if
    # (x + y) is in used_sums OR 2*x is in used_sums OR x+y equals 2*y (i.e., x==y, already excluded).
    # The key insight: x conflicts with y iff (x+y) in used_sums or (2*x) in used_sums.
    # We can precompute for each x, which y values would make x+y fall in a "forbidden" sum range.
    # But used_sums changes dynamically, so instead we just make the check O(|s|) but with early exit.
    
    def can_add(x: int, s: Set[int], used_sums: Set[int]) -> bool:
        # Fast path: check 2*x first
        if 2 * x in used_sums:
            return False
        # Then check x+y for each y in s
        for y in s:
            if x + y in used_sums:
                return False
        return True

    def add_to_sums(x: int, used_sums: Set[int], s: Set[int]) -> Set[int]:
        new_sums = set(used_sums)
        for y in s:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        return new_sums

    # --- Phase 1: Greedy initialization ---
    def greedy_init() -> List[int]:
        current_set = set(prescribed)
        used_sums = get_sums(current_set)
        for x in range(1, n + 1):
            if x in current_set:
                continue
            if can_add(x, current_set, used_sums):
                current_set.add(x)
                used_sums = add_to_sums(x, used_sums, current_set)
        return sorted(current_set)

    best_set = greedy_init()
    best_size = len(best_set)

    # --- Phase 2: Beam search with pruning ---
    # Use beam search instead of full backtracking to avoid exponential blowup
    # while still exploring more than pure greedy.
    
    def beam_search(target_size: int, max_time: float, beam_width: int = 50) -> bool:
        nonlocal best_set, best_size
        
        initial_set = set(prescribed)
        initial_sums = get_sums(initial_set)
        
        remaining_to_add = target_size - len(prescribed)
        if remaining_to_add <= 0:
            return True
        
        # Each beam entry: (frozenset, used_sums, max_element)
        beam = [(frozenset(initial_set), initial_sums, max(initial_set))]
        
        for depth in range(remaining_to_add):
            if time.time() > max_time:
                return False
            
            next_beam = []
            seen_states = set()
            
            for curr_set, curr_sums, curr_max in beam:
                # Try adding each candidate > curr_max
                for x in range(curr_max + 1, n + 1):
                    if not can_add(x, curr_set, curr_sums):
                        continue
                    
                    new_set = curr_set | {x}
                    new_sums = add_to_sums(x, curr_sums, curr_set)
                    
                    # Use frozenset as key for deduplication
                    state_key = frozenset(new_set)
                    if state_key in seen_states:
                        continue
                    seen_states.add(state_key)
                    
                    next_beam.append((new_set, new_sums, x))
            
            if not next_beam:
                return False
            
            # Prune beam: keep top beam_width states
            # Score: prefer states with smaller max element (more room) and
            # fewer "tight" sums (sums close to 2*n)
            def beam_score(entry):
                s, sums, mx = entry
                # Smaller max is better
                # Fewer high sums is better (more room for future additions)
                high_sums = sum(1 for sm in sums if sm > 2 * n - 20)
                return (mx, high_sums)
            
            next_beam.sort(key=beam_score)
            beam = next_beam[:beam_width]
            
            # Check if we found a complete solution
            if depth + 1 == remaining_to_add:
                # Any state in beam is a valid solution
                best_set = sorted(beam[0][0])
                best_size = target_size
                return True
        
        return False

    # Try to find sets of increasing size using beam search
    for target in range(best_size + 1, min(best_size + 6, 21)):
        if time.time() > deadline - 20:
            break
        
        # Allocate time for this target size
        time_alloc = min(25, (deadline - 20 - time.time()) * 0.4)
        
        if time_alloc < 5:
            break
        
        found = beam_search(target, time.time() + time_alloc)
        if found:
            continue
        else:
            break

    # --- Helper: greedy extend a set to local maximum ---
    def greedy_extend(s: Set[int], used_sums: Set[int]) -> Tuple[Set[int], Set[int]]:
        """Add as many valid elements as possible in increasing order."""
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if x in s:
                    continue
                if can_add(x, s, used_sums):
                    s.add(x)
                    used_sums = add_to_sums(x, used_sums, s)
                    changed = True
        return s, used_sums

    # --- Phase 3: Local search refinement ---
    # Use targeted swaps to escape local maxima
    if time.time() < deadline - 10:
        rng = random.Random(42)
        current_set = set(best_set)
        current_sums = get_sums(current_set)
        # First, greedily extend to make sure we're at a local max
        current_set, current_sums = greedy_extend(current_set, current_sums)
        if len(current_set) > best_size:
            best_set = sorted(current_set)
            best_size = len(best_set)

        # Precompute for each candidate x not in current_set, how many elements
        # in current_set it conflicts with (i.e., x+y in used_sums for some y, or 2x in used_sums).
        # This helps us pick "almost-fitting" candidates for swaps.
        
        def conflict_count(x: int, s: Set[int], used_sums: Set[int]) -> int:
            cnt = 0
            if 2 * x in used_sums:
                cnt += 1
            for y in s:
                if x + y in used_sums:
                    cnt += 1
            return cnt

        for iteration in range(80000):
            if time.time() > deadline - 5:
                break

            # Try a swap: remove one non-prescribed element, add one new one
            removable = [x for x in current_set if x not in prescribed_set]
            if not removable:
                break
            
            # Pick removal element: weight toward elements that "block" many candidates
            x_remove = rng.choice(removable)
            
            # Find candidates that would fit if we removed x_remove
            # Compute sums after removing x_remove
            reduced_set = set(current_set)
            reduced_set.discard(x_remove)
            reduced_sums = get_sums(reduced_set)
            
            # Find all valid additions to reduced_set
            valid_adds = []
            for x in range(1, n + 1):
                if x in current_set:
                    continue
                if can_add(x, reduced_set, reduced_sums):
                    valid_adds.append(x)
            
            if not valid_adds:
                continue
            
            # Prefer candidates that are "close" to being valid in the full set
            # (i.e., they only conflict with x_remove)
            scored_adds = []
            for x in valid_adds:
                # How many conflicts did x have with the full set?
                # If x is valid after removing x_remove, its only conflict was with x_remove
                # Score: prefer smaller x (more room for future additions)
                scored_adds.append((x, x))
            
            # Sort by value (smaller first) and pick randomly from top candidates
            scored_adds.sort(key=lambda t: t[0])
            # Pick from top 20% or all if fewer
            top_k = max(1, len(scored_adds) // 5)
            x_add = scored_adds[rng.randint(0, top_k - 1)][0]

            new_set = reduced_set | {x_add}
            new_sums = get_sums(new_set)

            # Try to extend: add any valid element in greedy order
            extended = False
            for y in range(1, n + 1):
                if y in new_set:
                    continue
                if can_add(y, new_set, new_sums):
                    new_set.add(y)
                    new_sums = get_sums(new_set)
                    if len(new_set) > best_size:
                        best_set = sorted(new_set)
                        best_size = len(best_set)
                    extended = True
                    break

            # Accept swap if it led to extension, or with simulated annealing probability
            if extended:
                current_set = new_set
                current_sums = new_sums
            else:
                # Simulated annealing: accept with decreasing probability
                # Temperature decreases over iterations
                temp = max(0.01, 0.5 * (1.0 - iteration / 80000.0))
                if rng.random() < temp:
                    current_set = new_set
                    current_sums = new_sums

    # --- Phase 4: Randomized greedy with perturbation ---
    if time.time() < deadline - 5:
        rng2 = random.Random(12345)
        for trial in range(50):
            if time.time() > deadline - 3:
                break
            current_set = set(prescribed)
            used_sums = get_sums(current_set)
            candidates = [x for x in range(1, n + 1) if x not in prescribed_set]
            # Mix of random and value-based ordering for diversity
            if trial % 3 == 0:
                # Pure random
                shuffled = candidates[:]
                rng2.shuffle(shuffled)
            elif trial % 3 == 1:
                # Mostly random, slightly biased toward smaller values
                shuffled = [(rng2.random() * 0.8 + x / n * 0.2, x) for x in candidates]
                shuffled.sort(key=lambda t: t[0])
                shuffled = [x for _, x in shuffled]
            else:
                # Mostly value-based, slightly random
                shuffled = [(x / n * 0.8 + rng2.random() * 0.2, x) for x in candidates]
                shuffled.sort(key=lambda t: t[0])
                shuffled = [x for _, x in shuffled]
            for x in shuffled:
                if can_add(x, current_set, used_sums):
                    current_set.add(x)
                    used_sums = add_to_sums(x, used_sums, current_set)
            current_set, used_sums = greedy_extend(current_set, used_sums)
            if len(current_set) > best_size:
                best_set = sorted(current_set)
                best_size = len(best_set)

    # Final verification
    final_set = set(best_set)
    sums = get_sums(final_set)
    expected = len(final_set) * (len(final_set) + 1) // 2
    if len(sums) != expected:
        best_set = list(prescribed)

    return best_set

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Size: {len(res)}, Time: {end-start:.2f}s")
    # Verify
    sums = set()
    valid = True
    r = sorted(res)
    for i in range(len(r)):
        for j in range(i, len(r)):
            s = r[i] + r[j]
            if s in sums:
                valid = False
            sums.add(s)
    if not all(x in res for x in [1, 2, 5, 11]):
        valid = False
    if not all(1 <= x <= 284 for x in res):
        valid = False
    if len(set(res)) != len(res):
        valid = False
    print(f"Valid: {valid}")
