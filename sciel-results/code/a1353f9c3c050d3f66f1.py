import time
import random
import math

N = 512

# Precompute conflicts: for each x, list of triples (u,v,w) such that {x,u,v,w} is a 4-AP
_conflicts = [[] for _ in range(N + 1)]

for a in range(1, N - 2):
    for d in range(1, (N - a) // 3 + 1):
        e = a + 3 * d
        if e > N:
            break
        b = a + d
        c = a + 2 * d
        ap = (a, b, c, e)
        for i in range(4):
            x = ap[i]
            others = tuple(ap[j] for j in range(4) if j != i)
            _conflicts[x].append(others)


def _can_add(x, S):
    """Check if x can be added to set S without creating a 4-AP."""
    for u, v, w in _conflicts[x]:
        if u in S and v in S and w in S:
            return False
    return True


def _greedy(order, S=None):
    """Greedy construction given an order of elements."""
    if S is None:
        S = set()
    for x in order:
        if x not in S and _can_add(x, S):
            S.add(x)
    return S


def _init_conflict_counts(S):
    """Calculate conflict counts for a given set S (assuming S is valid, i.e., no 4-APs)."""
    counts = [0] * (N + 1)
    # Since S is valid, counts should be 0 for all x in S.
    # For x not in S, count is number of APs containing x where other 3 are in S.
    for x in range(1, N + 1):
        if x in S:
            continue
        for u, v, w in _conflicts[x]:
            if u in S and v in S and w in S:
                counts[x] += 1
    return counts


def _simulated_annealing(start_set, deadline, rng):
    """
    Simulated Annealing to maximize size of 4-AP-free set.
    """
    S = set(start_set)
    conflict_counts = _init_conflict_counts(S)
    energy = 0 # Should be 0 for valid start
    
    best_S = set(S)
    best_size = len(S)
    
    T = 5.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute list of elements for fast random choice
    all_elems = list(range(1, N + 1))
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Choose a move: Add or Remove
        # Bias towards adding if size < N/2? No, just random.
        # Probability of add vs remove can be adjusted.
        # Let's say 50% add, 50% remove.
        
        if rng.random() < 0.5:
            # Try to add
            # Pick random x not in S
            # To avoid O(N) check for "not in S", we can maintain a list of candidates?
            # Or just pick random and check.
            # With N=512, picking random and checking is fast.
            x = rng.choice(all_elems)
            if x in S:
                continue
            
            delta_E = conflict_counts[x]
            
            # Acceptance
            accept = False
            if delta_E == 0:
                accept = True
            else:
                # Metropolis criterion
                # We want to allow uphill moves (increasing energy) to escape
                # But we prioritize size. 
                # Standard SA minimizes energy. Here we want to maximize size subject to energy=0.
                # A common heuristic: Accept if delta_E < 0.
                # If delta_E > 0, accept with prob exp(-delta_E / T).
                # This might get stuck in high energy states if T is too high.
                # Let's try standard SA on Energy, but track best valid set.
                if rng.random() < math.exp(-delta_E / T):
                    accept = True
            
            if accept:
                S.add(x)
                # Update counts
                # For each AP containing x, if others in S, they become active for others
                for u, v, w in _conflicts[x]:
                    if u in S and v in S and w in S:
                        conflict_counts[u] += 1
                        conflict_counts[v] += 1
                        conflict_counts[w] += 1
                # conflict_counts[x] is already correct (it was the delta_E)
                energy += delta_E
                
                if energy == 0:
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                        
        else:
            # Try to remove
            if not S:
                continue
            x = rng.choice(list(S))
            
            delta_E = -conflict_counts[x]
            
            # Removing always decreases or maintains energy (delta_E <= 0)
            # So always accept?
            # If we remove an element with 0 conflicts, energy stays 0, size decreases.
            # We should only accept removal if it allows future additions?
            # In SA, we accept downhill moves (decreasing energy) always.
            # But here, size is the objective.
            # Let's use a combined score?
            # Score = -Energy + lambda * Size?
            # Or just: Accept removal if delta_E < 0 (energy decreases).
            # If delta_E == 0 (removing non-critical element), accept with prob exp(-SizePenalty / T)?
            # Actually, removing an element with 0 conflicts just reduces size.
            # This is a "downhill" move in size, "neutral" in energy.
            # We should be reluctant to do this.
            # Let's say: Accept removal if delta_E < 0.
            # If delta_E == 0, accept with small probability to allow shaking.
            
            if delta_E < 0:
                accept = True
            else:
                # delta_E == 0
                # Accept with probability proportional to T?
                # Or just reject?
                # If we reject all 0-energy removals, we might get stuck.
                # Let's accept with prob 0.1 if T > 1, else 0.
                if T > 1.0 and rng.random() < 0.1:
                    accept = True
                else:
                    accept = False
            
            if accept:
                S.remove(x)
                # Update counts
                for u, v, w in _conflicts[x]:
                    if u in S and v in S and w in S:
                        conflict_counts[u] -= 1
                        conflict_counts[v] -= 1
                        conflict_counts[w] -= 1
                conflict_counts[x] = 0
                energy += delta_E
                
                if energy == 0:
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)

        T *= cooling
        
    return best_S


def _local_search(S, deadline, rng):
    """
    Local search using Set representation.
    Attempts to add elements, then performs swaps to escape local optima.
    """
    current = set(S)
    current_size = len(current)
    all_elems = list(range(1, N + 1))

    while time.time() < deadline:
        improved = False

        # 1. Try to add any element
        for x in all_elems:
            if x not in current and _can_add(x, current):
                current.add(x)
                current_size += 1
                improved = True

        if improved:
            continue

        # 2. If no additions, try swaps
        attempts = 0
        max_attempts = 200

        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            x = rng.choice(all_elems)
            if x in current:
                continue

            # Check if x is blocked
            blocking_elements = set()
            is_blocked = False
            for u, v, w in _conflicts[x]:
                if u in current and v in current and w in current:
                    is_blocked = True
                    blocking_elements.update([u, v, w])

            if not is_blocked:
                # Should have been added in step 1
                current.add(x)
                current_size += 1
                improved = True
                break

            if not blocking_elements:
                continue

            # Try to remove one of the blocking elements and add x
            rem = rng.choice(list(blocking_elements))

            # Check if removing rem allows adding x
            valid_swap = True
            for u, v, w in _conflicts[x]:
                if u in current and v in current and w in current:
                    if rem not in (u, v, w):
                        valid_swap = False
                        break

            if valid_swap:
                current.discard(rem)
                current.add(x)
                improved = True
                break

        if not improved:
            # Stuck. Try a "kick": remove a few random elements and re-greedy
            if current_size > 5:
                for _ in range(3):
                    elem = rng.choice(list(current))
                    current.discard(elem)
                current_size = len(current)
                # Re-greedy fill
                for x in all_elems:
                    if x not in current and _can_add(x, current):
                        current.add(x)
                current_size = len(current)

    return current


def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    rng = random.Random(42)

    # 1. Generate multiple seeds
    seeds = []

    # Forward greedy
    seeds.append(_greedy(range(1, N + 1)))

    # Reverse greedy
    seeds.append(_greedy(range(N, 0, -1)))

    # Random greedy (a few times)
    for _ in range(5):
        order = list(range(1, N + 1))
        rng.shuffle(order)
        seeds.append(_greedy(order))

    # Find best seed
    best = max(seeds, key=len)
    best_size = len(best)

    # 2. Local search on best seed
    if time.time() < deadline - 10:
        improved = _local_search(best, deadline - 5, rng)
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)

    # 3. Simulated Annealing restarts
    # Run SA for a portion of the remaining time
    sa_deadline = deadline - 5
    if time.time() < sa_deadline:
        # Run SA on the current best
        sa_best = _simulated_annealing(best, sa_deadline, rng)
        if len(sa_best) > best_size:
            best = sa_best
            best_size = len(best)

        # Run SA on other seeds if time permits
        # We can run parallel-ish by just iterating through seeds
        for seed in seeds:
            if time.time() >= sa_deadline:
                break
            # Skip if seed is too small
            if len(seed) < best_size - 5:
                continue
            
            # Give SA a small slice of time
            slice_deadline = min(sa_deadline, time.time() + 2.0)
            if time.time() >= slice_deadline:
                continue
            sa_candidate = _simulated_annealing(seed, slice_deadline, rng)
            if len(sa_candidate) > best_size:
                best = sa_candidate
                best_size = len(best)

    # 4. Final local search on the global best
    if time.time() < deadline - 1:
        improved = _local_search(best, deadline, rng)
        if len(improved) > best_size:
            best = improved

    # 5. Verify
    for x in best:
        for u, v, w in _conflicts[x]:
            if u in best and v in best and w in best:
                return list(_greedy(range(1, N + 1)))

    return sorted(list(best))


if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
