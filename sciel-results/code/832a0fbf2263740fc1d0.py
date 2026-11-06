import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin within 90s kill

    N = 256
    # Precompute incompatibility: incompat[i][j] = True if (i*j)%(i+j)==0
    # Use 1-indexed: index 0..256, ignore 0
    incompat = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                incompat[a].add(b)
                incompat[b].add(a)

    # Initial set: all odd numbers (known valid, size 128)
    S = set(range(1, N + 1, 2))
    S_list = list(S)
    best_S = set(S)
    best_size = len(best_S)

    # Greedy add: try to add evens in random order
    evens = list(range(2, N + 1, 2))
    random.shuffle(evens)
    for e in evens:
        if time.time() > deadline:
            break
        if not (incompat[e] & S):
            S.add(e)
            S_list.append(e)

    best_S = set(S)
    best_size = len(S)

    # Simulated annealing
    T = 50.0
    cooling = 0.9995
    swap_prob = 0.5
    min_T = 0.01

    while time.time() < deadline and T > min_T:
        # Try to add a random element not in S
        candidates = [x for x in range(1, N + 1) if x not in S]
        if candidates:
            x = random.choice(candidates)
            conflicts = incompat[x] & S
            if not conflicts:
                # Accept: improves size
                S.add(x)
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
            else:
                # Delta = -len(conflicts) (removing conflicts would be needed, but we don't remove)
                # For add attempt, if conflicts exist, reject (can't add without removing)
                pass

        # Try a swap: remove one from S, add one not in S
        if random.random() < swap_prob and len(S) > 0:
            # Pick a random element to remove
            to_remove = random.choice(list(S))
            # Pick a random element to add (not in S)
            candidates = [x for x in range(1, N + 1) if x not in S]
            if candidates:
                to_add = random.choice(candidates)
                # Compute delta: 
                # Removing to_remove: we lose conflicts with to_remove (but those are already in S, so they stay)
                # Adding to_add: we gain conflicts = incompat[to_add] & (S - {to_remove})
                new_conflicts = len(incompat[to_add] & (S - {to_remove}))
                # We also need to check that removing to_remove doesn't create issues (it doesn't, since it's just removal)
                # Delta in "energy" = new_conflicts - 0 (old had no conflict for to_remove since it was in S)
                # Actually, the "energy" of a set is the number of conflicting pairs. 
                # Current energy E. After swap: E' = E - (conflicts involving to_remove) + (conflicts involving to_add with S\{to_remove})
                # conflicts involving to_remove = len(incompat[to_remove] & S) - (1 if to_remove in incompat[to_remove] else 0) but we don't have self
                # Since to_remove is in S, conflicts_involving_to_remove = len(incompat[to_remove] & S)
                # But wait: incompat[to_remove] & S includes to_remove? No, since we didn't add self. So it's fine.
                old_conflicts_to_remove = len(incompat[to_remove] & S)
                delta = new_conflicts - old_conflicts_to_remove
                # Accept if delta <= 0 or with probability exp(-delta/T)
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    S.discard(to_remove)
                    S.add(to_add)
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)

        T *= cooling

    # Return the best set found
    return list(best_S)
