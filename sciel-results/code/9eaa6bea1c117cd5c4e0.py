"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import lcm

DENOMS = list(range(2, 49))
N = len(DENOMS)

L = 1
for d in DENOMS:
    L = lcm(L, d)

WEIGHTS = [L // d for d in DENOMS]

def _greedy_init():
    """Greedy: take largest weights that fit under L."""
    s = 0
    idxs = []
    for d in sorted(DENOMS, key=lambda x: -WEIGHTS[DENOMS.index(x)]):
        w = WEIGHTS[DENOMS.index(d)]
        if s + w < L:
            s += w
            idxs.append(DENOMS.index(d))
    return s, idxs

def _local_search(sum_w, indices, deadline):
    """Hill-climbing local search with 1-add, 1-swap, 2-swap moves."""
    in_set = set(indices)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding one element
        for i in range(N):
            if i not in in_set:
                new_s = sum_w + WEIGHTS[i]
                if new_s < L and new_s > sum_w:
                    sum_w = new_s
                    indices.append(i)
                    in_set.add(i)
                    improved = True
                    break
        # Try removing one element that allows adding a larger one
        for j in list(indices):
            wj = WEIGHTS[j]
            for i in range(N):
                if i in in_set or i == j:
                    continue
                wi = WEIGHTS[i]
                new_s = sum_w - wj + wi
                if new_s < L and new_s > sum_w:
                    indices.remove(j)
                    in_set.remove(j)
                    indices.append(i)
                    in_set.add(i)
                    sum_w = new_s
                    improved = True
                    break
            if improved:
                break
        # Try 2-for-1 swap: remove two, add one
        if not improved and len(indices) >= 2:
            for j1 in list(indices):
                wj1 = WEIGHTS[j1]
                for j2 in list(indices):
                    if j2 == j1:
                        continue
                    wj2 = WEIGHTS[j2]
                    for i in range(N):
                        if i in in_set:
                            continue
                        wi = WEIGHTS[i]
                        new_s = sum_w - wj1 - wj2 + wi
                        if new_s < L and new_s > sum_w:
                            indices.remove(j1)
                            in_set.remove(j1)
                            indices.remove(j2)
                            in_set.remove(j2)
                            indices.append(i)
                            in_set.add(i)
                            sum_w = new_s
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        # Try 1-for-2 swap: remove one, add two
        if not improved:
            for j in list(indices):
                wj = WEIGHTS[j]
                for i1 in range(N):
                    if i1 in in_set:
                        continue
                    wi1 = WEIGHTS[i1]
                    for i2 in range(N):
                        if i2 in in_set or i2 == i1:
                            continue
                        wi2 = WEIGHTS[i2]
                        new_s = sum_w - wj + wi1 + wi2
                        if new_s < L and new_s > sum_w:
                            indices.remove(j)
                            in_set.remove(j)
                            indices.append(i1)
                            in_set.add(i1)
                            indices.append(i2)
                            in_set.add(i2)
                            sum_w = new_s
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    return sum_w, indices

def _random_restart(deadline):
    """Generate a random feasible solution."""
    # Start with a random subset
    while time.time() < deadline:
        k = random.randint(3, 20)
        candidates = random.sample(range(N), k)
        s = 0
        valid = True
        for i in candidates:
            s += WEIGHTS[i]
            if s >= L:
                valid = False
                break
        if valid and s > 0:
            return s, candidates
    return 0, [0]  # fallback

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Initial greedy solution
    best_sum_w, best_indices = _greedy_init()
    
    # Local search from greedy
    best_sum_w, best_indices = _local_search(best_sum_w, best_indices, deadline)
    
    # Multiple random restarts with local search
    rng = random.Random(42)
    restarts = 0
    while time.time() < deadline - 2.0:
        restarts += 1
        s, idxs = _random_restart(deadline)
        if s == 0:
            continue
        s, idxs = _local_search(s, idxs, deadline)
        if s > best_sum_w:
            best_sum_w = s
            best_indices = idxs
            # Try to further improve with more aggressive local search
            best_sum_w, best_indices = _local_search(best_sum_w, best_indices, deadline)
    
    # Final local search polish
    best_sum_w, best_indices = _local_search(best_sum_w, best_indices, deadline)
    
    # Fallback
    if not best_indices:
        best_indices = [0]
    
    return best_indices
# EVOLVE-BLOCK-END
