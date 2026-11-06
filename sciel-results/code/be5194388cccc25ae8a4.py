import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)  # 560

    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute K(2,2,2) configurations
    # A K(2,2,2) is determined by 3 disjoint pairs, and the 8 transversal triples.
    k222_list = []
    triple_to_k222s = [[] for _ in range(num_triples)]
    k222_to_triples = []  # list of sets of triple indices

    for points in combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = points
        # All ways to partition 6 points into 3 pairs
        pairings = [
            ((p0,p1),(p2,p3),(p4,p5)),
            ((p0,p1),(p2,p4),(p3,p5)),
            ((p0,p1),(p2,p5),(p3,p4)),
            ((p0,p2),(p1,p3),(p4,p5)),
            ((p0,p2),(p1,p4),(p3,p5)),
            ((p0,p2),(p1,p5),(p3,p4)),
            ((p0,p3),(p1,p2),(p4,p5)),
            ((p0,p3),(p1,p4),(p2,p5)),
            ((p0,p3),(p1,p5),(p2,p4)),
            ((p0,p4),(p1,p2),(p3,p5)),
            ((p0,p4),(p1,p3),(p2,p5)),
            ((p0,p4),(p1,p5),(p2,p3)),
            ((p0,p5),(p1,p2),(p3,p4)),
            ((p0,p5),(p1,p3),(p2,p4)),
            ((p0,p5),(p1,p4),(p2,p3)),
        ]

        for pair1, pair2, pair3 in pairings:
            transversals = []
            for x in pair1:
                for y in pair2:
                    for z in pair3:
                        transversals.append(tuple(sorted((x, y, z))))

            trans_indices = [triple_to_idx[t] for t in transversals]
            trans_set = set(trans_indices)

            k_idx = len(k222_list)
            k222_list.append(trans_set)
            k222_to_triples.append(trans_indices)

            for idx in trans_indices:
                triple_to_k222s[idx].append(k_idx)

    num_k222s = len(k222_list)

    def time_left():
        return deadline - time.time()

    def greedy_init(order=None):
        active = [False] * num_triples
        counts = [0] * num_k222s
        current_size = 0
        if order is None:
            order = list(range(num_triples))
            random.shuffle(order)

        for t_idx in order:
            if time_left() < 5:
                break
            valid = True
            for k_idx in triple_to_k222s[t_idx]:
                if counts[k_idx] == 7:
                    valid = False
                    break
            if valid:
                active[t_idx] = True
                current_size += 1
                for k_idx in triple_to_k222s[t_idx]:
                    counts[k_idx] += 1

        return active, counts, current_size

    def local_search(active, counts, current_size, max_iters=500000):
        """Local search with swap moves. Returns improved solution."""
        no_improve = 0
        max_no_improve = 100000

        # Build list of active and inactive triples for efficient sampling
        # We'll use random sampling with retries

        for iteration in range(max_iters):
            if time_left() < 5:
                break
            if no_improve > max_no_improve:
                break

            # Try a swap: remove one active, add one inactive
            # Pick random active triple
            t_remove = -1
            for _ in range(30):
                t_remove = random.randint(0, num_triples - 1)
                if active[t_remove]:
                    break
            if t_remove == -1:
                continue

            # Pick random inactive triple
            t_add = -1
            for _ in range(30):
                t_add = random.randint(0, num_triples - 1)
                if not active[t_add]:
                    break
            if t_add == -1:
                continue

            # Check if swap is valid: after removing t_remove, can we add t_add?
            # Temporarily remove t_remove
            for k_idx in triple_to_k222s[t_remove]:
                counts[k_idx] -= 1

            valid_add = True
            for k_idx in triple_to_k222s[t_add]:
                if counts[k_idx] == 7:
                    valid_add = False
                    break

            if valid_add:
                # Perform the swap (size stays same)
                active[t_remove] = False
                active[t_add] = True
                for k_idx in triple_to_k222s[t_add]:
                    counts[k_idx] += 1
                # Size unchanged, but structure changed - this helps escape local optima
            else:
                # Restore
                for k_idx in triple_to_k222s[t_remove]:
                    counts[k_idx] += 1

            # After swaps, try to add more triples
            # Try a few random additions
            for _ in range(5):
                if time_left() < 5:
                    break
                t_idx = random.randint(0, num_triples - 1)
                if not active[t_idx]:
                    valid = True
                    for k_idx in triple_to_k222s[t_idx]:
                        if counts[k_idx] == 7:
                            valid = False
                            break
                    if valid:
                        for k_idx in triple_to_k222s[t_idx]:
                            counts[k_idx] += 1
                        active[t_idx] = True
                        current_size += 1
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1

        return active, counts, current_size

    def perturb(active, counts, current_size, num_remove):
        """Remove random triples to perturb."""
        removed = 0
        for _ in range(num_remove * 3):
            if time_left() < 5:
                break
            if removed >= num_remove:
                break
            t_idx = random.randint(0, num_triples - 1)
            if active[t_idx]:
                for k_idx in triple_to_k222s[t_idx]:
                    counts[k_idx] -= 1
                active[t_idx] = False
                current_size -= 1
                removed += 1
        return active, counts, current_size

    best_size = 0
    best_active = None
    best_counts = None

    # Multiple greedy inits with different seeds
    for r in range(20):
        if time_left() < 20:
            break

        active, counts, current_size = greedy_init()

        if current_size > best_size:
            best_size = current_size
            best_active = active[:]
            best_counts = counts[:]

        # Local search from this solution
        active, counts, current_size = local_search(active, counts, current_size, max_iters=200000)

        if current_size > best_size:
            best_size = current_size
            best_active = active[:]
            best_counts = counts[:]

    # Perturbation-based restarts from best
    for r in range(10):
        if time_left() < 20:
            break

        active = best_active[:]
        counts = best_counts[:]
        current_size = best_size

        # Perturb
        num_remove = random.randint(10, 30)
        active, counts, current_size = perturb(active, counts, current_size, num_remove)

        # Greedy fill
        order = list(range(num_triples))
        random.shuffle(order)
        for t_idx in order:
            if time_left() < 5:
                break
            if not active[t_idx]:
                valid = True
                for k_idx in triple_to_k222s[t_idx]:
                    if counts[k_idx] == 7:
                        valid = False
                        break
                if valid:
                    for k_idx in triple_to_k222s[t_idx]:
                        counts[k_idx] += 1
                    active[t_idx] = True
                    current_size += 1

        # Local search
        active, counts, current_size = local_search(active, counts, current_size, max_iters=300000)

        if current_size > best_size:
            best_size = current_size
            best_active = active[:]
            best_counts = counts[:]

    # Final local search on best
    if best_active is not None and time_left() > 10:
        best_active, best_counts, best_size = local_search(
            best_active[:], best_counts[:], best_size, max_iters=500000
        )

    # Construct result
    if best_active is None:
        active, counts, current_size = greedy_init()
        best_active = active

    result = []
    for i in range(num_triples):
        if best_active[i]:
            result.append(list(all_triples[i]))

    return result

if __name__ == "__main__":
    result = construct()
    print(f"Number of triples: {len(result)}")
