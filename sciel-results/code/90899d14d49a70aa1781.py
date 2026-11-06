import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115  # safety margin under 120s

    n = 16
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_id = {t: i for i, t in enumerate(triples)}

    # Precompute all K(2,2,2) configurations: each is a set of 8 triples
    # on 6 points partitioned into 3 pairs
    # For 6 points, number of ways to partition into 3 unordered pairs = 15
    # Number of 6-subsets = C(16,6) = 8008
    # Total K(2,2,2)s = 8008 * 15 = 120120

    # For each 6-subset, generate all 15 pairings
    def get_pairings(v):
        """Generate all 15 ways to partition 6 elements into 3 unordered pairs."""
        v = list(v)
        pairings = []
        # Fix v[0] paired with one of the other 5
        for i in range(1, 6):
            p1 = (v[0], v[i])
            remaining = [v[j] for j in range(1, 6) if j != i]
            # Pair remaining[0] with one of remaining[1:4]
            for j in range(1, 4):
                p2 = (remaining[0], remaining[j])
                p3 = (remaining[1 if j != 1 else 2], remaining[3 if j != 1 else 2])
                # Ensure p3 is canonical
                if p3[0] > p3[1]:
                    p3 = (p3[1], p3[0])
                pairings.append((p1, p2, p3))
        return pairings

    # Build K(2,2,2) list and reverse mapping
    k222_triples = []  # list of lists of 8 triple ids
    triple_to_k222s = [[] for _ in range(num_triples)]

    for six_subset in combinations(range(n), 6):
        for pairing in get_pairings(six_subset):
            t_ids = []
            for x in pairing[0]:
                for y in pairing[1]:
                    for z in pairing[2]:
                        t = tuple(sorted((x, y, z)))
                        t_id = triple_to_id[t]
                        t_ids.append(t_id)
            k_idx = len(k222_triples)
            k222_triples.append(t_ids)
            for t_id in t_ids:
                triple_to_k222s[t_id].append(k_idx)

    num_k222s = len(k222_triples)

    # Solver class for efficient add/remove with K(2,2,2) tracking
    class Solver:
        __slots__ = ['present', 'k_counts', 'present_list', 'missing_list', 'blocked']

        def __init__(self):
            self.present = set()
            self.k_counts = [0] * num_k222s
            self.present_list = []
            self.missing_list = list(range(num_triples))
            self.blocked = [False] * num_triples

        def can_add(self, t_id):
            if t_id in self.present:
                return True
            if self.blocked[t_id]:
                return False
            for k_idx in triple_to_k222s[t_id]:
                if self.k_counts[k_idx] == 7:
                    return False
            return True

        def add(self, t_id):
            if not self.can_add(t_id):
                return False
            self.present.add(t_id)
            self.present_list.append(t_id)
            self.blocked[t_id] = False
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] += 1
                if self.k_counts[k_idx] == 7:
                    for other_t in k222_triples[k_idx]:
                        if other_t != t_id and other_t not in self.present:
                            self.blocked[other_t] = True
            return True

        def remove(self, t_id):
            if t_id not in self.present:
                return
            self.present.remove(t_id)
            self.present_list.remove(t_id)
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] -= 1
                if self.k_counts[k_idx] == 6:
                    for other_t in k222_triples[k_idx]:
                        if other_t != t_id and other_t not in self.present:
                            self.blocked[other_t] = False
            self.blocked[t_id] = False

        def size(self):
            return len(self.present)

        def rebuild_missing(self):
            self.missing_list = [t for t in range(num_triples) if t not in self.present]

    best_size = 0
    best_set = set()

    rng = random.Random(42)

    def polish(solver, max_rounds=20):
        """Repeatedly try to add any unblocked missing triple until stuck."""
        for _ in range(max_rounds):
            if time.time() > DEADLINE:
                break
            added_any = False
            new_missing = []
            for t_id in solver.missing_list:
                if time.time() > DEADLINE:
                    break
                if t_id in solver.present:
                    continue
                if not solver.blocked[t_id]:
                    can = True
                    for k_idx in triple_to_k222s[t_id]:
                        if solver.k_counts[k_idx] == 7:
                            can = False
                            break
                    if can:
                        solver.add(t_id)
                        added_any = True
                    else:
                        new_missing.append(t_id)
                else:
                    new_missing.append(t_id)
            if not added_any:
                break
            solver.missing_list = new_missing
        solver.rebuild_missing()
        # Reset blocked flags for missing triples
        for t_id in solver.missing_list:
            solver.blocked[t_id] = False

    def greedy_init(rng, solver):
        """Initialize with greedy in random order."""
        order = list(range(num_triples))
        rng.shuffle(order)
        for t_id in order:
            if time.time() > DEADLINE:
                break
            solver.add(t_id)

    def try_swap(solver, t_out, t_in):
        """Try to remove t_out and add t_in."""
        if t_out not in solver.present or t_in in solver.present:
            return False
        solver.remove(t_out)
        if solver.add(t_in):
            return True
        else:
            solver.add(t_out)
            return False

    def multi_swap(solver, k_remove, rng):
        """Remove k random present triples, try to add k missing ones."""
        if len(solver.present_list) < k_remove:
            return False
        to_remove = rng.sample(solver.present_list, k_remove)
        for t in to_remove:
            solver.remove(t)
        added = 0
        # Try adding from missing list
        for t_in in solver.missing_list:
            if added >= k_remove:
                break
            if t_in in solver.present:
                continue
            if solver.add(t_in):
                added += 1
        if added == k_remove:
            solver.rebuild_missing()
            return True
        else:
            # Revert
            for t in to_remove:
                solver.add(t)
            solver.rebuild_missing()
            return False

    def simulated_annealing(solver, temp, steps, rng):
        """Run SA steps with given temperature."""
        for step in range(steps):
            if time.time() > DEADLINE:
                break
            move_type = rng.random()
            if move_type < 0.4:
                # Single swap
                if not solver.present_list:
                    continue
                t_out = rng.choice(solver.present_list)
                for _ in range(10):
                    t_in = rng.randint(0, num_triples - 1)
                    if t_in not in solver.present:
                        if try_swap(solver, t_out, t_in):
                            break
                        break
            elif move_type < 0.7:
                multi_swap(solver, 2, rng)
            else:
                multi_swap(solver, 3, rng)

    # Main loop: multiple restarts with greedy + polish + SA
    attempts = 0
    while time.time() < DEADLINE:
        solver = Solver()
        greedy_init(rng, solver)
        polish(solver)

        if solver.size() > best_size:
            best_size = solver.size()
            best_set = set(solver.present)

        # If we have time, do some SA to try to improve
        if time.time() < DEADLINE - 5:
            # Simulated annealing with decreasing temperature
            temp = 1.0
            sa_steps = 200
            simulated_annealing(solver, temp, sa_steps, rng)
            polish(solver)

            if solver.size() > best_size:
                best_size = solver.size()
                best_set = set(solver.present)

        attempts += 1
        # Stop if we've found a very good solution
        if best_size >= 520:
            break

    return [list(t) for t in best_set]
