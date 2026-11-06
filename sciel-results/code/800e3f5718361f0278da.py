import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 118

    n = 16
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_id = {t: i for i, t in enumerate(triples)}

    # Build K(2,2,2) conflict structure
    k222_triples = []
    triple_to_k222s = [[] for _ in range(num_triples)]

    def get_partitions(vertices):
        v = list(vertices)
        partitions = []
        for i in range(1, 6):
            pair1 = (v[0], v[i])
            remaining = [v[j] for j in range(1, 6) if j != i]
            for j in range(1, 4):
                pair2 = (remaining[0], remaining[j])
                if j == 1:
                    pair3 = (remaining[2], remaining[3])
                elif j == 2:
                    pair3 = (remaining[1], remaining[3])
                else:
                    pair3 = (remaining[1], remaining[2])
                partitions.append([pair1, pair2, pair3])
        return partitions

    for six_subset in combinations(range(n), 6):
        for partition in get_partitions(six_subset):
            t_ids = []
            for x in partition[0]:
                for y in partition[1]:
                    for z in partition[2]:
                        t = tuple(sorted((x, y, z)))
                        t_id = triple_to_id[t]
                        t_ids.append(t_id)

            k_idx = len(k222_triples)
            k222_triples.append(t_ids)

            for t_id in t_ids:
                triple_to_k222s[t_id].append(k_idx)

    num_k222s = len(k222_triples)
    k222_to_triples = [set(t_ids) for t_ids in k222_triples]

    # Build conflict graph: two triples conflict if they belong to the same K(2,2,2)
    # and adding both would complete a K(2,2,2) (i.e., 7 of 8 already present)
    # Actually, the constraint is: no K(2,2,2) can have all 8 triples present.
    # So we can have up to 7 from any K(2,2,2).
    # Conflict: triple t1 conflicts with t2 if there exists a K(2,2,2) containing both
    # where the other 6 are already in the set. But for the independent set formulation,
    # we need: a set of triples is valid iff no K(2,2,2) has all 8 present.
    # This is NOT a simple independent set problem. It's a hypergraph independent set.
    # We'll use local search.

    triple_neighbors = [set() for _ in range(num_triples)]
    for k_idx, t_ids in enumerate(k222_triples):
        t_list = list(t_ids)
        for i in range(len(t_list)):
            for j in range(i+1, 8):
                triple_neighbors[t_list[i]].add(t_list[j])
                triple_neighbors[t_list[j]].add(t_list[i])

    class Solver:
        def __init__(self):
            self.present = set()
            self.k_counts = [0] * num_k222s
            self.block_count = [0] * num_triples
            self.unblocked_missing = set()

        def init_from_set(self, present_set):
            self.present = set(present_set)
            self.k_counts = [0] * num_k222s
            for t_id in self.present:
                for k_idx in triple_to_k222s[t_id]:
                    self.k_counts[k_idx] += 1
            self.block_count = [0] * num_triples
            self.unblocked_missing = set()
            for t_id in range(num_triples):
                if t_id in self.present:
                    continue
                bc = 0
                for k_idx in triple_to_k222s[t_id]:
                    if self.k_counts[k_idx] == 7:
                        bc += 1
                self.block_count[t_id] = bc
                if bc == 0:
                    self.unblocked_missing.add(t_id)

        def can_add(self, t_id):
            if t_id in self.present:
                return True
            for k_idx in triple_to_k222s[t_id]:
                if self.k_counts[k_idx] == 7:
                    return False
            return True

        def add(self, t_id):
            if not self.can_add(t_id):
                return False
            self.present.add(t_id)
            self.unblocked_missing.discard(t_id)
            for k_idx in triple_to_k222s[t_id]:
                old_count = self.k_counts[k_idx]
                self.k_counts[k_idx] = old_count + 1
                if old_count == 6:
                    for other_t in k222_to_triples[k_idx]:
                        if other_t != t_id and other_t not in self.present:
                            self.block_count[other_t] += 1
                            if self.block_count[other_t] == 1:
                                self.unblocked_missing.discard(other_t)
            return True

        def remove(self, t_id):
            if t_id not in self.present:
                return
            self.present.discard(t_id)
            for k_idx in triple_to_k222s[t_id]:
                old_count = self.k_counts[k_idx]
                self.k_counts[k_idx] = old_count - 1
                if old_count == 7:
                    for other_t in k222_to_triples[k_idx]:
                        if other_t != t_id and other_t not in self.present:
                            self.block_count[other_t] -= 1
                            if self.block_count[other_t] == 0:
                                self.unblocked_missing.add(other_t)
            bc = 0
            for k_idx in triple_to_k222s[t_id]:
                if self.k_counts[k_idx] == 7:
                    bc += 1
            self.block_count[t_id] = bc
            if bc == 0:
                self.unblocked_missing.add(t_id)
            else:
                self.unblocked_missing.discard(t_id)

        def size(self):
            return len(self.present)

    best_size = 0
    best_set = set()

    def polish(solver, max_iterations=50):
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < DEADLINE:
            improved = False
            iterations += 1
            candidates = list(solver.unblocked_missing)
            random.Random(iterations * 137 + 7).shuffle(candidates)
            for t_id in candidates:
                if t_id in solver.present:
                    continue
                if solver.can_add(t_id):
                    solver.add(t_id)
                    improved = True
        return improved

    def simulated_annealing(solver, rng, iterations=5000):
        """Simulated annealing with adaptive temperature."""
        T = 1.0
        cooling = 0.9995
        best_local = solver.size()
        best_local_set = set(solver.present)

        for it in range(iterations):
            if time.time() > DEADLINE:
                break

            # Pick a triple to remove
            present_list = list(solver.present)
            if not present_list:
                break

            # Prefer tight triples (in K(2,2,2) with count 7)
            tight_triples = []
            for t_id in present_list:
                for k_idx in triple_to_k222s[t_id]:
                    if solver.k_counts[k_idx] == 7:
                        tight_triples.append(t_id)
                        break

            if tight_triples and rng.random() < 0.6:
                t_out = rng.choice(tight_triples)
            else:
                t_out = rng.choice(present_list)

            old_size = solver.size()
            solver.remove(t_out)

            # Try to add a different triple
            candidates = []
            for nb in triple_neighbors[t_out]:
                if nb not in solver.present:
                    if solver.can_add(nb):
                        candidates.append(nb)

            if len(candidates) < 5:
                # Sample some random non-present triples
                non_present_sample = rng.sample(
                    [t for t in range(num_triples) if t not in solver.present],
                    min(20, num_triples - len(solver.present))
                )
                for t_id in non_present_sample:
                    if t_id not in candidates and solver.can_add(t_id):
                        candidates.append(t_id)

            if candidates:
                # Score candidates by how many K(2,2,2)s they participate in
                scored = [(len(triple_to_k222s[t_id]), t_id) for t_id in candidates]
                scored.sort(reverse=True)
                t_in = scored[rng.randint(0, min(5, len(scored))-1)][1]
                if solver.add(t_in):
                    new_size = solver.size()
                    if new_size > best_local:
                        best_local = new_size
                        best_local_set = set(solver.present)
                    # Accept if better, or with probability exp(-delta/T)
                    delta = old_size - new_size
                    if delta < 0 or rng.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                        pass  # accept
                    else:
                        solver.remove(t_in)
                        solver.add(t_out)
                else:
                    solver.add(t_out)
            else:
                solver.add(t_out)

            T *= cooling

        # Restore best local
        if best_local_set != solver.present:
            solver.init_from_set(best_local_set)
        return best_local > solver.size() or True

    def kick_restart(solver, rng, num_kicks=3):
        """Remove several random triples to escape local optima."""
        for _ in range(num_kicks):
            if time.time() > DEADLINE:
                break
            present_list = list(solver.present)
            if not present_list:
                break
            t_out = rng.choice(present_list)
            solver.remove(t_out)

    def multi_swap(solver, rng, k=2):
        """Try to swap k triples out for k better ones."""
        if len(solver.present) < k:
            return False

        present_list = list(solver.present)
        tight_triples = []
        for t_id in present_list:
            for k_idx in triple_to_k222s[t_id]:
                if solver.k_counts[k_idx] == 7:
                    tight_triples.append(t_id)
                    break

        if tight_triples and len(tight_triples) >= k and rng.random() < 0.7:
            outs = rng.sample(tight_triples, k)
        else:
            outs = rng.sample(present_list, k)

        old_size = solver.size()
        for t in outs:
            solver.remove(t)

        candidates = set()
        for t_out in outs:
            for nb in triple_neighbors[t_out]:
                if nb not in solver.present and nb not in outs:
                    if solver.can_add(nb):
                        candidates.add(nb)

        if len(candidates) < k:
            non_present = [t for t in range(num_triples) if t not in solver.present and t not in set(outs)]
            sample = rng.sample(non_present, min(30, len(non_present)))
            for t_id in sample:
                if solver.can_add(t_id):
                    candidates.add(t_id)

        candidates = list(candidates)
        if len(candidates) < k:
            for t in outs:
                solver.add(t)
            return False

        # Try to add k candidates
        added = []
        rng.shuffle(candidates)
        for t_id in candidates:
            if len(added) == k:
                break
            if solver.add(t_id):
                added.append(t_id)

        if len(added) == k:
            return True
        else:
            for t in added:
                solver.remove(t)
            for t in outs:
                solver.add(t)
            return False

    # Main search loop
    num_restarts = 40
    for seed in range(num_restarts):
        if time.time() > DEADLINE:
            break

        rng = random.Random(seed * 7919 + 1)
        solver = Solver()

        # Greedy initialization with random order
        order = list(range(num_triples))
        rng.shuffle(order)
        for t_id in order:
            if time.time() > DEADLINE:
                break
            solver.add(t_id)

        solver.init_from_set(solver.present)
        polish(solver, max_iterations=30)

        if solver.size() > best_size:
            best_size = solver.size()
            best_set = set(solver.present)

        # Simulated annealing phase
        sa_iterations = max(2000, int(20000 / (1 + seed * 0.1)))
        simulated_annealing(solver, rng, iterations=sa_iterations)
        polish(solver, max_iterations=30)

        if solver.size() > best_size:
            best_size = solver.size()
            best_set = set(solver.present)

        # Local improvement with multi-swap
        for _ in range(50):
            if time.time() > DEADLINE:
                break
            if rng.random() < 0.5:
                multi_swap(solver, rng, k=2)
            else:
                multi_swap(solver, rng, k=3)
            polish(solver, max_iterations=10)

            if solver.size() > best_size:
                best_size = solver.size()
                best_set = set(solver.present)

            # Occasional kick
            if rng.random() < 0.2:
                kick_restart(solver, rng, num_kicks=2)
                polish(solver, max_iterations=20)

    # Final polish on best solution
    if best_set:
        final_solver = Solver()
        final_solver.init_from_set(best_set)
        polish(final_solver, max_iterations=200)

        final_rng = random.Random(12345)
        final_deadline = DEADLINE - 2
        while time.time() < final_deadline:
            simulated_annealing(final_solver, final_rng, iterations=2000)
            polish(final_solver, max_iterations=50)

            if final_solver.size() > best_size:
                best_size = final_solver.size()
                best_set = set(final_solver.present)

        polish(final_solver, max_iterations=100)

        if final_solver.size() > best_size:
            best_size = final_solver.size()
            best_set = set(final_solver.present)

    return [list(triples[t_id]) for t_id in best_set]
