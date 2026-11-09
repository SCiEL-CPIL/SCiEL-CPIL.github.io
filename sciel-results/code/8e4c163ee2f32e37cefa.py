"""Difference triangle set (7,5) – structural change:
   Represent each row as a difference signature (set of 15 pairwise differences).
   Use a genetic algorithm to evolve 7 pairwise-disjoint signatures that are
   geometrically consistent (embeddable into a strictly-increasing row starting at 0).
   This decouples the global uniqueness constraint from local geometric consistency.
"""
import time
import random
import math
import sys
from collections import defaultdict

_START = time.time()
_DEADLINE = _START + 145.0


def _elapsed():
    return time.time() - _START


def _dead():
    return time.time() >= _DEADLINE


def _diffs_of_row(row):
    d = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ai)
    return d


def _scope(rows):
    return max(max(r) for r in rows)


def _valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                if r[j] <= r[i]:
                    return False
        for d in _diffs_of_row(r):
            if d in used:
                return False
            used.add(d)
    return True


class DiffTracker:
    def __init__(self, max_diff):
        self.max_diff = max_diff
        self.used = bytearray(max_diff + 1)
        self.count = 0

    def add(self, diffs):
        for d in diffs:
            if d <= self.max_diff and not self.used[d]:
                self.used[d] = 1
                self.count += 1

    def remove(self, diffs):
        for d in diffs:
            if d <= self.max_diff and self.used[d]:
                self.used[d] = 0
                self.count -= 1

    def check(self, diffs):
        for d in diffs:
            if d <= self.max_diff and self.used[d]:
                return False
        return True


def _row_from_steps(steps, k):
    """Build a row from k steps (differences between consecutive entries)."""
    row = [0]
    for s in steps:
        row.append(row[-1] + s)
    return row


def _row_diffs_from_steps(steps, k):
    """Compute all pairwise differences from steps."""
    row = _row_from_steps(steps, k)
    return _diffs_of_row(row)


def _steps_from_row(row):
    """Extract steps from a row."""
    return [row[i + 1] - row[i] for i in range(len(row) - 1)]


def _greedy_row(tracker, k, max_mark, rnd=None, noise=0.0):
    row = [0]
    for _ in range(k):
        x = row[-1] + 1
        found = False
        while x <= max_mark:
            if noise > 0 and rnd and rnd.random() < noise:
                x += 1
                continue
            new_diffs = {x - a for a in row}
            if tracker.check(new_diffs):
                row.append(x)
                tracker.add(new_diffs)
                found = True
                break
            x += 1
        if not found:
            return None
    return row


def _interleaved_greedy(n_rows, k, max_mark, rnd=None):
    tracker = DiffTracker(max_mark + 1)
    rows = [[0] for _ in range(n_rows)]
    for _ in range(n_rows * k):
        candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
        if not candidates:
            break
        i = min(candidates, key=lambda idx: rows[idx][-1])
        row = rows[i]
        x = row[-1] + 1
        found = False
        while x <= max_mark:
            new_diffs = {x - a for a in row}
            if tracker.check(new_diffs):
                row.append(x)
                tracker.add(new_diffs)
                found = True
                break
            x += 1
        if not found:
            return None, None
    if all(len(r) == k + 1 for r in rows):
        return rows, tracker
    return None, None


def _signature_valid(steps, k, used_diffs):
    """Check if a set of steps produces a valid row with no conflicts."""
    diffs = _row_diffs_from_steps(steps, k)
    for d in diffs:
        if d in used_diffs:
            return False
    return True


def _ga_evolve(n_rows, k, max_mark, rnd, time_budget):
    """Genetic algorithm to evolve 7 pairwise-disjoint difference signatures."""
    deadline = time.time() + time_budget
    pop_size = 20
    best_rows = None
    best_scope = float('inf')
    best_fitness = float('-inf')

    # Initialize population with random step vectors
    population = []
    for _ in range(pop_size):
        rows = []
        used = set()
        valid = True
        for ri in range(n_rows):
            # Random steps between 1 and max_mark // (k * n_rows)
            max_step = max(1, max_mark // (k * n_rows * 2))
            steps = [rnd.randint(1, max_step) for _ in range(k)]
            diffs = _row_diffs_from_steps(steps, k)
            if diffs & used:
                valid = False
                break
            used |= diffs
            rows.append(_row_from_steps(steps, k))
        if valid:
            population.append(rows)

    # If initial population is empty, fill with greedy solutions
    if not population:
        for _ in range(pop_size):
            rows, _ = _interleaved_greedy(n_rows, k, max_mark, rnd)
            if rows:
                population.append(rows)
        if not population:
            return None, float('inf')

    # Evaluate initial population
    def fitness(rows):
        if not rows or len(rows) != n_rows:
            return float('-inf')
        # Check validity
        used = set()
        for r in rows:
            if len(r) != k + 1 or r[0] != 0:
                return float('-inf')
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    if r[j] <= r[i]:
                        return float('-inf')
            diffs = _diffs_of_row(r)
            if diffs & used:
                return float('-inf')
            used |= diffs
        scope = _scope(rows)
        # Lower scope is better; add small penalty for larger scope
        return -scope

    # Main GA loop
    gen = 0
    while time.time() < deadline:
        # Evaluate population
        scored = [(fitness(ind), ind) for ind in population]
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Update best
        if scored[0][0] > best_fitness:
            best_fitness = scored[0][0]
            best_scope = -best_fitness
            best_rows = [r[:] for r in scored[0][1]]

        if _dead():
            break

        # Create next generation
        new_pop = []
        
        # Elitism: keep top 20%
        elite_count = max(1, pop_size // 5)
        for i in range(elite_count):
            new_pop.append([r[:] for r in scored[i][1]])

        # Crossover and mutation to fill rest
        while len(new_pop) < pop_size:
            if _dead():
                break
            # Tournament selection
            def select():
                idx1 = rnd.randint(0, len(scored) - 1)
                idx2 = rnd.randint(0, len(scored) - 1)
                idx3 = rnd.randint(0, len(scored) - 1)
                best_idx = max(idx1, idx2, idx3, key=lambda i: scored[i][0])
                return scored[best_idx][1]

            parent1 = select()
            parent2 = select()

            # Crossover: mix rows from two parents
            child = []
            used = set()
            valid = True
            for ri in range(n_rows):
                if rnd.random() < 0.5:
                    row = parent1[ri][:]
                else:
                    row = parent2[ri][:]
                # Mutation: perturb one entry
                if rnd.random() < 0.3:
                    pos = rnd.randint(1, len(row) - 1)
                    delta = rnd.randint(-3, 3)
                    new_val = row[pos] + delta
                    if new_val > row[pos - 1] and (pos == len(row) - 1 or new_val < row[pos + 1]):
                        row[pos] = new_val
                # Check validity
                diffs = _diffs_of_row(row)
                if diffs & used:
                    valid = False
                    break
                used |= diffs
                child.append(row)
            
            if valid:
                new_pop.append(child)
            else:
                # Fallback: use a greedy row
                tracker = DiffTracker(max_mark + 1)
                for r in child:
                    tracker.add(_diffs_of_row(r))
                row = _greedy_row(tracker, k, max_mark, rnd, noise=0.1)
                if row:
                    child.append(row)
                    new_pop.append(child)
                else:
                    new_pop.append([r[:] for r in parent1])

        population = new_pop
        gen += 1

    return best_rows, best_scope


def _targeted_reduction(rows, n_rows, k, time_budget):
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    deadline = time.time() + time_budget
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for ri in range(n_rows):
            row = best[ri]
            for pi in range(len(row) - 1, 0, -1):
                old_val = row[pi]
                other_diffs = set()
                for r2 in best:
                    if r2 is not row:
                        other_diffs |= _diffs_of_row(r2)
                row_diffs_other = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        if i != pi and j != pi:
                            row_diffs_other.add(row[j] - row[i])
                forbidden = other_diffs | row_diffs_other
                for new_val in range(old_val - 1, row[pi - 1], -1):
                    if pi < len(row) - 1 and new_val >= row[pi + 1]:
                        continue
                    new_diffs_pi = set()
                    for j in range(len(row)):
                        if j != pi:
                            if j < pi:
                                new_diffs_pi.add(new_val - row[j])
                            else:
                                new_diffs_pi.add(row[j] - new_val)
                    if new_diffs_pi & forbidden:
                        continue
                    row[pi] = new_val
                    new_scope = _scope(best)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    improved = True
                    break
    return best, best_scope


def _iterated_local_search(rows, n_rows, k, max_mark, time_budget, rnd):
    """Simulated annealing with row-rebuild and entry-shift moves."""
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    tracker = DiffTracker(max_mark + 1)
    for r in best:
        tracker.add(_diffs_of_row(r))

    deadline = time.time() + time_budget
    T = 15.0
    cooling = 0.9995
    min_T = 0.05

    while time.time() < deadline and T > min_T:
        ri = rnd.randint(0, n_rows - 1)
        old_row = best[ri]
        old_diffs = _diffs_of_row(old_row)

        if rnd.random() < 0.65:
            tracker.remove(old_diffs)
            noise = 0.15 + 0.35 * (T / 15.0)
            new_row = _greedy_row(tracker, k, max_mark, rnd, noise=noise)
            if new_row is None:
                tracker.add(old_diffs)
                T *= cooling
                continue
            new_diffs = _diffs_of_row(new_row)
            new_scope = max(max(r2) for r2 in best)
            new_scope = max(new_scope, max(new_row))
            delta = new_scope - best_scope
            if delta <= 0 or rnd.random() < math.exp(-max(0, delta) / T):
                best[ri] = new_row
                tracker.add(new_diffs)
                if new_scope < best_scope:
                    best_scope = new_scope
                    T = 15.0
            else:
                tracker.add(old_diffs)
        else:
            pi = rnd.randint(1, len(old_row) - 1)
            old_val = old_row[pi]
            moved = False
            for offset in [-3, -2, -1, 1, 2, 3]:
                new_val = old_val + offset
                if new_val <= old_row[pi - 1]:
                    continue
                if pi < len(old_row) - 1 and new_val >= old_row[pi + 1]:
                    continue
                new_row = old_row[:]
                new_row[pi] = new_val
                new_diffs = _diffs_of_row(new_row)
                tracker.remove(old_diffs)
                conflict = False
                for d in new_diffs:
                    if d <= max_mark and tracker.used[d]:
                        conflict = True
                        break
                if not conflict:
                    new_scope = max(max(r2) for r2 in best)
                    new_scope = max(new_scope, max(new_row))
                    delta = new_scope - best_scope
                    if delta <= 0 or rnd.random() < math.exp(-max(0, delta) / T):
                        best[ri] = new_row
                        tracker.add(new_diffs)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            T = 15.0
                        moved = True
                        break
                    else:
                        tracker.add(old_diffs)
                        break
                else:
                    tracker.add(old_diffs)
                    break
            if not moved:
                T *= cooling
        T *= cooling

    return best, best_scope


def construct(n_rows, k):
    rnd = random.Random(42)
    best_rows = None
    best_scope = float('inf')

    # Phase 1: Multi-start interleaved greedy to get initial solutions
    for _ in range(20):
        if _dead():
            break
        rows, _ = _interleaved_greedy(n_rows, k, 200, rnd)
        if rows is not None:
            sc = _scope(rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = [r[:] for r in rows]

    if _dead():
        return best_rows

    # Phase 2: Genetic algorithm to evolve better solutions
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 20.0
        if remaining > 5.0:
            ga_rows, ga_scope = _ga_evolve(n_rows, k, best_scope, rnd, min(remaining, 60.0))
            if ga_rows is not None and ga_scope < best_scope:
                best_scope = ga_scope
                best_rows = ga_rows

    if _dead():
        return best_rows

    # Phase 3: Targeted reduction
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 10.0
        if remaining > 2.0:
            best_rows, best_scope = _targeted_reduction(best_rows, n_rows, k, remaining)

    if _dead():
        return best_rows

    # Phase 4: ILS with row-rebuild + entry-shift
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 15.0
        if remaining > 3.0:
            best_rows, best_scope = _iterated_local_search(
                best_rows, n_rows, k, best_scope, remaining, rnd
            )

    if _dead():
        return best_rows

    # Phase 5: Another targeted reduction pass
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 8.0
        if remaining > 1.5:
            best_rows, best_scope = _targeted_reduction(best_rows, n_rows, k, remaining)

    if _dead():
        return best_rows

    # Phase 6: Final ILS polish
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 3.0
        if remaining > 1.0:
            best_rows, best_scope = _iterated_local_search(
                best_rows, n_rows, k, best_scope, remaining, rnd
            )

    if _dead():
        return best_rows

    # Final validation
    if best_rows is not None and not _valid(best_rows, n_rows, k):
        rows, _ = _interleaved_greedy(n_rows, k, 200)
        if rows is not None:
            best_rows = rows
            best_scope = _scope(rows)

    return best_rows
