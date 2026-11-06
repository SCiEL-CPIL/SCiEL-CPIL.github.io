# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Construct two nonempty Sidon subsets A, B of [1, 256] with at most 64 elements each.
    Pair sums inside each set (including a+a) must be unique.
    Nonzero difference sets of A and B must be disjoint.
    Maximize |A| + |B|.
    A and B may overlap.
    """
    start_time = time.time()
    deadline = start_time + 55  # 55 seconds safety margin within 60s limit

    n = 256
    candidates = list(range(1, 257))

    class State:
        def __init__(self):
            self.A = []
            self.B = []
            self.sums_A = set()
            self.sums_B = set()
            self.diffs_A = set()
            self.diffs_B = set()
            self.set_A = set()
            self.set_B = set()

        def size(self):
            return len(self.A) + len(self.B)

        def copy(self):
            new_state = State()
            new_state.A = self.A[:]
            new_state.B = self.B[:]
            new_state.sums_A = self.sums_A.copy()
            new_state.sums_B = self.sums_B.copy()
            new_state.diffs_A = self.diffs_A.copy()
            new_state.diffs_B = self.diffs_B.copy()
            new_state.set_A = self.set_A.copy()
            new_state.set_B = self.set_B.copy()
            return new_state

    def can_add_to_A(state, x):
        if x in state.set_A:
            return False
        # Check differences: |a - x| for a in A must not be in diffs_B
        for a in state.A:
            d = abs(a - x)
            if d > 0 and d in state.diffs_B:
                return False
        # Check sums
        for a in state.A:
            s = a + x
            if s in state.sums_A:
                return False
        if (x + x) in state.sums_A:
            return False
        return True

    def can_add_to_B(state, x):
        if x in state.set_B:
            return False
        # Check differences: |b - x| for b in B must not be in diffs_A
        for b in state.B:
            d = abs(b - x)
            if d > 0 and d in state.diffs_A:
                return False
        # Check sums
        for b in state.B:
            s = b + x
            if s in state.sums_B:
                return False
        if (x + x) in state.sums_B:
            return False
        return True

    def add_to_A(state, x):
        state.A.append(x)
        state.set_A.add(x)
        state.sums_A.add(x + x)
        for a in state.A[:-1]:
            state.sums_A.add(a + x)
            d = abs(a - x)
            if d > 0:
                state.diffs_A.add(d)

    def add_to_B(state, x):
        state.B.append(x)
        state.set_B.add(x)
        state.sums_B.add(x + x)
        for b in state.B[:-1]:
            state.sums_B.add(b + x)
            d = abs(b - x)
            if d > 0:
                state.diffs_B.add(d)

    def remove_from_A(state, x):
        state.A.remove(x)
        state.set_A.remove(x)
        new_sums_A = set()
        new_diffs_A = set()
        for i in range(len(state.A)):
            for j in range(i, len(state.A)):
                new_sums_A.add(state.A[i] + state.A[j])
                if i != j:
                    new_diffs_A.add(abs(state.A[i] - state.A[j]))
        state.sums_A = new_sums_A
        state.diffs_A = new_diffs_A

    def remove_from_B(state, x):
        state.B.remove(x)
        state.set_B.remove(x)
        new_sums_B = set()
        new_diffs_B = set()
        for i in range(len(state.B)):
            for j in range(i, len(state.B)):
                new_sums_B.add(state.B[i] + state.B[j])
                if i != j:
                    new_diffs_B.add(abs(state.B[i] - state.B[j]))
        state.sums_B = new_sums_B
        state.diffs_B = new_diffs_B

    def local_improve(state, max_time):
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for x in candidates:
                if time.time() > max_time:
                    break
                if x in state.set_A:
                    pass  # Already in A, skip
                elif x in state.set_B:
                    pass  # Already in B, skip
                else:
                    if can_add_to_A(state, x):
                        add_to_A(state, x)
                        improved = True
                    elif can_add_to_B(state, x):
                        add_to_B(state, x)
                        improved = True
        return state

    def greedy_construct(prefer_A_first=True):
        state = State()
        order = list(range(1, 257))
        random.shuffle(order)

        for x in order:
            if prefer_A_first:
                if can_add_to_A(state, x):
                    add_to_A(state, x)
                elif can_add_to_B(state, x):
                    add_to_B(state, x)
            else:
                if can_add_to_B(state, x):
                    add_to_B(state, x)
                elif can_add_to_A(state, x):
                    add_to_A(state, x)

        local_improve(state, deadline)
        return state

    def beam_search(num_beams=50, max_time=None):
        """Beam search exploring multiple partial solutions simultaneously."""
        if max_time is None:
            max_time = deadline
        beam = [State() for _ in range(num_beams)]
        # Initialize with first element in each beam
        for i in range(num_beams):
            x = random.choice(candidates)
            add_to_A(beam[i], x)

        for step in range(1, 128):  # Try to build up to 128 total elements
            if time.time() > max_time:
                break
            new_beam = []
            for state in beam:
                # Generate candidates for this state
                candidates_to_try = [x for x in candidates if x not in state.set_A and x not in state.set_B]
                if not candidates_to_try:
                    new_beam.append(state.copy())
                    continue

                # Try adding each candidate to A or B
                best_child = None
                best_child_size = state.size()
                for x in candidates_to_try[:20]:  # Limit candidates per state for speed
                    if time.time() > max_time:
                        break
                    # Try adding to A
                    child_a = state.copy()
                    if can_add_to_A(child_a, x):
                        add_to_A(child_a, x)
                        if child_a.size() > best_child_size:
                            best_child = child_a
                            best_child_size = child_a.size()
                    # Try adding to B
                    child_b = state.copy()
                    if can_add_to_B(child_b, x):
                        add_to_B(child_b, x)
                        if child_b.size() > best_child_size:
                            best_child = child_b
                            best_child_size = child_b.size()

                if best_child is None:
                    new_beam.append(state.copy())
                else:
                    new_beam.append(best_child)

            # Keep top num_beams states by size
            new_beam.sort(key=lambda s: s.size(), reverse=True)
            beam = new_beam[:num_beams]

            # Early termination if no improvement possible
            if all(s.size() == beam[0].size() for s in beam):
                break

        # Local improve the best beam state
        best_state = beam[0].copy()
        local_improve(best_state, max_time)
        return best_state

    best_state = None
    best_size = 0

    # Initial greedy runs
    random.seed(42)
    for _ in range(10):
        state = greedy_construct(prefer_A_first=True)
        if state.size() > best_size:
            best_size = state.size()
            best_state = state.copy()

    for _ in range(10):
        state = greedy_construct(prefer_A_first=False)
        if state.size() > best_size:
            best_size = state.size()
            best_state = state.copy()

    # Beam search
    for _ in range(5):
        state = beam_search(num_beams=30, max_time=deadline)
        if state.size() > best_size:
            best_size = state.size()
            best_state = state.copy()

    # Simulated Annealing / Stochastic Local Search
    current_state = best_state.copy()
    current_size = current_state.size()

    # Temperature schedule
    T = 1.0
    T_min = 0.01
    cooling = 0.999

    while time.time() < deadline:
        # Perturbation: Remove k random elements
        k = random.randint(2, 5)
        perturbed_state = current_state.copy()

        for _ in range(k):
            if not perturbed_state.A and not perturbed_state.B:
                break
            if perturbed_state.A and (not perturbed_state.B or random.random() < 0.5):
                x = random.choice(perturbed_state.A)
                remove_from_A(perturbed_state, x)
            else:
                x = random.choice(perturbed_state.B)
                remove_from_B(perturbed_state, x)

        # Local improvement
        local_improve(perturbed_state, deadline)

        new_size = perturbed_state.size()

        # Acceptance criterion
        if new_size > current_size:
            current_state = perturbed_state
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_state = current_state.copy()
        else:
            # Accept worse solution with probability exp(-(delta)/T)
            delta = current_size - new_size
            if delta == 0:
                accept = True
            else:
                prob = pow(2.71828, -delta / T)
                accept = random.random() < prob
            if accept:
                current_state = perturbed_state
                current_size = new_size

        T *= cooling
        if T < T_min:
            T = T_min
            # Restart temperature occasionally to escape deep local minima
            if random.random() < 0.1:
                T = 0.5

    # Final polish: try to add any remaining elements
    local_improve(best_state, deadline)

    # Ensure nonempty
    if not best_state.A:
        best_state.A = [1]
    if not best_state.B:
        best_state.B = [2]

    return [best_state.A, best_state.B]
# EVOLVE-BLOCK-END
